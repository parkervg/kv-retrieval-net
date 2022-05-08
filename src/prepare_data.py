import json
import pandas as pd
import re
from typing import Tuple, List, Dict, Union
from attr import attrs, attrib
from transformers import BertTokenizer
import torch
import copy

from .utils.datasets import _KVRETDataset
from .utils import utils

# tuples for drive
tuples_d = [
    ("poi", "address", "val"),
    ("poi", "poi_type", "val"),
    ("poi", "traffic_info", "val"),
    ("poi", "distance", "val"),
]
# tuples for calendar
tuples_c = [
    ("event", "time", "val"),
    ("event", "room", "val"),
    ("event", "party", "val"),
    ("event", "agenda", "val"),
    ("event", "date", "val"),
]
# tuples for weather
tuples_w = [
    ("location", "monday", "val"),
    ("location", "tuesday", "val"),
    ("location", "wednesday", "val"),
    ("location", "thursday", "val"),
    ("location", "friday", "val"),
    ("location", "saturday", "val"),
    ("location", "sunday", "val"),
]
scenario_type_to_tuples = {"poi": tuples_d, "event": tuples_c, "location": tuples_w}


@attrs()
class KVRETDataset:
    train_path: str = attrib()
    dev_path: str = attrib()
    test_path: str = attrib()
    device: torch.device = attrib(default=torch.device("cpu"))
    max_len: Union[int, str] = attrib(default=32)
    reverse_input: bool = attrib(default=False)
    tokenizer_type: str = attrib(default="bert-base-uncased")
    include_context: bool = attrib(default=False)
    kb_vocab_start: int = attrib(init=False)
    train: _KVRETDataset = attrib(init=False)
    dev: _KVRETDataset = attrib(init=False)
    test: _KVRETDataset = attrib(init=False)
    tok2id: Dict[str, int] = attrib(init=False)
    id2tok: Dict[int, str] = attrib(init=False)

    def __attrs_post_init__(self):
        # print(f"Loading {self.tokenizer_type} tokenizer...")
        # self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_type, return_tensors="pt")
        self.train, self.tok2id = self.load_from_json(self.train_path, tok2id=None)
        self.dev, _ = self.load_from_json(self.dev_path, tok2id=self.tok2id)
        self.test, _ = self.load_from_json(self.test_path, tok2id=self.tok2id)
        self.id2tok = {v: k for k, v in self.tok2id.items()}

    def load_from_json(self, json_path: str, tok2id: Dict[str, int] = None):
        df = pd.read_json(json_path)
        kb = self.create_kb_tuples(df=df)
        chats = self.get_chats(df=df)
        chats, kb = self.make_canonical_chats(chats=chats, kb=kb)
        if tok2id is None:
            tok2id, self.kb_vocab_start = self.make_tok2id(kb, chats)
            tok2id["unk"] = len(tok2id)
        kb_as_ids = []
        kb_valid_vocab_idx = []
        for kb_i in kb:
            kb_ids = []
            kb_vocab = []
            for tup in kb_i:
                kb_token = utils.token_from_kb_tuple(tup)
                kb_vocab.append(tok2id.get(kb_token, len(tok2id) - 1))
                kb_ids.append(tuple(tok2id.get(x, len(tok2id) - 1) for x in tup))
            kb_as_ids.append(kb_ids)
            kb_valid_vocab_idx.append(kb_vocab)
        inputs, outputs, kb_as_ids, kb_valid_vocab_idx = self.get_inputs_outputs(
            df,
            chats,
            kb_as_ids,
            kb_valid_vocab_idx,
            tok2id=tok2id,
            include_context=self.include_context,
        )

        return (
            _KVRETDataset(
                x=inputs,
                y=outputs,
                kb=kb_as_ids,
                kb_valid_vocab_idx=kb_valid_vocab_idx,
                device=self.device,
                tok2id=tok2id,
                kb_vocab_start=self.kb_vocab_start,
                padding_strategy=self.max_len,
                reverse_input=self.reverse_input,
            ),
            tok2id,
        )

    def make_tok2id(
        self, kb: List[List[Tuple[str, str, str]]], chats: List[List[str]]
    ) -> Tuple[Dict[str, int], int]:
        chat_vocab = {"[EOS]", "[SOS]", "[PAD]", "*"}
        kb_vocab = set()
        for i in range(len(chats)):
            for j in range(len(chats[i])):
                # Update with chat tokens
                chat_vocab.update(utils.tokenize(chats[i][j]))
                # Update with kb tokens
                for tup in kb[i]:
                    chat_vocab.update(tup)
                    kb_vocab.add(utils.token_from_kb_tuple(tup))
        kb_vocab = sorted(kb_vocab)
        chat_vocab = sorted(
            [i for i in chat_vocab if i not in kb_vocab]
        )  # Sort so this is deterministic
        vocab = {k: idx for idx, k in enumerate(chat_vocab)}
        kb_vocab_start = len(vocab)
        # TODO: What should be done with tokens in both vocab?
        # Should we assume they're the same as the kb representation?
        for i, kb_item in zip(
            range(kb_vocab_start, kb_vocab_start + len(kb_vocab)), kb_vocab
        ):
            assert kb_item not in vocab
            vocab[kb_item] = i
        return (vocab, kb_vocab_start)

    @staticmethod
    def create_kb_tuples(df: pd.DataFrame) -> List[List[Tuple[str, str, str]]]:
        """
        From the paper:
            "We store every entry of our KB using a (subject, relation, object) representation"
        `kb` is a List[List[Tuple[str]]] containing such representations, for each training instance.
        e.g. ('manhattan', 'monday', 'stormy, low of 50F, high of 70F')
        """
        kb = []
        for i in range(len(df)):
            kb_i = []
            if df["scenario"][i]["kb"]["items"]:
                scenario_type = df["scenario"][i]["kb"]["column_names"][
                    0
                ]  # One of "poi", "event", "location"
                assert scenario_type in ["poi", "event", "location"]
                scenario_tuple_frames = scenario_type_to_tuples[scenario_type]
                for j in range(len(df["scenario"][i]["kb"]["items"])):
                    for k in range(len(scenario_tuple_frames)):
                        # e.g. ('manhattan', 'monday', 'stormy, low of 50F, high of 70F')
                        subj = df["scenario"][i]["kb"]["items"][j][
                            scenario_tuple_frames[k][0]
                        ]
                        rel = scenario_tuple_frames[k][1]
                        obj = df["scenario"][i]["kb"]["items"][j][
                            scenario_tuple_frames[k][1]
                        ]
                        kb_i.append((subj.lower(), rel.lower(), obj.lower()))
            kb.append(kb_i)
        return kb

    @staticmethod
    def get_chats(df: pd.DataFrame) -> List[List[str]]:
        chats = []
        for i in range(len(df)):
            chat = []
            for j in range(len(df.iloc[i]["dialogue"])):
                chat.append(
                    str(df.iloc[i]["dialogue"][j]["data"]["utterance"])
                    .strip('"')
                    .lower()
                )
            chats.append(chat)
        return chats

    @staticmethod
    def make_canonical_chats(
        chats: List[List[str]], kb: List[List[Tuple[str, str, str]]]
    ) -> List[List[str]]:
        """
        Replacing values in chats with their canonical representations.
        E.g.
            'the nearest parking garage is dish parking at 550 alester ave. would you like directions there?'
            -->
            'the nearest parking garage is dish parking at Dish_Parking_address. would you like directions there? '
        """
        # Replacing values with their canonical representations
        # TODO: maybe clean this up
        for i, (chat, kb_i) in enumerate(zip(chats, kb)):
            # if chat[0].startswith("find out if it's supposed to rain"):
            #     print()
            for j, _ in enumerate(chat):
                for kb_idx, ki in enumerate(kb_i):
                    # Convert objects to subject_relation form
                    chats[i][j] = re.sub(
                        ki[2],
                        utils.token_from_kb_tuple(ki),
                        chats[i][j],
                        flags=re.IGNORECASE,
                    )
                    # Convert multi-word keys to underscore form
                    # e.g. palo alto cafe -> palo_alto_cafe
                    joined_subj = "_".join(ki[0].split(" "))
                    chats[i][j] = re.sub(
                        ki[0], joined_subj, chats[i][j], flags=re.IGNORECASE
                    )
                    new_tup = list(kb[i][kb_idx])
                    new_tup[0] = joined_subj
            # Now merge kb subjects
            for kb_idx, ki in enumerate(kb_i):
                new_tup = list(kb[i][kb_idx])
                new_tup[0] = "_".join(new_tup[0].split(" "))
                kb[i][kb_idx] = new_tup
        # Ensure there's an even number of turns in each chat
        for i in range(len(chats)):
            if len(chats[i]) % 2 != 0:
                chats[i] = chats[i][:-1]

        return chats, kb

    @staticmethod
    def get_inputs_outputs(
        df: pd.DataFrame,
        chats: List[List[str]],
        kb_as_ids: List[List[Tuple[int, int, int]]],
        valid_kb_idx,
        tok2id: Dict[str, int],
        include_context: bool,
    ) -> Tuple[List[str], List[str], List[List[Tuple[int, int, int]]]]:
        # Separate out user (input) and system (output) dialogues
        inputs = []
        outputs = []
        if include_context:
            for i in range(len(chats)):
                sent = ""
                for j in range(0, len(chats[i]), 2):
                    if len(sent) > 1:
                        sent += " * " + chats[i][j]
                    else:
                        sent += chats[i][j]
                    inputs.append(
                        [tok2id.get(i, len(tok2id) - 1) for i in utils.tokenize(sent)]
                    )
                    outputs.append(
                        [
                            tok2id.get(i, len(tok2id) - 1)
                            for i in utils.tokenize(chats[i][j + 1])
                        ]
                    )
                    sent += " * " + chats[i][j + 1]
                    inputs[-1].insert(0, tok2id["[SOS]"])
                    outputs[-1].insert(0, tok2id["[SOS]"])
                    inputs[-1].append(tok2id["[EOS]"])
                    outputs[-1].append(tok2id["[EOS]"])
        else:  # Don't concatenate consecutive dialogue turns
            for i in range(len(chats)):
                for j in range(len(chats[i][::2])):
                    inputs.append(
                        [
                            tok2id.get(i, len(tok2id) - 1)
                            for i in utils.tokenize(chats[i][::2][j])
                        ]
                    )
                    outputs.append(
                        [
                            tok2id.get(i, len(tok2id) - 1)
                            for i in utils.tokenize(chats[i][1::2][j])
                        ]
                    )
                    inputs[-1].insert(0, tok2id["[SOS]"])
                    outputs[-1].insert(0, tok2id["[SOS]"])
                    inputs[-1].append(tok2id["[EOS]"])
                    outputs[-1].append(tok2id["[EOS]"])
        multiplied_kb = []
        multiplied_kb_idx = []
        for i in range(len(df)):
            kb_copies = (
                (len(df["dialogue"][i]) / 2)
                if len(df["dialogue"][i]) % 2 == 0
                else (len(df["dialogue"][i]) - 1) / 2
            )
            assert kb_copies.is_integer()
            for _ in range(int(kb_copies)):
                multiplied_kb.append(kb_as_ids[i])
                multiplied_kb_idx.append(valid_kb_idx[i])
        return (inputs, outputs, multiplied_kb, multiplied_kb_idx)
