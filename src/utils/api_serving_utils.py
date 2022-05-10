import random
import re
from typing import Dict, List, Tuple

import pandas as pd
import torch

from ..model import KVNetwork
from . import utils


def load_model(dataset, device):
    device = torch.device(device)
    # pretrained_weights = utils.get_pretrained_weights(200, dataset.tok2id, glove_dir="./data/glove")
    model = KVNetwork(
        num_vocab=len(dataset.tok2id),
        # Only important things in loading this base model are shapes here
        embed_size=200,
        hidden_size=200,
        num_layers=1,
        dropout=0.3,
        device=torch.device("cpu"),
        padding_idx=dataset.tok2id["[PAD]"],
        kb_vocab_start=dataset.kb_vocab_start,
        attention_type="bahdanau",
        pretrained_weights=None,
    ).to(device)
    return model


def load_state(path, model):
    new_state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(new_state_dict, strict=False)
    print(f"Model loaded from: '{path}'")
    return model


def get_kb_state(dataset, scenario_type: str):
    """
    Returns a random KB state from the test dataset.
    :param dataset:
    :return:
    """
    print(f"Received request for {scenario_type}")
    sep_token = dataset.tok2id["*"]
    valid_items = [
        i for i in dataset if torch.count_nonzero(i.get("input") == sep_token) > 2
    ]
    task_subset = [i for i in valid_items if i.get("scenario_type") == scenario_type]
    test_item_id = random.randint(0, len(task_subset) - 1)  # Since randint is inclusive
    chosen_item = task_subset[test_item_id]
    example_inputs = [
        i.strip()
        for i in re.split(
            "\*",
            utils.ids_to_text(
                dataset.id2tok,
                dataset.tok2id["[EOS]"],
                chosen_item.get("input"),
                reversed=True,
            ),
        )
    ][0::2]
    df = tuples_to_df(chosen_item.get("kb_tuples"))
    chosen_item["kb_mappings"] = get_kb_mappings(chosen_item.get("kb_tuples"))
    chosen_item.pop("output")
    chosen_item.pop("input_mask")
    chosen_item.pop("input_length")
    chosen_item.pop("input")
    # Fixing shapes, with batch_size = 1
    chosen_item["kb_mask"] = chosen_item["kb_mask"].unsqueeze(0)
    chosen_item["kb"] = chosen_item["kb"].unsqueeze(0)
    chosen_item["kb_vocab_mask"] = chosen_item["kb_vocab_mask"].unsqueeze(0)
    return chosen_item, df, example_inputs


def tuples_to_df(tuples: List[Tuple[str, str, str]]):
    columns = ["name"] + list(set(t[1] for t in tuples))
    curr_data = [None for _ in range(len(columns))]
    all_data = []
    for t in tuples:
        k = re.sub("_", " ", t[0])
        if curr_data[0] is None:
            curr_data[0] = k
        curr_data[columns.index(t[1])] = t[2]
        if all(x is not None for x in curr_data):
            all_data.append(curr_data)
            curr_data = [None for _ in range(len(columns))]
    df = pd.DataFrame(all_data, columns=columns)
    return df


def get_kb_mappings(tuples: List[Tuple[str, str, str]]):
    """
    Gets mapping from surface form of subj to underscore form the model is familiar with.
    Also, gets mapping from "subj_rel" key to value, so the model can realize their surface forms
    :param tuples:
    :return:
    """
    kb_mappings = {}
    subj_surface_to_normalized = {}
    canonical_key_to_value = {}
    for t in tuples:
        surface_subj = re.sub("_", " ", t[0])
        subj_surface_to_normalized[surface_subj] = t[0]
        canonical_key_to_value[utils.token_from_kb_tuple(t)] = t[2]
    kb_mappings["subj_surface_to_normalized"] = subj_surface_to_normalized
    # kb_mappings["normalized_to_subj_surface"] = {v:k for k, v in subj_surface_to_normalized.items()}
    kb_mappings["canonical_key_to_value"] = canonical_key_to_value
    # kb_mappings["value_to_canonical_key"] = {v:k for k, v in canonical_key_to_value.items()}
    return kb_mappings


def featurize(toks: List[str], tok2id: Dict[str, int]):
    tok_ids = [tok2id["[SOS]"]]
    for tok in toks:
        try:
            tok_ids.append(tok2id[tok.lower()])
        except KeyError:
            tok_ids.append(tok2id.get("unk"))
    return torch.Tensor(tok_ids).to(int).unsqueeze(0)


def recover_surface_forms(text, kb_mappings):
    """
    Uses mapping dictionary to convert normalized subj to surface form, and
    :param text:
    :param item:
    :return:
    """
    # First, convert canonical subj_rel forms to vals
    for canon_key, val in kb_mappings["canonical_key_to_value"].items():
        text = re.sub(canon_key, val, text)
    # Then, regex rule to replace any remaining underscores with whitespace
    # e.g. p.f._changs -> p.f. changs
    text = re.sub("_", " ", text)
    return text


def canonicalize_input(text, kb_mappings):
    """
    Use mapping dictionary to turn natural text into canonicalized forms for model predictions.
    :param text:
    :param kb_mappings:
    :return:
    """
    for canon_key, val in kb_mappings["canonical_key_to_value"].items():
        text = re.sub(val, canon_key, text)
    for subj_surface, normalized_subj in kb_mappings[
        "subj_surface_to_normalized"
    ].items():
        text = re.sub(subj_surface, normalized_subj, text)
    return text


def get_prediction_json(model: "KVNetwork", text: str, item: Dict, dataset):
    """
    :param model:
    :param text:
    :param item:
    :param dataset:
    :return:
        aggregated_out with canonicalized forms and combination of inputs, and system_output with cleaned surface forms.
    """
    sos_token_id = dataset.tok2id["[SOS]"]
    eos_token_id = dataset.tok2id["[EOS]"]
    toks = utils.tokenize(text)[::-1]
    print(toks[::-1])
    item["input"] = featurize(toks, dataset.tok2id)  # (1, max_seq_len)
    with torch.no_grad():
        outputs = model(item=item, teacher_forcing_ratio=0.0, sos_token_id=sos_token_id)
    pred_ids = outputs.argmax(1).squeeze()
    raw_output = utils.ids_to_text(
        id2tok=dataset.id2tok, eos_token_id=eos_token_id, ids=pred_ids, reversed=False
    )
    print(raw_output)
    system_output = recover_surface_forms(raw_output, item.get("kb_mappings"))
    aggregated_out = f"{text} * {raw_output}"
    return {"output": system_output}, aggregated_out
