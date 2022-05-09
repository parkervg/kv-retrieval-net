from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset


class _KVRETDataset(Dataset):
    def __init__(
        self,
        x,
        y,
        kb,
        kb_valid_vocab_idx,
        kb_tuples,
        tok2id,
        kb_vocab_start,
        scenario_types,
        device: torch.device = torch.device("cpu"),
        padding_strategy: Union[int, str] = "longest",
        reverse_input: bool = False,
        train: bool = True,
    ):
        self.device = device
        self.padding_strategy = padding_strategy
        self.reverse_input = reverse_input
        self.kb_vocab_start = kb_vocab_start
        self.kb_tuples = kb_tuples
        self.scenario_types = scenario_types
        if self.padding_strategy == "longest":
            max_len = max([len(i) for i in y])
            self.max_len = max(max_len, max([len(i) for i in kb]))
        elif isinstance(self.padding_strategy, int):
            self.max_len = self.padding_strategy
        print(f"Padding to {self.max_len}...")
        self.tok2id = tok2id
        self.id2tok = {v: k for k, v in self.tok2id.items()}
        self.x, _ = self.pad_tensors(x)
        self.y, self.input_mask = self.pad_tensors(y)
        self.kb, self.kb_mask, self.kb_vocab_mask = self.prepare_kb(
            kb, kb_valid_vocab_idx
        )
        if not len(self.x) == len(self.y) == len(self.kb):
            raise Exception("Mismatch in lengths between x, y and kb")
        self.train = train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        """
        'Key of an entry corresponds to the sum of the word embeddings of the subject and relation'
        """
        # List of tokenizer outputs
        # where each output contains token ids for subject, relation, object
        item = {}
        item["input"] = (
            self.x[index].__reversed__().to(self.device)
            if self.reverse_input
            else self.x[index].to(self.device)
        )
        item["output"] = self.y[index].to(self.device)
        item["input_mask"] = self.input_mask[index].to(self.device)
        item["input_length"] = torch.count_nonzero(self.input_mask[index]).to(
            self.device
        )
        item["kb"] = self.kb[index].to(self.device)
        item["kb_mask"] = self.kb_mask[index].to(self.device)
        item["kb_vocab_mask"] = self.kb_vocab_mask[index].to(self.device)
        if self.train == False:
            item["kb_tuples"] = self.kb_tuples[index]
            item["scenario_type"] = self.scenario_types[index]
        return item

    def pad_tensors(self, data):
        """
        Pads tensor to max_len with padding token "[PAD]".
        """
        all_padded = []
        all_mask = []
        for item in data:
            padded = []
            mask = []
            for i in range(self.max_len):
                try:
                    padded.append(item[i])
                    mask.append(1)
                except IndexError:
                    padded.append(self.tok2id["[PAD]"])
                    mask.append(0)
            all_padded.append(torch.tensor(padded))
            all_mask.append(torch.tensor(mask))
        return all_padded, all_mask

    def prepare_kb(self, kb, kb_valid_vocab_idx):
        all_padded = []
        all_mask = []
        all_vocab_mask = []
        for index, kb_i in enumerate(kb):
            padded = []
            mask = []
            vocab_mask = torch.zeros(
                (
                    len(
                        self.tok2id,
                    )
                    - self.kb_vocab_start
                ),
                device=self.device,
            )
            for i in range(self.max_len):
                try:
                    padded.append([kb_i[i][0], kb_i[i][1]])
                    mask.append(1)
                    vocab_mask[
                        np.array(kb_valid_vocab_idx[index]) - self.kb_vocab_start
                    ] = 1
                except IndexError:
                    padded.append([self.tok2id["[PAD]"]] * 2)
                    mask.append(0)
            all_padded.append(torch.tensor(padded))
            all_mask.append(torch.tensor(mask))
            all_vocab_mask.append(vocab_mask)
        return all_padded, all_mask, all_vocab_mask

    def get_item_summary(self, index):
        print(
            "\n".join(
                [
                    f"INPUT: \n \t {' '.join([self.id2tok[i.item()] for i in self.x[index] if i.item() != self.tok2id['[PAD]']])}",
                    f"OUTPUT: \n \t {' '.join([self.id2tok[i.item()] for i in self.y[index] if i.item() != self.tok2id['[PAD]']])}\n",
                    f"KB: \n \t: ",
                    " ".join(
                        [
                            f"{self.id2tok[self.kb[index][i, 0].item()]}: {self.id2tok[self.kb[index][i, 1].item()]} |"
                            for i in range(self.kb[index].shape[0])
                            if self.id2tok[self.kb[index][i, 0].item()]
                            != self.tok2id["[PAD]"]
                        ]
                    ),
                    f"KB tokens: \n \t"
                    f"{' | '.join([self.id2tok[idx + self.kb_vocab_start] for idx, i in enumerate(self.kb_vocab_mask[index]) if i.item() == 1])}",
                ]
            )
        )
