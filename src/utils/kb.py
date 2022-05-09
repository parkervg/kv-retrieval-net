from typing import Dict, List, Tuple

from attr import attrib, attrs

from . import utils


@attrs()
class KBItem:
    scenario_type: str = attrib(default=None)
    tuples: List[Tuple[str, str, str]] = attrib(init=False)

    def __attrs_post_init__(self):
        self.tuples = []

    def add_tuple(self, t: Tuple[str, str, str]):
        self.tuples.append(t)

    def merge_kb_subjects(self):
        # Convert multi-word keys to underscore form
        # e.g. palo alto cafe -> palo_alto_cafe
        for kb_idx, ki in enumerate(self.tuples):
            new_tup = list(ki)
            new_tup[0] = "_".join(new_tup[0].split(" "))
            self.tuples[kb_idx] = new_tup


@attrs()
class KBList(list):
    expanded_kb: List[KBItem] = attrib(init=False)

    def expand_as(self, copy_sizes: List[int]):
        self.expanded_kb = []
        count = 0
        for idx, (num_copies, kb_item) in enumerate(zip(copy_sizes, self)):
            for _ in range(num_copies):
                kb_item.expanded_idx = count
                self.expanded_kb.append(kb_item)
                count += 1

    def featurize(self, tok2id: Dict[str, int]):
        """
        Adds to ids and valid_vocab_idxs fields in each KBItem.
        :param tok2id:
        :return:
        """
        for kb_item in self:
            kb_ids = []
            kb_valid_vocab = []
            for tup in kb_item.tuples:
                kb_token = utils.token_from_kb_tuple(tup)
                kb_valid_vocab.append(tok2id.get(kb_token, len(tok2id) - 1))
                kb_ids.append(tuple(tok2id.get(x, len(tok2id) - 1) for x in tup))
            kb_item.ids = kb_ids
            kb_item.valid_vocab_idxs = kb_valid_vocab

    def get_valid_vocab_idx(self):
        out = []
        for kb_item in self.expanded_kb:
            out.append(kb_item.valid_vocab_idxs)
        return out

    def get_ids(self):
        out = []
        for kb_item in self.expanded_kb:
            out.append(kb_item.ids)
        return out

    def get_tuples(self):
        out = []
        for kb_item in self.expanded_kb:
            out.append(kb_item.tuples)
        return out

    def get_scenario_types(self):
        out = []
        for kb_item in self.expanded_kb:
            out.append(kb_item.scenario_type)
        return out
