import os
import string
from functools import partial
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Tuple

import sacrebleu
import torch
from nltk import word_tokenize
from tqdm import tqdm


def load_glove_embeddings(
    filepath: str, embed_dim: int
) -> Tuple[torch.tensor, List[str]]:
    print("Loading Glove...")

    def get_num_lines(f):
        """take a peek through file handle `f` for the total number of lines"""
        num_lines = sum(1 for _ in f)
        f.seek(0)
        return num_lines

    itos = []
    with open(filepath, "r") as f:
        num_lines = get_num_lines(f)
        vectors = torch.zeros(num_lines, embed_dim, dtype=torch.float32)
        for i, l in enumerate(tqdm(f, total=num_lines)):
            l = l.split(" ")  # using bytes here is tedious but avoids unicode error
            word, vector = l[0], l[1:]
            itos.append(word)
            vectors[i] = torch.tensor([float(x) for x in vector])
    print(f"{len(itos)} words loaded!")
    return (vectors, itos)


def get_pretrained_weights(
    embed_dim: int, tok2id: Dict[str, int], glove_dir: str = "./data/glove"
) -> torch.Tensor:
    glove_dir = Path(glove_dir)
    glove_path = glove_dir / f"glove.6B.{embed_dim}d.txt"
    if not os.path.exists(glove_path):
        raise ValueError(f"Glove file does not exist: {glove_path}")
    vectors, itos = load_glove_embeddings(filepath=glove_path, embed_dim=embed_dim)
    # So that we can initiate OOV words (relative to glove) with same distribution as others
    glove_mean = vectors.mean()
    glove_std = vectors.std()

    weights = torch.zeros((len(tok2id), embed_dim), dtype=torch.float32)
    found = 0
    for tok, ix in tok2id.items():
        tok = tok.lower()
        if tok in itos:
            weights[ix, :] = vectors[itos.index(tok)]
            found += 1
        else:
            # print(f"Word not in glove: {tok}")
            weights[ix, :] = torch.normal(glove_mean, glove_std, size=(embed_dim,))
    print(f"{found} out of {len(tok2id)} words found.")
    return weights


def token_from_kb_tuple(tup: Tuple[str, str, str]):
    return ("_".join(tup[0].split(" ")) + "_" + tup[1]).lower()


def tokenize(text: str, lower: bool = True) -> List[str]:
    if lower:
        text = text.lower()
    return word_tokenize(text)


def get_whitespace_markers(tokens: List[str]) -> Generator[int, None, None]:
    """
    Returns list of integers corresponding to whether or not a token should get a whitespace before it.
    """
    for idx, i in enumerate(tokens):
        if (i.startswith("'") or i in string.punctuation or i == "n't") and (i != "*"):
            yield 0
        elif idx > 0 and (i == "na" and tokens[idx - 1] == "gon"):
            yield 0
        else:
            yield 1


def ids_to_text(
    id2tok: Dict[int, str],
    eos_token_id: int,
    ids: Iterable[int],
    reversed: bool = False,
):
    if torch.is_tensor(ids):
        ids = ids.tolist()
    if reversed:
        ids = ids[::-1]
    eos_index = (
        ids.index(eos_token_id) if eos_token_id in ids else len(ids)
    )  # Index returns first occurrence
    toks = [id2tok[ids[i]] for i in range(1, eos_index)]
    output = []
    for tok, marker in zip(toks, get_whitespace_markers(toks)):
        if marker:
            output.append(" ")
        output.append(tok)
    return "".join(output).lstrip()


def print_results(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    gold: torch.Tensor,
    id2tok: Dict[int, str],
    eos_token_id: int,
    k: int = 3,
    reversed: bool = False,
):
    _ids_to_text = partial(ids_to_text, id2tok, eos_token_id)
    for i in range(k):
        print("Input: \n \t")
        print(_ids_to_text(inputs[i, :], reversed=reversed))
        print()
        print("Output: \n \t")
        print(_ids_to_text(torch.argmax(outputs[i, :], dim=0)))
        print()
        print("Expected: \n \t")
        print(_ids_to_text(gold[i, :]))
        print(
            "__________________________________________________________________________________________"
        )


def evaluate(model, dataloader, sos_token_id, eos_token_id, id2tok):
    num_correct = 0
    num_tokens = 0
    hypotheses = []
    references = []
    _ids_to_text = partial(ids_to_text, id2tok, eos_token_id)
    with torch.no_grad():
        model.eval()
        for idx, item in enumerate(dataloader):
            outputs = model(
                item=item,
                teacher_forcing_ratio=0.0,
                sos_token_id=sos_token_id,
            )  # (batch_size, num_vocab, max_len)
            preds = outputs.argmax(1)  # (batch_size, seq_len)
            num_correct += (
                (preds == item.get("output")) * item.get("input_mask")
            ).sum()
            num_tokens += item.get("input_mask").sum()
            hypotheses.extend(
                [_ids_to_text(preds[i, :]) for i in range(preds.shape[0])]
            )
            references.extend(
                [
                    _ids_to_text(item.get("output")[i, :])
                    for i in range(item.get("output").shape[0])
                ]
            )
    references = [
        i if i else "." for i in references
    ]  # Hack to prevent empty references
    acc = (num_correct / num_tokens).item()
    assert len(hypotheses) == len(references)
    bleu = raw_corpus_bleu(hypotheses, references)
    return {
        "acc": acc,
        "bleu": bleu,
        "hypotheses": hypotheses,
        "references": references,
    }


def raw_corpus_bleu(hypotheses, references, offset=0.01):
    """
    Simple wrapper around sacreBLEU's BLEU without tokenization and smoothing.

    from https://github.com/awslabs/sockeye/blob/master/sockeye/evaluate.py#L37

    :param hypotheses: Hypotheses stream.
    :param references: Reference stream.
    :param offset: Smoothing constant.
    :return: BLEU score
    """
    return sacrebleu.raw_corpus_bleu(
        hypotheses, [references], smooth_value=offset
    ).score
