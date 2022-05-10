import pickle
from pathlib import Path
from typing import Union

import attr
import torch
from attr import attrib
from torch.utils.data import DataLoader

from src.model import *
from src.prepare_data import KVRETDataset
from src.utils import utils

"""
'For each time-step of decoding, the cell state is used to compute an attention over the encoder states
and a separate attention over the key of each entry in the KB.

Attention over the encoder are used to generate context vector, combined with the cell state to get a distribution over the normal vocabulary.

The attentions over the keys of the KB become the logits for associated values and are separate entities in a now augmented vocabulary that we argmax over'

TODO:
    - Fix punctuation issues
        - "round_table_address." not aligning with "round_table_address"
    - Make sure "webster_garage_distance" is "webster_garage" + "distance"
    - Fix 'the_clement_hotel\tis'
"""


@attr.s
class Args:
    kvret_path: str = attrib()
    model_name: str = attrib()
    include_context: bool = attrib(default=True)
    reverse_input: bool = attrib(default=False)
    epochs: int = attrib(default=10)
    embed_size: int = attrib(default=300)
    hidden_size: int = attrib(default=768)
    num_layers: int = attrib(default=1)
    max_length: Union[int, str] = attrib(default=32)
    lr: float = attrib(default=0.0001)
    batch_size: int = attrib(default=32)
    dropout: float = attrib(default=0.1)
    attention_type: str = attrib(default="bahdanau")
    weight_decay: float = attrib(default=0)
    grad_clip: float = attrib(default=10.0)
    model_save_dir: str = attrib(default="models/")
    teacher_forcing_ratio: float = attrib(default=0.5)
    use_pretrained: bool = attrib(default=False)
    device: torch.device = attrib(default=torch.device("cpu"))


if __name__ == "__main__":
    MODEL_NAME = "bahdanau_base"
    args = Args(
        kvret_path="data/kvret_dataset_public/kvret_{}_public.json",
        model_name=MODEL_NAME,
    )
    #####################################################################################################################
    args.epochs = 20
    args.embed_size = 200
    args.hidden_size = 200
    args.teacher_forcing_ratio = 0.5
    args.lr = 0.005
    args.max_length = "longest"
    args.attention_type = "bahdanau"
    args.use_pretrained = False
    args.dropout = 0.5
    args.batch_size = 64
    args.num_layers = 1
    args.include_context = True
    args.reverse_input = True
    args.device = (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    )
    #####################################################################################################################

    dataset = KVRETDataset(
        train_path=args.kvret_path.format("train"),
        dev_path=args.kvret_path.format("dev"),
        test_path=args.kvret_path.format("test"),
        device=args.device,
        include_context=args.include_context,
        max_len=args.max_length,
        reverse_input=args.reverse_input,
    )

    eos_token_id = dataset.tok2id["[EOS]"]
    sos_token_id = dataset.tok2id["[SOS]"]
    pad_id = dataset.tok2id["[PAD]"]

    train_dataloader = DataLoader(
        dataset.train, batch_size=args.batch_size, shuffle=True
    )
    dev_dataloader = DataLoader(dataset.dev, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset.test, batch_size=args.batch_size, shuffle=True)

    if args.use_pretrained:
        pretrained_weights = utils.get_pretrained_weights(
            args.embed_size, dataset.tok2id
        )
    else:
        pretrained_weights = None
    model = KVNetwork(
        num_vocab=len(dataset.id2tok),
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        device=args.device,
        num_layers=args.num_layers,
        padding_idx=pad_id,
        kb_vocab_start=dataset.kb_vocab_start,
        attention_type=args.attention_type,
        pretrained_weights=pretrained_weights,
    )
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    evaluate_output = utils.evaluate(
        model=model,
        dataloader=dev_dataloader,
        sos_token_id=sos_token_id,
        eos_token_id=eos_token_id,
        id2tok=dataset.id2tok,
    )
    best_model = model
    best_bleu = 0
    for epoch in range(args.epochs):
        print(f"EPOCH {epoch}")
        print("___________________________________________________________________")
        epoch_loss = 0
        num_correct = num_tokens = 0
        model.train()
        for idx, item in enumerate(train_dataloader):
            outputs = model(
                item=item,
                teacher_forcing_ratio=args.teacher_forcing_ratio,
                sos_token_id=sos_token_id,
            )  # (batch_size, num_vocab, max_len)

            # Loss, but with input_mask
            # also, do not consider [SOS] token in loss
            masked_model_output = (
                outputs * item.get("input_mask").unsqueeze(1).expand_as(outputs)
            )[:, :, 1:]
            masked_gold_output = (item.get("output") * item.get("input_mask"))[:, 1:]
            loss = criterion(masked_model_output, masked_gold_output)
            epoch_loss += loss
            preds = outputs.argmax(1)  # (batch_size, seq_len)
            num_correct += (
                (preds == item.get("output")) * item.get("input_mask")
            ).sum()
            num_tokens += item.get("input_mask").sum()

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            del loss
        acc = num_correct.item() / num_tokens.item()
        print()
        print(
            " Training Loss: {:.2f} \t Token-level Accuracy: {:.3f}".format(
                epoch_loss, acc
            )
        )

        print()
        print("***********************************************************")
        utils.print_results(
            inputs=item.get("input"),
            outputs=outputs,
            gold=item.get("output"),
            id2tok=dataset.id2tok,
            eos_token_id=eos_token_id,
            reversed=args.reverse_input,
            k=1,
        )
        print("Evaluating on dev set...")
        evaluate_output = utils.evaluate(
            model=model,
            dataloader=dev_dataloader,
            sos_token_id=sos_token_id,
            eos_token_id=eos_token_id,
            id2tok=dataset.id2tok,
        )
        print(
            " Token-level Accuracy: {:.3f} \t BLEU: {}".format(
                evaluate_output.get("acc"), evaluate_output.get("bleu")
            )
        )
        print()
        for i in range(5):
            print("\t")
            print(f"Reference: {evaluate_output.get('references')[i]}")
            print(f"Hypothesis: {evaluate_output.get('hypotheses')[i]}")
            print("\n\n")
        if evaluate_output.get("bleu") > best_bleu:
            print(f"New best bleu score! {best_bleu}")
            best_bleu = evaluate_output.get("bleu")
            best_model = model

    print("Evaluating best model on test set...")
    evaluate_output = utils.evaluate(
        model=best_model,
        dataloader=test_dataloader,
        sos_token_id=sos_token_id,
        eos_token_id=eos_token_id,
        id2tok=dataset.id2tok,
    )
    print(
        " Token-level Accuracy: {:.3f} \t BLEU: {}".format(
            evaluate_output.get("acc"), evaluate_output.get("bleu")
        )
    )
    print(f"Saving model to {args.model_save_dir}...")
    # Save relevant .pkl files
    model_save_dir = Path(args.model_save_dir)
    with open(model_save_dir / f"{args.model_name}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(model_save_dir / f"{args.model_name}/dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
