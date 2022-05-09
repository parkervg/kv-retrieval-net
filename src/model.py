import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/sunnysai12345/KVMemnn
class Encoder(nn.Module):
    def __init__(
        self,
        num_vocab: int,
        embed_size: int,
        padding_idx: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        pretrained_weights=None,
    ):
        super().__init__()

        if pretrained_weights is not None:
            print("Using pretrained weights for encoder...")
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weights, freeze=False, padding_idx=padding_idx
            )
        else:
            self.embedding = nn.Embedding(
                num_vocab, embed_size, padding_idx=padding_idx
            )

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        # Adding bias of 1 to LSTM cell forget gate (Pham et. al. 2014)
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/4
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.0)

    def forward(self, input):
        """
        Passes x in as dialogue turn.
        :param x: A line of dialogue, either from user (input) or system (output)
        :return:
        """
        embeds = self.embedding(input)
        # x_packed = pack_padded_sequence(x_embed, item.get("length"), batch_first=True, enforce_sorted=False)
        # output_packed, (hidden_enc, cell_state) = self.lstm(x_packed)
        # return pad_packed_sequence(output_packed, batch_first=True)[0], (hidden_enc, cell_state)
        return self.lstm(self.dropout(embeds))


class Decoder(nn.Module):
    def __init__(
        self,
        num_vocab: int,
        embed_size: int,
        padding_idx: int,
        kb_vocab_start: int,
        attention_type: str,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.2,
        pretrained_weights=None,
    ):
        super().__init__()
        self.kb_vocab_start = kb_vocab_start
        self.num_vocab = num_vocab
        self.attention_type = attention_type
        kb_vocab_size = num_vocab - kb_vocab_start
        base_vocab_size = num_vocab - kb_vocab_size

        if pretrained_weights is not None:
            print("Using pretrained weights for decoder...")
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_weights, freeze=False, padding_idx=padding_idx
            )
        else:
            self.embedding = nn.Embedding(
                num_vocab, embed_size, padding_idx=padding_idx
            )

        self.base_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.kb_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        self.base_out = nn.Linear(hidden_size * 2, base_vocab_size)
        self.kb_out = nn.Linear(hidden_size * 2, kb_vocab_size)

        if self.attention_type == "bahdanau":
            print("using bahdanau attention...")
            self.base_attn = BahdanauAttention(hidden_size)
            self.kb_attn = BahdanauAttention(hidden_size)
        elif self.attention_type == "luong":
            print("using luong attention...")
            self.base_attn = LuongAttention(hidden_size)
            self.kb_attn = LuongAttention(hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        for l in [self.base_lstm, self.kb_lstm]:
            for name, param in l.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
        for o in [self.base_out, self.kb_out]:
            nn.init.xavier_uniform_(o.weight)
            nn.init.constant_(o.bias, 0.0)
        # Adding bias of 1 to LSTM cell forget gate (Pham et. al. 2014)
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/4
        for l in [self.base_lstm, self.kb_lstm]:
            for names in l._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(l, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)

    def forward(self, input, item, states, encoder_outputs, base_vocab_mask):
        """
        Concatenate hidden state of decoder with attention
        Use this to make predictions
        :param preds: (batch_size)
        :param encoder_outputs:
        """
        input.shape[0]
        encoder_outputs.shape[1]

        kb = item.get("kb")  # (batch_size, max_seq_len, 2)
        kb_mask = item.get("kb_mask")
        kb_vocab_mask = item.get("kb_vocab_mask")
        input_mask = item.get("input_mask")

        # Get embeds for input
        base_embeds = self.dropout(
            self.embedding(input).unsqueeze(1)
        )  # (batch_size, 1, embed_size)

        # Pass embeds through lstm
        base_decoder_output, states = self.base_lstm(base_embeds, states)

        """
        Base vocab logits
        """
        if self.attention_type == "bahdanau":
            base_attn_args = {
                "q": base_decoder_output,
                "k": encoder_outputs,
                "v": encoder_outputs,
                "mask": input_mask,
            }
        else:
            base_attn_args = {
                "q": base_decoder_output,
                "k": encoder_outputs,
                "mask": input_mask,
            }
        # Compute attention over given decoder hidden state, and encoder outputs
        context, base_att_weights = self.base_attn(
            **base_attn_args,
        )  # (batch_size, 1, hidden_size)

        # noinspection PyArgumentList
        feats = torch.cat(
            (base_decoder_output, context), axis=-1
        )  # (batch_size, 1, hidden_size * 2)

        # Calculate logits for `normal` vocab, from encoder
        # base_vocab_logits = self.fc_out(self.base_out, feats, base_vocab_mask)
        base_vocab_logits = self.dropout(self.base_out(feats)).squeeze(1)

        """
        KB vocab logits
        """
        # First, avg. up obj/rel embeddings
        kb_embeds = self.embedding(kb)  # (batch_size, max_seq_len, 2, embed_size)
        avg_kb_embeds = torch.mean(
            kb_embeds, axis=2
        )  # (batch_size, max_seq_len, embed_size)

        # kb_decoder_output, kb_states = self.kb_lstm(avg_kb_embeds, states)
        if self.attention_type == "bahdanau":
            kb_attn_args = {
                "q": base_decoder_output,
                "k": avg_kb_embeds,
                "v": avg_kb_embeds,
                "mask": kb_mask,
            }
        else:
            kb_attn_args = {
                "q": base_decoder_output,
                "k": avg_kb_embeds,
                "mask": kb_mask,
            }

        context, kb_att_weights = self.kb_attn(
            **kb_attn_args
        )  # (batch_size, 1, hidden_size)

        feats = torch.cat(
            (base_decoder_output, context), axis=-1
        )  # (batch_size, 1, hidden_size * 2)

        # Calculate logits for `kb` vocab, add to sparse base
        kb_vocab_logits = self.dropout(self.kb_out(feats)).squeeze(1)
        kb_vocab_logits *= kb_vocab_mask

        # Join the two logit vectors
        logits = torch.cat([base_vocab_logits, kb_vocab_logits], axis=-1)
        assert logits.shape[-1] == self.num_vocab

        return logits, states, (base_att_weights, kb_att_weights)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, q, k, v, mask=None):
        """
        :param q: (b, 1, hidden_size)
        :param k: (b, max_seq_len, hidden_size)
        :param v: (b, max_seq_len, hidden_size)
        :return:
        """
        query = self.W_q(q)  # (batch_size, 1, hidden_size)
        key = self.W_k(k)  # (batch_size, max_seq_len, hidden_size)
        features = torch.tanh(query + key)  # (batch_size, max_seq_len, hidden_size)
        scores = self.W_v(features).swapaxes(1, 2)  # (batch_size, 1, max_seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-1e20"))
        scores = F.softmax(scores, dim=1)
        context = torch.bmm(scores, v)  # (batch_size, 1, hidden_size)
        return context, scores


class LuongAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, q, k, mask=None):
        """
        :param q: (b, 1, hidden_size)
        :param k: (b, max_seq_len, hidden_size)
        :return:
        """
        scores = torch.einsum("boh,bsh->bos", [q, k])  # (batch_size, 1, max_seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-1e20"))
        scores = F.softmax(scores, dim=1)
        context = torch.bmm(scores, k)  # (batch_size, 1, hidden_size)
        return context, scores


class KVNetwork(nn.Module):
    def __init__(
        self,
        num_vocab: int,
        embed_size: int,
        hidden_size: int,
        dropout: float,
        num_layers: int,
        device,
        padding_idx: int,
        kb_vocab_start: int,
        attention_type: str,
        pretrained_weights=None,
    ):
        super().__init__()
        self.num_vocab = num_vocab
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.device = device
        self.padding_idx = padding_idx
        self.kb_vocab_start = kb_vocab_start

        self.encoder = Encoder(
            num_vocab=self.num_vocab,
            embed_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            padding_idx=self.padding_idx,
            pretrained_weights=pretrained_weights,
        )

        self.decoder = Decoder(
            num_vocab=self.num_vocab,
            embed_size=self.embed_size,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            padding_idx=self.padding_idx,
            kb_vocab_start=self.kb_vocab_start,
            attention_type=attention_type,
            pretrained_weights=pretrained_weights,
        )

    def forward(self, item, sos_token_id: int, teacher_forcing_ratio: float = 0.0):
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        max_len = item.get("input").shape[1]
        batch_size = item.get("input").shape[0]
        # Creating base vocab masks
        base_vocab_mask = torch.ones((batch_size, self.num_vocab), device=self.device)
        base_vocab_mask[:, self.kb_vocab_start :] = 0

        encoder_outputs, states = self.encoder(item.get("input"))
        outputs = torch.zeros((max_len, batch_size, self.num_vocab), device=self.device)
        outputs[0, :, sos_token_id] = 1
        input = torch.full((batch_size,), sos_token_id, device=self.device)
        for i in range(1, max_len):
            logits, states, _ = self.decoder(
                input=input,
                item=item,
                states=states,
                encoder_outputs=encoder_outputs,
                base_vocab_mask=base_vocab_mask,
            )
            outputs[i] = logits
            if use_teacher_forcing:
                input = item.get("output")[:, i]
            else:
                input = torch.argmax(logits, dim=-1)
        return outputs.permute(1, 2, 0)
