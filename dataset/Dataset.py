import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.sos_id = tokenizer_tgt.token_to_id("[SOS]")
        self.eos_id = tokenizer_tgt.token_to_id("[EOS]")
        self.pad_id = tokenizer_tgt.token_to_id("[PAD]")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        src_text = item["translation"][self.src_lang]
        tgt_text = item["translation"][self.tgt_lang]
        enc_tokens = self.tokenizer_src.encode(src_text).ids
        dec_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        enc_tokens = enc_tokens[: self.seq_len - 2]
        dec_tokens = dec_tokens[: self.seq_len - 1]
        encoder_input = (
            [self.sos_id]
            + enc_tokens
            + [self.eos_id]
            + [self.pad_id] * (self.seq_len - len(enc_tokens) - 2)
        )
        decoder_input = (
            [self.sos_id]
            + dec_tokens
            + [self.pad_id] * (self.seq_len - len(dec_tokens) - 1)
        )
        label = (
            dec_tokens
            + [self.eos_id]
            + [self.pad_id] * (self.seq_len - len(dec_tokens) - 1)
        )
        encoder_input = torch.tensor(encoder_input, dtype=torch.long)
        decoder_input = torch.tensor(decoder_input, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        encoder_mask = (encoder_input != self.pad_id).unsqueeze(0).unsqueeze(0)  
        decoder_mask = (
            (decoder_input != self.pad_id).unsqueeze(0).unsqueeze(0)
            & causal_mask(self.seq_len)
        )  
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def causal_mask(size):
    return torch.tril(torch.ones((1, 1, size, size), dtype=torch.bool))
