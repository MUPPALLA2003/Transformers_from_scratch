from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from dataset.Dataset import TranslationDataset

def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]

def get_or_build_tokenizer(config, ds, lang, vocab):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=vocab,min_frequency=2,special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"])
        tokenizer.train_from_iterator(get_all_sentences(ds, lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'],config['src_vocab_size'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'],config['tgt_vocab_size'])
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    train_ds = TranslationDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['src_seq_len'],config['tgt_seq_len'])
    val_ds = TranslationDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['src_seq_len'],config['tgt_seq_len'])
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt