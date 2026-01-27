import torch
import torch.nn as nn
from src.models.model import build_transformer
from tokenizer.Tokenizer import get_ds
from validate.Validate import run_validation
from configs.training import get_weights_file_path,latest_weights_file_path
from torch.utils.data import Dataset, DataLoader, random_split
from src.utils.checkpoints import save_checkpoint,load_checkpoint
from torch.optim.lr_scheduler import LambdaLR
import warnings
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter

def get_model(config, vocab_src_len, vocab_tgt_len):
    return build_transformer(vocab_src_len,vocab_tgt_len,config["seq_len"],config["seq_len"],d_model=config["d_model"])

def train_model(config):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    print("Using device:", device)
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)
    writer = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"],eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"),label_smoothing=0.1).to(device)
    initial_epoch = 0
    global_step = 0
    preload = config["preload"]
    checkpoint_path = (latest_weights_file_path(config) if preload == "latest" else get_weights_file_path(config, preload) if preload else None)

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        epoch, global_step = load_checkpoint(model,optimizer,checkpoint_path,device)
        initial_epoch = epoch + 1
    else:
        print("No checkpoint found. Training from scratch.")

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader,desc=f"Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask)
            logits = model.project(decoder_output)
            loss = loss_fn(logits.view(-1, logits.size(-1)),label.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1
            batch_iterator.set_postfix(loss=f"{loss.item():.4f}")

        run_validation(model,val_dataloader,tokenizer_src,tokenizer_tgt,config["seq_len"],device,lambda msg: batch_iterator.write(msg),global_step,writer)
        epoch_ckpt = get_weights_file_path(config, f"{epoch:02d}")
        latest_ckpt = latest_weights_file_path(config)
        save_checkpoint(model=model,optimizer=optimizer,epoch=epoch,global_step=global_step,path=epoch_ckpt)
        print(f"Checkpoint saved for epoch {epoch}")

    