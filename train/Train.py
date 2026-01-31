import torch
import torch.nn as nn
from src.models.model import build_transformer
from tokenizer.Tokenizer import get_ds
from validate.Validate import run_validation
from configs.training import get_weights_file_path,latest_weights_file_path,get_config 
from src.utils.checkpoints import save_checkpoint,load_checkpoint
from tqdm import tqdm
from pathlib import Path
from src.utils.logging import setup_logger
from torch.utils.tensorboard import SummaryWriter
from src.utils.seed import set_global_seed

def get_model(config, vocab_src_len, vocab_tgt_len):
    return build_transformer(vocab_src_len,vocab_tgt_len,config["src_seq_len"],config["d_model"],config["N"],config["h"],config["dropout"],config["d_ff"],config["d_k"])

def train_model(config):
    set_global_seed(get_config()['seed'])
    logger = setup_logger(name=config["experiment_name"],log_dir=f"{config['datasource']}_{config['model_folder']}")
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    print("Using device:", device)
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)
    logger.info(f"Model initialized | "f"Src vocab: {tokenizer_src.get_vocab_size()} | "f"Tgt vocab: {tokenizer_tgt.get_vocab_size()}")
    writer = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"],eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"),label_smoothing=0.1).to(device)
    initial_epoch = 0
    global_step = 0
    preload = config["preload"]
    checkpoint_path = (latest_weights_file_path(config) if preload == "latest" else get_weights_file_path(config, preload) if preload else None)

    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        epoch, global_step = load_checkpoint(model,optimizer,checkpoint_path,device)
        initial_epoch = epoch + 1
    else:
        logger.info("No checkpoint found. Training from scratch.")

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        logger.info(f"Starting epoch {epoch}")
        batch_iterator = tqdm(train_dataloader,desc=f"Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask)
            logits = model.projection(decoder_output)
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
        logger.info(f"Checkpoint saved: {epoch_ckpt}")

    