import torch
import os
import torchmetrics
from dataset.Dataset import causal_mask

def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.tensor([[sos_idx]], device=device)
    while decoder_input.size(1) < max_len:
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)
        out = model.decode(encoder_output,source_mask,decoder_input,decoder_mask)
        prob = model.project(out[:, -1])
        next_word = prob.argmax(dim=1).item()
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_word]], device=device)],dim=1)
        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)

def run_validation(model,validation_ds,tokenizer_tgt,max_len,device,print_msg,global_step,writer,num_examples=2,):
    model.eval()
    predicted, expected = [], []
    try:
        _, console_width = os.popen("stty size", "r").read().split()
        console_width = int(console_width)
    except:
        console_width = 80
    with torch.no_grad():
        for i, batch in enumerate(validation_ds):
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            assert encoder_input.size(0) == 1, "Validation batch size must be 1"
            model_out = greedy_decode(model,encoder_input,encoder_mask,tokenizer_tgt,max_len,device,)
            src_text = batch["src_text"][0]
            tgt_text = batch["tgt_text"][0]
            pred_text = tokenizer_tgt.decode(model_out.cpu().numpy())
            predicted.append(pred_text)
            expected.append(tgt_text)
            print_msg("-" * console_width)
            print_msg(f"{'SOURCE:':>12} {src_text}")
            print_msg(f"{'TARGET:':>12} {tgt_text}")
            print_msg(f"{'PREDICTED:':>12} {pred_text}")
            if i + 1 == num_examples:
                break

    if writer:
        cer = torchmetrics.CharErrorRate()(predicted, expected)
        wer = torchmetrics.WordErrorRate()(predicted, expected)
        bleu = torchmetrics.BLEUScore()(predicted, expected)
        writer.add_scalar("validation/CER", cer, global_step)
        writer.add_scalar("validation/WER", wer, global_step)
        writer.add_scalar("validation/BLEU", bleu, global_step)
        writer.flush()
