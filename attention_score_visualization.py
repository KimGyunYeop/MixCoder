from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders
from transformers import PreTrainedTokenizerFast, BartTokenizer, GPT2Tokenizer, GPT2Model, BartModel, BartForConditionalGeneration
from modeling_mc_for_visualization import MixcoderForConditionalGeneration, MixcoderConfig
import torch
from matplotlib import pyplot as plt
import argparse
from datasets import load_dataset
import custom_tokenizer
import custom_datasets
import os
from torch.functional import F
from torchinfo import summary

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_name", type=str, default="wmt14")
argparser.add_argument("--subset", type=str, default="de-en")
argparser.add_argument("--src_lang", type=str, default="en")
argparser.add_argument("--tgt_lang", type=str, default="de")
argparser.add_argument("--batch_size", type=int, default=16)
argparser.add_argument("--tokenizer_path", type=str, default="tokenizer/wmt14_de-en_BPEtokenizer.json")
argparser.add_argument("--save_path", type=str, default="./results_base/wmt14_en-de/baseline-/1000000")
argparser.add_argument("--gpu", type=int, default=0)

args = argparser.parse_args()

os.makedirs("figs", exist_ok=True)

args.save_path = "/home/nlplab/hdd1/gyop/research/GenrateFromCurrentPosition/results_base/wmt14_en-de/-avg_prev_token-share_att-indi_self_q-indi_self_out-share_cross_att-indi_cross_q-indi_cross_out-hidden_cross_att/1000000"
# args.save_path = "/home/nlplab/hdd1/gyop/research/GenrateFromCurrentPosition/results_base/wmt14_en-de/-new_token-share_att-indi_self_q-indi_self_out-share_cross_att-indi_cross_q-indi_cross_out-hidden_cross_att/1000000"
# args.save_path = "/home/nlplab/hdd1/gyop/research/GenrateFromCurrentPosition/results_base/wmt14_en-de/baseline-/1000000"

dataset = load_dataset(args.data_name, args.subset, split="test")
print("before filtering:")
print(dataset)

tokenizer = custom_tokenizer.get_tokenizer(args.tokenizer_path)

# model = BartForConditionalGeneration.from_pretrained(args.save_path, local_files_only=True)

model = MixcoderForConditionalGeneration.from_pretrained(args.save_path, local_files_only=True)
if model.config.next_token_type == "new_token":
    tokenizer.add_tokens("<next>", special_tokens=True)
    next_token_id = tokenizer.convert_tokens_to_ids("<next>")
else:
    next_token_id = None

print(sum(p.numel() for p in model.parameters()))
print(model.num_parameters(only_trainable=True, exclude_embeddings=True))
print(summary(model))
quit()

test_dataset = custom_datasets.WmtDataset(dataset, tokenizer=tokenizer, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=test_dataset.collate_fn)

print(model)
print(len(tokenizer))
input()

for idx, batch in enumerate(test_dataloader):
    print(tokenizer.batch_decode(batch["input_ids"]))
    print(tokenizer.batch_decode(torch.where(batch["labels"] == -100, tokenizer.pad_token_id, batch["labels"]), skip_special_tokens=False))
    out = model(**batch, output_attentions=True, output_hidden_states=True)
    print(out.keys())
    att = torch.stack(out.decoder_attentions)
    print(torch.stack(out.decoder_attentions).shape)
    # att = torch.stack(out.attentions)
    # print(torch.stack(out.attentions).shape)
    print(torch.mean(torch.mean(att, dim=0), dim=1).unsqueeze(0))

    os.makedirs("figs/"+str(idx), exist_ok=True)

    plt.matshow(torch.mean(torch.mean(att, dim=0), dim=1).squeeze()[:,:].detach().numpy())
    plt.savefig(f"figs/{idx}/att.png")
    plt.clf()
    for i in range(att.size(0)):
        print(torch.mean(att[i,0,:,:], dim=0).unsqueeze(0))
        plt.matshow(torch.mean(att[i,0,:,:,:], dim=0).squeeze()[:,:].detach().numpy())
        plt.savefig(f"figs/{idx}/att_{i}.png")
        plt.clf()
    
    print(tokenizer.batch_decode(torch.where(batch["labels"] == -100, tokenizer.pad_token_id, batch["labels"]), skip_special_tokens=True))
    print(tokenizer.batch_decode(out["logits"].argmax(-1), skip_special_tokens=True))
    print(idx)

    input_embs = out.decoder_hidden_states[0]
    print(input_embs.shape)
    last_hidden_state = out.decoder_hidden_states[-1]
    print(last_hidden_state.shape)
    
    l2d = (input_embs - last_hidden_state).pow(2).sum(2).sqrt().T
    cs = F.cosine_similarity(input_embs, last_hidden_state, dim=-1).T

    print(l2d)
    print(torch.mean(l2d, dim=0))
    print(cs)
    print(torch.mean(cs, dim=0))
    
    input()