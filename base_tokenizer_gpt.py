from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from datasets import load_dataset
import os
from tqdm import tqdm

from transformers import AutoTokenizer

# data_name = "wmt14"
# subset = "de-en"
# src_lang = "en"
# tgt_lang = "de"

data_name = "wmt16"
subset = "ro-en"
src_lang = "ro"
tgt_lang = "en"

# data_name = "wmt14"
# subset = "fr-en"
# src_lang = "en"
# tgt_lang = "fr"

def to_list_casting(data):
    data[src_lang] = wmt14["train"]["translation"][src_lang]
    data[tgt_lang] = wmt14["train"]["translation"][tgt_lang]
    return data

# Load the WMT14 en-de dataset
wmt14 = load_dataset(data_name, subset)

data = []
for i in tqdm(range(len(wmt14["train"]["translation"]))):
    data.append(wmt14["train"][i]["translation"][src_lang])
    data.append(wmt14["train"][i]["translation"][tgt_lang])

# Initialize a tokenizer
# tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
# tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# # Set up pre-tokenization
# # tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# # Initialize a trainer
# trainer = trainers.BpeTrainer(vocab_size=37000, min_frequency=2, special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<sep>"], show_progress=True)

from tokenizers.processors import TemplateProcessing
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
old_tokenizer.add_special_tokens({"sep_token": "<sep>", "pad_token":"<pad>"})
old_tokenizer._tokenizer.post_processor = TemplateProcessing(
    single=old_tokenizer.bos_token + " $A " + old_tokenizer.sep_token,
    pair = old_tokenizer.bos_token +" $A " + old_tokenizer.sep_token + " $B "+ old_tokenizer.eos_token,
    special_tokens=[(old_tokenizer.eos_token, old_tokenizer.eos_token_id), (old_tokenizer.bos_token, old_tokenizer.bos_token_id), (old_tokenizer.sep_token, old_tokenizer.sep_token_id)],
)

tokenizer = old_tokenizer.train_new_from_iterator(data, 37000)

os.makedirs("tokenizer", exist_ok=True)
# Save the trained tokenizer

print(tokenizer)
print(tokenizer("hello, Nice to meet you"))
print(tokenizer.batch_decode(tokenizer("hello, Nice to meet you")["input_ids"], skip_special_tokens = False))

tokenizer.save_pretrained(f"tokenizer/GPT_{data_name}_{subset}_BPEtokenizer")

print(tokenizer)
a = tokenizer(["hello, Nice to meet you","hellos"], ["and you","bro"], add_special_tokens=True)
print(tokenizer(["hello, Nice to meet you","hellos"], ["and you","bro"], add_special_tokens=True))
print(tokenizer.batch_decode(a["input_ids"], skip_special_tokens = False))

# old_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
# old_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
# print(old_tokenizer)
# a = old_tokenizer(["hello, Nice to meet you","hellos"], ["and you","bro"], return_tensors="pt", add_special_tokens=True, padding=True)
# print(old_tokenizer(["hello, Nice to meet you","hellos"], ["and you","bro"], return_tensors="pt", add_special_tokens=True, padding=True))

# print(old_tokenizer.batch_decode(a["input_ids"], skip_special_tokens = False))

# from tokenizers.processors import TemplateProcessing
# tokenizer = AutoTokenizer.from_pretrained("./tokenizer/GPT_wmt16_ro-en_BPEtokenizer.json")
# tokenizer.add_special_tokens({"sep_token": "<sep>"})
# print(tokenizer.bos_token)
# print(tokenizer.bos_token_id)
# tokenizer._tokenizer.post_processor = TemplateProcessing(
#     single=tokenizer.bos_token + " $A " + tokenizer.eos_token,
#     pair = tokenizer.bos_token +" $A " + tokenizer.sep_token + " $B "+ tokenizer.eos_token,
#     special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id), (tokenizer.bos_token, tokenizer.bos_token_id), (tokenizer.sep_token, tokenizer.sep_token_id)],
# )
# print(tokenizer)
# a = tokenizer(["hello, Nice to meet you","hellos"], ["and you","bro"], add_special_tokens=True)
# print(tokenizer(["hello, Nice to meet you","hellos"], ["and you","bro"], add_special_tokens=True))

# print(tokenizer.batch_decode(a["input_ids"], skip_special_tokens = False))