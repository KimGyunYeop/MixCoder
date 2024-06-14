import json
import argparse

import evaluate
import custom_tokenizer
import sacrebleu

from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize


savepath = "./results_base/wmt14_en-de/baseline-"
data = json.load(open(f"{savepath}/test_result.json", "r"))
args = argparse.Namespace(**json.load(open(f"{savepath}/args.json", "r")))

tokenizer = custom_tokenizer.get_tokenizer(args.tokenizer_path)
matric_scarebleu = evaluate.load("sacrebleu")

for _, values in data.items():
    # tokenized_pred = [tokenizer.tokenize(values["pred"])]
    # tokenized_ref = [tokenizer.tokenize(values["ref"])]
    # print(values)
    matric_scarebleu.add(predictions=values["pred"], references=values["ref"])

print(matric_scarebleu.compute(tokenize="intl"))

# 모든 예측과 참조를 저장할 리스트 초기화
predictions = []
references = []
bleus = []

# 데이터 처리
for _, values in data.items():
    # 예측값과 참조값을 토크나이즈
    tokenized_pred = tokenizer.tokenize(values["pred"])
    tokenized_ref = tokenizer.tokenize(values["ref"])
    bleus.append(sentence_bleu([tokenized_ref], tokenized_pred))


print(sum(bleus) / len(bleus))
