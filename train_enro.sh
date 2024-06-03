#bin/bash

python -u train_mixcoder.py \
        --data_name wmt16 \
        --subset ro-en \
        --src_lang ro \
        --tgt_lang en \
        --tokenizer_path tokenizer/wmt16_ro-en_BPEtokenizer.json \
        --next_token_type new_token \
        --pass_hidden_to_cross_att \
        --gpu 1