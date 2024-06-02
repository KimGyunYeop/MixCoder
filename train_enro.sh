#bin/bash

python -u train_mixcoder.py \
        --data_name wmt16 \
        --src_lang en \
        --tgt_lang ro \
        --tokenizer_path tokenizer/wmt16_ro-en_BPEtokenizer.json \
        --next_token_type avg_prev_token \
        --share_self_attention_module \
        --indi_self_q \
        --indi_self_out \
        --share_cross_attention_module \
        --indi_cross_q \
        --indi_cross_out \
        --pass_hidden_to_cross_att \
        --share_ffnn 