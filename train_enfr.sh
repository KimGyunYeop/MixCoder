#bin/bash

gpu=1
setting=1
next_token_type=avg_prev_token
# ["new_token", "avg_prev_token"]
python -u train_mixcoder_for_ptbart.py \
        --data_name wmt14 \
        --subset fr-en \
        --src_lang en \
        --tgt_lang fr \
        --tokenizer_path tokenizer/wmt14_fr-en_BPEtokenizer.json \
        --next_token_type $next_token_type \
        --weight_decay 0.1 \
        --setting $setting \
        --base \
        --gpu $gpu