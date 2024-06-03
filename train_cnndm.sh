#bin/bash

# --next_token_type avg_prev_token \
# --share_self_attention_module \
# --indi_self_q \
# --indi_self_out \
# --share_cross_attention_module \
# --indi_cross_q \
# --indi_cross_out \
# --pass_hidden_to_cross_att \
# --share_ffnn \

python -u train_summarization_mixcoder.py \
        --data_name cnn_dailymail \
        --tokenizer_path tokenizer/cnn_dailymail_3.0.0_BPEtokenizer.json \
        --next_token_type avg_prev_token \
        --share_self_attention_module \
        --indi_self_q \
        --indi_self_out \
        --share_cross_attention_module \
        --indi_cross_q \
        --indi_cross_out \
        --pass_hidden_to_cross_att \
        --gpu 2