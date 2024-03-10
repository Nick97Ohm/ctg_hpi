

python ./mix_match_code/batched_MH/scripts/sample_batched_bias_mask_opt.py \
--max_iter 15 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./mix_match_code/batched_MH/output_samples/bias \
--alpha 1  \
--beta 1 \
--delta 0 \
--gamma 0 \
--data_name bias_test_bi_mask_src \
--disc_name  bias_1050 \
--disc_dir  C:/Users/z0048p4n/Downloads/mix_match_chkpts/mix_match_chkpts/bias/models/checkpoint-1050 \
--data_path ./data/bias/test_mask_short.txt \
--src_path ./data/bias/test_bi.txt \
--attr_path ./data/bias/test_bi.attr \
