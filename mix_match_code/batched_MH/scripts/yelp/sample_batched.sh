python ./mix_match_code/batched_MH/scripts/sample_batched_input_improved.py \
--single true \
--normalize true \
--fluency pyplexity \
--max_iter 4 \
--shuffle_positions \
--temperature 1.0 \
--out_path ./mix_match_code/batched_MH/output_samples/yelp \
--alpha 4  \
--beta 1 \
--delta 2 \
--data_name yelp_li_test \
--disc_name  yelp_100 \
--disc_dir  C:/Users/z0048p4n/Downloads/mix_match_model/mix_match_chkpts/mix_match_chkpts/yelp_cls_2/models/checkpoint-100 \
--data_path ./data/yelp/test_li.txt \
--attr_path ./data/yelp/test_li.attr \

