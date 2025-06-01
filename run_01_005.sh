CUDA_VISIBLE_DEVICES=7 python run.py --data_path data_new_v2.json --n_iter 100 --pop_size 50 --algorithm NSGAII --pct_words_to_swap 0.1 --start_idx 0 --end_idx 10 --reader_name vicuna-7b 
CUDA_VISIBLE_DEVICES=7 python run.py --data_path data_new_v2.json --n_iter 100 --pop_size 50 --algorithm NSGAII --pct_words_to_swap 0.05 --start_idx 0 --end_idx 10 --reader_name vicuna-7b

