gpu=$1
openai_api_key=$2

#chatgpt
CUDA_VISIBLE_DEVICES=${gpu} OPENAI_API_KEY=${openai_api_key} python3 rl4lms/testing/baselines/chatgpt.py output/imdb_test_1000/chatgpt
#gpt2-large pret
CUDA_VISIBLE_DEVICES=${gpu} python3 rl4lms/testing/baselines/gpt2_generate.py --model_name_or_path gpt2-large --output_dir output/imdb_test_1000/gpt2-large-pret
#gpt2-large ft
CUDA_VISIBLE_DEVICES=${gpu} python3 rl4lms/testing/baselines/gpt2_generate.py --model_name_or_path /data/dheeraj/GPT23/saved_models/gpt2_large_imdb/ --output_dir output/imdb_test_1000/gpt2-large-ft
#gpt2-large ft rlhf
CUDA_VISIBLE_DEVICES=${gpu} OPENAI_API_KEY=${openai_api_key} python3 rl4lms/testing/gpt23.py --model_name_or_path /data/dheeraj/GPT23/saved_models/gpt2_large_imdb_rlhf/gpt23/baseline_gpt2large_sup_rlhf/model/ --tokenizer_name /data/dheeraj/GPT23/saved_models/gpt2_large_imdb/ --output_dir output/imdb_test_1000/gpt2-large-ft-rlhf
#gpt2-large ft chatgpt first
CUDA_VISIBLE_DEVICES=${gpu} python3 rl4lms/testing/baselines/gpt2_generate_chatgpt_first.py --model_name_or_path /data/dheeraj/GPT23/saved_models/gpt2_large_imdb/ --chatgpt_gen_file output/imdb_test_1000/chatgpt/out.csv --output_dir output/imdb_test_1000/gpt2-large-ft-chatgpt-first
#gpt2-large ft chatgpt random
CUDA_VISIBLE_DEVICES=${gpu} OPENAI_API_KEY=${openai_api_key} python3 rl4lms/testing/baselines/gpt2_generate_chatgpt_random.py --model_name_or_path /data/dheeraj/GPT23/saved_models/gpt2_large_imdb/ --output_dir output/imdb_test_1000/gpt2-large-ft-chatgpt-random
#gpt23
CUDA_VISIBLE_DEVICES=${gpu} OPENAI_API_KEY=${openai_api_key} python3 rl4lms/testing/gpt23.py --model_name_or_path /data/dheeraj/GPT23/saved_models/gpt2_3_large_imdb_random_rlhf_all_topk_50/gpt23/rlhf_imdb_random_all_topk_50/model/ --tokenizer_name /data/dheeraj/GPT23/saved_models/gpt2_3_large_imdb_random/ --output_dir output/imdb_test_1000/gpt23-large-topk-50

