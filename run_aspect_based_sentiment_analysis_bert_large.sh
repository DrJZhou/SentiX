#base_bert_path="/home/zhoujie/bert_large_model/"
base_bert_path="/gruntdata/zhoujie/bert_large_model/"
bert_path="state_dict/large_bert_PretrainBERT_data_all_val_acc_0.9331_0.6973_Rating_Mask_290696/"
batch_size=8
bert_dim=1024
cuda_id="2"
python aspect_train.py --device cuda:${cuda_id} --bert_dim $bert_dim --cross_val_fold 0 --dataset res14 --batch_size $batch_size --learning_rate 2e-5 --pretrained_bert_name ${bert_path} --max_seq_len 120 >> log_res14.txt
python aspect_train.py --device cuda:${cuda_id} --bert_dim $bert_dim --cross_val_fold 0 --dataset lap14 --batch_size $batch_size --learning_rate 2e-5 --pretrained_bert_name ${bert_path} --max_seq_len 120 >> log_lap14.txt
python aspect_train.py --device cuda:${cuda_id} --bert_dim $bert_dim --cross_val_fold 0 --dataset twitter --batch_size $batch_size --learning_rate 2e-5 --pretrained_bert_name ${bert_path} --max_seq_len 120 >> log_twitter.txt

python aspect_train.py --device cuda:${cuda_id} --bert_dim $bert_dim --cross_val_fold 0 --dataset res14 --batch_size $batch_size --learning_rate 2e-5 --pretrained_bert_name ${bert_path} --max_seq_len 120 --fix_bert >> log_res14.txt
python aspect_train.py --device cuda:${cuda_id} --bert_dim $bert_dim --cross_val_fold 0 --dataset lap14 --batch_size $batch_size --learning_rate 2e-5 --pretrained_bert_name ${bert_path} --max_seq_len 120 --fix_bert >> log_lap14.txt
python aspect_train.py --device cuda:${cuda_id} --bert_dim $bert_dim --cross_val_fold 0 --dataset twitter --batch_size $batch_size --learning_rate 2e-5 --pretrained_bert_name ${bert_path} --max_seq_len 120 --fix_bert >> log_twitter.txt

#python aspect_train.py --device cuda:${cuda_id} --bert_dim $bert_dim --cross_val_fold 0 --dataset res14 --batch_size $batch_size --learning_rate 2e-5 --pretrained_bert_name ${base_bert_path} --max_seq_len 120 >> log_res14.txt
#python aspect_train.py --device cuda:${cuda_id} --bert_dim $bert_dim --cross_val_fold 0 --dataset lap14 --batch_size $batch_size --learning_rate 2e-5 --pretrained_bert_name ${base_bert_path}  --max_seq_len 120 >> log_lap14.txt
#python aspect_train.py --device cuda:${cuda_id} --bert_dim $bert_dim --cross_val_fold 0 --dataset twitter --batch_size $batch_size --learning_rate 2e-5 --pretrained_bert_name ${base_bert_path} --max_seq_len 120 >> log_twitter.txt

#python aspect_train.py --device cuda:${cuda_id} --bert_dim $bert_dim --cross_val_fold 0 --dataset res14 --batch_size $batch_size --learning_rate 2e-5 --pretrained_bert_name ${base_bert_path} --max_seq_len 120 --fix_bert >> log_res14.txt
#python aspect_train.py --device cuda:${cuda_id} --bert_dim $bert_dim --cross_val_fold 0 --dataset lap14 --batch_size $batch_size --learning_rate 2e-5 --pretrained_bert_name ${base_bert_path} --max_seq_len 120 --fix_bert >> log_lap14.txt
#python aspect_train.py --device cuda:${cuda_id} --bert_dim $bert_dim --cross_val_fold 0 --dataset twitter --batch_size $batch_size --learning_rate 2e-5 --pretrained_bert_name ${base_bert_path} --max_seq_len 120 --fix_bert >> log_twitter.txt

