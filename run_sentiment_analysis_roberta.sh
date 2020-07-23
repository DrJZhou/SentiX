#base_bert_path="/home/zhoujie/roberta_model/"
base_bert_path="/gruntdata/zhoujie/roberta_model/"
bert_path="state_dict/large_bert_PretrainBERT_data_all_val_acc_0.8691_0.6663_Rating_Mask_41528/"
batch_size=4
bert_dim=768
device_id=7
datasets=(SST-2-root) #SST-2-all
for dataset in ${datasets[@]}
do
#  python train_roberta.py --dataset $dataset --bert_dim $bert_dim --sentiment_class 2 --bert_path ${bert_path} --learning_rate 1e-5 --batch_size $batch_size --device cuda:$device_id
#  python train_roberta.py --dataset $dataset --bert_dim $bert_dim --sentiment_class 2 --bert_path ${bert_path} --learning_rate 1e-5 --fix_bert --batch_size $batch_size --device cuda:$device_id
  python train_roberta.py --dataset $dataset --bert_dim $bert_dim --sentiment_class 2 --bert_path ${base_bert_path} --learning_rate 1e-5 --batch_size $batch_size --device cuda:$device_id
  python train_roberta.py --dataset $dataset --bert_dim $bert_dim --sentiment_class 2 --bert_path ${base_bert_path} --learning_rate 1e-5 --fix_bert --batch_size $batch_size --device cuda:$device_id
done

datasets=(SST-5-root) #SST-5-all
for dataset in ${datasets[@]}
do
#  python train_roberta.py --dataset $dataset --bert_dim $bert_dim --bert_path ${bert_path} --learning_rate 1e-5 --batch_size $batch_size --device cuda:$device_id
#  python train_roberta.py --dataset $dataset --bert_dim $bert_dim --bert_path ${bert_path} --learning_rate 1e-5 --fix_bert --batch_size $batch_size --device cuda:$device_id
  python train_roberta.py --dataset $dataset --bert_dim $bert_dim --bert_path ${base_bert_path} --learning_rate 1e-5 --batch_size $batch_size --device cuda:$device_id
  python train_roberta.py --dataset $dataset --bert_dim $bert_dim --bert_path ${base_bert_path} --learning_rate 1e-5 --fix_bert --batch_size $batch_size --device cuda:$device_id
done

datasets=(IMDB Yelp)
for dataset in ${datasets[@]}
do
#  python train_roberta.py --dataset $dataset --bert_dim $bert_dim --cross_val_fold 0 --sentiment_class 2 --bert_path ${bert_path} --learning_rate 1e-5 --batch_size $batch_size --device cuda:$device_id
#  python train_roberta.py --dataset $dataset --bert_dim $bert_dim --cross_val_fold 0 --sentiment_class 2 --bert_path ${bert_path} --learning_rate 1e-5 --fix_bert --batch_size $batch_size --device cuda:$device_id
  python train_roberta.py --dataset $dataset --bert_dim $bert_dim --cross_val_fold 0 --sentiment_class 2 --bert_path ${base_bert_path} --learning_rate 1e-5 --batch_size $batch_size --device cuda:$device_id
  python train_roberta.py --dataset $dataset --bert_dim $bert_dim --cross_val_fold 0 --sentiment_class 2 --bert_path ${base_bert_path} --learning_rate 1e-5 --fix_bert --batch_size $batch_size --device cuda:$device_id
done
