base_bert_path="/home/zhoujie/bert_large_model/"
#base_bert_path="/gruntdata/zhoujie/bert_large_model/"
bert_path="state_dict/large_bert_PretrainBERT_data_all_val_acc_0.8691_0.6663_Rating_Mask_41528/"
batch_size=8
bert_dim=1024
datasets=(SST-2-root) #SST-2-all
for dataset in ${datasets[@]}
do
#  python train.py --dataset $dataset --bert_dim $bert_dim --sentiment_class 2 --bert_path ${bert_path} --learning_rate 1e-5 --batch_size $batch_size
#  python train.py --dataset $dataset --bert_dim $bert_dim --sentiment_class 2 --bert_path ${bert_path} --learning_rate 1e-5 --fix_bert --batch_size $batch_size
  python train.py --dataset $dataset --bert_dim $bert_dim --sentiment_class 2 --bert_path ${base_bert_path} --learning_rate 1e-5 --batch_size $batch_size
  python train.py --dataset $dataset --bert_dim $bert_dim --sentiment_class 2 --bert_path ${base_bert_path} --learning_rate 1e-5 --fix_bert --batch_size $batch_size
done

datasets=(SST-5-root) #SST-5-all
for dataset in ${datasets[@]}
do
#  python train.py --dataset $dataset --bert_dim $bert_dim --bert_path ${bert_path} --learning_rate 1e-5 --batch_size $batch_size
#  python train.py --dataset $dataset --bert_dim $bert_dim --bert_path ${bert_path} --learning_rate 1e-5 --fix_bert --batch_size $batch_size
  python train.py --dataset $dataset --bert_dim $bert_dim --bert_path ${base_bert_path} --learning_rate 1e-5 --batch_size $batch_size
  python train.py --dataset $dataset --bert_dim $bert_dim --bert_path ${base_bert_path} --learning_rate 1e-5 --fix_bert --batch_size $batch_size
done

datasets=(IMDB Yelp)
for dataset in ${datasets[@]}
do
#  python train.py --dataset $dataset --bert_dim $bert_dim --cross_val_fold 0 --sentiment_class 2 --bert_path ${bert_path} --learning_rate 1e-5 --batch_size $batch_size
#  python train.py --dataset $dataset --bert_dim $bert_dim --cross_val_fold 0 --sentiment_class 2 --bert_path ${bert_path} --learning_rate 1e-5 --fix_bert --batch_size $batch_size
  python train.py --dataset $dataset --bert_dim $bert_dim --cross_val_fold 0 --sentiment_class 2 --bert_path ${base_bert_path} --learning_rate 1e-5 --batch_size $batch_size
  python train.py --dataset $dataset --bert_dim $bert_dim --cross_val_fold 0 --sentiment_class 2 --bert_path ${base_bert_path} --learning_rate 1e-5 --fix_bert --batch_size $batch_size
done

datasets=(SST-2-all) #SST-2-all
for dataset in ${datasets[@]}
do
#  python train.py --dataset $dataset --bert_dim $bert_dim --sentiment_class 2 --bert_path ${bert_path} --learning_rate 1e-5 --batch_size $batch_size
#  python train.py --dataset $dataset --bert_dim $bert_dim --sentiment_class 2 --bert_path ${bert_path} --learning_rate 1e-5 --fix_bert --batch_size $batch_size
  python train.py --dataset $dataset --bert_dim $bert_dim --sentiment_class 2 --bert_path ${base_bert_path} --learning_rate 1e-5 --batch_size $batch_size
  python train.py --dataset $dataset --bert_dim $bert_dim --sentiment_class 2 --bert_path ${base_bert_path} --learning_rate 1e-5 --fix_bert --batch_size $batch_size
done

datasets=(SST-5-all) #SST-5-all
for dataset in ${datasets[@]}
do
#  python train.py --dataset $dataset --bert_dim $bert_dim --bert_path ${bert_path} --learning_rate 1e-5 --batch_size $batch_size
#  python train.py --dataset $dataset --bert_dim $bert_dim --bert_path ${bert_path} --learning_rate 1e-5 --fix_bert --batch_size $batch_size
  python train.py --dataset $dataset --bert_dim $bert_dim --bert_path ${base_bert_path} --learning_rate 1e-5 --batch_size $batch_size
  python train.py --dataset $dataset --bert_dim $bert_dim --bert_path ${base_bert_path} --learning_rate 1e-5 --fix_bert --batch_size $batch_size
done
