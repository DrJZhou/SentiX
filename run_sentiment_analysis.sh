base_bert_path="/home/zhoujie/bert_model/"
bert_path="state_dict/bert_PretrainBERT_data_all_val_acc_0.9317_0.0065_Rating_Mask_373752/"
batch_size=16
#datasets=(SST-2-root) #SST-2-all
#for dataset in ${datasets[@]}
#do
#  python train.py --dataset $dataset --sentiment_class 2 --bert_path ${bert_path} --batch_size $batch_size
#  python train.py --dataset $dataset --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size $batch_size
#  python train.py --dataset $dataset --sentiment_class 2 --bert_path ${base_bert_path} --batch_size $batch_size
#  python train.py --dataset $dataset --sentiment_class 2 --bert_path ${base_bert_path} --fix_bert --batch_size $batch_size
#done

#datasets=(SST-5-root) #SST-5-all
#for dataset in ${datasets[@]}
#do
#  python train.py --dataset $dataset --bert_path ${bert_path} --learning_rate 2e-5 --batch_size $batch_size
#  python train.py --dataset $dataset --bert_path ${bert_path} --learning_rate 2e-5 --fix_bert --batch_size $batch_size
#  python train.py --dataset $dataset --bert_path ${base_bert_path} --learning_rate 2e-5 --batch_size $batch_size
#  python train.py --dataset $dataset --bert_path ${base_bert_path} --learning_rate 2e-5 --fix_bert --batch_size $batch_size
#done

datasets=(IMDB)
for dataset in ${datasets[@]}
do
  python train.py --dataset $dataset --cross_val_fold 0 --sentiment_class 2 --bert_path ${bert_path} --batch_size $batch_size
  python train.py --dataset $dataset --cross_val_fold 0 --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size $batch_size
  python train.py --dataset $dataset --cross_val_fold 0 --sentiment_class 2 --bert_path ${base_bert_path} --batch_size $batch_size
  python train.py --dataset $dataset --cross_val_fold 0 --sentiment_class 2 --bert_path ${base_bert_path} --fix_bert --batch_size $batch_size
done

datasets=(Yelp)
for dataset in ${datasets[@]}
do
  python train.py --dataset $dataset --cross_val_fold 0 --sentiment_class 2 --bert_path ${bert_path} --batch_size $batch_size
  python train.py --dataset $dataset --cross_val_fold 0 --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size $batch_size
  python train.py --dataset $dataset --cross_val_fold 0 --sentiment_class 2 --bert_path ${base_bert_path} --batch_size $batch_size
  python train.py --dataset $dataset --cross_val_fold 0 --sentiment_class 2 --bert_path ${base_bert_path} --fix_bert --batch_size $batch_size
done

datasets=(SST-2-all) #SST-2-all
for dataset in ${datasets[@]}
do
  python train.py --dataset $dataset --sentiment_class 2 --bert_path ${bert_path} --batch_size $batch_size
  python train.py --dataset $dataset --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size $batch_size
  python train.py --dataset $dataset --sentiment_class 2 --bert_path ${base_bert_path} --batch_size $batch_size
  python train.py --dataset $dataset --sentiment_class 2 --bert_path ${base_bert_path} --fix_bert --batch_size $batch_size
done

datasets=(SST-5-all) #SST-5-all
for dataset in ${datasets[@]}
do
  python train.py --dataset $dataset --bert_path ${bert_path} --learning_rate 2e-5 --batch_size $batch_size
  python train.py --dataset $dataset --bert_path ${bert_path} --learning_rate 2e-5 --fix_bert --batch_size $batch_size
  python train.py --dataset $dataset --bert_path ${base_bert_path} --learning_rate 2e-5 --batch_size $batch_size
  python train.py --dataset $dataset --bert_path ${base_bert_path} --learning_rate 2e-5 --fix_bert --batch_size $batch_size
done
