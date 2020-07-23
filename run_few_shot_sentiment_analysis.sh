base_bert_path="/home/zhoujie/bert_model/"
bert_path="state_dict/bert_PretrainBERT_data_all_val_f1_0.4717_meanf1_0.6693_Rating_Mask/"
sample_num=(10 50 100 200 500 1000)
datasets=(SST-2-root)
for dataset in ${datasets[@]}
do
  for num in ${sample_num[@]}
  do
    python few_shot_train.py --dataset $dataset --sentiment_class 2 --bert_path ${bert_path} --batch_size 8 --few_shot_num $num
    python few_shot_train.py --dataset $dataset --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size 8 --few_shot_num $num
    python few_shot_train.py --dataset $dataset --sentiment_class 2 --bert_path ${base_bert_path} --batch_size 8 --few_shot_num $num
    python few_shot_train.py --dataset $dataset --sentiment_class 2 --bert_path ${base_bert_path} --fix_bert --batch_size 8 --few_shot_num $num
    echo $num
  done
done


datasets=(SST-5-root)
for dataset in ${datasets[@]}
do
  for num in ${sample_num[@]}
  do
    python few_shot_train.py --dataset $dataset --sentiment_class 5 --bert_path ${bert_path} --batch_size 8 --few_shot_num $num
    python few_shot_train.py --dataset $dataset --sentiment_class 5 --bert_path ${bert_path} --fix_bert --batch_size 8 --few_shot_num $num
    python few_shot_train.py --dataset $dataset --sentiment_class 5 --bert_path ${base_bert_path} --batch_size 8 --few_shot_num $num
    python few_shot_train.py --dataset $dataset --sentiment_class 5 --bert_path ${base_bert_path} --fix_bert --batch_size 8 --few_shot_num $num
    echo $num
  done
done


datasets=(IMDB Yelp)
for dataset in ${datasets[@]}
do
  for num in ${sample_num[@]}
  do
    python few_shot_train.py --dataset $dataset --cross_val_fold 0 --sentiment_class 2 --bert_path ${bert_path} --batch_size 8 --few_shot_num $num
    python few_shot_train.py --dataset $dataset --cross_val_fold 0 --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size 8 --few_shot_num $num
    python few_shot_train.py --dataset $dataset --cross_val_fold 0 --sentiment_class 2 --bert_path ${base_bert_path} --batch_size 8 --few_shot_num $num
    python few_shot_train.py --dataset $dataset --cross_val_fold 0 --sentiment_class 2 --bert_path ${base_bert_path} --fix_bert --batch_size 8 --few_shot_num $num
    echo $num
  done
done

