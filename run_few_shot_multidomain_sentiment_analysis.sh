base_bert_path="/gruntdata/zhoujie/bert_model/"
bert_path="state_dict/bert_PretrainBERT_data_all_val_acc_0.9294_0.6922_Rating_Mask_Multidomain"
sample_rate=(0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
cuda_id="5"
#datasets=('books')
#for dataset in ${datasets[@]}
#do
#  for rate in ${sample_rate[@]}
#  do
#    python multidomain_train_few_shot.py --dataset $dataset --few_shot_rate $rate --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --batch_size 8
#    python multidomain_train_few_shot.py --dataset $dataset --few_shot_rate $rate --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size 8
#    python multidomain_train_few_shot.py --dataset $dataset --few_shot_rate $rate --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --batch_size 8
#    python multidomain_train_few_shot.py --dataset $dataset --few_shot_rate $rate --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --fix_bert --batch_size 8
#    echo $rate
#  done
#done

#datasets=('dvd')
#for dataset in ${datasets[@]}
#do
#  for rate in ${sample_rate[@]}
#  do
#    python multidomain_train_few_shot.py --dataset $dataset --few_shot_rate $rate --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --batch_size 8
#    python multidomain_train_few_shot.py --dataset $dataset --few_shot_rate $rate --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size 8
#    python multidomain_train_few_shot.py --dataset $dataset --few_shot_rate $rate --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --batch_size 8
#    python multidomain_train_few_shot.py --dataset $dataset --few_shot_rate $rate --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --fix_bert --batch_size 8
#    echo $rate
#  done
#done

#datasets=('electronics')
#for dataset in ${datasets[@]}
#do
#  for rate in ${sample_rate[@]}
#  do
#    python multidomain_train_few_shot.py --dataset $dataset --few_shot_rate $rate --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --batch_size 8
#    python multidomain_train_few_shot.py --dataset $dataset --few_shot_rate $rate --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size 8
#    python multidomain_train_few_shot.py --dataset $dataset --few_shot_rate $rate --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --batch_size 8
#    python multidomain_train_few_shot.py --dataset $dataset --few_shot_rate $rate --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --fix_bert --batch_size 8
#    echo $rate
#  done
#done

datasets=('kitchen_&_housewares')
for dataset in ${datasets[@]}
do
  for rate in ${sample_rate[@]}
  do
    python multidomain_train_few_shot.py --dataset $dataset --few_shot_rate $rate --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --batch_size 8
    python multidomain_train_few_shot.py --dataset $dataset --few_shot_rate $rate --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size 8
    python multidomain_train_few_shot.py --dataset $dataset --few_shot_rate $rate --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --batch_size 8
    python multidomain_train_few_shot.py --dataset $dataset --few_shot_rate $rate --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --fix_bert --batch_size 8
    echo $rate
  done
done