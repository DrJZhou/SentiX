base_bert_path="/home/zhoujie/bert_model/"
bert_path="state_dict/bert_PretrainBERT_data_all_val_f1_0.4717_meanf1_0.6693_Rating_Mask/"
#'books', 'dvd', 'electronics', 'kitchen_&_housewares'
cuda_id="0"
python multidomain_train.py --dataset books --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --batch_size 8
python multidomain_train.py --dataset dvd --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --batch_size 8
python multidomain_train.py --dataset electronics --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --batch_size 8
python multidomain_train.py --dataset 'kitchen_&_housewares' --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --batch_size 8

python multidomain_train.py --dataset books --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size 8
python multidomain_train.py --dataset dvd --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size 8
python multidomain_train.py --dataset electronics --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size 8
python multidomain_train.py --dataset 'kitchen_&_housewares' --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size 8

python multidomain_train.py --dataset books --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --batch_size 8
python multidomain_train.py --dataset dvd --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --batch_size 8
python multidomain_train.py --dataset electronics --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --batch_size 8
python multidomain_train.py --dataset 'kitchen_&_housewares' --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --batch_size 8

python multidomain_train.py --dataset books --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --fix_bert --batch_size 8
python multidomain_train.py --dataset dvd --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --fix_bert --batch_size 8
python multidomain_train.py --dataset electronics --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --fix_bert --batch_size 8
python multidomain_train.py --dataset 'kitchen_&_housewares' --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --fix_bert --batch_size 8
