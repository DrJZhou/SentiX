base_bert_path="/gruntdata/zhoujie/bert_model/"
bert_path="state_dict/bert_PretrainBERT_data_all_val_acc_0.9294_0.6922_Rating_Mask_Multidomain/"
#'books', 'dvd', 'electronics', 'kitchen_&_housewares'
cuda_id="7"
python multidomain_inference.py --dataset books --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --batch_size 8
python multidomain_inference.py --dataset dvd --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --batch_size 8
python multidomain_inference.py --dataset electronics --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --batch_size 8
python multidomain_inference.py --dataset 'kitchen_&_housewares' --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --batch_size 8

python multidomain_inference.py --dataset books --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size 8
python multidomain_inference.py --dataset dvd --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size 8
python multidomain_inference.py --dataset electronics --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size 8
python multidomain_inference.py --dataset 'kitchen_&_housewares' --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size 8

bert_path="state_dict/bert_PretrainBERT_data_all_val_acc_0.9409_MLM/"
python multidomain_inference.py --dataset books --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --batch_size 8
python multidomain_inference.py --dataset dvd --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --batch_size 8
python multidomain_inference.py --dataset electronics --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --batch_size 8
python multidomain_inference.py --dataset 'kitchen_&_housewares' --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --batch_size 8

python multidomain_inference.py --dataset books --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size 8
python multidomain_inference.py --dataset dvd --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size 8
python multidomain_inference.py --dataset electronics --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size 8
python multidomain_inference.py --dataset 'kitchen_&_housewares' --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${bert_path} --fix_bert --batch_size 8


python multidomain_inference.py --dataset books --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --batch_size 8
python multidomain_inference.py --dataset dvd --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --batch_size 8
python multidomain_inference.py --dataset electronics --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --batch_size 8
python multidomain_inference.py --dataset 'kitchen_&_housewares' --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --batch_size 8

python multidomain_inference.py --dataset books --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --fix_bert --batch_size 8
python multidomain_inference.py --dataset dvd --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --fix_bert --batch_size 8
python multidomain_inference.py --dataset electronics --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --fix_bert --batch_size 8
python multidomain_inference.py --dataset 'kitchen_&_housewares' --device cuda:${cuda_id} --sentiment_class 2 --bert_path ${base_bert_path} --fix_bert --batch_size 8
