# SentiX
This is the source code of our COLING 2020 paper <SentiX: A Sentiment-Aware Pre-Trained Model for Cross-domain Sentiment Analysis>


## Requirements

* Python 3.6 or higher.
* [PyTorch](http://pytorch.org/) 1.0.0+.
* [transformers](https://github.com/huggingface/transformers) PyTorch 1.2.0.

## Pre-trained Models (PyTorch)
The following pre-trained models are available for download from Google Drive:
* [`SentiX`](...): 
PyTorch version, same setting with BERT-base，loading model with transformers.

## Training

To train SentiX, simply run:
```
sh run_sentix.sh
```
## Evaluation Instructions

To test after setting model path:
```
sh test_sentix.sh
```

## File Structure
- pretrain_multigpu_final.py: main file of pre-trained model
- data_processing: data processing
- models： our models
- data_utils_pretrain_final.py: data_loader


