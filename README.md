# Text-Classification-of-the-shelf-with-a-normalizer-for-Arabic-text-

*This is a simple text classification library based on keras associated with a text normalization script for Arabic*

# Current Implemented Models:

1- Word Level CNN based on:
"Convultion Neural Network for Text Classificartion"
url: http://www.aclweb.org/anthology/D14-1181

2- Word Level C-LSTM based on:
"A C-LSTM Neural Network for Text Classification"
url:https://arxiv.org/pdf/1511.08630.pdf

3- Recurrent Network and its variants (BiLSTM, LSTM, GRU, BiGRU, Attention-BiLSTM)

4- Models implemented but currently not supported in options (Attention-LSTM,Attention-BiGRU).

5- Not yet tested  (char level CNN). 

# Requirements

1- keras (2.0 or above)
2- gensim
3- numpy
4- pandas

# General Usage:
* Tested with python 3.4 *
python test_baselines.py --train training_file --Ar='True' --dev Dev_File --test test_file --model_type=model_selection --static=Trainable_embeddings --rand=Random_Embeddings --embedding=External_Embedding_model --model_file=Output_model_file_inJson

# Options details #

- train: training file assuming in csv format, text, label
- Ar: if True then Arabic normalization is applied (should be true in case of external embeddings)
- dev: Development file in csv format 
- test: test file in csv format
- model_type: currently support those type of models: (cnn: word level cnn, clstm: word level clstm, lstm: vanilla lstm architecture, blstm: Vanilla bidirectional LSTM, bigru: Vanilla BiGated Recurrent unit, attbilstm: BiLSTM with self attention mechanism)
- static: used in case of external embedding, if True: External Embeddings are not fine tuned during training, if False: External EMbeddings are fine tuned during testing). 
- rand: if True, No external embedding is applied, randomly initialized embedding 
- embedding: External embedding model in gensim format
- model_file: Output model file in Json.

