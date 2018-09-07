from BaselineModels import BasicBiGRUs,BasicBiLSTM
import pandas as pd
import data_helpers as dh
from keras.preprocessing.text import Tokenizer

train_corpus="/home/mohamed/PhD/DATA/WASSA/train.csv"
dev_corpus="/home/mohamed/PhD/DATA/WASSA/val.csv"

def LoadData(Corpus):
    DF=pd.read_csv(Corpus,converters={'text': str})
    X=DF['text'].tolist()
    Y=DF['label'].tolist()
    return X,Y


print('----- Load Train and Test Data --------')
#X_train,Y_train,Y_train_true=dh.LoadData(train_corpus,ClassesDict=dh.get_classes(),Arabic=False)
#X_valid,Y_valid,Y_valid_true=dh.LoadData(dev_corpus,ClassesDict=dh.get_classes(),Arabic=False)
X_train,Ytrain=LoadData(train_corpus)
X_valid,Y_valid=LoadData(dev_corpus)
X_train, X_valid,X_test,wordmap = dh.tokenizeData(X_train, X_valid, vocab_size=dh.get_vocab_size(),X_test=X_valid)
X_train, X_valid,X_test = dh.paddingSequence(X_train, X_valid, maxLen=30,X_test=X_test)
n_symbols,word_map=dh.get_word_map_num_symbols(train_corpus)

basicGRUS=BasicBiGRUs(BiGRU_rand=False,STATIC=False,
                      ExternalEmbeddingModel='/home/mohamed/PhD/DATA_DER/EmbeddingModels/fastText_reduced_embedding.txt',
                      EmbeddingType='fastText',n_symbols=n_symbols,wordmap=word_map)

basicGRUS.model=basicGRUS.Load_model(ModelFileName='models/bigru_fasttext_wassa')

