# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import subprocess
import os
from six.moves import urllib
import tempfile
import sys
import numpy as np
import re
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_non_alphanum
from unidecode import unidecode
import pickle
import spacy
# sp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

def strip_custom(token):
    token = token.replace('&Reg;', " ")
    token = token.replace("&lt;", "<")
    token = token.replace("&times;", "")
    token = token.replace("&gt;", ">")
    token = token.replace("&quot;", "")
    token = token.replace('&nbsp', " ")
    token = token.replace('&copy;', " ")
    token = token.replace('&reg', " ")
    token = token.replace('%20', " ")
    # this has to be last:
    token = token.replace("&amp;", "&")
    token = token.replace("â\x80¢", " ")
#     token = token.replace("Â®", " ")
#     token = token.replace("Ã©", " ")
#     token = token.replace("®", " ")
#     token = token.replace("©", " ")
#     token = token.replace("™", " ")
#     token = token.replace("•", "")
#     token = token.replace("�", "")
    
    token = token.replace("width:99pt", "")
    token = token.replace('class="xl66">', '')
    token = token.replace('&#160;', ' ')
    return token


def string_processor(token):
    str = unidecode(token)
    str = str.lower()
    
    str = strip_custom(str)
    str = remove_stopwords(str)
    str = remove_tags(str)
    str = strip_punctuation(str)
    str = strip_non_alphanum(str)
    str = strip_multiple_whitespaces(str)

    return str

def processing_sequences(x_test,maxlen=16):
    print(len(x_test), 'sample sequences')

    print('Average sample sequence length: {}'.format(
        np.mean(list(map(len, x_test)))))
    print('Pad sequences (samples x time)')
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    print('X shape:', x_test.shape)
    return x_test

def get_token_for_sentence(corpus,name,save_dir):

    vocabulary_size = 30000
    tokenizer = Tokenizer(num_words= vocabulary_size, 
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
                          lower=True, 
                          split=' ')

    tokenizer.fit_on_texts(corpus)
    sequences = tokenizer.texts_to_sequences(corpus)

    top_word = len(tokenizer.index_word) +1
    print("vocab:",top_word)

    # saving
    saveing_name = name+'tokenizer.pickle'
    with open(saveing_name, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    subprocess.check_call(['gsutil', 'cp', saveing_name, save_dir],
        stderr=sys.stdout)
        
    return np.array(sequences),tokenizer.word_index

def getEBDmatrix(maxWords,word_index,embeddings_index):

    print('Preparing embedding matrix.')

    # prepare embedding matrix
    num_words = min(maxWords, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, 300))
    for word, i in word_index.items():
        if i >= maxWords:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            
    return embedding_matrix



def load_data(data_url,data_filename,job_dir,feature_column):

# # [START download-data]
# data_filename = 'fastText_data.csv'
# data_url = 'gs://enrich_xingming/fast_text_data/test_30k.csv'

# gsutil outputs everything to stderr so we need to divert it to stdout.
    subprocess.check_call(['gsutil', 'cp', data_url,
                           data_filename], stderr=sys.stdout)

#download embedding
    MAX_NUM_WORDS = 30000
    glove_filename = 'glove.42B.300d.zip'
    
    glove_url = 'gs://enrich_xingming/glove/glove.42B.300d.zip'
    subprocess.check_call(['gsutil', 'cp', glove_url,
                           glove_filename], stderr=sys.stdout)
    subprocess.check_call(['unzip', glove_filename], stderr=sys.stdout)

    print('Indexing word vectors.')

    embeddings_index = {}
    with open('glove.42B.300d.txt') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))
    
# [END download-data]
    df = pd.read_csv(data_filename)

    if feature_column == 'name':
        print("\nUse name as feature\n")
        df = df[['bucket_name','product_name']]
        maxlen=16

    if feature_column == 'description':
        print("\nUse description as feature\n")
        df = df[['bucket_name','description']]
        maxlen=75
        
    df.columns = ['bucket_name', 'feature'] 
    
    print("before drop: ",len(df))
    df = df.dropna()
    print("after drop: ",len(df))

    
    ####Text preprocessing
    print("preprocessing text")
    # to lowercase
    df.bucket_name = df.bucket_name.apply(lambda x : x.lower())
    df.feature = df.feature.apply(lambda x : string_processor(x))
    print("\n\nFinished preprocessing text")
    ####Text preprocessing
    
    
    
    CLASSES = LabelEncoder()
    df['label']=CLASSES.fit_transform(df['bucket_name'])
    NUM_OF_CLASS=len(CLASSES.classes_)
    
    encoder_filename = feature_column+'saved_encoder.npy'
    np.save(encoder_filename, CLASSES.classes_)
    subprocess.check_call(['gsutil', 'cp', encoder_filename, job_dir],
        stderr=sys.stdout)

    label = df.label.tolist()
    feature_token, feature_word_index = get_token_for_sentence(df.feature.tolist(),feature_column,job_dir)
    X = processing_sequences(feature_token,maxlen)
    y = np.array(label)


    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y, 
                                                        test_size=0.15)
    
    
    # Reshape label columns for use with tf.data.Dataset
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    # save encoder in the end
    
    EB_feature = getEBDmatrix(MAX_NUM_WORDS,feature_word_index,embeddings_index)
    
    return X_train, y_train, X_test, y_test, NUM_OF_CLASS,EB_feature,MAX_NUM_WORDS


