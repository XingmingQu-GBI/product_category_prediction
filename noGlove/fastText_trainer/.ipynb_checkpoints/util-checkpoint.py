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
    token = token.replace("Â®", " ")
    token = token.replace("Ã©", " ")
    token = token.replace("®", " ")
    token = token.replace("©", " ")
    token = token.replace("™", " ")
    token = token.replace("•", "")
    token = token.replace("�", "")
    
    token = token.replace("width:99pt", "")
    token = token.replace('class="xl66">', '')
    #token = re.sub(r"\'", '', token)
    token = token.replace('&#160;', ' ')
    return token


def string_processor(token):

    str = unidecode(token)
    str = strip_custom(str)
    str = remove_stopwords(str)
    str = strip_punctuation(str)
    str = strip_non_alphanum(str)
#     tokens = sp(str)
#     tokens = [token.lemma_ for token in tokens]
#     str = " ".join(tokens)
    str = strip_multiple_whitespaces(str)

    return str

def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.

    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


# In[33]:


def processing_sequences(x_train,x_test,maxlen=16,ngram_range=1):
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    print('Average train sequence length: {}'.format(
        np.mean(list(map(len, x_train)))))
    print('Average test sequence length: {}'.format(
        np.mean(list(map(len, x_test)))))

    if ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in x_train:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        start_index = max_features + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        # max_features is the highest integer that could be found in the dataset.
        max_features = np.max(list(indice_token.keys())) + 1
        print("max_features:",max_features)

        # Augmenting x_train and x_test with n-grams features
        x_train = add_ngram(x_train, token_indice, ngram_range)
        x_test = add_ngram(x_test, token_indice, ngram_range)
        print('Average train sequence length: {}'.format(
            np.mean(list(map(len, x_train)), dtype=int)))
        print('Average test sequence length: {}'.format(
            np.mean(list(map(len, x_test)), dtype=int)))

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    return x_train,x_test


# In[35]:


def load_data(data_url,data_filename):

# # [START download-data]
# data_filename = 'fastText_data.csv'
# data_url = 'gs://enrich_xingming/fast_text_data/test_30k.csv'

# gsutil outputs everything to stderr so we need to divert it to stdout.
    subprocess.check_call(['gsutil', 'cp', data_url,
                           data_filename], stderr=sys.stdout)

# [END download-data]
    df = pd.read_csv(data_filename)
    df = df[['bucket_name','product_name']]
    print("before drop: ",len(df))
    df = df.dropna()
    print("after drop: ",len(df))
    
    ####Text preprocessing
    print("preprocessing text")
    # to lowercase
    df.bucket_name = df.bucket_name.apply(lambda x : x.lower())
    df.product_name = df.product_name.apply(lambda x : str(x).lower())
    df.product_name = df.product_name.apply(lambda x : string_processor(x))
    print("\n\nFinished preprocessing text")
    ####Text preprocessing
    
    
    
    CLASSES = LabelEncoder()
    df['label']=CLASSES.fit_transform(df['bucket_name'])
    NUM_OF_CLASS=len(CLASSES.classes_)
#     np.save('saved_labelEncoder.npy', CLASSES.classes_)
    
    corpus = df.product_name.tolist()
    label = df.label.tolist()

    vocabulary_size = 20000
    tokenizer = Tokenizer(num_words= vocabulary_size, 
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
                          lower=True, 
                          split=' ')

    tokenizer.fit_on_texts(corpus)
    sequences = tokenizer.texts_to_sequences(corpus)

    X = np.array(sequences)
    y = np.array(label)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y, 
                                                        test_size=0.2)
    
    X_train,X_test=processing_sequences(X_train,X_test,maxlen=16,ngram_range=1)
    
    # Reshape label columns for use with tf.data.Dataset
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    # save encoder in the end
    return X_train, y_train, X_test, y_test, NUM_OF_CLASS,CLASSES


