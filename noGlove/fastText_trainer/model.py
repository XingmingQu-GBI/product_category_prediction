from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf




def input_fn(features, labels, shuffle, num_epochs, batch_size):
    """Generates an input function to be used for model training.
    Args:
    features: numpy array of features used for training or inference
    labels: numpy array of labels for each example
    shuffle: boolean for whether to shuffle the data or not (set True for
      training, False for evaluation)
    num_epochs: number of epochs to provide the data for
    batch_size: batch size for training
    Returns:
    A tf.data.Dataset that can provide data to the Keras model for training or
      evaluation
    """
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def create_keras_model(max_features, embedding_dims,maxlen,NUM_OF_CLASS):
    print('Build model...')
#     model = Sequential()

#     # we start off with an efficient embedding layer which maps
#     # our vocab indices into embedding_dims dimensions
#     model.add(Embedding(max_features,
#                         embedding_dims,
#                         input_length=maxlen))

#     # we add a GlobalAveragePooling1D, which will average the embeddings
#     # of all words in the document
#     model.add(GlobalAveragePooling1D())

#     # We project onto a single unit output layer, and squash it with a sigmoid:
#     model.add(Dense(NUM_OF_CLASS, activation='softmax'))
    
    Dense = tf.keras.layers.Dense
    Embedding = tf.keras.layers.Embedding
    GlobalAveragePooling1D = tf.keras.layers.GlobalAveragePooling1D
    
#     Dense = tf.keras.layers.Dense
    model = tf.keras.Sequential(
      [
          Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen),
          
          GlobalAveragePooling1D(),
          
          
#           Dense(100, activation=tf.nn.relu, kernel_initializer='uniform',
#                   input_shape=(16,)),
          Dense(NUM_OF_CLASS, activation=tf.nn.softmax)


      ])

    

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model