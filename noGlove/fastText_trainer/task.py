

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

# import model as md
# import util as ut
import subprocess
from . import model as md
from . import util as ut
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import sys

def get_args():
    """Argument parser.
  Returns:
    Dictionary of arguments.
  """

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--job-dir', type=str, required=True,
                        help='local or GCS location for writing checkpoints and exporting models'
                        )
    
    parser.add_argument('--data-dir', type=str, required=True,
                        help='local or GCS location for training CVS file'
                        )
    
    parser.add_argument('--num-epochs', type=int, default=35,
                        help='number of times to go through the data, default=20'
                        )
    parser.add_argument('--batch-size', default=32, type=int,
                        help='number of records to read during each training step, default=128'
                        )
#     parser.add_argument('--learning-rate', default=.01, type=float,
#                         help='learning rate for gradient descent, default=.01'
#                         )

    parser.add_argument('--verbosity', choices=['DEBUG', 'ERROR',
                        'FATAL', 'INFO', 'WARN'], default='INFO')
    (args, _) = parser.parse_known_args()
    return args


def train_and_evaluate(args):
    """Trains and evaluates the Keras model.
  Uses the Keras model defined in model.py and trains on data loaded and
  preprocessed in util.py. Saves the trained model in TensorFlow SavedModel
  format to the path defined in part by the --job-dir argument.
  Args:
    args: dictionary of arguments - see get_args() for details
  """
    data_filename = 'training_data.csv'
#     data_url = 'gs://enrich_xingming/fast_text_data/beauty.csv'
    data_url = args.data_dir
    (train_x, train_y, eval_x, eval_y, NUM_OF_CLASS,saved_encoder) = ut.load_data(data_url,data_filename)

  # dimensions

    (num_train_examples, input_dim) = train_x.shape
    num_eval_examples = eval_x.shape[0]
    print(num_train_examples,num_eval_examples, input_dim)

  # Create the Keras Model
    max_features = 20000
    embedding_dims = 75
    maxlen = 16

    keras_model = md.create_keras_model(max_features, embedding_dims,maxlen,NUM_OF_CLASS)
    print(keras_model.summary())

#   # Pass a numpy array by passing DataFrame.values
    print(train_x.shape, train_y.shape, eval_x.shape, eval_y.shape)
    print(type(train_x), type(train_y), eval_x.shape, eval_y.shape)

    training_dataset = md.input_fn(features=train_x,
            labels=train_y, shuffle=True, num_epochs=args.num_epochs,
            batch_size=args.batch_size)

#   # Pass a numpy array by passing DataFrame.values

    validation_dataset = md.input_fn(features=eval_x,
            labels=eval_y, shuffle=False, num_epochs=args.num_epochs,
            batch_size=num_eval_examples)

#   # Setup Learning Rate decay.

#     lr_decay_cb = \
#         tf.keras.callbacks.LearningRateScheduler(lambda epoch: \
#             args.learning_rate + 0.02 * 0.5 ** (1 + epoch),
#             verbose=True)

  # Setup TensorBoard callback.

    tensorboard_cb = \
        tf.keras.callbacks.TensorBoard(os.path.join(args.job_dir,
            'keras_tensorboard'), histogram_freq=1)

#   # Train model
    current_model_name = "best.h5"
    checkpoint = ModelCheckpoint(current_model_name, monitor='val_loss', verbose=1, save_best_only=True)
    
#     for features, labels in training_dataset.take(10):
#         print(features)
    
    keras_model.fit(
        training_dataset,
        steps_per_epoch=int(num_train_examples / args.batch_size),
        epochs=args.num_epochs,
        validation_data=validation_dataset,
        validation_steps=1,
        verbose=1,
        callbacks=[ EarlyStopping(monitor='val_loss', patience=3),checkpoint, tensorboard_cb],
        )
    
    keras_model.load_weights(current_model_name)

    export_path = os.path.join(args.job_dir, 'keras_export')
#     tf.keras.models.save_model(keras_model, export_path)
    tf.contrib.saved_model.save_keras_model(keras_model, export_path)
    
    # [START encoder]
    # Upload the saved model file to Cloud Storage
    encoder_filename = 'saved_encoder.npy'
    np.save(encoder_filename, saved_encoder.classes_)
    subprocess.check_call(['gsutil', 'cp', encoder_filename, args.job_dir],
        stderr=sys.stdout)
    # [END upload-model]

    print('Model exported to: {}'.format(export_path))


if __name__ == '__main__':
#     config_cmd='python -m spacy download en_core_web_sm'
#     flag=subprocess.call(config_cmd,shell=True)
#     print('\n\n flag = ',flag)
    args = get_args()
    tf.logging.set_verbosity(args.verbosity)
    train_and_evaluate(args)
