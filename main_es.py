from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import random
from collections import OrderedDict
from six.moves import xrange
import json
import math
import time
import datetime
import model
#import model_new as model

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='The path to the CIFAR-10 data directory.')

parser.add_argument('--model_dir', type=str, default='/tmp/cifar10_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--num_nodes', type=int, default=7,
                    help='The number of nodes in a cell.')

parser.add_argument('--arch', type=str, default="convx3-reducx1-convx3-reducx1-convx3",
                    help='The architecture of the network.')

parser.add_argument('--filters', type=int, default=36,
                    help='The numer of filters.')

parser.add_argument('--dropout', type=float, default=0.4,
                    help='Dropout rate.')

parser.add_argument('--train_epochs', type=int, default=310,
                    help='The number of epochs to train.')

parser.add_argument('--epochs_per_eval', type=int, default=10,
                    help='The number of epochs to run in between evaluations.')

parser.add_argument('--batch_size', type=int, default=128,
                    help='The number of images per batch.')

parser.add_argument('--random_dag', type=bool, default=False,
                    help='Random sample a dag to run.')

parser.add_argument('--dag', type=int, default=0,
                    help='Dag to run.')

parser.add_argument(
    '--data_format', type=str, default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')

parser.add_argument('--lr_schedule', type=str, default='cosine',
                    choices=['cosine', 'constant', 'decay'],
                    help='Learning rate schedule schema.')

parser.add_argument('--lr', type=float, default='0.1',
                    help='Learning rate when learning rate schedule is constant.')

parser.add_argument('--lr_max', type=float, default=0.05,
                    help='Max learning rate.')

parser.add_argument('--lr_min', type=float, default=0.001,
                    help='Min learning rate.')

parser.add_argument('--T_0', type=int, default=10,
                    help='Epochs of the first cycle.')

parser.add_argument('--T_mul', type=int, default=2,
                    help='Multiplicator for the cycle.')

_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}


def record_dataset(filenames):
  """Returns an input pipeline Dataset from `filenames`."""
  record_bytes = _HEIGHT * _WIDTH * _DEPTH + 1
  return tf.data.FixedLengthRecordDataset(filenames, record_bytes)


def get_filenames(is_training, data_dir):
  """Returns a list of filenames."""
  data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

  assert os.path.exists(data_dir), (
      'Run cifar10_download_and_extract.py first to download and extract the '
      'CIFAR-10 data.')

  if is_training:
    return [
        os.path.join(data_dir, 'data_batch_%d.bin' % i)
        for i in range(1, _NUM_DATA_FILES + 1)
    ]
  else:
    return [os.path.join(data_dir, 'test_batch.bin')]


def parse_record(raw_record):
  """Parse CIFAR-10 image and label from a raw record."""
  # Every record consists of a label followed by the image, with a fixed number
  # of bytes for each.
  label_bytes = 1
  image_bytes = _HEIGHT * _WIDTH * _DEPTH
  record_bytes = label_bytes + image_bytes

  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.decode_raw(raw_record, tf.uint8)

  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  label = tf.cast(record_vector[0], tf.int32)
  label = tf.one_hot(label, _NUM_CLASSES)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      record_vector[label_bytes:record_bytes], [_DEPTH, _HEIGHT, _WIDTH])

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

  return image, label


def preprocess_image(image, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, _HEIGHT + 8, _WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  dataset = record_dataset(get_filenames(is_training, data_dir))

  if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance. Because CIFAR-10
    # is a relatively small dataset, we choose to shuffle the full epoch.
    dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

  dataset = dataset.map(parse_record)
  dataset = dataset.map(
      lambda image, label: (preprocess_image(image, is_training), label))

  dataset = dataset.prefetch(2 * batch_size)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)

  # Batch results by up to batch_size, and then fetch the tuple from the
  # iterator.
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, labels

def _log_variable_sizes(var_list, tag):
  """Log the sizes and shapes of variables, and the total size.

    Args:
      var_list: a list of varaibles
      tag: a string
  """
  name_to_var = {v.name: v for v in var_list}
  total_size = 0
  for v_name in sorted(list(name_to_var)):
    v = name_to_var[v_name]
    v_size = int(np.prod(np.array(v.shape.as_list())))
    tf.logging.info("Weight    %s\tshape    %s\tsize    %d",
      v.name[:-2].ljust(80),
      str(v.shape).ljust(20), v_size)
    total_size += v_size
  tf.logging.info("%s Total size: %d", tag, total_size)

def cifar10_model_fn(features, labels, mode, params):
  """Model function for CIFAR-10."""
  tf.summary.image('images', features, max_outputs=6)

  inputs = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _DEPTH])
  logits = model.build_model(inputs, params, mode == tf.estimator.ModeKeys.TRAIN)

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # Add weight decay to the loss.
  loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])


  _log_variable_sizes(tf.trainable_variables(), "Trainable Variables")

  if mode == tf.estimator.ModeKeys.TRAIN:
    # Scale the learning rate linearly with the batch size. When the batch size
    # is 128, the learning rate should be 0.1.
    global_step = tf.train.get_or_create_global_step()

    if FLAGS.lr_schedule == 'cosine':
      lr_max = FLAGS.lr_max
      lr_min = FLAGS.lr_min
      T_0 = tf.constant(FLAGS.T_0, dtype=tf.float32)
      T_mul = tf.constant(FLAGS.T_mul, dtype=tf.float32)
      batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']
      
      cur_epoch = tf.cast(global_step, dtype=tf.float32) / batches_per_epoch + 1.0
      cur_i = tf.ceil(tf.log((T_mul - 1.0) * (cur_epoch / T_0 + 1.0)) / tf.log(2.0) - 1.0)
      T_beg = T_0 * (tf.pow(T_mul, cur_i) - 1.0) / (T_mul - 1.0)
      T_i = T_0 * tf.pow(T_mul, cur_i)
      
      T_cur = cur_epoch - T_beg
      learning_rate = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(T_cur / T_i * np.pi))
    elif FLAGS.lr_schedule == 'decay':
      batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']
      boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 200, 300]]
      values = [FLAGS.lr * decay for decay in [1, 0.1, 0.01, 0.001]]
      learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)
    else:
      learning_rate = FLAGS.lr
    """
    initial_learning_rate = 0.1 * params['batch_size'] / 128
    batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']
    global_step = tf.train.get_or_create_global_step()
    # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
    boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
    values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
    learning_rate = tf.train.piecewise_constant(
      tf.cast(global_step, tf.int32), boundaries, values)
    """

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM)

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])
  metrics = {'accuracy': accuracy}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)

def build_dag():
  if FLAGS.random_dag:
    conv_dag = get_dag(FLAGS.num_nodes, 'conv_dag')
    reduc_dag = get_dag(FLAGS.num_nodes, 'reduc_dag') 
  else:
    if FLAGS.dag == 0:
      conv_dag = OrderedDict()
      conv_dag['node_1'] = model.Node('node_1', None, None, None, None)
      conv_dag['node_2'] = model.Node('node_2', None, None, None, None)
      conv_dag['node_3'] = model.Node('node_3', 'node_2', 'node_2', 'sep_conv 3x3', 'identity')
      conv_dag['node_4'] = model.Node('node_4', 'node_2', 'node_1', 'sep_conv 5x5', 'identity')
      conv_dag['node_5'] = model.Node('node_5', 'node_1', 'node_2', 'avg_pool 3x3', 'sep_conv 3x3')
      conv_dag['node_6'] = model.Node('node_6', 'node_1', 'node_2', 'sep_conv 3x3', 'avg_pool 3x3')
      conv_dag['node_7'] = model.Node('node_7', 'node_2', 'node_1', 'sep_conv 5x5', 'avg_pool 3x3')

      reduc_dag = OrderedDict()
      reduc_dag['node_1'] = model.Node('node_1', None, None, None, None)
      reduc_dag['node_2'] = model.Node('node_2', None, None, None, None)
      reduc_dag['node_3'] = model.Node('node_3', 'node_1', 'node_2', 'sep_conv 5x5', 'avg_pool 3x3')
      reduc_dag['node_4'] = model.Node('node_4', 'node_2', 'node_2', 'sep_conv 3x3', 'avg_pool 3x3')
      reduc_dag['node_5'] = model.Node('node_5', 'node_2', 'node_2', 'avg_pool 3x3', 'sep_conv 3x3')
      reduc_dag['node_6'] = model.Node('node_6', 'node_5', 'node_2', 'sep_conv 5x5', 'avg_pool 3x3')
      reduc_dag['node_7'] = model.Node('node_7', 'node_6', 'node_1', 'sep_conv 3x3', 'sep_conv 5x5')
    else:
      conv_dag = OrderedDict()
      conv_dag['node_1'] = model.Node('node_1', None, None, None, None)
      conv_dag['node_2'] = model.Node('node_2', None, None, None, None)
      conv_dag['node_3'] = model.Node('node_3', 'node_2', 'node_2', 'conv 3x3', 'identity')
      conv_dag['node_4'] = model.Node('node_4', 'node_2', 'node_1', 'conv 5x5', 'identity')
      conv_dag['node_5'] = model.Node('node_5', 'node_1', 'node_2', 'avg_pool 3x3', 'conv 3x3')
      conv_dag['node_6'] = model.Node('node_6', 'node_1', 'node_2', 'conv 3x3', 'avg_pool 3x3')
      conv_dag['node_7'] = model.Node('node_7', 'node_2', 'node_1', 'conv 5x5', 'avg_pool 3x3')

      reduc_dag = OrderedDict()
      reduc_dag['node_1'] = model.Node('node_1', None, None, None, None)
      reduc_dag['node_2'] = model.Node('node_2', None, None, None, None)
      reduc_dag['node_3'] = model.Node('node_3', 'node_1', 'node_2', 'conv 5x5', 'avg_pool 3x3')
      reduc_dag['node_4'] = model.Node('node_4', 'node_2', 'node_2', 'conv 3x3', 'avg_pool 3x3')
      reduc_dag['node_5'] = model.Node('node_5', 'node_2', 'node_2', 'avg_pool 3x3', 'conv 3x3')
      reduc_dag['node_6'] = model.Node('node_6', 'node_5', 'node_2', 'conv 5x5', 'avg_pool 3x3')
      reduc_dag['node_7'] = model.Node('node_7', 'node_6', 'node_1', 'conv 3x3', 'conv 5x5')


  with open(os.path.join(FLAGS.model_dir, 'model_dag.json'), 'w') as f:
    dag = OrderedDict()
    dag['conv_dag'] = conv_dag
    dag['reduc_dag'] = reduc_dag
    json.dump(dag, f)
  return conv_dag, reduc_dag

def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  conv_dag, reduc_dag = build_dag()

  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
  cifar_classifier = tf.estimator.Estimator(
      model_fn=cifar10_model_fn, model_dir=FLAGS.model_dir, config=run_config,
      params={
        'arch': FLAGS.arch,
        'num_nodes': FLAGS.num_nodes,
        'num_classes': _NUM_CLASSES,
        'filters': FLAGS.filters,
        'dropout': FLAGS.dropout,
        'conv_dag': conv_dag,
        'reduc_dag': reduc_dag,
        'data_format': FLAGS.data_format,
        'batch_size': FLAGS.batch_size,
      })

  for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy'
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    cifar_classifier.train(
        input_fn=lambda: input_fn(
            True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_results = cifar_classifier.evaluate(
        input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size))
    print(eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
