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
import dag
#from utils import data_parallelism

_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_NUM_DATA_FILES = 5

_WEIGHT_DECAY = 5e-4 #1e-4
_MOMENTUM = 0.9

_NUM_IMAGES = {
    'train': 45000,
    'valid': 5000,
    'test': 10000,
}

_TEST_BATCH_SIZE = 100

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='The path to the CIFAR-10 data directory.')

parser.add_argument('--dataset', type=str, default='cifar10',
                    help='CIFAR-10, CIFAR-100.')

parser.add_argument('--model_dir', type=str, default='/tmp/cifar10_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--seed', type=int, default=None,
                    help='Seed to use.')

parser.add_argument(
    '--data_format', type=str, default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')


def record_dataset(filenames, dataset, mode):
  """Returns an input pipeline Dataset from `filenames`."""
  if dataset == 'cifar10':
    record_bytes = _HEIGHT * _WIDTH * _DEPTH + 1
    return tf.data.FixedLengthRecordDataset(filenames, record_bytes)
  elif dataset == 'cifar100':
    record_bytes = _HEIGHT * _WIDTH * _DEPTH + 2
    return tf.data.FixedLengthRecordDataset(filenames, record_bytes)
  
  
def get_filenames(split, mode, data_dir, dataset):
  """Returns a list of filenames."""
  if dataset == 'cifar10':
    if not split:
      data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

    assert os.path.exists(data_dir), (
        'Run cifar10_download_and_extract.py first to download and extract the '
        'CIFAR-10 data.')

    if split:
      if mode == 'train':
        return [
          os.path.join(data_dir, 'train_batch_%d.bin' % i)
          for i in range(1, _NUM_DATA_FILES + 1)]
      elif mode == 'valid':
        return [os.path.join(data_dir, 'valid_batch.bin')]
      else:
        return [os.path.join(data_dir, 'test_batch.bin')]
    else:
      if mode == 'train':
        return [
          os.path.join(data_dir, 'data_batch_%d.bin' % i)
          for i in range(1, _NUM_DATA_FILES + 1)
    ]
      else:
        return [os.path.join(data_dir, 'test_batch.bin')]
  
  elif dataset == 'cifar100':
    data_dir = os.path.join(data_dir, 'cifar-100-binary')

    assert os.path.exists(data_dir)

    if mode == 'train':
      return [os.path.join(data_dir, 'train.bin')]
    else:
      return [os.path.join(data_dir, 'test.bin')]


def parse_record(raw_record, dataset):
  #Parse CIFAR image and label from a raw record.
  # Every record consists of a label followed by the image, with a fixed number
  # of bytes for each.
  if dataset == 'cifar10':
    label_bytes = 1
  elif dataset == 'cifar100':
    label_bytes = 2
  image_bytes = _HEIGHT * _WIDTH * _DEPTH
  record_bytes = label_bytes + image_bytes
  
  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.decode_raw(raw_record, tf.uint8)
  
  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  if dataset == 'cifar10':
    label = tf.cast(record_vector[0], tf.int32)
    label = tf.one_hot(label, 10)
  elif dataset == 'cifar100':
    label = tf.cast(record_vector[1], tf.int32)
    label = tf.one_hot(label, 100)
  
  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      record_vector[label_bytes:record_bytes], [_DEPTH, _HEIGHT, _WIDTH])
  
  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
  
  return image, label
  
  
def preprocess_image(image, mode, cutout_size):
  """Preprocess a single image of layout [height, width, depth]."""
  if mode == 'train':
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, _HEIGHT + 8, _WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)

  if mode == 'train' and cutout_size is not None:
    mask = tf.ones([cutout_size, cutout_size], dtype=tf.int32)
    start = tf.random_uniform([2], minval=0, maxval=32, dtype=tf.int32)
    mask = tf.pad(mask, [[cutout_size + start[0], 32 - start[0]],
                        [cutout_size + start[1], 32 - start[1]]])
    mask = mask[cutout_size: cutout_size + 32,
                cutout_size: cutout_size + 32]
    mask = tf.reshape(mask, [32, 32, 1])
    mask = tf.tile(mask, [1, 1, 3])
    image = tf.where(tf.equal(mask, 0), x=image, y=tf.zeros_like(image))
  return image


def input_fn(split, mode, data_dir, dataset, batch_size, cutout_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    mode: train, valid or test.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  data_set = record_dataset(get_filenames(split, mode, data_dir, dataset), dataset, mode)

  if mode == 'train':
      data_set = data_set.shuffle(buffer_size=50000)

  data_set = data_set.map(lambda x:parse_record(x, dataset), num_parallel_calls=16)
  data_set = data_set.map(
      lambda image, label: (preprocess_image(image, mode, cutout_size), label),
      num_parallel_calls=4)

  data_set = data_set.repeat(num_epochs)
  data_set = data_set.batch(batch_size)
  data_set = data_set.prefetch(10)
  iterator = data_set.make_one_shot_iterator()
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


def get_test_ops(x, y, params, reuse=False):
  with tf.device('/gpu:0'):
    inputs = tf.reshape(x, [-1, _HEIGHT, _WIDTH, _DEPTH])
    labels = y
    res = model.build_model(inputs, params, False, reuse)
    logits = res['logits']
    cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)
    # Add weight decay to the loss.
    loss = cross_entropy + params['weight_decay'] * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    if 'aux_logits' in res:
      aux_logits = res['aux_logits']
      aux_loss = tf.losses.softmax_cross_entropy(
        logits=aux_logits, onehot_labels=labels, weights=params['aux_head_weight'])
      loss += aux_loss

    predictions = tf.argmax(logits, axis=1)
    labels = tf.argmax(y, axis=1)
    test_accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), dtype=tf.float32))
    return loss, test_accuracy


def test(params):
  g = tf.Graph()
  with g.as_default(), tf.device('/cpu:0'):
    tf.set_random_seed(params['seed'])
    x_test, y_test = input_fn(False, 'test', params['data_dir'], params['dataset'], 100, None, None)
    _log_variable_sizes(tf.trainable_variables(), 'Trainable Variables')
    test_loss, test_accuracy = get_test_ops(x_test, y_test, params, True)
    tf.logging.info('Starting Session')
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.train.SingularMonitoredSession(
      config=config, checkpoint_dir=params['model_dir']) as sess:
      test_ops = [
        test_loss, test_accuracy
      ]
      test_start_time = time.time()
      test_loss_list = []
      test_accuracy_list = []
      for _ in range(_NUM_IMAGES['test'] // 100):
        test_loss_v, test_accuracy_v = sess.run(test_ops)
        test_loss_list.append(test_loss_v)
        test_accuracy_list.append(test_accuracy_v)
      test_time = time.time() - test_start_time
      log_string =  "Evaluation on test data\n"
      log_string += "loss={:<6f} ".format(np.mean(test_loss_list))
      log_string += "test_accuracy={:<8.6f} ".format(np.mean(test_accuracy_list))
      log_string += "secs={:<10.2f}".format((test_time))
      tf.logging.info(log_string)


def get_params():
  if not os.path.exists(os.path.join(FLAGS.model_dir, 'hparams.json')):
    raise ValueError('model_dir/hparams.json does not exist ')
  with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'r') as f:
    params = json.load(f)
  params.update({
    'model_dir': FLAGS.model_dir,
    'data_dir': FLAGS.data_dir,
    'data_format': FLAGS.data_format,
    'dataset': FLAGS.dataset,
  })
  return params 


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  params = get_params()
  test(params)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
