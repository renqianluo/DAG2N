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

_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

_WEIGHT_DECAY = 5e-4 #1e-4
_MOMENTUM = 0.9

_NUM_IMAGES = {
    'train': 45000,
    'valid': 5000,
    'test': 10000,
}


parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'test'],
                    help='Train, or test.')

parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='The path to the CIFAR-10 data directory.')

parser.add_argument('--model_dir', type=str, default='/tmp/cifar10_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--previous_steps', type=int, default=0,
                    help='Previous steps when restored.')

parser.add_argument('--num_nodes', type=int, default=7,
                    help='The number of nodes in a cell.')

parser.add_argument('--N', type=int, default=6,
                    help='The number of stacked convolution cell.')

parser.add_argument('--filters', type=int, default=36,
                    help='The numer of filters.')

parser.add_argument('--drop_path_keep_prob', type=float, default=0.6,
                    help='Dropout rate.')

parser.add_argument('--dense_dropout_keep_prob', type=float, default=1.0,
                    help='Dropout rate.')

parser.add_argument('--stem_multiplier', type=float, default=3.0,
                    help='Stem convolution multiplier. Default is 3.0 for CIFAR-10. 1.0 is for ImageNet.')

parser.add_argument('--train_epochs', type=int, default=310,
                    help='The number of epochs to train.')

parser.add_argument('--epochs_per_eval', type=int, default=10,
                    help='The number of epochs to run in between evaluations.')

parser.add_argument('--eval_after', type=int, default=0,
                    help='The number of epochs to run before evaluations.')

parser.add_argument('--batch_size', type=int, default=128,
                    help='The number of images per batch.')

parser.add_argument('--dag', type=str, default=None,
                    help='Default dag to run.')

parser.add_argument('--hparams', type=str, default=None,
                    help='hparams file. All the params will be overrided by this file.')

parser.add_argument('--split_train_valid', action='store_true', default=False,
                    help='Split training data to train set and valid set.')

parser.add_argument('--activation', type=str, default=None,
          help='Activation function for convolutions.')

parser.add_argument('--use_nesterov', action='store_true', default=False,
                    help='Use nesterov in Momentum Optimizer.')

parser.add_argument('--use_aux_head', action='store_true', default=False,
                    help='Use auxillary head.')

parser.add_argument('--aux_head_weight', type=float, default=0.4,
                    help='Weight of auxillary head loss.')

parser.add_argument('--weight_decay', type=float, default=_WEIGHT_DECAY,
                    help='Weight decay.')

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

parser.add_argument('--lr_max', type=float, default=0.025,  #0.05 in ENAS
                    help='Max learning rate.')

parser.add_argument('--lr_min', type=float, default=0.0, #0.001 in ENAS
                    help='Min learning rate.')

parser.add_argument('--T_0', type=int, default=10,
                    help='Epochs of the first cycle.')

parser.add_argument('--T_mul', type=int, default=2,
                    help='Multiplicator for the cycle.')


def record_dataset(filenames):
  """Returns an input pipeline Dataset from `filenames`."""
  record_bytes = _HEIGHT * _WIDTH * _DEPTH + 1
  return tf.data.FixedLengthRecordDataset(filenames, record_bytes)


def get_filenames(split, mode, data_dir):
  """Returns a list of filenames."""
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


def input_fn(split, mode, data_dir, batch_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  dataset = record_dataset(get_filenames(split, mode, data_dir))
  is_training = mode in ['train', 'valid']


  if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance. Because CIFAR-10
    # is a relatively small dataset, we choose to shuffle the full epoch.
    if split:
      dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])
    else:
      dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train']+_NUM_IMAGES['valid'])

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
  res = model.build_model(inputs, params, mode == tf.estimator.ModeKeys.TRAIN)
  logits = res['logits']

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
  loss = cross_entropy + params['weight_decay'] * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

  if 'aux_logits' in res:
    aux_logits = res['aux_logits']
    aux_loss = tf.losses.softmax_cross_entropy(
      logits=aux_logits, onehot_labels=labels, weights=params['aux_head_weight'])
    loss += aux_loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()

    num_images = _NUM_IMAGES['train'] if params['split_train_valid'] else _NUM_IMAGES['train'] + _NUM_IMAGES['valid']

    if params['lr_schedule'] == 'cosine':
      lr_max = params['lr_max']
      lr_min = params['lr_min']
      T_0 = tf.constant(params['T_0'], dtype=tf.float32)
      T_mul = tf.constant(params['T_mul'], dtype=tf.float32)
      batches_per_epoch = math.ceil(num_images / params['batch_size'])
      
      cur_epoch = tf.floor(tf.cast(global_step, dtype=tf.float32) / batches_per_epoch)
      if params['T_mul'] == 1:
        cur_i = tf.floor(cur_epoch / T_0)
        T_beg = T_0 * cur_i
        T_i = T_0
      else:
        cur_i = tf.ceil(tf.log((T_mul - 1.0) * (cur_epoch / T_0 + 1.0)) / tf.log(2.0))
        T_beg = T_0 * (tf.pow(T_mul, cur_i) - 1.0) / (T_mul - 1.0)
        T_i = T_0 * tf.pow(T_mul, cur_i)
      
      T_cur = cur_epoch - T_beg
      learning_rate = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(T_cur / T_i * np.pi))
    elif params['lr_schedule'] == 'decay':
      batches_per_epoch = num_images / params['batch_size']
      boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 200, 300]]
      values = [params['lr'] * decay for decay in [1, 0.1, 0.01, 0.001]]
      learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)
    else:
      learning_rate = params['lr']

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM,
        use_nesterov=params['use_nesterov'])

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      #train_op = optimizer.minimize(loss, global_step)
      gradients, variables = zip(*optimizer.compute_gradients(loss))
      gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
      train_op = optimizer.apply_gradients(zip(gradients, variables), global_step)
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


def build_dag(dag_name_or_path):
  try:
    conv_dag, reduc_dag = eval('dag.{}()'.format(dag_name_or_path))
  except:
    conv_dag, reduc_dag = None, None

  return conv_dag, reduc_dag


def get_params():
  conv_dag, reduc_dag = build_dag(FLAGS.dag)
  
  if FLAGS.split_train_valid:
    total_steps = int(FLAGS.train_epochs * _NUM_IMAGES['train'] / float(FLAGS.batch_size))
  else:
    total_steps = int(FLAGS.train_epochs * (_NUM_IMAGES['train'] + _NUM_IMAGES['valid']) / float(FLAGS.batch_size))
  
  params = vars(FLAGS)
  params['num_classes'] = _NUM_CLASSES
  params['conv_dag'] = conv_dag
  params['reduc_dag'] = reduc_dag
  params['total_steps'] = total_steps

  if FLAGS.hparams is not None:
    with open(os.path.join(FLAGS.hparams), 'r') as f:
      hparams = json.load(f)
      params.update(hparams)
 
  if params['conv_dag'] is None or params['reduc_dag'] is None:
    raise ValueError('You muse specify a registered model name or provide a model in the hparams.')
  
  return params 


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  if FLAGS.mode == 'train':
    params = get_params()

    cifar10_model_fn(tf.zeros([32 ,32 ,32 ,3]),
      tf.one_hot(tf.ones([32], dtype=tf.uint8), _NUM_CLASSES),
      tf.estimator.ModeKeys.TRAIN, params)

    _log_variable_sizes(tf.trainable_variables(), 'Trainable Variables')

    with open(os.path.join(params['model_dir'], 'hparams.json'), 'w') as f:
      json.dump(params, f)
    
    if os.path.exists(os.path.join(params['model_dir'], 'checkpoint')):
      with open(os.path.join(params['model_dir'], 'checkpoint'), 'r') as f:
        line = f.readline()
        line = line.strip().split(' ')[-1]
        line = line.split('-')[-1][:-1]
        previous_step = int(line)
        num_images = _NUM_IMAGES['train'] if params['split_train_valid'] else _NUM_IMAGES['train'] + _NUM_IMAGES['valid']
        batches_per_epoch = num_images / params['batch_size']
        start_epoch = previous_step // batches_per_epoch
    else:
      start_epoch = 0

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
    cifar_classifier = tf.estimator.Estimator(
      model_fn=cifar10_model_fn, model_dir=params['model_dir'], config=run_config,
      params=params)
    if start_epoch < params['eval_after']: 
      start_epoch_loop = int(start_epoch // 10)#FLAGS.epochs_per_eval)
      for _ in range(start_epoch_loop, params['eval_after'] // 10):
        tensors_to_log = {
            'learning_rate': 'learning_rate',
            'cross_entropy': 'cross_entropy',
            'train_accuracy': 'train_accuracy'
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

        cifar_classifier.train(
            input_fn=lambda: input_fn(
              params['split_train_valid'], 'train', params['data_dir'], params['batch_size'],10),
              hooks=[logging_hook])

      rest_epochs = params['train_epochs'] - params['eval_after']
      start_epoch_loop = 0
      for _ in range(start_epoch_loop, rest_epochs // params['epochs_per_eval']):
        tensors_to_log = {
            'learning_rate': 'learning_rate',
            'cross_entropy': 'cross_entropy',
            'train_accuracy': 'train_accuracy'
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

        cifar_classifier.train(
            input_fn=lambda: input_fn(
              params['split_train_valid'], 'train', params['data_dir'], params['batch_size'], params['epochs_per_eval']),
              hooks=[logging_hook])

        if params['split_train_valid']:
          # Valid the model and print results
          eval_results = cifar_classifier.evaluate(
              input_fn=lambda: input_fn(params['split_train_valid'], 'valid', params['data_dir'], params['batch_size']))
          tf.logging.info('Evaluation on valid data set')
          print(eval_results)
      
        # Evaluate the model and print results
        eval_results = cifar_classifier.evaluate(
            input_fn=lambda: input_fn(params['split_train_valid'], 'test', params['data_dir'], params['batch_size']))
        tf.logging.info('Evaluation on test data set')
        print(eval_results)

    else:
      eval_epochs = params['train_epochs'] - params['eval_after']
      start_epoch_from_eval = start_epoch - params['eval_after']
      start_epoch_loop = int(start_epoch_from_eval // params['epochs_per_eval'])
      for _ in range(start_epoch_loop, eval_epochs // params['epochs_per_eval']):
        tensors_to_log = {
            'learning_rate': 'learning_rate',
            'cross_entropy': 'cross_entropy',
            'train_accuracy': 'train_accuracy'
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

        cifar_classifier.train(
            input_fn=lambda: input_fn(
              params['split_train_valid'], 'train', params['data_dir'], params['batch_size'], params['epochs_per_eval']),
              hooks=[logging_hook])

        if params['split_train_valid']:
          # Valid the model and print results
          eval_results = cifar_classifier.evaluate(
              input_fn=lambda: input_fn(params['split_train_valid'], 'valid', params['data_dir'], params['batch_size']))
          tf.logging.info('Evaluation on valid data set')
          print(eval_results)
      
        # Evaluate the model and print results
        eval_results = cifar_classifier.evaluate(
            input_fn=lambda: input_fn(params['split_train_valid'], 'test', params['data_dir'], params['batch_size']))
        tf.logging.info('Evaluation on test data set')
        print(eval_results)
  elif FLAGS.mode == 'test':
    if not os.path.exists(os.path.join(FLAGS.model_dir, 'hparams.json')):
      raise ValueError('No hparams.json found in {0}'.format(FLAGS.model_dir))
    with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'r') as f:
      params = json.load(f)
  
    cifar_classifier = tf.estimator.Estimator(
      model_fn=cifar10_model_fn, model_dir=FLAGS.model_dir, params=params)
    eval_results = cifar_classifier.evaluate(
          input_fn=lambda: input_fn(False, 'test', FLAGS.data_dir, params['batch_size']))
    tf.logging.info('Evaluation on test data set')
    print(eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
