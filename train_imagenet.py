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

_WEIGHT_DECAY = 3e-5
_NUM_CLASSES = 1000

_NUM_IMAGES = {
    'train': 1281167,
    'valid': 0,
    'test': 50000,
}

_MOMENTUM = 0.9
_TEST_BATCH_SIZE = 100

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'test'],
                    help='Train, or test.')

parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='The path to the CIFAR-10 data directory.')

parser.add_argument('--dataset', type=str, default='imagenet',
                    help='imagenet.')

parser.add_argument('--model_dir', type=str, default='/tmp/cifar10_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--num_nodes', type=int, default=7,
                    help='The number of nodes in a cell.')

parser.add_argument('--N', type=int, default=4,
                    help='The number of stacked convolution cell.')

parser.add_argument('--filters', type=int, default=36,
                    help='The numer of filters.')

parser.add_argument('--input_size', type=int, default=224,
                    help='The size of input image. 331 for large imagenet setting, 224 for mobile imagenet setting.')

parser.add_argument('--drop_path_keep_prob', type=float, default=0.6,
                    help='Dropout rate.')

parser.add_argument('--dense_dropout_keep_prob', type=float, default=0.5,
                    help='Dropout rate.')

parser.add_argument('--stem_multiplier', type=float, default=3.0,
                    help='Stem convolution multiplier.')

parser.add_argument('--label_smoothing', type=float, default=0.1,
                    help='Label smoothing.')

parser.add_argument('--train_epochs', type=int, default=300,
                    help='The number of epochs to train.')

parser.add_argument('--epochs_per_eval', type=int, default=1,
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

parser.add_argument('--use_aux_head', action='store_true', default=False,
                    help='Use auxillary head.')

parser.add_argument('--aux_head_weight', type=float, default=0.4,
                    help='Weight of auxillary head loss.')

parser.add_argument('--weight_decay', type=float, default=_WEIGHT_DECAY,
                    help='Weight decay.')

parser.add_argument('--cutout_size', type=int, default=None,
                    help='Size of cutout. Default to None, means no cutout.')

parser.add_argument('--num_gpus', type=int, default=1,
                    help='Number of GPU to use.')

parser.add_argument('--seed', type=int, default=None,
                    help='Seed to use.')

parser.add_argument(
    '--data_format', type=str, default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')

parser.add_argument('--optimizer', type=str, default='sgd',
                    help='Optimizer to use.')

parser.add_argument('--lr', type=float, default='0.001',
                    help='Learning rate when learning rate schedule is constant.')

parser.add_argument('--decay_every', type=int, default=2,
                    help='Epochs that learning rate decays.')

parser.add_argument('--decay_rate', type=float, default=0.97,
                    help='Decay rate of learning rate.')

parser.add_argument('--clip', type=float, default=10.0,
                    help='Clip gradients according to global norm.')


def record_dataset(filenames, mode):
  """Returns an input pipeline Dataset from `filenames`."""
  tf.logging.info("Reading data files from %s", filenames)
  data_files = tf.contrib.slim.parallel_reader.get_data_files(filenames)
  if mode == 'train':
    random.shuffle(data_files)
  return tf.data.TFRecordDataset(data_files)
  

def get_filenames(split, mode, data_dir):
  assert os.path.exists(data_dir)
  if mode == 'train':
    filepattern = os.path.join(data_dir, 'train-*')
  else:
    filepattern = os.path.join(data_dir, 'valid-*')
  return filepattern

def parse_record(raw_record):
  data_fields = {
    'image/encoded': tf.FixedLenFeature(
      (), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature(
      (), tf.string, default_value='jpeg'),
    'image/class/label': tf.FixedLenFeature(
      [], dtype=tf.int64, default_value=-1),
  }
  data_items_to_decoders = {
    'image': tf.contrib.slim.tfexample_decoder.Image('image/encoded', 'image/format'),
    'label': tf.contrib.slim.tfexample_decoder.Tensor('image/class/label'),
  }
  decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
    data_fields, data_items_to_decoders)
  decode_items = list(data_items_to_decoders)
  decoded = decoder.decode(raw_record, items=decode_items)
  image, label = decoded
  label = label - 1
  label = tf.one_hot(label, _NUM_CLASSES)
  return image, label
    

def preprocess_image(image, mode, input_size, cutout_size):
  """Preprocess a single image of layout [height, width, depth]."""
  # convert to [0,1]
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  if mode == 'train':
    image = tf.image.resize_images(image, [input_size, input_size])
    image.set_shape([input_size, input_size, 3])
    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    #image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])
    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32./255.)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    
  else:
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [input_size, input_size], align_corners=False)
    image = tf.squeeze(image, [0])

  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  
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


def input_fn(split, mode, data_dir, batch_size, input_size, cutout_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for  dataset.

  Args:
    mode: train, valid or test.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  data_set = record_dataset(get_filenames(split, mode, data_dir, ), mode)

  if mode == 'train':
      data_set = data_set.shuffle(buffer_size=50000)

  data_set = data_set.map(parse_record, num_parallel_calls=16)
  data_set = data_set.map(
      lambda image, label: (preprocess_image(image, mode, input_size, cutout_size), label),
      num_parallel_calls=16)

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


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def get_train_ops(x, y, params, reuse=False):
  global_step = tf.train.get_or_create_global_step()

  num_images = _NUM_IMAGES['train'] if params['split_train_valid'] else _NUM_IMAGES['train'] + _NUM_IMAGES['valid']

  lr = params['lr']
  batches_per_epoch = math.ceil(num_images / params['batch_size'])
  params['batches_per_epoch'] = batches_per_epoch
  learning_rate = tf.train.exponential_decay(
    learning_rate = lr,
    global_step = global_step,
    decay_steps = batches_per_epoch * params['decay_every'],
    decay_rate = params['decay_rate'],
    staircase=True,
    name="learning_rate",
  )

  tf.summary.scalar('learning_rate', learning_rate)

  if params['optimizer'] == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate,
        decay = 0.9,
        epsilon = 1.0)
  elif params['optimizer'] == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
      learning_rate=learning_rate,
      momentum=_MOMENTUM)
  elif params['optimizer'] == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=learning_rate
    )
  else:
    raise ValueError('Unrecoginized optimizer: {}'.format(params['optimizer']))

  inputs = tf.reshape(x, [-1, params['input_size'], params['input_size'], 3])
  labels = y
  inputs_sharded = tf.split(inputs, params['num_gpus'], axis=0)
  labels_sharded = tf.split(labels, params['num_gpus'], axis=0)
  loss_sharded = []
  cross_entropy_sharded = []
  tower_grads = []
  train_top1_accuracy_sharded = []
  train_top5_accuracy_sharded = []
  for i in range(params['num_gpus']):
    with tf.device('/gpu:%d'%i):
      with tf.name_scope('shard_%d'%i):
        res = model.build_model(inputs_sharded[i], params, True, reuse if i==0 else True)
        logits = res['logits']
        cross_entropy = tf.losses.softmax_cross_entropy(
          logits=logits, onehot_labels=labels_sharded[i], label_smoothing=params['label_smoothing'])
        if 'aux_logits' in res:
          aux_logits = res['aux_logits']
          aux_loss = tf.losses.softmax_cross_entropy(
            logits=aux_logits, onehot_labels=labels_sharded[i], weights=params['aux_head_weight'])
        loss = cross_entropy + aux_loss
        loss = loss + params['weight_decay'] * tf.add_n(
          [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        grads = optimizer.compute_gradients(loss)
        tower_grads.append(grads)
        loss_sharded.append(loss)
        cross_entropy_sharded.append(cross_entropy)
        predictions = logits
        targets = tf.argmax(labels_sharded[i], axis=1)
        train_top1_accuracy = tf.reduce_mean(
          tf.cast(tf.nn.in_top_k(predictions, targets, 1, 'top1'), dtype=tf.float32))
        train_top5_accuracy = tf.reduce_mean(
          tf.cast(tf.nn.in_top_k(predictions, targets, 5, 'top5'), dtype=tf.float32))
        train_top1_accuracy_sharded.append(train_top1_accuracy)
        train_top5_accuracy_sharded.append(train_top5_accuracy)

  loss = tf.reduce_mean(loss_sharded, axis=0)
  cross_entropy = tf.reduce_mean(cross_entropy_sharded, axis=0)
  tf.summary.scalar('training_loss', loss)
  train_top1_accuracy = tf.reduce_mean(train_top1_accuracy_sharded, axis=0)
  train_top5_accuracy = tf.reduce_mean(train_top5_accuracy_sharded, axis=0)
  tf.summary.scalar('train_top1_accuracy', train_top1_accuracy)
  tf.summary.scalar('train_top5_accuracy', train_top5_accuracy)
  grads = average_gradients(tower_grads)
  # Batch norm requires update ops to be added as a dependency to the train_op
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    #gradients, variables = zip(*optimizer.compute_gradients(loss))
    #gradients, _ = tf.clip_by_global_norm(gradients, params['clip'])
    #train_op = optimizer.apply_gradients(zip(gradients, variables), global_step)
    gradients, variables = zip(*grads)
    gradients, _ = tf.clip_by_global_norm(gradients, params['clip'])
    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step)
  
  return cross_entropy, loss, learning_rate, train_top1_accuracy, train_top5_accuracy, train_op, global_step

def get_valid_ops(x, y, params, reuse=False):
  with tf.device('/gpu:0'):
    inputs = tf.reshape(x, [-1, params['input_size'], params['input_size'], 3])
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
    predictions = logits
    labels = tf.argmax(y, axis=1)
    top1_accuracy = tf.reduce_mean(
      tf.cast(tf.nn.in_top_k(predictions, labels, 1, 'top1'), dtype=tf.float32))
    top5_accuracy = tf.reduce_mean(
      tf.cast(tf.nn.in_top_k(predictions, labels, 5, 'top5'), dtype=tf.float32))
    return cross_entropy, loss, top1_accuracy, top5_accuracy

def get_test_ops(x, y, params, reuse=False):
  with tf.device('/gpu:0'):
    inputs = tf.reshape(x, [-1, params['input_size'], params['input_size'], 3])
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

    predictions = logits
    labels = tf.argmax(y, axis=1)
    top1_accuracy = tf.reduce_mean(
      tf.cast(tf.nn.in_top_k(predictions, labels, 1, 'top1'), dtype=tf.float32))
    top5_accuracy = tf.reduce_mean(
      tf.cast(tf.nn.in_top_k(predictions, labels, 5, 'top5'), dtype=tf.float32))
    return cross_entropy, loss, top1_accuracy, top5_accuracy

def train(params):
  g = tf.Graph()
  with g.as_default(), tf.device('/cpu:0'):
    tf.set_random_seed(params['seed'])
    x_train, y_train = input_fn(params['split_train_valid'], 'train', params['data_dir'], params['batch_size'], params['input_size'], params['cutout_size'], None)
    if params['split_train_valid']:
      x_valid, y_valid = input_fn(params['split_train_valid'], 'valid', params['data_dir'], 100, params['input_size'], None, None)
    else:
      x_valid, y_valid = None, None
    x_test, y_test = input_fn(False, 'test', params['data_dir'], 100, params['input_size'], None, None)
    train_cross_entropy, train_loss, learning_rate, train_top1_accuracy, train_top5_accuracy, train_op, global_step = get_train_ops(x_train, y_train, params)
    _log_variable_sizes(tf.trainable_variables(), 'Trainable Variables')
    if params['split_train_valid']:
      valid_cross_entropy, valid_loss, valid_top1_accuracy, valid_top5_accuracy = get_valid_ops(x_valid, y_valid, params, True)
    test_cross_entropy, test_loss, test_top1_accuracy, test_top5_accuracy = get_test_ops(x_test, y_test, params, True)
    saver = tf.train.Saver(max_to_keep=10)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(
      params['model_dir'], save_steps=params['batches_per_epoch'], saver=saver)
    hooks = [checkpoint_saver_hook]
    tf.logging.info('Starting Session')
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.train.SingularMonitoredSession(
      config=config, hooks=hooks, checkpoint_dir=params['model_dir']) as sess:
      start_time = time.time()
      while True:
        run_ops = [
          train_cross_entropy,
          train_loss,
          learning_rate,
          train_top1_accuracy,
          train_top5_accuracy,
          train_op,
          global_step
        ]
        train_cross_entropy_v, train_loss_v, learning_rate_v, train_top1_accuracy_v, train_top5_accuracy_v, _, global_step_v = sess.run(run_ops)

        epoch = global_step_v // params['batches_per_epoch'] 
        curr_time = time.time()
        if global_step_v % 100 == 0:
          log_string = "epoch={:<6d} ".format(epoch)
          log_string += "step={:<6d} ".format(global_step_v)
          log_string += "cross_entropy={:<6f} ".format(train_cross_entropy_v)
          log_string += "loss={:<6f} ".format(train_loss_v)
          log_string += "learning_rate={:<8.4f} ".format(learning_rate_v)
          log_string += "training_top1_accuracy={:<8.4f} ".format(train_top1_accuracy_v)
          log_string += "training_top5_accuracy={:<8.4f} ".format(train_top5_accuracy_v)
          log_string += "mins={:<10.2f}".format((curr_time - start_time) / 60)
          tf.logging.info(log_string)
        if global_step_v % params['batches_per_epoch'] == 0:
          if params['split_train_valid']:
            valid_ops = [
              valid_cross_entropy, valid_loss, valid_top1_accuracy, valid_top5_accuracy,
            ]
            valid_start_time = time.time()
            valid_loss_list = []
            valid_cross_entropy_list = []
            valid_top1_accuracy_list = []
            valid_top5_accuracy_list = []
            for _ in range(_NUM_IMAGES['valid'] // 100):
              valid_cross_entropy_v, valid_loss_v, valid_top1_accuracy_v, valid_top5_accuracy_v = sess.run(valid_ops)
              valid_cross_entropy_list.append(valid_cross_entropy_v)
              valid_loss_list.append(valid_loss_v)
              valid_top1_accuracy_list.append(valid_top1_accuracy_v)
              valid_top5_accuracy_list.append(valid_top5_accuracy_v)
            valid_time = time.time() - valid_start_time
            log_string =  "Evaluation on valid data\n"
            log_string += "epoch={:<6d} ".format(epoch)
            log_string += "step={:<6d} ".format(global_step_v)
            log_string += "cross_entropy={:<6f} ".format(np.mean(valid_cross_entropy_list))
            log_string += "loss={:<6f} ".format(np.mean(valid_loss_list))
            log_string += "learning_rate={:<8.6f} ".format(learning_rate_v)
            log_string += "valid_top1_accuracy={:<8.6f} ".format(np.mean(valid_top1_accuracy_list))
            log_string += "valid_top5_accuracy={:<8.6f} ".format(np.mean(valid_top5_accuracy_list))
            log_string += "secs={:<10.2f}".format((valid_time))
            tf.logging.info(log_string)
          
          test_ops = [
            test_cross_entropy, test_loss, test_top1_accuracy, test_top5_accuracy,
          ]
          test_start_time = time.time()
          test_cross_entropy_list = []
          test_loss_list = []
          test_top1_accuracy_list = []
          test_top5_accuracy_list = []
          for _ in range(_NUM_IMAGES['test'] // 100):
            test_cross_entropy_v, test_loss_v, test_top1_accuracy_v, test_top5_accuracy_v = sess.run(test_ops)
            test_cross_entropy_list.append(test_cross_entropy_v)
            test_loss_list.append(test_loss_v)
            test_top1_accuracy_list.append(test_top1_accuracy_v)
            test_top5_accuracy_list.append(test_top5_accuracy_v)
          test_time = time.time() - test_start_time
          log_string =  "Evaluation on test data\n"
          log_string += "epoch={:<6d} ".format(epoch)
          log_string += "step={:<6d} ".format(global_step_v)
          log_string += "cross_entropy={:<6f} ".format(np.mean(test_cross_entropy_list))
          log_string += "loss={:<6f} ".format(np.mean(test_loss_list))
          log_string += "learning_rate={:<8.6f} ".format(learning_rate_v)
          log_string += "test_top1_accuracy={:<8.6f} ".format(np.mean(test_top1_accuracy_list))
          log_string += "test_top5_accuracy={:<8.6f} ".format(np.mean(test_top5_accuracy_list))
          log_string += "secs={:<10.2f}".format((test_time))
          tf.logging.info(log_string)
        if epoch >= params['train_epochs']:
          break


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
    with open(os.path.join(params['model_dir'], 'hparams.json'), 'w') as f:
      json.dump(params, f)
    train(params)

  elif FLAGS.mode == 'test':
    pass


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
