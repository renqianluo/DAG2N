from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import json
import math
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from slim.datasets import imagenet
from slim.preprocessing import inception_preprocessing
import dag
import model

#from utils import data_parallelism

_WEIGHT_DECAY = 4e-5
_NUM_CLASSES = 1000
_NUM_IMAGES = {
    'train': 1281167,
    'valid': 0,
    'test': 50000,
}
_MOMENTUM = 0.9
_TEST_BATCH_SIZE = 100
model._BATCH_NORM_DECAY = 0.9997
model._BATCH_NORM_EPSILON = 0.001

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'test'],
                    help='Train, or test.')

parser.add_argument('--data_dir', type=str, default='data/imagenet/2012/data',
                    help='The path to the imagenet data directory.')

parser.add_argument('--model_dir', type=str, default='models',
                    help='The directory where the model will be stored.')

parser.add_argument('--labels_offset', type=int, default=1,
                    help='.')

parser.add_argument('--num_nodes', type=int, default=5,
                    help='The number of nodes in a cell.')

parser.add_argument('--N', type=int, default=4,
                    help='The number of stacked convolution cell.')

parser.add_argument('--filters', type=int, default=36,
                    help='The numer of filters.')

parser.add_argument('--skip_reduction_layer_input', type=int, default=0,
                    help='0 for mobile setting, and 1 for large setting.')

parser.add_argument('--train_image_size', type=int, default=224,
                    help='The size of input image. 331 for large imagenet setting, 224 for mobile imagenet setting.')

parser.add_argument('--eval_image_size', type=int, default=224,
                    help='The size of input image. 331 for large imagenet setting, 224 for mobile imagenet setting.')

parser.add_argument('--drop_path_keep_prob', type=float, default=1.0,
                    help='Dropout rate.')

parser.add_argument('--dense_dropout_keep_prob', type=float, default=0.5,
                    help='Dropout rate.')

parser.add_argument('--stem_multiplier', type=float, default=1.0,
                    help='Stem convolution multiplier. 3.0 for large setting, 1.0 for mobile setting.')

parser.add_argument('--label_smoothing', type=float, default=0.1,
                    help='Label smoothing.')

parser.add_argument('--train_epochs', type=int, default=312,
                    help='The number of epochs to train.')

parser.add_argument('--epochs_per_eval', type=int, default=1,
                    help='The number of epochs to run in between evaluations.')

parser.add_argument('--eval_after', type=int, default=0,
                    help='The number of epochs to run before evaluations.')

parser.add_argument('--batch_size', type=int, default=32,
                    help='The number of images per batch.')

parser.add_argument('--arch', type=str, default=None,
                    help='Default dag to run.')

parser.add_argument('--hparams', type=str, default=None,
                    help='hparams file. All the params will be overrided by this file.')

parser.add_argument('--use_aux_head', action='store_true', default=False,
                    help='Use auxillary head.')

parser.add_argument('--aux_head_weight', type=float, default=0.4,
                    help='Weight of auxillary head loss.')

parser.add_argument('--weight_decay', type=float, default=_WEIGHT_DECAY,
                    help='Weight decay.')

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

parser.add_argument('--optimizer', type=str, default='momentum',
                    help='Optimizer to use.')

parser.add_argument('--lr', type=float, default='0.04',
                    help='Learning rate when learning rate schedule is constant.')

parser.add_argument('--decay_every', type=float, default=2.4,
                    help='Epochs that learning rate decays.')

parser.add_argument('--decay_rate', type=float, default=0.97,
                    help='Decay rate of learning rate.')

parser.add_argument('--clip', type=float, default=10.0,
                    help='Clip gradients according to global norm.')
    

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

  num_images = _NUM_IMAGES['train']

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

  inputs = x #tf.reshape(x, [-1, params['input_size'], params['input_size'], 3])
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
    inputs = x #tf.reshape(x, [-1, params['input_size'], params['input_size'], 3])
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
    inputs = x #tf.reshape(x, [-1, params['input_size'], params['input_size'], 3])
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

def load_pb(pb):
  with tf.gfile.GFile(pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='')
    return graph

def calcluate_flops(g, sess):
  output_graph_def = graph_util.convert_variables_to_constants(sess, g.as_default_def(), ['output'])
  with tf.gfile.GFile('graph.pb', 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  g2 = load_pb('./graph.pb')
  with g2.as_default():
    flops = tf.profiler.profile(g2, options = tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOP is {}'.format(flops))

def train(params):
  g = tf.Graph()
  with g.as_default(), tf.device('/cpu:0'):
    tf.set_random_seed(params['seed'])
    dataset_train = imagenet.get_split('train', params['data_dir'])
    provider_train = tf.contrib.slim.dataset_data_provider.DatasetDataProvider(
      dataset_train,
      num_readers=4,
      common_queue_capacity=20*params['batch_size'],
      common_queue_min=10*params['batch_size'],
    )
    [image, label] = provider_train.get(['image', 'label'])
    label -= params['labels_offset'] #[1,1000] to [0,999]
    image = inception_preprocessing.preprocess_image(image, params['train_image_size'], params['train_image_size'], True)
    images_train, labels_train = tf.train.batch(
      [image, label],
      batch_size=params['batch_size'],
      num_threads=4,
      capacity=5 * params['batch_size'])
    labels_train = tf.contrib.slim.one_hot_encoding(
      labels_train, dataset_train.num_classes - params['labels_offset'])
    
    dataset_valid = imagenet.get_split('validation', params['data_dir'], 'valid')
    provider_valid = tf.contrib.slim.dataset_data_provider.DatasetDataProvider(
      dataset_valid,
      num_readers=4,
      common_queue_capacity=20 * 100,
      common_queue_min=10 * 100,
    )
    [image, label] = provider_valid.get(['image', 'label'])
    label -= params['labels_offset']  # [1,1000] to [0,999]
    image = inception_preprocessing.preprocess_image(image, params['eval_image_size'], params['eval_image_size'], False)
    images_valid, labels_valid = tf.train.batch(
      [image, label],
      batch_size=100,
      num_threads=4,
      capacity=5 * 100)
    labels_valid = tf.contrib.slim.one_hot_encoding(
      labels_valid, dataset_valid.num_classes - params['labels_offset'])
    
    train_cross_entropy, train_loss, learning_rate, train_top1_accuracy, train_top5_accuracy, train_op, global_step = get_train_ops(images_train, labels_train, params)
    _log_variable_sizes(tf.trainable_variables(), 'Trainable Variables')
    test_cross_entropy, test_loss, test_top1_accuracy, test_top5_accuracy = get_test_ops(images_valid, labels_valid, params, True)
    saver = tf.train.Saver(max_to_keep=30)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(
      params['model_dir'], save_steps=params['batches_per_epoch'], saver=saver)
    hooks = [checkpoint_saver_hook]
    tf.logging.info('Starting Session')
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.train.SingularMonitoredSession(
      config=config, hooks=hooks, checkpoint_dir=params['model_dir']) as sess:
      start_time = time.time()
      calcluate_flops(g, sess)
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
    try:
      with open(os.path.join(dag_name_or_path), 'r') as f:
        content = json.load(f)
        conv_dag, reduc_dag = content['conv_dag'], content['reduc_dag']
    except:
      conv_dag, reduc_dag = None, None

  return conv_dag, reduc_dag


def get_params():
  conv_dag, reduc_dag = build_dag(FLAGS.arch)
  total_steps = int(FLAGS.train_epochs * (_NUM_IMAGES['train']  / float(FLAGS.batch_size)))
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
