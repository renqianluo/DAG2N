from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
from collections import namedtuple, OrderedDict
import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

_OPERATIONS=OrderedDict()
_OPERATIONS['identity'] = lambda inputs, filters, strides, depth_multiplier, data_format, is_training : tf.identity(inputs)
_OPERATIONS['sep_conv 5x5'] = lambda inputs, filters, strides, depth_multiplier, data_format, is_training : separable_conv2d(inputs, filters, 5, strides, depth_multiplier, data_format, is_training)
_OPERATIONS['sep_conv 3x3'] = lambda inputs, filters, strides, depth_multiplier, data_format, is_training : separable_conv2d(inputs, filters, 3, strides, depth_multiplier, data_format, is_training)
_OPERATIONS['avg_pool 3x3'] = lambda inputs, filters, strides, depth_multiplier, data_format, is_training : average_pooling2d(inputs, 3, strides, data_format)
_OPERATIONS['max_pool 3x3'] = lambda inputs, filters, strides, depth_multiplier, data_format, is_training : max_pooling2d(inputs, 3, strides, data_format)
_OPERATIONS['conv 5x5'] = lambda inputs, filters, strides, depth_multiplier, data_format, is_training : conv2d(inputs, filters, 5, strides, data_format, is_training)
_OPERATIONS['conv 3x3'] = lambda inputs, filters, strides, depth_multiplier, data_format, is_training : conv2d(inputs, filters, 3, strides, data_format, is_training)

Node = namedtuple('Node', ['name', 'previous_node_1', 'previous_node_2', 'operation_1', 'operation_2'])

def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
  inputs: A tensor of size [batch, channels, height_in, width_in] or
    [batch, height_in, width_in, channels] depending on data_format.
  kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
         Should be a positive integer.
  data_format: The input format ('channels_last' or 'channels_first').

  Returns:
  A tensor with the same format as the input with the data either intact
  (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                  [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                  [pad_beg, pad_end], [0, 0]])
  return padded_inputs

def batch_normalization(inputs, data_format, is_training):
  inputs = tf.layers.batch_normalization(
    inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
    momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
    scale=True, training=is_training, fused=True)
  return inputs

def _separable_conv2d(inputs, filters, kernel_size, strides, depth_multiplier, data_format, is_training):
  inputs = tf.nn.relu(inputs) 
    
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)
  
  inputs = tf.layers.separable_conv2d(
    inputs=inputs, filters=filters, kernel_size=kernel_size, 
    strides=strides, depth_multiplier=depth_multiplier,
    padding=('SAME' if strides == 1 else 'VALID'), #use_bias=False,
    depthwise_initializer=tf.variance_scaling_initializer(),
    pointwise_initializer=tf.variance_scaling_initializer(),
    data_format=data_format)

  inputs = batch_normalization(inputs, data_format, is_training)
  return inputs

def separable_conv2d(inputs, filters, kernel_size, strides, depth_multiplier, data_format, is_training):
  inputs = _separable_conv2d(inputs, filters, kernel_size, strides, depth_multiplier, data_format, is_training)
  inputs = _separable_conv2d(inputs, filters, kernel_size, 1, depth_multiplier, data_format, is_training)

  return inputs

def average_pooling2d(inputs, pool_size, strides, data_format):
  if strides > 1:
    inputs = fixed_padding(inputs, pool_size, data_format)

  inputs = tf.layers.average_pooling2d(
    inputs=inputs, pool_size=pool_size, strides=strides,
    padding=('SAME' if strides == 1 else 'VALID'),
    data_format=data_format)
  return inputs

def max_pooling2d(inputs, pool_size, strides, data_format):
  if strides > 1:
    inputs = fixed_padding(inputs, pool_size, data_format)

  inputs = tf.layers.max_pooling2d(
    inputs=inputs, pool_size=pool_size, strides=strides,
    padding=('SAME' if strides == 1 else 'VALID'),
    data_format=data_format)
  return inputs

def _conv2d(inputs, filters, kernel_size, strides, data_format, is_training):
  inputs = tf.nn.relu(inputs) 
    
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)
  
  inputs = tf.layers.conv2d(
    inputs=inputs, filters=filters, kernel_size=kernel_size, 
    strides=strides,
    padding=('SAME' if strides == 1 else 'VALID'), #use_bias=False,
    kernel_initializer=tf.variance_scaling_initializer(),
    data_format=data_format)

  inputs = batch_normalization(inputs, data_format, is_training)
  return inputs

def conv2d(inputs, filters, kernel_size, strides, data_format, is_training):
  inputs = _conv2d(inputs, filters, kernel_size, strides, data_format, is_training)
  inputs = _conv2d(inputs, filters, kernel_size, 1, data_format, is_training)
  return inputs

def convolution_cell(last_inputs, inputs, params, is_training):
  # node 1 and node 2 are last_inputs and inputs respectively
  # begin processing from node 3
  num_nodes = params['num_nodes']
  data_format = params['data_format']
  filters = params['filters']
  depth_multiplier = params['depth_multiplier']
  dag = params['conv_dag']

  assert num_nodes == len(dag), 'num_nodes of convolution cell is not equal to number of nodes in convolution DAG!'

  h = {}
  leaf_nodes = ['node_%d' % i for i in xrange(1, num_nodes+1)]
  for i in xrange(1, num_nodes+1):
    name = 'node_%d' % i
    with tf.variable_scope(name):
      node = dag[name]
      assert name == node.name, 'name incompatible with node.name'
      if i == 1:
        h[name] = last_inputs
        continue
      elif i == 2:
        h[name] = inputs
        continue
      previous_node_1, previous_node_2 = node.previous_node_1, node.previous_node_2
      input_1, input_2 = h[previous_node_1], h[previous_node_2]
      if previous_node_1 in leaf_nodes:
        leaf_nodes.remove(previous_node_1)
      if previous_node_2 in leaf_nodes:
        leaf_nodes.remove(previous_node_2)
      operation_1, operation_2 = node.operation_1, node.operation_2
      with tf.variable_scope('input_1'):
        output_1 = _OPERATIONS[operation_1](input_1, filters, 1, depth_multiplier, data_format, is_training)
      with tf.variable_scope('input_2'):
        output_2 = _OPERATIONS[operation_2](input_2, filters, 1, depth_multiplier, data_format, is_training)
      output = tf.identity(output_1 + output_2, 'output')
      h[name] = output
  output = tf.concat([h[name] for name in leaf_nodes], axis=1 if data_format == 'channels_first' else 3, name='concat_leaf_nodes_output')
  with tf.variable_scope('cell_output'):
    output = tf.layers.conv2d(
      inputs=output, filters=filters, kernel_size=1, strides=1,
      padding='SAME', #use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)
    output = tf.nn.relu(batch_normalization(output, data_format, is_training))
  return inputs, output
  

def reduction_cell(last_inputs, inputs, params, is_training):
  # node 1 and node 2 are last_inputs and inputs respectively
  # begin processing from node 3
  num_nodes = params['num_nodes']
  data_format = params['data_format']
  filters = params['filters']
  depth_multiplier = params['depth_multiplier']
  dag = params['reduc_dag']
  
  assert num_nodes == len(dag), 'num_nodes is not equal to number of nodes in reduction DAG!'

  h = {}
  leaf_nodes = ['node_%d' % i for i in xrange(1, num_nodes+1)]
  for i in xrange(1, num_nodes+1):
    name = 'node_%d' % i
    with tf.variable_scope(name):
      node = dag[name]
      assert name == node.name, 'name incompatible with node.name'
      if i == 1:
        h[name] = last_inputs
      elif i == 2:
        h[name] = inputs
      else:
        previous_node_1, previous_node_2 = node.previous_node_1, node.previous_node_2
        input_1, input_2 = h[previous_node_1], h[previous_node_2]
        if previous_node_1 in leaf_nodes:
          leaf_nodes.remove(previous_node_1)
        if previous_node_2 in leaf_nodes:
          leaf_nodes.remove(previous_node_2)
        operation_1, operation_2 = node.operation_1, node.operation_2
        with tf.variable_scope('input_1'):
          output_1 = _OPERATIONS[operation_1](input_1, filters, 2 if previous_node_1 in ['node_1', 'node_2'] else 1, depth_multiplier, data_format, is_training)
        with tf.variable_scope('input_2'):
          output_2 = _OPERATIONS[operation_2](input_2, filters, 2 if previous_node_2 in ['node_1', 'node_2'] else 1, depth_multiplier, data_format, is_training)
        output = tf.identity(output_1 + output_2, 'output')
        h[name] = output
  output = tf.concat([h[name] for name in leaf_nodes], axis=1 if data_format == 'channels_first' else 3, name='concat_leaf_nodes_output')
  with tf.variable_scope('cell_output'):
    output = tf.layers.conv2d(
      inputs=output, filters=filters, kernel_size=1, strides=1,
      padding='SAME', #use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)
    output = tf.nn.relu(batch_normalization(output, data_format, is_training))
  return inputs, output

def build_block(inputs, params, is_training):
  num_cells = params['num_cells']
  # first convolution_cell
  with tf.variable_scope('convolution_cell_1'):
    last_inputs, inputs = convolution_cell(inputs, inputs, params, is_training)
  for i in xrange(1, num_cells):
    with tf.variable_scope('convolution_cell_%d' % (i+1)):
      last_inputs, inputs = convolution_cell(last_inputs, inputs, params, is_training)
  if params['reduc_dag'] is not None:
    with tf.variable_scope('reduction_cell'):
      last_inputs, inputs = reduction_cell(last_inputs, inputs, params, is_training)
  return inputs

def build_model(inputs, params, is_training, reuse=False) -> 'Get logits from inputs':
  """Generator for net.

  Args:
  inputs: inputs
  params: A dict containing following keys:
    num_blocks: A single integer for the number of blocks.
    num_cells: A single integer for the number of convolution cells.
    num_nodes: A single integer for the number of nodes.
    num_classes: The number of possible classes for image classification.
    filters: The numer of filters
    conv_dag: The DAG of the convolution cell
    reduc_dag: The DAG of the reduction cell
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
  is_training: boolean, whether is in training mode

  Returns:
  The model function that takes in `inputs` and `is_training` and
  returns the output tensor of the model.
  """
  if params['data_format'] is None:
    params['data_format'] = (
      'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

  data_format = params['data_format']
  num_classes = params['num_classes']
  filters = params['filters']
  num_blocks = params['num_blocks']
  
  if data_format == 'channels_first':
    # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
    # This provides a large performance boost on GPU. See
    # https://www.tensorflow.org/performance/performance_guide#data_formats
    inputs = tf.transpose(inputs, [0, 3, 1, 2])
 
  with tf.variable_scope('body', reuse=reuse):
    with tf.variable_scope('input_convonlution'):
      inputs = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=1, strides=1,
        padding='SAME', #use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)
      inputs = tf.nn.relu(batch_normalization(inputs, data_format, is_training))
    for i in xrange(1, num_blocks + 1):
      with tf.variable_scope('block_%d' % i):
        inputs = build_block(inputs, params, is_training)
    with tf.variable_scope('final_global_average_pooling'):
      inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=32//(2**num_blocks), strides=1, padding='VALID', data_format=data_format)
    inputs = tf.reshape(inputs, [-1, filters])
    with tf.variable_scope('fully_connected_layer'):
      inputs = tf.layers.dense(inputs=inputs, units=num_classes)
  return inputs
