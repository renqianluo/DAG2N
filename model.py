from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
from collections import namedtuple, OrderedDict
import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

_OPERATIONS=[
  'identity',
  'sep_conv 5x5',
  'sep_conv 3x3',
  'avg_pool 3x3',
  'max_pool 3x3',
  'conv 5x5',
  'conv 3x3'
]

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


def get_channel_dim(shape, data_format='INVALID'):
  assert data_format != 'INVALID'
  assert len(shape) == 4
  if data_format == 'channels_first':
    return int(shape[1])
  else:
    return int(shape[3])


def get_channel_index(data_format='INVALID'):
  assert data_format != 'INVALID'
  axis = 1 if data_format == 'channels_first' else 3
  return axis


def reduce_prev_layer(prev_layer, curr_layer, filters, activation, data_format, is_training):
  if prev_layer is None:
    return curr_layer
  #curr_num_filters = get_channel_dim(curr_layer.shape, data_format)
  curr_num_filters = filters
  prev_num_filters = get_channel_dim(prev_layer.shape, data_format)
  curr_filter_shape = int(curr_layer.shape[2])
  prev_filter_shape = int(prev_layer.shape[2])
  if curr_filter_shape != prev_filter_shape:
    prev_layer = tf.nn.relu(prev_layer)
    prev_layer = factorized_reduction(prev_layer, curr_num_filters, 2, activation, data_format, is_training)
  elif curr_num_filters != prev_num_filters:
    prev_layer = tf.nn.relu(prev_layer)
    prev_layer = tf.layers.conv2d(
      inputs=prev_layer, filters=curr_num_filters, kernel_size=1, 
      strides=1, padding='SAME',
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format,
      activation=activation)
    prev_layer = batch_normalization(prev_layer, data_format, is_training)
  return prev_layer


def batch_normalization(inputs, data_format, is_training):
  inputs = tf.layers.batch_normalization(
    inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
    momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
    scale=True, training=is_training, fused=True)
  return inputs


def pooling(operation, inputs, strides, data_format):
  pooling_type, pooling_size = _operation_to_pooling_info(operation)
  if strides > 1:
    inputs = fixed_padding(inputs, pooling_size, data_format)
  if pooling_type == 'avg_pool':
    inputs = tf.layers.average_pooling2d(
      inputs=inputs, pool_size=pooling_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'),
    data_format=data_format)
  elif pooling_type == 'max_pool':
    inputs = tf.layers.max_pooling2d(
      inputs=inputs, pool_size=pooling_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'),
      data_format=data_format)
  else:
    raise NotImplementedError('Unimplemented pooling type: ', pooling_type)
  return inputs


def separable_conv2d(operation, inputs, filters, strides, activation, data_format, is_training):
  kernel_size, num_layers = _operation_to_info(operation)
  for layer_num in range(num_layers - 1):
    inputs = tf.nn.relu(inputs) 
    if strides > 1:
      inputs = fixed_padding(inputs, kernel_size, data_format)
    with tf.variable_scope('separable_conv_{0}x{0}_{1}'.format(kernel_size, layer_num+1)):
      inputs = tf.layers.separable_conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, 
        strides=strides, depth_multiplier=1,
        padding=('SAME' if strides == 1 else 'VALID'),
        depthwise_initializer=tf.variance_scaling_initializer(),
        pointwise_initializer=tf.variance_scaling_initializer(),
        data_format=data_format,
        activation=activation)
    with tf.variable_scope('bn_sep_{0}x{0}_{1}'.format(kernel_size, layer_num+1)):
      inputs = batch_normalization(inputs, data_format, is_training)
    strides = 1

  inputs = tf.nn.relu(inputs)
  with tf.variable_scope('separable_conv_{0}x{0}_{1}'.format(kernel_size, num_layers)):
    inputs = tf.layers.separable_conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, 
      strides=strides, depth_multiplier=1,
      padding=('SAME' if strides == 1 else 'VALID'),
      depthwise_initializer=tf.variance_scaling_initializer(),
      pointwise_initializer=tf.variance_scaling_initializer(),
      data_format=data_format,
      activation=activation)
  with tf.variable_scope('bn_sep_{0}x{0}_{1}'.format(kernel_size, num_layers)):
    inputs = batch_normalization(inputs, data_format, is_training)

  return inputs


def conv2d(operation, inputs, filters, strides, activation, data_format, is_training):
  kernel_size, num_layers = _operation_to_info(operation)
  for layer_num in range(num_layers - 1):
    inputs = tf.nn.relu(inputs) 
    if strides > 1:
      inputs = fixed_padding(inputs, kernel_size, data_format)
    with tf.variable_scope('conv_{0}x{0}_{1}'.format(kernel_size, layer_num+1)):
      inputs = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, 
        strides=strides, padding=('SAME' if strides == 1 else 'VALID'),
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format,
        activation=activation)
    with tf.variable_scope('bn_conv_{0}x{0}_{1}'.format(kernel_size, layer_num+1)):
      inputs = batch_normalization(inputs, data_format, is_training)
    strides = 1

  inputs = tf.nn.relu(inputs)
  with tf.variable_scope('conv_{0}x{0}_{1}'.format(kernel_size, num_layers)):
    inputs = tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, 
      strides=strides, padding=('SAME' if strides == 1 else 'VALID'),
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format,
      activation=activation)
  with tf.variable_scope('bn_conv_{0}x{0}_{1}'.format(kernel_size, num_layers)):
    inputs = batch_normalization(inputs, data_format, is_training)
  return inputs


def _operation_to_filter_shape(operation):
  splitted_operation = operation.split('x')
  filter_shape = int(splitted_operation[0][-1])
  assert filter_shape == int(
      splitted_operation[1][0]), 'Rectangular filters not supported.'
  return filter_shape


def _operation_to_num_layers(operation):
  splitted_operation = operation.split(' ')
  if 'x' in splitted_operation[-1]:
    return 1
  return int(splitted_operation[-1])


def _operation_to_info(operation):
  num_layers = 2
  filter_shape = _operation_to_filter_shape(operation)
  return filter_shape, num_layers


def _operation_to_pooling_type(operation):
  splitted_operation = operation.split(' ')
  return splitted_operation[0]


def _operation_to_pooling_shape(operation):
  splitted_operation = operation.split(' ')
  shape = splitted_operation[-1]
  assert 'x' in shape
  filter_height, filter_width = shape.split('x')
  assert filter_height == filter_width
  return int(filter_height)


def _operation_to_pooling_info(operation):
  pooling_type = _operation_to_pooling_type(operation)
  pooling_shape = _operation_to_pooling_shape(operation)
  return pooling_type, pooling_shape


def apply_operation(operation, inputs, filters, strides, activation, is_from_original_input, data_format, is_training):
  if strides > 1 and not is_from_original_input:
    strides = 1
  input_filters = get_channel_dim(inputs.shape, data_format)
  if 'sep_conv' in operation:
    inputs = separable_conv2d(operation, inputs, filters, strides, activation, data_format, is_training)
  elif 'conv' in operation:
    inputs = conv2d(operation, inputs, filers, strides, activation, data_format, is_training)
  elif 'identity' in operation:
    if strides > 1 or (input_filters != filters):
      inputs = tf.nn.relu(inputs)
      inputs = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=1, 
        strides=strides, padding='SAME',
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format,
        activation=activation)
      inputs = batch_normalization(inputs, data_format, is_training)
  elif 'pool' in operation:
    inputs = pooling(operation, inputs, strides, data_format)
    if input_filters != filters:
      inputs = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=1, 
        strides=1, padding='SAME',
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format,
        activation=activation)
      inputs = batch_normalization(inputs, data_format, is_training)
  else:
    raise ValueError('Unimplemented operation', operation)

  return inputs


def factorized_reduction(inputs, filters, strides, activation, data_format, is_training):
  assert filters % 2 == 0, (
    'Need even number of filters when using this factorized reduction')
  if strides == 1:
    with tf.variable_scope('path_conv'):
      inputs = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, 
        strides=strides, padding=('SAME' if strides == 1 else 'VALID'),
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format,
        activation=activation)
    with tf.variable_scope('path_bn'):
      inputs = batch_normalization(inputs, data_format, is_training)
    return inputs

  path1 = tf.layers.average_pooling2d(inputs, pool_size=1, strides=strides, padding='VALID', data_format=data_format)
  with tf.variable_scope('path1_conv'):
    path1 = tf.layers.conv2d(
      inputs=path1, filters=int(filters / 2), kernel_size=1, 
      strides=1, padding='SAME',
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format,
      activation=activation)

  if data_format == 'channels_first':
    pad_arr = [[0, 0], [0, 0], [0, 1], [0, 1]]
    path2 = tf.pad(inputs, pad_arr)[:, :, 1:, 1:]
  else:
    pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
    path2 = tf.pad(inputs, pad_arr)[:, 1:, 1:, :]

  path2 = tf.layers.average_pooling2d(path2, pool_size=1, strides=strides, padding='VALID', data_format=data_format)
  with tf.variable_scope('path2_conv'):
    path2 = tf.layers.conv2d(
      inputs=path2, filters=int(filters / 2), kernel_size=1, 
      strides=1, padding='SAME',
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format,
      activation=activation)

  final_path = tf.concat(values=[path1, path2], axis=get_channel_index(data_format))
  with tf.variable_scope('final_bn'):
    inputs = batch_normalization(final_path, data_format, is_training)

  return inputs


def apply_drop_path(inputs)


def cell_base(last_inputs, inputs, filters, activation, data_format, is_training):
  with tf.variable_scope('transforme_last_inputs'):
    last_inputs = reduce_prev_layer(last_inputs, inputs, filters, activation, data_format, is_training)
  with tf.variable_scope('transforme_inputs'):
    inputs = tf.nn.relu(inputs)
    inputs = tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=1, 
      strides=1, padding='SAME',
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format,
      activation=activation)
    inputs = batch_normalization(inputs, data_format, is_training)
  return last_inputs, inputs


def convolution_cell(last_inputs, inputs, params, is_training):
  # node 1 and node 2 are last_inputs and inputs respectively
  # begin processing from node 3
  num_nodes = params['num_nodes']
  data_format = params['data_format']
  filters = params['filters']
  dag = params['conv_dag']
  activation = params['activation']

  assert num_nodes == len(dag), 'num_nodes of convolution cell is not equal to number of nodes in convolution DAG!'

  curr_inputs = inputs
  last_inputs, inputs = cell_base(last_inputs, inputs, filters, activation, data_format, is_training)

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
      h1, h2 = h[previous_node_1], h[previous_node_2]
      if previous_node_1 in leaf_nodes:
        leaf_nodes.remove(previous_node_1)
      if previous_node_2 in leaf_nodes:
        leaf_nodes.remove(previous_node_2)
      operation_1, operation_2 = node.operation_1, node.operation_2
      with tf.variable_scope('input_1'):
        is_from_original_input = int(previous_node_1.split('_')[-1]) < 3
        h1 = apply_operation(operation_1, h1, filters, 1, activation, is_from_original_input, data_format, is_training)
      with tf.variable_scope('input_2'):
        is_from_original_input = int(previous_node_2.split('_')[-1]) < 3
        h2 = apply_operation(operation_2, h2, filters, 1, activation, is_from_original_input, data_format, is_training)
      output = tf.identity(h1 + h2, 'output')
      h[name] = output

  output = tf.concat([h[name] for name in leaf_nodes], axis=get_channel_index(data_format))
  
  return curr_inputs, output
  

def reduction_cell(last_inputs, inputs, params, is_training):
  # node 1 and node 2 are last_inputs and inputs respectively
  # begin processing from node 3
  num_nodes = params['num_nodes']
  data_format = params['data_format']
  filters = params['filters']
  dag = params['reduc_dag']
  
  assert num_nodes == len(dag), 'num_nodes is not equal to number of nodes in reduction DAG!'

  curr_inputs = inputs
  last_inputs, inputs = cell_base(last_inputs, inputs, filters, data_format, is_training)

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
        h1, h2 = h[previous_node_1], h[previous_node_2]
        if previous_node_1 in leaf_nodes:
          leaf_nodes.remove(previous_node_1)
        if previous_node_2 in leaf_nodes:
          leaf_nodes.remove(previous_node_2)
        operation_1, operation_2 = node.operation_1, node.operation_2
        with tf.variable_scope('input_1'):
          is_from_original_input = int(previous_node_1.split('_')[-1]) < 3
          h1 = apply_operation(operation_1, h1, filters, 2, is_from_original_input, data_format, is_training)
        with tf.variable_scope('input_2'):
          is_from_original_input = int(previous_node_2.split('_')[-1]) < 3
          h2 = apply_operation(operation_2, h2, filters, 2, is_from_original_input, data_format, is_training)
        output = tf.identity(h1 + h2, 'output')
        h[name] = output

  output = tf.concat([h[name] for name in leaf_nodes], axis=get_channel_index(data_format))
  
  return curr_inputs, output

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
  arch = params['arch']
  if params['activation'] is None:
    params['activation'] = None
  elif params['activation'] == 'relu':
    params['activation'] = tf.nn.relu
  else:
    raise ValueError('Unsorported activation function: ', params['activation'])
  
  if data_format == 'channels_first':
    # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
    # This provides a large performance boost on GPU. See
    # https://www.tensorflow.org/performance/performance_guide#data_formats
    inputs = tf.transpose(inputs, [0, 3, 1, 2])
 
  with tf.variable_scope('body', reuse=reuse):
    last_inputs = None
    with tf.variable_scope('stem_conv_3x3'):
      inputs = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        padding='SAME', #use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format,
        activation=params['activation'])
    with tf.variable_scope('bn'):
      inputs = batch_normalization(inputs, data_format, is_training)

    arch = arch.split('-')
    conv_cell_count, reduc_cell_count = 0, 0
    for cell_com in arch:
      cell, num = cell_com.split('x')
      num = int(num)
      if cell == 'conv':
        for i in xrange(conv_cell_count + 1, conv_cell_count + 1 + num):
          with tf.variable_scope('convolution_cell_%d' % i):
            last_inputs, inputs = convolution_cell(last_inputs, inputs, params, is_training)
        conv_cell_count += num
      elif cell == 'reduc':
        for i in xrange(reduc_cell_count + 1, reduc_cell_count + 1 + num):
          with tf.variable_scope('reduction_cell_%d' % i):
            params['filters'] *= 2
            last_inputs, inputs = reduction_cell(last_inputs, inputs, params, is_training)
        reduc_cell_count += num
     
    inputs = tf.nn.relu(inputs)
        
    if data_format == 'channels_first':
      inputs = tf.reduce_mean(inputs, axis=[2,3])
    else:
      inputs = tf.reduce_mean(inputs, axis=[1,2])
      
    #inputs = tf.reshape(inputs, [-1, inputs.shape[-1]])

    with tf.variable_scope('fully_connected_layer'):
      inputs = tf.layers.dense(inputs=inputs, units=num_classes)

  return inputs
