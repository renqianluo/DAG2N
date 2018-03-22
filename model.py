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


def batch_normalization(inputs, data_format, is_training):
  inputs = tf.layers.batch_normalization(
    inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
    momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
    scale=True, training=is_training, fused=True)
  return inputs


def _pooling(operation, inputs, strides, data_format):
  pooling_type, pooling_size = _operation_to_pooling_info(operation)
  if pooling_type == 'avg_pool':
    inputs = tf.layers.average_pooling2d(
      inputs=inputs, pool_size=pooling_size, strides=strides,
      padding='SAME',
    data_format=data_format)
  elif pooling_type == 'max_pool':
    inputs = tf.layers.max_pooling2d(
      inputs=inputs, pool_size=pooling_size, strides=strides,
      padding='SAME',
      data_format=data_format)
  else:
    raise NotImplementedError('Unimplemented pooling type: ', pooling_type)
  return inputs


def _separable_conv2d(operation, inputs, filters, strides, activation, data_format, is_training):
  kernel_size, num_layers = _operation_to_info(operation)

  for layer_num in range(num_layers - 1):
    inputs = tf.nn.relu(inputs) 
    with tf.variable_scope('separable_conv_{0}x{0}_{1}'.format(kernel_size, layer_num+1)):
      inputs = tf.layers.separable_conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, 
        strides=strides, depth_multiplier=1,
        padding='SAME',
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
      padding='SAME',
      depthwise_initializer=tf.variance_scaling_initializer(),
      pointwise_initializer=tf.variance_scaling_initializer(),
      data_format=data_format,
      activation=activation)
  with tf.variable_scope('bn_sep_{0}x{0}_{1}'.format(kernel_size, num_layers)):
    inputs = batch_normalization(inputs, data_format, is_training)

  return inputs


def _conv2d(operation, inputs, filters, strides, activation, data_format, is_training):
  kernel_size, num_layers = _operation_to_info(operation)
  for layer_num in range(num_layers - 1):
    inputs = tf.nn.relu(inputs) 
    with tf.variable_scope('conv_{0}x{0}_{1}'.format(kernel_size, layer_num+1)):
      inputs = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, 
        strides=strides, padding='SAME',
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
      strides=strides, padding='SAME',
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


def factorized_reduction(inputs, filters, strides, activation, data_format, is_training):
  assert filters % 2 == 0, (
    'Need even number of filters when using this factorized reduction')
  if strides == 1:
    with tf.variable_scope('path_conv'):
      inputs = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=1, 
        strides=strides, padding='SAME',
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


def drop_path(inputs, keep_prob, is_training=True):
  if is_training:
    batch_size = tf.shape(inputs)[0]
    noise_shape = [batch_size, 1, 1, 1]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape, dtype=tf.float32)
    binary_tensor = tf.floor(random_tensor)
    inputs = tf.div(inputs, keep_prob) * binary_tensor
  return inputs


class ENASCell(object):
  def __init__(self, filters, dag, num_nodes, drop_path_keep_prob, num_cells,
    total_steps,activation, data_format, is_training):
    self._filters = filters
    self._dag = dag
    self._num_nodes = num_nodes
    self._drop_path_keep_prob = drop_path_keep_prob
    self._num_cells = num_cells
    self._total_steps = total_steps
    self._is_training = is_training
    self._data_format = data_format
    self._activation = activation

  def _reduce_prev_layer(self, prev_layer, curr_layer):
    if prev_layer is None:
      return curr_layer

    curr_num_filters = self._filter_size
    data_format = self._data_format
    activation = self._activation
    is_training = self._is_training

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


  def _cell_base(self, last_inputs, inputs):
    filters = self._filter_size
    activation = self._activation
    data_format = self._data_format
    is_training = self._is_training

    with tf.variable_scope('transforme_last_inputs'):
      last_inputs = self._reduce_prev_layer(last_inputs, inputs)
    with tf.variable_scope('transforme_inputs'):
      with tf.variable_scope('1x1'):
        inputs = tf.nn.relu(inputs)
        inputs = tf.layers.conv2d(
          inputs=inputs, filters=filters, kernel_size=1, 
          strides=1, padding='SAME',
          kernel_initializer=tf.variance_scaling_initializer(),
          data_format=data_format,
          activation=activation)
      with tf.variable_scope('beginning_bn'):
        inputs = batch_normalization(inputs, data_format, is_training)
    return last_inputs, inputs


  def __call__(self, inputs, filter_scaling=1, strides=1,
    last_inputs=None, cell_num=-1):
    self._cell_num = cell_num
    self._filter_scaling = filter_scaling
    self._filter_size = int(self._filters * filter_scaling)
    num_nodes = self._num_nodes
    dag = self._dag
    data_format = self._data_format

    # node 1 and node 2 are last_inputs and inputs respectively
    # begin processing from node 3

    assert num_nodes == len(dag), 'num_nodes of convolution cell is not equal to number of nodes in convolution DAG!'

    curr_inputs = inputs
    last_inputs, inputs = self._cell_base(last_inputs, inputs)

    h = {}
    loose_nodes = ['node_%d' % i for i in xrange(1, num_nodes+1)]
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
        if previous_node_1 in loose_nodes:
          loose_nodes.remove(previous_node_1)
        if previous_node_2 in loose_nodes:
          loose_nodes.remove(previous_node_2)
        operation_1, operation_2 = node.operation_1, node.operation_2
        with tf.variable_scope('input_1'):
          is_from_original_input = int(previous_node_1.split('_')[-1]) < 3
          h1 = self._apply_operation(operation_1, h1, strides, is_from_original_input)
        with tf.variable_scope('input_2'):
          is_from_original_input = int(previous_node_2.split('_')[-1]) < 3
          h2 = self._apply_operation(operation_2, h2, strides, is_from_original_input)
        with tf.variable_scope('combine'):
          output = h1 + h2
        h[name] = output

    with tf.variable_scope('cell_output'):
      output = self._combine_unused_states(h, loose_nodes)
    
    return curr_inputs, output


  def _apply_operation(self, operation, inputs, strides, 
    is_from_original_input):
    filters = self._filter_size
    activation = self._activation
    data_format = self._data_format
    is_training = self._is_training

    if strides > 1 and not is_from_original_input:
      strides = 1
    input_filters = get_channel_dim(inputs.shape, data_format)
    if 'sep_conv' in operation:
      inputs = _separable_conv2d(operation, inputs, filters, strides, activation, data_format, is_training)
    elif 'conv' in operation:
      inputs = _conv2d(operation, inputs, filters, strides, activation, data_format, is_training)
    elif 'identity' in operation:
      if strides > 1 or (input_filters != filters):
        inputs = tf.nn.relu(inputs)
        with tf.variable_scope('1x1'):
          inputs = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=1, 
            strides=strides, padding='SAME',
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=data_format,
            activation=activation)
        with tf.variable_scope('bn_1'):
          inputs = batch_normalization(inputs, data_format, is_training)
    elif 'pool' in operation:
      inputs = _pooling(operation, inputs, strides, data_format)
      if input_filters != filters:
        with tf.variable_scope('1x1'):
          inputs = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=1, 
            strides=1, padding='SAME',
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=data_format,
            activation=activation)
        with tf.variable_scope('bn_1'):
          inputs = batch_normalization(inputs, data_format, is_training)
    else:
      raise ValueError('Unimplemented operation', operation)

    if operation != 'identity':
      inputs = self._apply_drop_path(inputs, is_training)

    return inputs


  def _combine_unused_states(self, h, loose_nodes):
    activation = self._activation
    data_format = self._data_format
    is_training = self._is_training

    final_height = int(h['node_%d'%self._num_nodes].shape[2])
    final_filters = get_channel_dim(h['node_%d'%self._num_nodes].shape, self._data_format)

    for i in range(1, self._num_nodes+1):
      node_name = 'node_%d'%i
      curr_height = int(h[node_name].shape[2])
      curr_filters = get_channel_dim(h[node_name].shape, data_format)

      # Determine if a reduction should be applied to make the number of filters match.
      should_reduce = final_filters != curr_filters
      should_reduce = (final_height != curr_height) or should_reduce
      should_reduce = should_reduce and node_name in loose_nodes
      if should_reduce:
        strides = 2 if final_height != curr_height else 1
        with tf.variable_scope('reduction_{}'.format(i)):
          h[node_name] = factorized_reduction(h[node_name], final_filters, strides, activation, data_format, is_training)

    with tf.variable_scope('concat_loose_ends'):
      output = tf.concat([h[name] for name in loose_nodes], axis=get_channel_index(data_format))
    return output

  def _apply_drop_path(self, inputs, is_training, current_step=None, use_summaries=False, drop_connect_version='v3'):
    drop_path_keep_prob = self._drop_path_keep_prob
    if drop_path_keep_prob < 1.0:
      assert drop_connect_version in ['v1', 'v2', 'v3']
      if drop_connect_version in ['v2', 'v3']:
        # Scale keep prob by layer number
        assert self._cell_num != -1
        num_cells = self._num_cells
        layer_ratio = (self._cell_num + 1) / float(num_cells)
        if use_summaries:
          with tf.device('/cpu:0'):
            tf.summary.scalar('layer_ratio', layer_ratio)
        drop_path_keep_prob = 1 - layer_ratio * (1 - drop_path_keep_prob)
      if drop_connect_version in ['v1', 'v3']:
        # Decrease the keep probability over time
        if not current_step:
          current_step = tf.cast(tf.train.get_or_create_global_step(),
                                   tf.float32)
        drop_path_burn_in_steps = self._total_steps
        current_ratio = current_step / drop_path_burn_in_steps
        current_ratio = tf.minimum(1.0, current_ratio)
        if use_summaries:
          with tf.device('/cpu:0'):
            tf.summary.scalar('current_ratio', current_ratio)
        drop_path_keep_prob = (1 - current_ratio * (1 - drop_path_keep_prob))
      if use_summaries:
        with tf.device('/cpu:0'):
          tf.summary.scalar('drop_path_keep_prob', drop_path_keep_prob)
      inputs = drop_path(inputs, drop_path_keep_prob, is_training)
    return inputs


def build_model(inputs, params, is_training, reuse=False) -> 'Get logits from inputs':
  """Generator for net.

  Args:
  inputs: inputs
  params: A dict containing following keys:
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
  
  filters = params['filters']
  conv_dag = params['conv_dag']
  reduc_dag = params['reduc_dag']
  arch = params['arch']
  num_nodes = params['num_nodes']
  if not is_training:
    drop_path_keep_prob = 1.0
    dense_dropout_keep_prob = 1.0
  else:
    drop_path_keep_prob = params['drop_path_keep_prob']
    dense_dropout_keep_prob = params['dense_dropout_keep_prob']
  total_steps = params['total_steps']
  if params['activation'] is None:
    activation = None
  elif params['activation'] == 'relu':
    activation = tf.nn.relu
  else:
    raise ValueError('Unsorported activation function: ', params['activation'])
  if params['data_format'] is None:
    data_format = 'channels_first' if tf.test.is_built_with_cuda() else 'channels_last'
  num_classes = params['num_classes']

  
  if data_format == 'channels_first':
    # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
    # This provides a large performance boost on GPU. See
    # https://www.tensorflow.org/performance/performance_guide#data_formats
    inputs = tf.transpose(inputs, [0, 3, 1, 2])
 
  arch = arch.split('-')
  num_cells = 0
  for e in arch:
    _, num = e.split('x')
    num = int(num)
    num_cells += num

  convolution_cell = ENASCell(filters, conv_dag, num_nodes, drop_path_keep_prob, num_cells,
    total_steps,activation, data_format, is_training)
  reduction_cell = ENASCell(filters, reduc_dag, num_nodes, drop_path_keep_prob, num_cells,
    total_steps,activation, data_format, is_training)

  with tf.variable_scope('body', reuse=reuse):
    last_inputs = None
    with tf.variable_scope('stem_conv_3x3'):
      inputs = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        padding='SAME',
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format,
        activation=params['activation'])
    with tf.variable_scope('bn'):
      inputs = batch_normalization(inputs, data_format, is_training)

    conv_cell_count, reduc_cell_count, filter_scaling = 0, 0, 1

    cell_num = 0
    for cell_com in arch:
      cell, num = cell_com.split('x')
      num = int(num)
      if cell == 'conv':
        for i in xrange(conv_cell_count + 1, conv_cell_count + 1 + num):
          with tf.variable_scope('convolution_cell_%d' % i):
            last_inputs, inputs = convolution_cell(inputs, filter_scaling, 1, last_inputs, cell_num)
          cell_num += 1
        conv_cell_count += num
      elif cell == 'reduc':
        for i in xrange(reduc_cell_count + 1, reduc_cell_count + 1 + num):
          with tf.variable_scope('reduction_cell_%d' % i):
            filter_scaling *= 2
            last_inputs, inputs = reduction_cell(inputs, filter_scaling, 2, last_inputs, cell_num)
          cell_num += 1
        reduc_cell_count += num
     
    inputs = tf.nn.relu(inputs)
        
    if data_format == 'channels_first':
      inputs = tf.reduce_mean(inputs, axis=[2,3])
    else:
      inputs = tf.reduce_mean(inputs, axis=[1,2])
      
    inputs = tf.nn.dropout(inputs, dense_dropout_keep_prob)

    with tf.variable_scope('fully_connected_layer'):
      inputs = tf.layers.dense(inputs=inputs, units=num_classes)

  return inputs
