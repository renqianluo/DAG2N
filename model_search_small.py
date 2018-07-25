from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
from collections import namedtuple, OrderedDict
import tensorflow as tf
from tensorflow.python.training import moving_averages

_BATCH_NORM_DECAY = 0.9 #0.997
_BATCH_NORM_EPSILON = 1e-5
_USE_BIAS = False
_KERNEL_INITIALIZER=tf.variance_scaling_initializer(mode='fan_out')

def sample_arch(num_cells):
  #arc_seq = tf.TensorArray(tf.int32, size=num_cells * 4)
  arc_seq = []
  for cell_id in range(num_cells):
    for branch_id in range(2):
      index = tf.random_uniform([1], minval=0, maxval=cell_id+1, dtype=tf.int32)
      arc_seq.append(index)
      config_id = tf.random_uniform([1], minval=0, maxval=11, dtype=tf.int32)#11
      arc_seq.append(config_id)
  arc_seq = tf.concat(arc_seq, axis=0)
  return arc_seq

def sample_arch_from_pool(arch_pool, prob=None):
  N = len(arch_pool)
  if prob is not None:
    prob = tf.expand_dims(tf.squeeze(prob),axis=0)
    index = tf.multinomial(prob, 1)[0][0]
  else:
    index = tf.random_uniform([], minval=0, maxval=N, dtype=tf.int32)
  return arch_pool[index]

def create_weight(name, shape, initializer=None, trainable=True, seed=None):
  if initializer is None:
    initializer = _KERNEL_INITIALIZER
  return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)


def create_bias(name, shape, initializer=None):
  if initializer is None:
    initializer = tf.constant_initializer(0.0, dtype=tf.float32)
  return tf.get_variable(name, shape, initializer=initializer)


def get_channel_dim(x, data_format='INVALID'):
  assert data_format != 'INVALID'
  assert x.shape.ndims == 4
  if data_format == 'channels_first':
    return x.shape[1].value
  else:
    return x.shape[3].value


def get_channel_index(data_format='INVALID'):
  assert data_format != 'INVALID'
  axis = 1 if data_format == 'channels_first' else 3
  return axis


def batch_normalization(x, data_format, is_training):
  if data_format == "channels_first":
    shape = [x.get_shape()[1]]
  elif data_format == "channels_last":
    shape = [x.get_shape()[3]]
  else:
    raise NotImplementedError("Unknown data_format {}".format(data_format))

  with tf.variable_scope('batch_normalization'):#, reuse=None if is_training else True):
    offset = tf.get_variable(
      "offset", shape,
      initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    scale = tf.get_variable(
      "scale", shape,
      initializer=tf.constant_initializer(1.0, dtype=tf.float32))
    moving_mean = tf.get_variable(
      "moving_mean", shape, trainable=False,
      initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    moving_variance = tf.get_variable(
      "moving_variance", shape, trainable=False,
      initializer=tf.constant_initializer(1.0, dtype=tf.float32))

    if is_training:
      x, mean, variance = tf.nn.fused_batch_norm(
        x, scale, offset, epsilon=_BATCH_NORM_EPSILON,
        data_format='NCHW' if data_format == "channels_first" else 'NHWC',
        is_training=True)
      update_mean = moving_averages.assign_moving_average(
        moving_mean, mean, _BATCH_NORM_DECAY)
      update_variance = moving_averages.assign_moving_average(
        moving_variance, variance, _BATCH_NORM_DECAY)
      with tf.control_dependencies([update_mean, update_variance]):
        x = tf.identity(x)
    else:
      x, _, _ = tf.nn.fused_batch_norm(x, scale, offset, mean=moving_mean,
                                       variance=moving_variance,
                                       epsilon=_BATCH_NORM_EPSILON,
                                       data_format='NCHW' if data_format == "channels_first" else 'NHWC',
                                       is_training=False)
  return x


def factorized_reduction(inputs, filters, strides, data_format, is_training):
  assert filters % 2 == 0, (
    'Need even number of filters when using this factorized reduction')
  if strides == 1:
    with tf.variable_scope('path_conv'):
      inputs = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=1, 
        strides=strides, padding='SAME', use_bias=_USE_BIAS,
        kernel_initializer=_KERNEL_INITIALIZER,
        data_format=data_format)
    with tf.variable_scope('path_bn'):
      inputs = batch_normalization(inputs, data_format, is_training)
    return inputs

  path1 = tf.layers.average_pooling2d(inputs, pool_size=1, strides=strides, padding='VALID', data_format=data_format)
  with tf.variable_scope('path1_conv'):
    path1 = tf.layers.conv2d(
      inputs=path1, filters=int(filters / 2), kernel_size=1, 
      strides=1, padding='SAME', use_bias=_USE_BIAS,
      kernel_initializer=_KERNEL_INITIALIZER,
      data_format=data_format)

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
      strides=1, padding='SAME', use_bias=_USE_BIAS,
      kernel_initializer=_KERNEL_INITIALIZER,
      data_format=data_format)

  final_path = tf.concat(values=[path1, path2], axis=get_channel_index(data_format))
  with tf.variable_scope('final_path_bn'):
    inputs = batch_normalization(final_path, data_format, is_training)

  return inputs


class NASCell(object):
  def __init__(self, filters, dag, num_nodes, drop_path_keep_prob, num_cells,
    total_steps, data_format, is_training):
    self._filters = filters
    self._dag = dag
    self._num_nodes = num_nodes
    self._drop_path_keep_prob = drop_path_keep_prob
    self._num_cells = num_cells
    self._total_steps = total_steps
    self._is_training = is_training
    self._data_format = data_format

  def _reduce_prev_layer(self, prev_layer, curr_layer, is_training):
    if prev_layer is None:
      return curr_layer

    curr_num_filters = self._filter_size
    data_format = self._data_format
    #is_training = self._is_training

    prev_num_filters = get_channel_dim(prev_layer, data_format)
    curr_filter_shape = int(curr_layer.shape[2])
    prev_filter_shape = int(prev_layer.shape[2])
    if curr_filter_shape != prev_filter_shape:
      prev_layer = tf.nn.relu(prev_layer)
      prev_layer = factorized_reduction(prev_layer, curr_num_filters, 2, data_format, is_training)
    elif curr_num_filters != prev_num_filters:
      prev_layer = tf.nn.relu(prev_layer)
      with tf.variable_scope('prev_1x1'):
        prev_layer = tf.layers.conv2d(
          inputs=prev_layer, filters=curr_num_filters, kernel_size=1, 
          strides=1, padding='SAME', use_bias=_USE_BIAS,
          kernel_initializer=_KERNEL_INITIALIZER,
          data_format=data_format)
      with tf.variable_scope('prev_bn'):
        prev_layer = batch_normalization(prev_layer, data_format, is_training)
    return prev_layer


  def _nas_conv(self, x, curr_cell, prev_cell, filter_size, out_filters, stack_conv=1):
    with tf.variable_scope("conv_{0}x{0}".format(filter_size)):
      num_possible_inputs = curr_cell + 2
      for conv_id in range(stack_conv):
        with tf.variable_scope("stack_{0}".format(conv_id)):
          # create params and pick the correct path
          inp_c = get_channel_dim(x, self._data_format)
          w = create_weight(
            "w", [num_possible_inputs, filter_size * filter_size * inp_c * out_filters],
            initializer=_KERNEL_INITIALIZER)
          w = w[prev_cell, :]
          w = tf.reshape(
            w, [filter_size, filter_size, inp_c, out_filters])

          with tf.variable_scope("bn"):
            zero_init = tf.initializers.zeros(dtype=tf.float32)
            one_init = tf.initializers.ones(dtype=tf.float32)
            offset = create_weight(
              "offset", [num_possible_inputs, out_filters],
              initializer=zero_init)
            scale = create_weight(
              "scale", [num_possible_inputs, out_filters],
              initializer=one_init)
            offset = offset[prev_cell]
            scale = scale[prev_cell]
          
          # the computations
          x = tf.nn.relu(x)
          x = tf.nn.conv2d(
            x,
            filter=w,
            strides=[1, 1, 1, 1], padding="SAME",
            data_format='NCHW' if self._data_format=='channels_first' else 'NHWC')
          #x = batch_normalization(x, self._data_format, self._is_training)
          x, _, _ = tf.nn.fused_batch_norm(
            x, scale, offset, epsilon=_BATCH_NORM_EPSILON, is_training=True,
            data_format='NCHW' if self._data_format=='channels_first' else 'NHWC')
    return x


  def _nas_sep_conv(self, x, curr_cell, prev_cell, filter_size, out_filters, stack_conv=2):
    with tf.variable_scope("sep_conv_{0}x{0}".format(filter_size)):
      num_possible_inputs = curr_cell + 2
      for conv_id in range(stack_conv):
        with tf.variable_scope("stack_{0}".format(conv_id)):
          # create params and pick the correct path
          inp_c = get_channel_dim(x, self._data_format)
          w_depthwise = create_weight(
            "w_depth", [num_possible_inputs, filter_size * filter_size * inp_c],
            initializer=_KERNEL_INITIALIZER)
          w_depthwise = w_depthwise[prev_cell, :]
          w_depthwise = tf.reshape(
            w_depthwise, [filter_size, filter_size, inp_c, 1])

          w_pointwise = create_weight(
            "w_point", [num_possible_inputs, inp_c * out_filters],
            initializer=_KERNEL_INITIALIZER)
          w_pointwise = w_pointwise[prev_cell, :]
          w_pointwise = tf.reshape(w_pointwise, [1, 1, inp_c, out_filters])

          with tf.variable_scope("bn"):
            zero_init = tf.initializers.zeros(dtype=tf.float32)
            one_init = tf.initializers.ones(dtype=tf.float32)
            offset = create_weight(
              "offset", [num_possible_inputs, out_filters],
              initializer=zero_init)
            scale = create_weight(
              "scale", [num_possible_inputs, out_filters],
              initializer=one_init)
            offset = offset[prev_cell]
            scale = scale[prev_cell]

          # the computations
          x = tf.nn.relu(x)
          x = tf.nn.separable_conv2d(
            x,
            depthwise_filter=w_depthwise,
            pointwise_filter=w_pointwise,
            strides=[1, 1, 1, 1], padding="SAME",
            data_format='NCHW' if self._data_format=='channels_first' else 'NHWC')
          #x = batch_normalization(x, self._data_format, self._is_training)
          x, _, _ = tf.nn.fused_batch_norm(
            x, scale, offset, epsilon=_BATCH_NORM_EPSILON, is_training=True,
            data_format='NCHW' if self._data_format=='channels_first' else 'NHWC')
    return x

  def _nas_cell(self, x, curr_cell, prev_cell, op_id, out_filters):
    num_possible_inputs = curr_cell + 1
    
    with tf.variable_scope('max_pool_3x3'):
      max_pool_3 = tf.layers.max_pooling2d(
        x, [3, 3], [1, 1], "SAME", data_format=self._data_format)
      max_pool_c = get_channel_dim(max_pool_3, self._data_format)
      if max_pool_c != out_filters:
        with tf.variable_scope("conv"):
          w = create_weight(
            "w", [num_possible_inputs, max_pool_c * out_filters],
            initializer=_KERNEL_INITIALIZER)
          w = w[prev_cell]
          w = tf.reshape(w, [1, 1, max_pool_c, out_filters])
          max_pool_3 = tf.nn.relu(max_pool_3)
          max_pool_3 = tf.nn.conv2d(max_pool_3, w, strides=[1, 1, 1, 1], padding="SAME",
                                    data_format='NCHW' if self._data_format == 'channels_first' else 'NHWC')
          max_pool_3 = batch_normalization(max_pool_3, is_training=True, #self._is_training,
                                           data_format=self._data_format)
    
    with tf.variable_scope('avg_pool_3x3'):
      avg_pool_3 = tf.layers.average_pooling2d(
        x, [3, 3], [1, 1], "SAME", data_format=self._data_format)
      avg_pool_c = get_channel_dim(avg_pool_3, self._data_format)
      if avg_pool_c != out_filters:
        with tf.variable_scope("conv"):
          w = create_weight(
            "w", [num_possible_inputs, avg_pool_c * out_filters],
            initializer=_KERNEL_INITIALIZER)
          w = w[prev_cell]
          w = tf.reshape(w, [1, 1, avg_pool_c, out_filters])
          avg_pool_3 = tf.nn.relu(avg_pool_3)
          avg_pool_3 = tf.nn.conv2d(avg_pool_3, w, strides=[1, 1, 1, 1], padding="SAME",
                                    data_format='NCHW' if self._data_format == 'channels_first' else 'NHWC')
          avg_pool_3 = batch_normalization(avg_pool_3, is_training=True, #self._is_training,
                                           data_format=self._data_format)

    x_c = get_channel_dim(x, self._data_format)
    if x_c != out_filters:
      with tf.variable_scope("x_conv"):
        w = create_weight("w", [num_possible_inputs, x_c * out_filters],
            initializer=_KERNEL_INITIALIZER)
        w = w[prev_cell]
        w = tf.reshape(w, [1, 1, x_c, out_filters])
        x = tf.nn.relu(x)
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME",
                         data_format='NCHW' if self._data_format == 'channels_first' else 'NHWC')
        x = batch_normalization(x, is_training=True, data_format=self._data_format)

    out = [
      x,
      self._nas_sep_conv(x, curr_cell, prev_cell, 3, out_filters),
      self._nas_sep_conv(x, curr_cell, prev_cell, 5, out_filters),
      max_pool_3,
      avg_pool_3
    ]

    out = tf.stack(out, axis=0)
    out = out[op_id, :, :, :, :]
    return out

  def _cell_base(self, last_inputs, inputs, is_training):
    filters = self._filter_size
    data_format = self._data_format
    #is_training = self._is_training

    with tf.variable_scope('transforme_last_inputs'):
      if last_inputs is None:
        last_inputs = inputs
      last_inputs = self._reduce_prev_layer(last_inputs, inputs, is_training)
    with tf.variable_scope('transforme_inputs'):
      inputs = tf.nn.relu(inputs)
      with tf.variable_scope('1x1'):
        inputs = tf.layers.conv2d(
          inputs=inputs, filters=filters, kernel_size=1, 
          strides=1, padding='SAME', use_bias=_USE_BIAS,
          kernel_initializer=_KERNEL_INITIALIZER,
          data_format=data_format)
      with tf.variable_scope('bn'):
        inputs = batch_normalization(inputs, data_format, is_training=is_training)
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

    last_inputs, inputs = self._cell_base(last_inputs, inputs, is_training=True)
    layers = [last_inputs, inputs]
    used = []
    for i in xrange(num_nodes):
      prev_layers = tf.stack(layers, axis=0)
      with tf.variable_scope('cell_{}'.format(i+1)):
        with tf.variable_scope('x'):
          x_id = dag[4*i]
          x_op = dag[4*i+1]
          x = prev_layers[x_id, :, :, :, :]
          x = self._nas_cell(x, i, x_id, x_op, self._filter_size)
          x_used = tf.one_hot(x_id, depth=num_nodes+2, dtype=tf.int32)
        with tf.variable_scope('y'):
          y_id = dag[4*i+2]
          y_op = dag[4*i+3]
          y = prev_layers[y_id, :, :, :, :]
          y = self._nas_cell(y, i, y_id, y_op, self._filter_size)
          y_used = tf.one_hot(y_id, depth=num_nodes+2, dtype=tf.int32)
        
        output = x + y
        used.extend([x_used, y_used])
        layers.append(output)

    used = tf.add_n(used)
    indices = tf.where(tf.equal(used, 0))
    indices = tf.to_int32(indices)
    indices = tf.reshape(indices, [-1])
    num_outs = tf.size(indices)
    out = tf.stack(layers, axis=0)
    out = tf.gather(out, indices, axis=0)

    inp = prev_layers[0]
    if self._data_format == "channels_last":
      N = tf.shape(inp)[0]
      H = tf.shape(inp)[1]
      W = tf.shape(inp)[2]
      C = tf.shape(inp)[3]
      out = tf.transpose(out, [1, 2, 3, 0, 4])
      out = tf.reshape(out, [N, H, W, num_outs * self._filter_size])
    elif self._data_format == "channels_first":
      N = tf.shape(inp)[0]
      C = tf.shape(inp)[1]
      H = tf.shape(inp)[2]
      W = tf.shape(inp)[3]
      out = tf.transpose(out, [1, 0, 2, 3, 4])
      out = tf.reshape(out, [N, num_outs * self._filter_size, H, W])
    else:
      raise ValueError("Unknown data_format '{0}'".format(self._data_format))

    with tf.variable_scope("final_conv"):
      w = create_weight("w",
                        [self._num_nodes + 2, self._filter_size * self._filter_size],
                        initializer=_KERNEL_INITIALIZER)
      w = tf.gather(w, indices, axis=0)
      w = tf.reshape(w, [1, 1, num_outs * self._filter_size, self._filter_size])
      out = tf.nn.relu(out)
      out = tf.nn.conv2d(out, w, strides=[1, 1, 1, 1], padding="SAME",
                         data_format='NCHW' if self._data_format == 'channels_first' else 'NHWC')
      out = batch_normalization(out, is_training=True, data_format=self._data_format)

    out = tf.reshape(out, tf.shape(prev_layers[0]))

    return out


def _build_aux_head(aux_net, num_classes, params, data_format, is_training):
  with tf.variable_scope('aux_head'):
    aux_logits = tf.nn.relu(aux_net)
    aux_logits = tf.layers.average_pooling2d(
      inputs=aux_logits, 
      pool_size=5, strides=3, padding='VALID', data_format=data_format)
    with tf.variable_scope('proj'):
      aux_logits = tf.layers.conv2d(
        inputs=aux_logits, filters=128, kernel_size=1, 
        strides=1, padding='SAME', use_bias=_USE_BIAS,
        kernel_initializer=_KERNEL_INITIALIZER, 
        data_format=data_format)
      aux_logits = batch_normalization(aux_logits, data_format, is_training)
      aux_logits = tf.nn.relu(aux_logits)
      
    with tf.variable_scope('avg_pool'):
      shape = aux_logits.shape
      if data_format == 'channels_first':
        shape = shape[2:4]
      else:
        shape = shape[1:3]
      aux_logits = tf.layers.conv2d(
        inputs=aux_logits, filters=768, kernel_size=shape, 
        strides=1, padding='VALID', use_bias=_USE_BIAS, 
        kernel_initializer=_KERNEL_INITIALIZER, 
        data_format=data_format)
      aux_logits = batch_normalization(aux_logits, data_format, is_training)
      aux_logits = tf.nn.relu(aux_logits)

    with tf.variable_scope('fc'):
      if data_format == 'channels_first':
        aux_logits = tf.reduce_mean(aux_logits, axis=[2,3])
      else:
        aux_logits = tf.reduce_mean(aux_logits, axis=[1,2])
      aux_logits = tf.layers.dense(inputs=aux_logits, units=num_classes)#, use_bias=_USE_BIAS)
  return aux_logits


def build_model(inputs, params, is_training, reuse=False):
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
  N = params['N']
  num_nodes = params['num_nodes']
  if params['conv_dag'] is None or params['reduc_dag'] is None:
    if params['arch_pool'] is None:
      conv_dag = sample_arch(num_nodes)
      reduc_dag = sample_arch(num_nodes)
    else:
      conv_dag, reduc_dag = sample_arch_from_pool(params['arch_pool'], params['prob'])
  else:
    conv_dag = params['conv_dag']
    reduc_dag = params['reduc_dag']
  if is_training:
    drop_path_keep_prob = params['drop_path_keep_prob']
  else:
    drop_path_keep_prob = 1.0
  dense_dropout_keep_prob = params['dense_dropout_keep_prob']
  total_steps = params['total_steps']
  if params['data_format'] is None:
    data_format = 'channels_first' if tf.test.is_built_with_cuda() else 'channels_last'
  else:
    data_format = params['data_format']
  num_classes = params['num_classes']
  stem_multiplier = params['stem_multiplier']
  use_aux_head = params['use_aux_head']

  
  if data_format == 'channels_first':
    # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
    # This provides a large performance boost on GPU. See
    # https://www.tensorflow.org/performance/performance_guide#data_formats
    inputs = tf.transpose(inputs, [0, 3, 1, 2])
 
  num_cells = N * 3
  total_num_cells = num_cells + 2

  convolution_cell = NASCell(filters, conv_dag, num_nodes, drop_path_keep_prob, total_num_cells,
    total_steps, data_format, is_training)
  reduction_cell = NASCell(filters, reduc_dag, num_nodes, drop_path_keep_prob, total_num_cells,
    total_steps, data_format, is_training)

  reduction_layers = []
  for pool_num in range(1, 3):
    layer_num = (float(pool_num) / (2 + 1)) * num_cells
    layer_num = int(layer_num)
    reduction_layers.append(layer_num)

  if len(reduction_layers) >= 2:
    aux_head_ceill_index = reduction_layers[1]  #- 1

  with tf.variable_scope('body', reuse=reuse):
    with tf.variable_scope('layer_1_stem_conv_3x3'):
      inputs = tf.layers.conv2d(
        inputs=inputs, filters=int(filters*stem_multiplier), kernel_size=3, strides=1,
        padding='SAME', use_bias=_USE_BIAS,
        kernel_initializer=_KERNEL_INITIALIZER,
        data_format=data_format)
    with tf.variable_scope('layer_1_stem_bn'):
      inputs = batch_normalization(inputs, data_format, is_training)

    layers = [None, inputs]

    true_cell_num, filter_scaling = 0, 1

    for cell_num in range(num_cells):
      strides = 1
      if cell_num in reduction_layers:
        filter_scaling *= 2
        with tf.variable_scope('reduction_cell_%d' % (reduction_layers.index(cell_num)+1)):
          #inputs = factorized_reduction(inputs, filters * filter_scaling, 2, data_format, is_training)
          #layers = [layers[-1], inputs]
          inputs = reduction_cell(layers[-1], filter_scaling, 2, layers[-2], true_cell_num)
        layers = [layers[-1], inputs]
        true_cell_num += 1
      with tf.variable_scope('convolution_cell_%d' % (cell_num+1)):
        inputs = convolution_cell(layers[-1], filter_scaling, strides, layers[-2], true_cell_num)
      layers = [layers[-1], inputs]
      true_cell_num += 1
      if use_aux_head and aux_head_ceill_index == cell_num and num_classes and is_training:
        aux_logits = _build_aux_head(inputs, num_classes, params, data_format, is_training)

    inputs = tf.nn.relu(inputs)

    assert inputs.shape.ndims == 4
        
    if data_format == 'channels_first':
      inputs = tf.reduce_mean(inputs, axis=[2,3])
    else:
      inputs = tf.reduce_mean(inputs, axis=[1,2])
      
    # tf.layers.dropout(inputs, rate) where rate is the drop rate
    # tf.nn.dropout(inputs, rate) where rate is the keep prob
    inputs = tf.layers.dropout(inputs, 1 - dense_dropout_keep_prob, training=is_training)

    with tf.variable_scope('fully_connected_layer'):
      inputs = tf.layers.dense(inputs=inputs, units=num_classes)#, use_bias=_USE_BIAS)

  res = {'logits': inputs,
         'conv_dag': conv_dag,
         'reduc_dag': reduc_dag}
  if use_aux_head and is_training:
    res['aux_logits'] = aux_logits
  return res
