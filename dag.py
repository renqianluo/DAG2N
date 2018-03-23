import model

def ENAS():
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
  return conv_dag, reduc_dag

def AmoebaNet_A():
  conv_dag = OrderedDict()
  conv_dag['node_1'] = model.Node('node_1', None, None, None, None)
  conv_dag['node_2'] = model.Node('node_2', None, None, None, None)
  conv_dag['node_3'] = model.Node('node_3', 'node_1', 'node_1', 'avg_pool 3x3', 'max_pool 3x3')
  conv_dag['node_4'] = model.Node('node_4', 'node_3', 'node_1', 'sep_conv 5x5', 'sep_conv 3x3')
  conv_dag['node_5'] = model.Node('node_5', 'node_4', 'node_1', 'avg_pool 3x3', 'sep_conv 3x3')
  conv_dag['node_6'] = model.Node('node_6', 'node_2', 'node_2', 'sep_conv 3x3', 'identity')
  conv_dag['node_7'] = model.Node('node_7', 'node_2', 'node_1', 'avg_pool 3x3', 'identity')

  reduc_dag = OrderedDict()
  reduc_dag['node_1'] = model.Node('node_1', None, None, None, None)
  reduc_dag['node_2'] = model.Node('node_2', None, None, None, None)
  reduc_dag['node_3'] = model.Node('node_3', 'node_1', 'node_2', 'avg_pool 3x3', 'sep_conv 3x3')
  reduc_dag['node_4'] = model.Node('node_4', 'node_1', 'node_3', 'max_pool 3x3', 'sep_conv 7x7')
  reduc_dag['node_5'] = model.Node('node_5', 'node_1', 'node_2', 'sep_conv 7x7', 'avg_pool 3x3')
  reduc_dag['node_6'] = model.Node('node_6', 'node_2', 'node_1', 'max_pool 3x3', 'max_pool 3x3')
  reduc_dag['node_7'] = model.Node('node_7', 'node_6', 'node_1', 'sep_conv 3x3', 'conv 1x7+7x1')
  return conv_dag, reduc_dag

def AmoebaNet_B():
  conv_dag = OrderedDict()
  conv_dag['node_1'] = model.Node('node_1', None, None, None, None)
  conv_dag['node_2'] = model.Node('node_2', None, None, None, None)
  conv_dag['node_3'] = model.Node('node_3', 'node_1', 'node_2', 'sep_conv 3x3', 'identity')
  conv_dag['node_4'] = model.Node('node_4', 'node_2', 'node_2', 'max_pool 3x3', 'conv 1x1')
  conv_dag['node_5'] = model.Node('node_5', 'node_2', 'node_1', 'conv 1x1', 'sep_conv 3x3')
  conv_dag['node_6'] = model.Node('node_6', 'node_4', 'node_4', 'identity', 'conv 1x1')
  conv_dag['node_7'] = model.Node('node_7', 'node_2', 'node_6', 'avg_pool 3x3', 'conv 1x1')

  reduc_dag = OrderedDict()
  reduc_dag['node_1'] = model.Node('node_1', None, None, None, None)
  reduc_dag['node_2'] = model.Node('node_2', None, None, None, None)
  reduc_dag['node_3'] = model.Node('node_3', 'node_1', 'node_1', 'max_pool 2x2', 'max_pool 3x3')
  reduc_dag['node_4'] = model.Node('node_4', 'node_3', 'node_3', 'dil_sep_conv 5x5', 'max_pool 3x3')
  reduc_dag['node_5'] = model.Node('node_5', 'node_3', 'node_2', 'identity', 'conv 3x3')
  reduc_dag['node_6'] = model.Node('node_6', 'node_4', 'node_5', 'avg_pool 3x3', 'conv 1x1')
  reduc_dag['node_7'] = model.Node('node_7', 'node_5', 'node_2', 'identity', 'sep_conv 3x3')
  return conv_dag, reduc_dag

def NASNet_A():
  conv_dag = OrderedDict()
  conv_dag['node_1'] = model.Node('node_1', None, None, None, None)
  conv_dag['node_2'] = model.Node('node_2', None, None, None, None)
  conv_dag['node_3'] = model.Node('node_3', 'node_2', 'node_2', 'sep_conv 3x3', 'identity')
  conv_dag['node_4'] = model.Node('node_4', 'node_1', 'node_2', 'sep_conv 3x3', 'sep_conv 5x5')
  conv_dag['node_5'] = model.Node('node_5', 'node_2', 'node_1', 'avg_pool 3x3', 'identity')
  conv_dag['node_6'] = model.Node('node_6', 'node_1', 'node_1', 'avg_pool 3x3', 'avg_pool 3x3')
  conv_dag['node_7'] = model.Node('node_7', 'node_1', 'node_1', 'sep_conv 5x5', 'sep_conv 3x3')

  reduc_dag = OrderedDict()
  reduc_dag['node_1'] = model.Node('node_1', None, None, None, None)
  reduc_dag['node_2'] = model.Node('node_2', None, None, None, None)
  reduc_dag['node_3'] = model.Node('node_3', 'node_1', 'node_2', 'sep_conv 7x7', 'sep_conv 5x5')
  reduc_dag['node_4'] = model.Node('node_4', 'node_2', 'node_1', 'max_pool 3x3', 'sep_conv 7x7')
  reduc_dag['node_5'] = model.Node('node_5', 'node_2', 'node_1', 'avg_pool 3x3', 'sep_conv 5x5')
  reduc_dag['node_6'] = model.Node('node_6', 'node_2', 'node_3', 'max_pool 3x3', 'sep_conv 3x3')
  reduc_dag['node_7'] = model.Node('node_7', 'node_3', 'node_4', 'avg_pool 3x3', 'identity')
  return conv_dag, reduc_dag