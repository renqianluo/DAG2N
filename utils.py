from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def _add_variable_proxy_methods(var, proxy_tensor):
  """Proxy methods of underlying variable.

  This enables our custom getters to still work with, e.g., batch norm.

  Args:
    var: Variable to proxy
    proxy_tensor: Tensor that is identity of var
  """
  proxy_tensor.read_value = lambda: tf.identity(proxy_tensor)
  proxy_tensor.assign_sub = var.assign_sub

def transpose_list_of_lists(lol):
  """Transpose a list of equally-sized python lists.

  Args:
    lol: a list of lists
  Returns:
    a list of lists
  """
  assert lol, "cannot pass the empty list"
  return [list(x) for x in zip(*lol)]

class Parallelism(object):
  """Helper class for creating sets of parallel function calls.

  The purpose of this class is to replace this code:

      e = []
      f = []
      for i in xrange(len(devices)):
        with tf.device(devices[i]):
          e_, f_ = func(a[i], b[i], c)
          e.append(e_)
          f.append(f_)

  with this code:

      e, f = expert_utils.Parallelism(devices)(func, a, b, c)
  """

  def __init__(self,
               device_names_or_functions,
               reuse=None,
               caching_devices=None,
               daisy_chain_variables=False):
    """Create a Parallelism.

    Args:
      device_names_or_functions: A list of length n, containing device names
        or device functions (see `tf.device`)
      reuse: True or None.  Whether to reuse variables created in the first
        replica in the subsequent replicas.
      caching_devices: Either `None`, or a list of length n containing device
        names.
      daisy_chain_variables: a boolean - if true, then copies variables in a
        daisy chain between devices.

    Returns:
      a Parallelism.
    """
    assert device_names_or_functions
    self._devices = device_names_or_functions
    self._n = len(device_names_or_functions)
    self._reuse = reuse
    self._caching_devices = self._maybe_repeat(caching_devices)
    self._daisy_chain_variables = daisy_chain_variables

  def __call__(self, fn, *args, **kwargs):
    """A parallel set of function calls (using the specified devices).

    Args:
      fn: a function or a list of n functions.
      *args: additional args.  Each arg should either be not a list, or a list
         of length n.
      **kwargs: additional keyword args.  Each arg should either be not a
         list, or a list of length n.

    Returns:
      either a single list of length n (if fn does not return a tuple), or a
      tuple of lists of length n (if fn returns a tuple).
    """
    # Construct lists or args and kwargs for each function.
    if args:
      my_args = transpose_list_of_lists(
          [self._maybe_repeat(arg) for arg in args])
    else:
      my_args = [[] for _ in xrange(self.n)]
    my_kwargs = [{} for _ in xrange(self.n)]
    for k, v in six.iteritems(kwargs):
      vals = self._maybe_repeat(v)
      for i in xrange(self.n):
        my_kwargs[i][k] = vals[i]

    # Construct lists of functions.
    fns = self._maybe_repeat(fn)

    # Now make the parallel call.
    outputs = []
    cache = {}
    for i in xrange(self.n):

      def daisy_chain_getter(getter, name, *args, **kwargs):
        """Get a variable and cache in a daisy chain."""
        device_var_key = (self._devices[i], name)
        if device_var_key in cache:
          # if we have the variable on the correct device, return it.
          return cache[device_var_key]
        if name in cache:
          # if we have it on a different device, copy it from the last device
          v = tf.identity(cache[name])
        else:
          var = getter(name, *args, **kwargs)
          v = tf.identity(var._ref())  # pylint: disable=protected-access
          _add_variable_proxy_methods(var, v)
        # update the cache
        cache[name] = v
        cache[device_var_key] = v
        return v

      # Variable scope will not reset caching_device on reused variables,
      # so we make a custom getter that uses identity to cache the variable.
      # pylint: disable=cell-var-from-loop
      def caching_getter(getter, name, *args, **kwargs):
        """Cache variables on device."""
        key = (self._caching_devices[i], name)
        if key in cache:
          return cache[key]

        v = getter(name, *args, **kwargs)
        with tf.device(self._caching_devices[i]):
          ret = tf.identity(v._ref())  # pylint: disable=protected-access
        _add_variable_proxy_methods(v, ret)
        cache[key] = ret
        return ret

      if self._daisy_chain_variables:
        custom_getter = daisy_chain_getter
      elif self._caching_devices[i]:
        custom_getter = caching_getter
      else:
        custom_getter = None
      # pylint: enable=cell-var-from-loop
      with tf.name_scope("parallel_%d" % i):
        with tf.variable_scope(
            tf.get_variable_scope() if self._reuse else "parallel_%d" % i,
            reuse=True if i > 0 and self._reuse else None,
            caching_device=self._caching_devices[i],
            custom_getter=custom_getter):
          # TODO(noam, epot, avaswani)
          # Allows for passing no device in case you want to default to the
          # existing device. This is needed when we put all experts on a single
          # device, for example in local_moe.
          if self._devices[i] != DEFAULT_DEV_STRING:
            with tf.device(self._devices[i]):
              outputs.append(fns[i](*my_args[i], **my_kwargs[i]))
          else:
            outputs.append(fns[i](*my_args[i], **my_kwargs[i]))
    if isinstance(outputs[0], tuple):
      outputs = list(zip(*outputs))
      outputs = tuple([list(o) for o in outputs])
    return outputs

  @property
  def n(self):
    return self._n

  @property
  def devices(self):
    return self._devices

  def _maybe_repeat(self, x):
    """Utility function for processing arguments that are singletons or lists.

    Args:
      x: either a list of self.n elements, or not a list.

    Returns:
      a list of self.n elements.
    """
    if isinstance(x, list):
      assert len(x) == self.n
      return x
    else:
      return [x] * self.n


def data_parallelism(worker_gpu=1):
  """Over which devices do we split each training batch.
  This function returns an expert_utils.Parallelism object, which can be used
  to build the model.  It is configured in a way that any variables created
  by `tf.get_variable` will be assigned to the parameter servers and shared
  between datashards.

  Returns:
    a expert_utils.Parallelism.
  """
  if worker_gpu > 1:
    datashard_devices = [
        "/GPU:%d" % d
          for d in range(worker_gpu)
    ]
    caching_devices = ["/GPU:0"] * worker_gpu
  else:
    datashard_devices = ['']
    caching_devices = None
  tf.logging.info("datashard_devices: %s", datashard_devices)
  tf.logging.info("caching_devices: %s", caching_devices)
  return Parallelism(
      datashard_devices,
      reuse=True,
      caching_devices=caching_devices,
      daisy_chain_variables=True)