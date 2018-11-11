# pylint: disable=undefined-variable
""" Contains base class for tensorflow models """

import os
import glob
import re
import json
import threading
import contextlib

import numpy as np
import tensorflow as tf

from ... import is_best_practice, Config
from ..utils import unpack_fn_from_config
from ..base import BaseModel
from .layers import mip, conv_block, upsample
from .losses import softmax_cross_entropy, dice
from .train import piecewise_constant


LOSSES = {
    'mse': tf.losses.mean_squared_error,
    'bce': tf.losses.sigmoid_cross_entropy,
    'ce': softmax_cross_entropy,
    'crossentropy': softmax_cross_entropy,
    'absolutedifference': tf.losses.absolute_difference,
    'l1': tf.losses.absolute_difference,
    'cosine': tf.losses.cosine_distance,
    'cos': tf.losses.cosine_distance,
    'hinge': tf.losses.hinge_loss,
    'huber': tf.losses.huber_loss,
    'logloss': tf.losses.log_loss,
    'dice': dice,
}

DECAYS = {
    'exp': tf.train.exponential_decay,
    'invtime': tf.train.inverse_time_decay,
    'naturalexp': tf.train.natural_exp_decay,
    'const': piecewise_constant,
    'poly': tf.train.polynomial_decay,
}


class TFModel(BaseModel):
    r""" Base class for all tensorflow models

    **Configuration**

    ``build`` and ``load`` are inherited from :class:`.BaseModel`.

    device : str or callable
        if str, a device name (e.g. '/device:GPU:0').
        if callable, a function which takes an operation and returns a device name for it.
        See `tf.device <https://www.tensorflow.org/api_docs/python/tf/device>`_ for details.

    session : dict
        `Tensorflow session parameters <https://www.tensorflow.org/api_docs/python/tf/Session#__init__>`_.

    inputs : dict
        model inputs (see :meth:`.TFModel._make_inputs`)

    loss - a loss function, might be defined in one of three formats:
        - name
        - tuple (name, args)
        - dict {'name': name, \**args}

        where name might be one of:
            - short name (`'mse'`, `'ce'`, `'l1'`, `'cos'`, `'hinge'`, `'huber'`, `'logloss'`, `'dice'`)
            - a function name from `tf.losses <https://www.tensorflow.org/api_docs/python/tf/losses>`_
              (e.g. `'absolute_difference'` or `'sparse_softmax_cross_entropy'`)
            - callable

        If loss is a callable, then it should add the result to a loss collection.
        Otherwise, ``add_loss`` should be set to True. An optional collection might also be specified through
        ``loss_collection`` parameter.

        .. note:: Losses from non-default collections won't be detected automatically,
                  so you should process them within your code.

        Examples:

        - ``{'loss': 'mse'}``
        - ``{'loss': {'name': 'sigmoid_cross_entropy', 'label_smoothing': 1e-6}}``
        - ``{'loss': (tf.losses.huber_loss, {'reduction': tf.losses.Reduction.MEAN})}``
        - ``{'loss': external_loss_fn_with_add_loss_inside}``
        - ``{'loss': external_loss_fn_without_add_loss, 'add_loss': True}``
        - ``{'loss': external_loss_fn_to_collection, 'add_loss': True, 'loss_collection': tf.GraphKeys.LOSSES}``

    decay - a learning rate decay algorithm might be defined in one of three formats:
        - name
        - tuple (name, args)
        - dict {'name': name, **args}

        where name might be one of:

        - short name ('exp', 'invtime', 'naturalexp', 'const', 'poly')
        - a function name from `tf.train <https://www.tensorflow.org/api_docs/python/tf/train>`_
          (e.g. 'exponential_decay')
        - a callable

        Examples:

        - ``{'decay': 'exp'}``
        - ``{'decay': ('polynomial_decay', {'decay_steps': 10000})}``
        - ``{'decay': {'name': tf.train.inverse_time_decay, 'decay_rate': .5}``

    optimizer - an optimizer might be defined in one of three formats:
            - name
            - tuple (name, args)
            - dict {'name': name, \**args}

            where name might be one of:

            - short name (e.g. 'Adam', 'Adagrad', any optimizer from
              `tf.train <https://www.tensorflow.org/api_docs/python/tf/train>`_ without a word `Optimizer`)
            - a function name from `tf.train <https://www.tensorflow.org/api_docs/python/tf/train>`_
              (e.g. 'FtlrOptimizer')
            - a callable

        Examples:

        - ``{'optimizer': 'Adam'}``
        - ``{'optimizer': ('Ftlr', {'learning_rate_power': 0})}``
        - ``{'optimizer': {'name': 'Adagrad', 'initial_accumulator_value': 0.01}``
        - ``{'optimizer': functools.partial(tf.train.MomentumOptimizer, momentum=0.95)}``
        - ``{'optimizer': some_optimizer_fn}``

    common : dict
        default parameters for all :func:`.conv_block`

    initial_block : dict
        parameters for the input block, usually :func:`.conv_block` parameters.

        The only required parameter here is ``initial_block/inputs`` which should contain a name or
        a list of names from ``inputs`` which tensors will be passed to ``initial_block`` as ``inputs``.

        Examples:

        - ``{'initial_block/inputs': 'images'}``
        - ``{'initial_block': dict(inputs='features')}``
        - ``{'initial_block': dict(inputs='images', layout='nac nac', filters=64, kernel_size=[7, 3], strides=[1, 2])}``

    body : dict
        parameters for the base network layers, usually :func:`.conv_block` parameters

    head : dict
        parameters for the head layers, usually :func:`.conv_block` parameters

    predictions : str or callable
        an operation applied to the head output to make the predictions tensor which is used in the loss function.

    output : dict or list
        auxiliary operations

    For more details about predictions and auxiliary output operations see :meth:`.TFModel.output`.

    **How to create your own model**

    #. Take a look at :func:`~.layers.conv_block` since it is widely used as a building block almost everywhere.

    #. Define model defaults (e.g. number of filters, batch normalization options, etc)
       by overriding :meth:`.TFModel.default_config`.
       Or skip it and hard code all the parameters in unpredictable places without the possibility to
       change them easily through model's config.

    #. Define build configuration (e.g. number of classes, etc)
       by overriding :meth:`~.TFModel.build_config`.

    #. Override :meth:`~.TFModel.initial_block`, :meth:`~.TFModel.body` and :meth:`~.TFModel.head`, if needed.
       In many cases defaults and build config are just enough to build a network without additional code writing.

    Things worth mentioning:

    #. Input data and its parameters should be defined in configuration under ``inputs`` key.
       See :meth:`.TFModel._make_inputs` for details.

    #. You might want to use a convenient multidimensional :func:`.conv_block`,
       as well as :func:`~.layers.global_average_pooling`, :func:`~.layers.mip`, or other predefined layers.
       Of course, you can use usual `tensorflow layers <https://www.tensorflow.org/api_docs/python/tf/layers>`_.

    #. If you make dropout, batch norm, etc by hand, you might use a predefined ``self.is_training`` tensor.

    #. For decay and training control there is a predefined ``self.global_step`` tensor.

    #. In many cases there is no need to write a loss function, learning rate decay and optimizer
       as they might be defined through config.

    #. For a configured loss to work one of the inputs should have a name ``targets`` and
       one of the tensors in your model should have a name ``predictions``.
       They will be used in a loss function.

    #. If you have defined your own loss function, call `tf.losses.add_loss(...)
       <https://www.tensorflow.org/api_docs/python/tf/losses/add_loss>`_.

    #. If you need to use your own optimizer, assign ``self.train_step`` to the train step operation.
       Don't forget about `UPDATE_OPS control dependency
       <https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization>`_.
    """

    def __init__(self, *args, **kwargs):
        self.session = kwargs.get('session', None)
        self.graph = tf.Graph() if self.session is None else self.session.graph
        self._graph_context = None
        self.is_training = None
        self.global_step = None
        self.loss = None
        self.train_step = None
        self._train_lock = threading.Lock()
        self._attrs = []
        self._saver = None
        self._to_classes = {}
        self._inputs = {}
        self.inputs = None

        super().__init__(*args, **kwargs)

    def build(self, *args, **kwargs):
        """ Build the model

        #. Create a graph
        #. Define ``is_training`` and ``global_step`` tensors
        #. Define a model architecture by calling :meth:``._build``
        #. Create a loss function from config
        #. Create an optimizer and define a train step from config
        #. `Set UPDATE_OPS control dependency on train step
           <https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization>`_
        #. Create a tensorflow session
        """

        def _device_context():
            if 'device' in self.config:
                device = self.config.get('device')
                context = self.graph.device(device)
            else:
                context = contextlib.ExitStack()
            return context

        with self.graph.as_default(), _device_context():
            with tf.variable_scope(self.__class__.__name__):
                with tf.variable_scope('globals'):
                    if self.is_training is None:
                        self.store_to_attr('is_training', tf.placeholder(tf.bool, name='is_training'))
                    if self.global_step is None:
                        self.store_to_attr('global_step', tf.Variable(0, trainable=False, name='global_step'))

                config = self.build_config()
                self._build(config)

                if self.train_step is None:
                    self._make_loss(config)
                    self.store_to_attr('loss', tf.losses.get_total_loss())

                    optimizer = self._make_optimizer(config)

                    if optimizer:
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        with tf.control_dependencies(update_ops):
                            train_step = optimizer.minimize(self.loss, global_step=self.global_step)
                            self.store_to_attr('train_step', train_step)
                else:
                    self.store_to_attr('train_step', self.train_step)

            if self.session is None:
                self.create_session(config)
                self.reset()


    def create_session(self, config=None):
        """ Create TF session """
        config = config if config is not None else self.config
        session_config = config.get('session', default={})
        self.session = tf.Session(**session_config)

    def reset(self):
        """ Reset the trained model to allow a new training from scratch """
        with self.session.graph.as_default():
            self.session.run(tf.global_variables_initializer())

    def _make_inputs(self, names=None, config=None):
        """ Create model input data from config provided

        In the config's inputs section it looks for ``names``, creates placeholders required, and
        makes some typical transformations (like one-hot-encoding), if needed.

        **Configuration**

        inputs : dict
            - key : str
                a placeholder name
            - values : str or dict or tuple
                each input's config

        Input config:

        ``dtype`` : str or tf.DType (by default 'float32')
            data type

        ``shape`` : int, tuple, list or None (default)
            a tensor shape which includes the number of channels/classes and doesn't include a batch size.

        ``classes`` : int, array-like or None (default)
            a number of class or an array of class labels if data labels are strings or anything else except
            ``np.arange(num_classes)``.

        ``data_format`` : str {'channels_first', 'channels_last'} or {'f', 'l'}
            The ordering of the dimensions in the inputs. Default is 'channels_last'.
            For brevity ``data_format`` may be shortened to ``df``.

        ``transform`` : str or callable
            Predefined transforms are

            - ``'ohe'`` - one-hot encoding
            - ``'mip @ d'`` - maximum intensity projection :func:`~.layers.mip`
              with depth ``d`` (should be int)

        ``name`` : str
            a name for the transformed tensor.

        If an input config is a tuple, it should contain all items exactly in the order shown above:
        dtype, shape, classes, data_format, transform, name.
        If an item is None, the default value will be used instead.

        **How it works**

        A placholder with ``dtype``, ``shape`` and with a name ``key`` is created first.
        Then it is transformed with a ``transform`` function in accordance with ``data_format``.
        The resulting tensor will have the name ``name``.

        **Aliases**
        If an input config is a string, it should point to another key from inputs dict.
        This creates an alias to another input which might be convenient to substitute tensor names.

        By default, `targets` is aliased to `labels` or `masks` if present.

        Parameters
        ----------
        names : list
            placeholder names that are expected in the config's 'inputs' section

        Raises
        ------
        KeyError if there is any name missing in the config's 'inputs' section.
        ValueError if there are duplicate names.

        Returns
        -------
        placeholders : dict
            key : str
                a placeholder name
            value : tf.Tensor
                placeholder tensor
        tensors : dict
            key : str
                a placeholder name
            value : tf.Tensor
                an input tensor after transformations
        """
        # pylint:disable=too-many-statements
        config = config.get('inputs')

        names = names or []
        missing_names = set(names) - set(config.keys())
        if len(missing_names) > 0:
            raise KeyError("Inputs should contain {} names".format(missing_names))

        placeholder_names = set(config.keys())
        tensor_names = set(x.get('name') for x in config.values() if isinstance(x, dict) and x.get('name'))
        wrong_names = placeholder_names & tensor_names
        if len(wrong_names) > 0:
            raise ValueError('Inputs contain duplicate names:', wrong_names)

        # add default aliases
        if 'labels' in config and 'targets' not in config:
            config['targets'] = 'labels'
        elif 'masks' in config and 'targets' not in config:
            config['targets'] = 'masks'
        # if targets is defined in the input dict, these implicit aliases will be overwritten.

        param_names = ('dtype', 'shape', 'classes', 'data_format', 'transform', 'name')
        defaults = dict(data_format='channels_last')

        placeholders = dict()
        tensors = dict()

        for input_name, input_config in config.items():
            if isinstance(input_config, str):
                continue
            elif isinstance(input_config, (tuple, list)):
                input_config = list(input_config) + [None for _ in param_names]
                input_config = input_config[:len(param_names)]
                input_config = dict(zip(param_names, input_config))
                input_config = dict((k, v) for k, v in input_config.items() if v is not None)
            input_config = {**defaults, **input_config}

            reshape = None
            shape = input_config.get('shape')
            if isinstance(shape, int):
                shape = (shape,)
            if shape:
                input_config['shape'] = shape
                shape = [None] + list(shape)

            self._inputs[input_name] = dict(config=input_config)

            if self.has_classes(input_name):
                dtype = input_config.get('dtype', tf.int64)
                shape = shape or (None,)
            else:
                dtype = input_config.get('dtype', 'float')
            tensor = tf.placeholder(dtype, shape, input_name)
            placeholders[input_name] = tensor
            self.store_to_attr(input_name, tensor)

            if 'df' in input_config and 'data_format' not in input_config:
                input_config['data_format'] = input_config['df']
            if input_config.get('data_format') == 'l':
                input_config['data_format'] = 'channels_last'
            elif input_config.get('data_format') == 'f':
                input_config['data_format'] = 'channels_first'

            self._inputs[input_name] = dict(config=input_config)
            tensor = self._make_transform(input_name, tensor, input_config)

            if isinstance(reshape, (list, tuple)):
                tensor = tf.reshape(tensor, [-1] + list(reshape))

            name = input_config.get('name')
            if name is not None:
                tensor = tf.identity(tensor, name=name)
                self.store_to_attr(name, tensor)

            tensors[input_name] = tensor

            self._inputs[input_name] = dict(config=input_config, placeholder=placeholders[input_name], tensor=tensor)
            if name is not None:
                self._inputs[name] = self._inputs[input_name]

        # check for aliases
        for input_name, input_config in config.items():
            if isinstance(input_config, str) and input_name not in self._inputs:
                self._inputs[input_name] = self._inputs[input_config]
                tensors[input_name] = tensors[input_config]
                placeholders[input_name] = placeholders[input_config]
                self.store_to_attr(input_name, tensors[input_name])

        self.inputs = tensors

        return placeholders, tensors

    def _make_transform(self, input_name, tensor, config):
        if config is not None:
            transform_names = config.get('transform')
            if not isinstance(transform_names, list):
                transform_names = [transform_names]
            for transform_name in transform_names:
                if isinstance(transform_name, str):
                    transforms = {
                        'ohe': self._make_ohe,
                        'mip': self._make_mip,
                        'mask_downsampling': self._make_mask_downsampling
                    }

                    kwargs = dict()
                    if transform_name.startswith('mip'):
                        parts = transform_name.split('@')
                        transform_name = parts[0].strip()
                        kwargs['depth'] = int(parts[1])

                    tensor = transforms[transform_name](input_name, tensor, config, **kwargs)
                elif callable(transform_name):
                    tensor = transform_name(tensor)
                elif transform_name:
                    raise ValueError("Unknown transform {}".format(transform_name))
        return tensor

    def _make_ohe(self, input_name, tensor, config):
        if config.get('shape') is None and config.get('classes') is None:
            raise ValueError("shape and classes cannot be both None for input " +
                             "'{}' with one-hot-encoding transform".format(input_name))

        num_classes = self.num_classes(input_name)
        axis = -1 if self.data_format(input_name) == 'channels_last' else 1
        tensor = tf.one_hot(tensor, depth=num_classes, axis=axis)
        return tensor

    def _make_mask_downsampling(self, input_name, tensor, config):
        """ Perform mask downsampling with factor from config of tensor. """
        _ = input_name
        factor = config.get('factor')
        size = self.shape(tensor, False)
        if None in size[1:]:
            size = self.shape(tensor, True)
        size = size / factor
        size = tf.cast(size, tf.int32)
        tensor = tf.expand_dims(tensor, -1)
        tensor = tf.image.resize_nearest_neighbor(tensor, size)
        tensor = tf.squeeze(tensor, [-1])
        return tensor

    def to_classes(self, tensor, input_name, name=None):
        """ Convert tensor with labels to classes of ``input_name`` """
        if tensor.dtype in [tf.float16, tf.float32, tf.float64]:
            tensor = tf.argmax(tensor, axis=-1, name=name)
        if self.has_classes(input_name):
            self._to_classes.update({tensor: input_name})
        return tensor

    def _make_mip(self, input_name, tensor, config, depth):
        # mip has to know shape
        if config.get('shape') is None:
            raise ValueError('mip transform requires shape specified in the inputs config')
        if depth is None:
            raise ValueError("mip should be specified as mip @ depth, e.g. 'mip @ 3'")
        tensor = mip(tensor, depth=depth, data_format=self.data_format(input_name))
        return tensor

    def _check_tensor(self, name):
        try:
            tensor = getattr(self, name)
        except AttributeError:
            raise KeyError("Model %s does not have '%s' tensor" % (type(self).__name__, name))
        return tensor

    def _make_loss(self, config):
        loss, args = unpack_fn_from_config('loss', config)

        add_loss = False
        if loss is None:
            pass
        elif isinstance(loss, str):
            loss = LOSSES.get(re.sub('[-_ ]', '', loss).lower(), None)
        elif isinstance(loss, str) and hasattr(tf.losses, loss):
            loss = getattr(tf.losses, loss)
        elif callable(loss):
            pass
        else:
            raise ValueError("Unknown loss", loss)

        if loss is None:
            if len(tf.losses.get_losses()) == 0:
                raise ValueError("Loss is not defined in the model %s" % self)
        else:
            predictions = self._check_tensor('predictions')
            targets = self._check_tensor('targets')

            add_loss = args.pop('add_loss', False)
            if add_loss:
                loss_collection = args.pop('loss_collection', None)
            tensor_loss = loss(targets, predictions, **args)
            if add_loss:
                if loss_collection:
                    tf.losses.add_loss(tensor_loss, loss_collection)
                else:
                    tf.losses.add_loss(tensor_loss)

    def _make_decay(self, config):
        decay_name, decay_args = unpack_fn_from_config('decay', config)

        if decay_name is None:
            pass
        elif callable(decay_name):
            pass
        elif isinstance(decay_name, str) and hasattr(tf.train, decay_name):
            decay_name = getattr(tf.train, decay_name)
        elif decay_name in DECAYS:
            decay_name = DECAYS.get(re.sub('[-_ ]', '', decay_name).lower(), None)
        else:
            raise ValueError("Unknown learning rate decay method", decay_name)

        return decay_name, decay_args

    def _make_optimizer(self, config):
        optimizer_name, optimizer_args = unpack_fn_from_config('optimizer', config)

        if optimizer_name is None or callable(optimizer_name):
            pass
        elif isinstance(optimizer_name, str) and hasattr(tf.train, optimizer_name):
            optimizer_name = getattr(tf.train, optimizer_name)
        elif isinstance(optimizer_name, str) and hasattr(tf.train, optimizer_name + 'Optimizer'):
            optimizer_name = getattr(tf.train, optimizer_name + 'Optimizer')
        else:
            raise ValueError("Unknown optimizer", optimizer_name)

        decay_name, decay_args = self._make_decay(config)
        if decay_name is not None:
            optimizer_args['learning_rate'] = decay_name(**decay_args, global_step=self.global_step)

        if optimizer_name:
            optimizer = optimizer_name(**optimizer_args)
        else:
            optimizer = None

        return optimizer

    def get_number_of_trainable_vars(self):
        """ Return the number of trainable variable in the model graph """
        arr = np.asarray([np.prod(v.get_shape().as_list()) for v in self.graph.get_collection('trainable_variables')])
        return np.sum(arr)

    def get_tensor_config(self, tensor, **kwargs):
        """ Return tensor configuration

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        dict
            tensor config (see :meth:`.TFModel._make_inputs`)

        Raises
        ------
        ValueError shape in tensor configuration isn't int, tuple or list
        """
        if isinstance(tensor, tf.Tensor):
            names = [n for n, i in self._inputs.items() if tensor in [i['placeholder'], i['tensor']]]
            if len(names) > 0:
                input_name = names[0]
            else:
                input_name = tensor.name
        elif isinstance(tensor, str):
            if tensor in self._inputs:
                input_name = tensor
            else:
                input_name = self._map_name(tensor)
        else:
            raise TypeError("Tensor can be tf.Tensor or string, but given %s" % type(tensor))

        if input_name in self._inputs:
            config = self._inputs[input_name]['config']
            shape = config.get('shape')
            if isinstance(shape, int):
                shape = (shape,)
            if shape:
                kwargs['shape'] = shape
        elif isinstance(input_name, str):
            try:
                tensor = self.graph.get_tensor_by_name(input_name)
            except KeyError:
                config = {}
            else:
                shape = tensor.get_shape().as_list()[1:]
                config = dict(dtype=tensor.dtype, shape=shape, name=tensor.name, data_format='channels_last')
        else:
            config = {}

        config = {**config, **kwargs}
        return config

    def _map_name(self, name):
        if isinstance(name, str):
            if hasattr(self, name):
                return getattr(self, name)

            if ':' in name:
                return name

            tensors = tf.get_collection(name)
            if len(tensors) != 0:
                return tensors
            return name + ':0'
        return name

    def _fill_feed_dict(self, feed_dict=None, is_training=True):
        feed_dict = feed_dict or {}
        _feed_dict = {}
        for placeholder, value in feed_dict.items():
            if self.has_classes(placeholder):
                classes = self.classes(placeholder)
                get_indices = np.vectorize(lambda c, arr=classes: np.where(c == arr)[0])
                value = get_indices(value)
            placeholder = self._map_name(placeholder)
            value = self._map_name(value)
            _feed_dict.update({placeholder: value})
        if self.is_training not in _feed_dict:
            _feed_dict.update({self.is_training: is_training})
        return _feed_dict

    def _fill_fetches(self, fetches=None, default=None):
        fetches = fetches or default
        if isinstance(fetches, str):
            _fetches = self._map_name(fetches)
        elif isinstance(fetches, (tuple, list)):
            _fetches = []
            for fetch in fetches:
                _fetches.append(self._map_name(fetch))
        elif isinstance(fetches, dict):
            _fetches = dict()
            for key, fetch in fetches.items():
                _fetches.update({key: self._map_name(fetch)})
        else:
            _fetches = fetches
        return _fetches


    def _recast_output(self, out, ix=None, fetches=None):
        if isinstance(out, np.ndarray):
            fetch = fetches[ix] if ix is not None else fetches
            if isinstance(fetch, str):
                fetch = self.graph.get_tensor_by_name(fetch)
            if fetch in self._to_classes:
                return self.classes(self._to_classes[fetch])[out]
        return out

    def _fill_output(self, output, fetches):
        if isinstance(output, (tuple, list)):
            _output = []
            for i, o in enumerate(output):
                _output.append(self._recast_output(o, i, fetches))
            output = type(output)(_output)
        elif isinstance(output, dict):
            _output = type(output)()
            for k, v in output.items():
                _output.update({k: self._recast_output(v, k, fetches)})
        else:
            output = self._recast_output(output, fetches=fetches)

        return output

    def train(self, fetches=None, feed_dict=None, use_lock=False, **kwargs):
        """ Train the model with the data provided

        Parameters
        ----------
        fetches : tuple, list
            a sequence of `tf.Operation` and/or `tf.Tensor` to calculate
        feed_dict : dict
            input data, where key is a placeholder name and value is a numpy value
        use_lock : bool
            if True, the whole train step is locked, thus allowing for multithreading.

        Returns
        -------
        Calculated values of tensors in `fetches` in the same structure

        See also
        --------
        `Tensorflow Session run <https://www.tensorflow.org/api_docs/python/tf/Session#run>`_

        Notes
        -----
        ``feed_dict`` is not required as all placeholder names and their data can be passed directly.

        Examples
        --------

        ::

            model.train(fetches='loss', feed_dict={'images': B('images'), 'labels': B('labels')})

        The same as::

            model.train(fetches='loss', images=B('images'), labels=B('labels'))
        """
        with self.graph.as_default():
            feed_dict = {} if feed_dict is None else feed_dict
            feed_dict = {**feed_dict, **kwargs}
            _feed_dict = self._fill_feed_dict(feed_dict, is_training=True)
            if fetches is None:
                _fetches = tuple()
            else:
                _fetches = self._fill_fetches(fetches, default=None)

            if use_lock:
                self._train_lock.acquire()

            _all_fetches = []
            if self.train_step:
                _all_fetches += [self.train_step]
            if _fetches is not None:
                _all_fetches += [_fetches]
            if len(_all_fetches) > 0:
                _, output = self.session.run(_all_fetches, feed_dict=_feed_dict)
            else:
                output = None

            if use_lock:
                self._train_lock.release()

            return self._fill_output(output, _fetches)

    def predict(self, fetches=None, feed_dict=None, **kwargs):
        """ Get predictions on the data provided

        Parameters
        ----------
        fetches : tuple, list
            a sequence of `tf.Operation` and/or `tf.Tensor` to calculate
        feed_dict : dict
            input data, where key is a placeholder name and value is a numpy value

        Returns
        -------
        Calculated values of tensors in `fetches` in the same structure

        See also
        --------
        `Tensorflow Session run <https://www.tensorflow.org/api_docs/python/tf/Session#run>`_

        Notes
        -----
        ``feed_dict`` is not required as all placeholder names and their data can be passed directly.

        Examples
        --------

        ::

            model.predict(fetches='loss', feed_dict={'images': B('images'), 'labels': B('labels')})

        The same as::

            model.predict(fetches='loss', images=B('images'), labels=B('labels'))

        """
        with self.graph.as_default():
            feed_dict = {} if feed_dict is None else feed_dict
            feed_dict = {**feed_dict, **kwargs}
            _feed_dict = self._fill_feed_dict(feed_dict, is_training=False)
            _fetches = self._fill_fetches(fetches, default='predictions')
            output = self.session.run(_fetches, _feed_dict)
        return self._fill_output(output, _fetches)

    def save(self, path, *args, **kwargs):
        """ Save tensorflow model.

        Parameters
        ----------
        path : str
            a path to a directory where all model files will be stored

        Examples
        --------
        >>> tf_model = ResNet34()

        Now save the model

        >>> tf_model.save('/path/to/models/resnet34')

        The model will be saved to /path/to/models/resnet34
        """
        with self.graph.as_default():
            if not os.path.exists(path):
                os.makedirs(path)
            if self._saver is None:
                self._saver = tf.train.Saver()
            self._saver.save(self.session, os.path.join(path, 'model'), *args, global_step=self.global_step, **kwargs)
            with open(os.path.join(path, 'attrs.json'), 'w') as f:
                json.dump(self._attrs, f)

    def load(self, path, graph=None, checkpoint=None, *args, **kwargs):
        """ Load a TensorFlow model from files

        Parameters
        ----------
        path : str
            a directory where a model is stored
        graph : str
            a filename for a metagraph file
        checkpoint : str
            a checkpoint file name or None to load the latest checkpoint

        Examples
        --------
        >>> resnet = ResNet34(load=dict(path='/path/to/models/resnet34'))

        >>> tf_model.load(path='/path/to/models/resnet34')
        """
        _ = args, kwargs
        self.graph = tf.Graph()

        with self.graph.as_default():
            if graph is None:
                graph_files = glob.glob(os.path.join(path, '*.meta'))
                graph_files = [os.path.splitext(os.path.basename(graph))[0] for graph in graph_files]
                all_steps = []
                for _graph in graph_files:
                    try:
                        step = int(_graph.split('-')[-1])
                    except ValueError:
                        pass
                    else:
                        all_steps.append(step)
                graph = '-'.join(['model', str(max(all_steps))]) + '.meta'

            graph_path = os.path.join(path, graph)
            saver = tf.train.import_meta_graph(graph_path)

            if checkpoint is None:
                checkpoint_path = tf.train.latest_checkpoint(path)
            else:
                checkpoint_path = os.path.join(path, checkpoint)

            self.create_session()
            saver.restore(self.session, checkpoint_path)

        with open(os.path.join(path, 'attrs.json'), 'r') as json_file:
            self._attrs = json.load(json_file)
        with self.graph.as_default():
            for attr, graph_item in zip(self._attrs, tf.get_collection('attrs')):
                setattr(self, attr, graph_item)

    def store_to_attr(self, attr, graph_item):
        """ Make a graph item (variable or operation) accessible as a model attribute """
        with self.graph.as_default():
            setattr(self, attr, graph_item)
            self._attrs.append(attr)
            tf.get_collection_ref('attrs').append(graph_item)

    @classmethod
    def crop(cls, inputs, resize_to, data_format='channels_last'):
        """ Crop input tensor to a shape of a given image.
        If resize_to does not have a fully defined shape (resize_to.get_shape() has at least one None),
        the returned tf.Tensor will be of unknown shape except the number of channels.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        resize_to : tf.Tensor
            a tensor which shape the inputs should be resized to
        data_format : str {'channels_last', 'channels_first'}
            data format
        """

        static_shape = cls.spatial_shape(resize_to, data_format, False)
        dynamic_shape = cls.spatial_shape(resize_to, data_format, True)

        if None in cls.shape(inputs) + static_shape:
            return cls._dynamic_crop(inputs, static_shape, dynamic_shape, data_format)
        return cls._static_crop(inputs, static_shape, data_format)

    @classmethod
    def _static_crop(cls, inputs, shape, data_format='channels_last'):
        input_shape = np.array(cls.spatial_shape(inputs, data_format))

        if np.abs(input_shape - shape).sum() > 0:
            begin = [0] * inputs.shape.ndims
            if data_format == "channels_last":
                size = [-1] + shape + [-1]
            else:
                size = [-1, -1] + shape
            x = tf.slice(inputs, begin=begin, size=size)
        else:
            x = inputs
        return x

    @classmethod
    def _dynamic_crop(cls, inputs, static_shape, dynamic_shape, data_format='channels_last'):
        input_shape = cls.spatial_shape(inputs, data_format, True)
        n_channels = cls.num_channels(inputs, data_format)
        if data_format == 'channels_last':
            slice_size = [(-1,), dynamic_shape, (n_channels,)]
            output_shape = [None] * (len(static_shape) + 1) + [n_channels]
        else:
            slice_size = [(-1, n_channels), dynamic_shape]
            output_shape = [None, n_channels] + [None] * len(static_shape)

        begin = [0] * len(inputs.get_shape().as_list())
        size = tf.concat(slice_size, axis=0)
        cond = tf.reduce_sum(tf.abs(input_shape - dynamic_shape)) > 0
        x = tf.cond(cond, lambda: tf.slice(inputs, begin=begin, size=size), lambda: inputs)
        x.set_shape(output_shape)
        return x

    @classmethod
    def initial_block(cls, inputs, name='initial_block', **kwargs):
        """ Transform inputs with a convolution block

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Notes
        -----
        For other parameters see :func:`.conv_block`.

        Returns
        -------
        tf.Tensor
        """
        kwargs = cls.fill_params('initial_block', **kwargs)
        if kwargs.get('layout'):
            return conv_block(inputs, name=name, **kwargs)
        return inputs

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        """ Base layers which produce a network embedding

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Notes
        -----
        For other parameters see :func:`.conv_block`.

        Returns
        -------
        tf.Tensor

        Examples
        --------
        ::

            MyModel.body(inputs, layout='ca ca ca', filters=[128, 256, 512], kernel_size=3)
        """
        kwargs = cls.fill_params('body', **kwargs)
        if kwargs.get('layout'):
            return conv_block(inputs, name=name, **kwargs)
        return inputs

    @classmethod
    def head(cls, inputs, name='head', **kwargs):
        """ The last network layers which produce predictions

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Notes
        -----
        For other parameters see :func:`.conv_block`.

        Returns
        -------
        tf.Tensor

        Examples
        --------
        A fully convolutional head with 3x3 and 1x1 convolutions and global max pooling:

            MyModel.head(network_embedding, layout='cacaP', filters=[128, num_classes], kernel_size=[3, 1])

        A fully connected head with dropouts, a dense layer with 1000 units and final dense layer with class logits::

            MyModel.head(network_embedding, layout='dfadf', units=[1000, num_classes], dropout_rate=.15)
        """
        kwargs = cls.fill_params('head', **kwargs)
        if kwargs.get('layout'):
            return conv_block(inputs, name=name, **kwargs)
        return inputs

    def output(self, inputs, predictions=None, ops=None, prefix=None, **kwargs):
        """ Add output operations to the model graph, like predicted probabilities or labels, etc.

        Parameters
        ----------
        inputs : tf.Tensor or a sequence of tf.Tensors
            input tensors

        predictions : str or callable
            an operation applied to inputs to get `predictions` tensor which is used in a loss function:

            - 'sigmoid' - ``sigmoid(inputs)``
            - 'proba' - ``softmax(inputs)``
            - 'labels' - ``argmax(inputs)``
            - callable - a user-defined operation

        ops : a sequence of operations or an ordered dict
            auxiliary operations

            If dict:

            - key - a prefix for each input
            - value - a sequence of aux operations

        Raises
        ------
        ValueError if the number of inputs does not equal to the number of prefixes
        TypeError if inputs is not a Tensor or a sequence of Tensors

        Examples
        --------

        ::

            config = {
                'output': ['proba', 'labels']
            }

        However, if one of the placeholders also has a name 'labels', then it will be lost as the model
        will rewrite the name 'labels' with an output.

        That is where a dict might be convenient::

            config = {
                'output': {'predicted': ['proba', 'labels']}
            }

        Now the output will be stored under names 'predicted_proba' and 'predicted_labels'.

        For multi-output models ensure that an ordered dict is used (e.g. :class:`~collections.OrderedDict`).
        """
        if ops is None:
            ops = []
        elif not isinstance(ops, (dict, tuple, list)):
            ops = [ops]
        if not isinstance(ops, dict):
            ops = {'': ops}

        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]

        for i, tensor in enumerate(inputs):
            if not isinstance(tensor, tf.Tensor):
                raise TypeError("Network output is expected to be a Tensor, but given {}".format(type(inputs)))

            prefix = [*ops.keys()][i]
            if prefix:
                ctx = tf.variable_scope(prefix)
                ctx.__enter__()
            else:
                ctx = None
            attr_prefix = prefix + '_' if prefix else ''

            self._add_output_op(tensor, predictions, 'predictions', '', **kwargs)
            for oper in ops[prefix]:
                self._add_output_op(tensor, oper, oper, attr_prefix, **kwargs)

            if ctx:
                ctx.__exit__(None, None, None)

    def _add_output_op(self, inputs, oper, name, attr_prefix, **kwargs):
        if oper is None:
            self._add_output_identity(inputs, name, attr_prefix, **kwargs)
        elif oper == 'sigmoid':
            self._add_output_sigmoid(inputs, name, attr_prefix, **kwargs)
        elif oper == 'proba':
            self._add_output_proba(inputs, name, attr_prefix, **kwargs)
        elif oper == 'labels':
            self._add_output_labels(inputs, name, attr_prefix, **kwargs)
        elif callable(oper):
            self._add_output_callable(inputs, oper, None, attr_prefix, **kwargs)

    def _add_output_identity(self, inputs, name, attr_prefix, **kwargs):
        _ = kwargs
        x = tf.identity(inputs, name=name)
        self.store_to_attr(attr_prefix + name, x)
        return x

    def _add_output_sigmoid(self, inputs, name, attr_prefix, **kwargs):
        _ = kwargs
        proba = tf.sigmoid(inputs, name=name)
        self.store_to_attr(attr_prefix + name, proba)

    def _add_output_proba(self, inputs, name, attr_prefix, **kwargs):
        axis = self.channels_axis(kwargs['data_format'])
        proba = tf.nn.softmax(inputs, name=name, dim=axis)
        self.store_to_attr(attr_prefix + name, proba)

    def _add_output_labels(self, inputs, name, attr_prefix, **kwargs):
        class_axis = self.channels_axis(kwargs.get('data_format'))
        predicted_classes = tf.argmax(inputs, axis=class_axis, name=name)
        self.store_to_attr(attr_prefix + name, predicted_classes)

    def _add_output_callable(self, inputs, oper, name, attr_prefix, **kwargs):
        _ = kwargs
        x = oper(inputs)
        name = name or oper.__name__
        self.store_to_attr(attr_prefix + name, x)
        return x

    @classmethod
    def default_config(cls):
        """ Define model defaults

        You need to override this method if you expect your model or its blocks to serve as a base for other models
        (e.g. VGG for FCN, ResNet for LinkNet, etc).

        Put here all constants (like the number of filters, kernel sizes, block layouts, strides, etc)
        specific to the model, but independent of anything else (like image shapes, number of classes, etc).

        These defaults can be changed in :meth:`~.TFModel.build_config` or when calling :meth:`.Pipeline.init_model`.

        Usually, it looks like::

            @classmethod
            def default_config(cls):
                config = TFModel.default_config()
                config['initial_block'] = dict(layout='cnap', filters=16, kernel_size=7, strides=2,
                                               pool_size=3, pool_strides=2)
                config['body/filters'] = 32
                config['head'] = dict(layout='cnadV', dropout_rate=.2)
                return config
        """
        config = Config()
        config['inputs'] = {}
        config['common'] = {}
        config['initial_block'] = {}
        config['body'] = {}
        config['head'] = {}
        config['predictions'] = None
        config['output'] = None
        config['optimizer'] = ('Adam', dict())

        if is_best_practice():
            config['common'] = {'batch_norm': {'momentum': .1}}
            config['optimizer'][1].update({'use_locking': True})

        return config

    @classmethod
    def fill_params(cls, _name, **kwargs):
        """ Fill block params from default config and kwargs """
        config = cls.default_config()
        _config = config.get(_name)
        config = {**config['common'], **_config, **kwargs}
        return config

    def build_config(self, names=None):
        """ Define a model architecture configuration

        It takes just 2 steps:

        #. Define names for all placeholders and make input tensors by calling ``super().build_config(names)``.

           If the model config does not contain any name from ``names``, :exc:`KeyError` is raised.

           See :meth:`.TFModel._make_inputs` for details.

        #. Define parameters for :meth:`.TFModel.initial_block`, :meth:`.TFModel.body`, :meth:`.TFModel.head`
           which depend on inputs.

        #. Don't forget to return ``config``.

        Typically it looks like this::

            def build_config(self, names=None):
                names = names or ['images', 'labels']
                config = super().build_config(names)
                config['head']['num_classes'] = self.num_classes('targets')
                return config
        """
        config = self.default_config()

        config = config + self.config

        if config.get('inputs'):
            with tf.variable_scope('inputs'):
                self._make_inputs(names, config)
            inputs = self.get('initial_block/inputs', config)

            if isinstance(inputs, str):
                config['common/data_format'] = self.data_format(inputs)
                config['initial_block/inputs'] = self.inputs[inputs]
            elif isinstance(inputs, list):
                config['initial_block/inputs'] = [self.inputs[name] for name in inputs]
            else:
                raise ValueError('initial_block/inputs should be specified with a name or a list of names.')

        return config


    def _build(self, config=None):
        defaults = {'is_training': self.is_training, **config['common']}
        config['initial_block'] = {**defaults, **config['initial_block']}
        config['body'] = {**defaults, **config['body']}
        config['head'] = {**defaults, **config['head']}

        x = self.initial_block(**config['initial_block'])
        x = self.body(inputs=x, **config['body'])
        output = self.head(inputs=x, **config['head'])
        self.output(output, predictions=config['predictions'], ops=config['output'], **defaults)

    @classmethod
    def channels_axis(cls, data_format):
        """ Return the channels axis for the tensor

        Parameters
        ----------
        data_format : str {'channels_last', 'channels_first'}

        Returns
        -------
        number of channels : int
        """
        return 1 if data_format == "channels_first" or data_format.startswith("NC") else -1

    def data_format(self, tensor, **kwargs):
        """ Return the tensor data format (channels_last or channels_first)

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        data_format : str
        """
        config = self.get_tensor_config(tensor, **kwargs)
        return config.get('data_format')

    def has_classes(self, tensor):
        """ Check if a tensor has classes defined in the config """
        config = self.get_tensor_config(tensor)
        has = config.get('classes') is not None
        return has

    def classes(self, tensor):
        """ Return the classes """
        config = self.get_tensor_config(tensor)
        classes = config.get('classes')
        if isinstance(classes, int):
            return np.arange(classes)
        return np.asarray(classes)

    def num_classes(self, tensor):
        """ Return the  number of classes """
        if self.has_classes(tensor):
            classes = self.classes(tensor)
            return classes if isinstance(classes, int) else len(classes)
        return self.get_num_channels(tensor)

    def get_num_channels(self, tensor, **kwargs):
        """ Return the number of channels in the tensor

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        number of channels : int
        """
        config = self.get_tensor_config(tensor, **kwargs)
        shape = (None,) + config.get('shape')
        channels_axis = self.channels_axis(tensor, **kwargs)
        return shape[channels_axis] if shape else None

    @classmethod
    def num_channels(cls, tensor, data_format='channels_last'):
        """ Return number of channels in the input tensor

        Parameters
        ----------
        tensor : tf.Tensor

        Returns
        -------
        shape : tuple of ints
        """
        shape = tensor.get_shape().as_list()
        axis = cls.channels_axis(data_format)
        return shape[axis]

    def get_shape(self, tensor, **kwargs):
        """ Return the tensor shape without batch dimension

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        shape : tuple
        """
        config = self.get_tensor_config(tensor, **kwargs)
        return config.get('shape')

    @classmethod
    def shape(cls, tensor, dynamic=False):
        """ Return shape of the input tensor without batch size

        Parameters
        ----------
        tensor : tf.Tensor

        dynamic : bool
            if True, returns tensor which represents shape. If False, returns list of ints and/or Nones

        Returns
        -------
        shape : tf.Tensor or list
        """
        if dynamic:
            shape = tf.shape(tensor)
        else:
            shape = tensor.get_shape().as_list()
        return shape[1:]

    def get_spatial_dim(self, tensor, **kwargs):
        """ Return the tensor spatial dimensionality (without batch and channels dimensions)

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        number of spatial dimensions : int
        """
        config = self.get_tensor_config(tensor, **kwargs)
        return len(config.get('shape')) - 1

    @classmethod
    def spatial_dim(cls, tensor):
        """ Return spatial dim of the input tensor (without channels and batch dimension)

        Parameters
        ----------
        tensor : tf.Tensor

        Returns
        -------
        dim : int
        """
        return len(tensor.get_shape().as_list()) - 2

    def get_spatial_shape(self, tensor, **kwargs):
        """ Return the tensor spatial shape (without batch and channels dimensions)

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        spatial shape : tuple
        """
        config = self.get_tensor_config(tensor, **kwargs)
        data_format = config.get('data_format')
        shape = config.get('shape')[:-1] if data_format == 'channels_last' else config.get('shape')[1:]
        return shape

    @classmethod
    def spatial_shape(cls, tensor, data_format='channels_last', dynamic=False):
        """ Return spatial shape of the input tensor

        Parameters
        ----------
        tensor : tf.Tensor

        dynamic : bool
            if True, returns tensor which represents shape. If False, returns list of ints and/or Nones

        Returns
        -------
        shape : tf.Tensor or list
        """
        if dynamic:
            shape = tf.shape(tensor)
        else:
            shape = tensor.get_shape().as_list()
        axis = slice(1, -1) if data_format == "channels_last" else slice(2, None)
        return shape[axis]

    def get_batch_size(self, tensor):
        """ Return batch size (the length of the first dimension) of the input tensor

        Parameters
        ----------
        tensor : str or tf.Tensor

        Returns
        -------
        batch size : int or None
        """
        if isinstance(tensor, tf.Tensor):
            pass
        elif isinstance(tensor, str):
            if tensor in self._inputs:
                tensor = self._inputs[tensor]['placeholder']
            else:
                input_name = self._map_name(tensor)
                if input_name in self._inputs:
                    tensor = self._inputs[input_name]
                else:
                    tensor = self.graph.get_tensor_by_name(input_name)
        else:
            raise TypeError("Tensor can be tf.Tensor or string, but given %s" % type(tensor))

        return tensor.get_shape().as_list()[0]

    @classmethod
    def batch_size(cls, tensor):
        """ Return batch size (the length of the first dimension) of the input tensor

        Parameters
        ----------
        tensor : tf.Tensor

        Returns
        -------
        batch size : int or None
        """
        return tensor.get_shape().as_list()[0]

    @classmethod
    def se_block(cls, inputs, ratio, name='se', **kwargs):
        """ Squeeze and excitation block

        Hu J. et al. "`Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`_"

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        ratio : int
            squeeze ratio for the number of filters

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            data_format = kwargs.get('data_format')
            in_filters = cls.num_channels(inputs, data_format)
            x = conv_block(inputs,
                           **{**kwargs, 'layout': 'Vfafa', 'units': [in_filters//ratio, in_filters],
                              'name': 'se', 'activation': [tf.nn.relu, tf.nn.sigmoid]})

            shape = [-1] + [1] * (cls.spatial_dim(inputs) + 1)
            axis = cls.channels_axis(data_format)
            shape[axis] = in_filters
            scale = tf.reshape(x, shape)
            x = inputs * scale
        return x

    @classmethod
    def upsample(cls, inputs, factor=None, resize_to=None, layout='b', name='upsample', **kwargs):
        """ Upsample input tensor

        Parameters
        ----------
        inputs : tf.Tensor or tuple of two tf.Tensor
            a tensor to resize and a tensor which size to resize to
        factor : int
            an upsamping scale
        resize_to : tf.Tensor
            a tensor which shape the output should be resized to
        layout : str
            a resizing technique, a sequence of:

            - R - use residual connection with bilinear additive upsampling (must be the first symbol)
            - b - bilinear resize
            - B - bilinear additive upsampling
            - N - nearest neighbor resize
            - t - transposed convolution
            - X - subpixel convolution

        Returns
        -------
        tf.Tensor
        """
        if np.all(factor == 1):
            return inputs

        if kwargs.get('filters') is None:
            kwargs['filters'] = cls.num_channels(inputs, kwargs['data_format'])

        x = upsample(inputs, factor=factor, layout=layout, name=name, **kwargs)
        if resize_to is not None:
            x = cls.crop(x, resize_to, kwargs['data_format'])
        return x
