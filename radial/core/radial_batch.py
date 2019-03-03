""" Batch class for radial regime regression. """

from functools import wraps

import random
import numpy as np
import scipy as sc

from . import radial_batch_tools as bt
from ..batchflow import Batch, ImagesBatch, action, inbatch_parallel, any_action_failed, FilesIndex, DatasetIndex, R

def _safe_make_array(dst, len_src):
    """ Makes array from dst data. Raises exception if length of resulting array
    is not equal to len_src. If dst is None returns array of None of length len_src.
    Parameters
    ----------
    dst : str or None or list, tuple, np.ndarray of str
        Contains names of batch component(s) to be processed
    len_src : int
        Desired length of resulting array
    Returns
    -------
    np.array
    """
    if not isinstance(dst, (list, tuple, np.ndarray)):
        if not dst:
            dst = np.array([None] * len_src)
        else:
            dst = np.asarray(dst).reshape(-1)
    elif not len(dst) == len_src:
        raise ValueError('Number of given components must be equal')
    return dst

def safe_src_dst_preprocess(method):
    """ Decorator used for preprocessing  kwargs such as src, dst, src_range, dst_range.
    Modifies str to list of str, inserts default values and raises ValueError if mandatory
    parameters are missing.

    Parameters
    ----------
    method : method to be decorated

    Returns
    -------
    Method with updated kwargs
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        """ Wrapper
        """
        src = kwargs.get('src', None)
        dst = kwargs.get('dst', None)
        src_range = kwargs.get('src_range', None)
        dst_range = kwargs.get('dst_range', None)

        src = np.asarray(src).reshape(-1)
        dst = _safe_make_array(dst, len(src))

        src_range = _safe_make_array(src_range, len(src))
        dst_range = _safe_make_array(dst_range, len(src))

        for i, component in enumerate(src):
            if not component:
                raise ValueError('Required argument `src` (pos 1) not found')
            if not hasattr(self, component):
                raise ValueError('Component passed in src does not exist')
            if not dst[i]:
                dst[i] = component
            if not hasattr(self, dst[i]):
                setattr(self, dst[i], self.array_of_nones)

            if src_range[i] and not hasattr(self, src_range[i]):
                raise ValueError('Component provided in src_range does not exist')
            if dst_range[i] and not hasattr(self, dst_range[i]):
                setattr(self, dst_range[i], self.array_of_nones)
        kwargs.update(src=src, dst=dst, src_range=src_range, dst_range=dst_range)
        return method(self, *args, **kwargs)
    return wrapper


class RadialBatch(Batch):
    """
    Batch class that stores multiple time series along with other parameters
    for radial flow regime regression.

    Parameters
    ----------
    index : DatasetIndex
        Unique identifiers of rig data in the batch.
    preloaded : tuple, optional
        Data to put in the batch if given. Defaults to ``None``.

    Attributes
    ----------
    index : DatasetIndex
        Unique identifiers of rig data in the batch.
    """
    def __init__(self, index, preloaded=None):
        super().__init__(index, preloaded)
        self.time = self.array_of_nones
        self.derivative = self.array_of_nones
        self.rig_type = self.array_of_nones
        self.target = self.array_of_nones
        self.predictions = self.array_of_nones

    @property
    def components(self):
        """tuple of str: Data components names."""
        return "time", "derivative", "rig_type", "target", "predictions"

    @property
    def array_of_nones(self):
        """1-D ndarray: ``NumPy`` array with ``None`` values."""
        return np.array([None] * len(self.index))

    def _reraise_exceptions(self, results):
        """Reraise all exceptions in the ``results`` list.

        Parameters
        ----------
        results : list
            Post function computation results.

        Raises
        ------
        RuntimeError
            If any paralleled action raised an ``Exception``.
        """
        if any_action_failed(results):
            all_errors = self.get_errors(results)
            raise RuntimeError("Cannot assemble the batch", all_errors)

    @action
    def load(self, fmt=None, components=None, *args, **kwargs):
        """Load given batch components.

        This method supports loading of data from npz format.

        Parameters
        ----------
        fmt : str, optional
            Source format.
        components : str or array-like, optional
            Components to load.

        Returns
        -------
        batch : RadialBatch
            Batch with loaded components. Changes batch data inplace.
        """
        _ = args, kwargs
        components = [components] if isinstance(components, str) else components

        if fmt == 'csv':
            return self._load_table(fmt=fmt, components=components, *args, **kwargs)
        if fmt == 'npz':
            if components is None:
                components = ["time", "derivative", "rig_type", "target"]
            return self._load(fmt, components)

    @inbatch_parallel(init="indices", post="_assemble_load", target="threads")
    def _load(self, indice, fmt=None, components=None, *args, ** kwargs):
        """
        Load given components from file.

        Parameters
        ----------
        fmt : str, optional
            Source format.
        components : iterable, optional

        Returns
        -------
        batch : RadialBatch
            Batch with loaded components. Changes batch data inplace.

        Raises
        ------
        ValueError
            If source path is not specified and batch's ``index`` is not a
            ``FilesIndex``.
        """
        loaders = {'npz': bt.load_npz}

        if isinstance(self.index, FilesIndex):
            path = self.index.get_fullpath(indice)  # pylint: disable=no-member
        else:
            raise ValueError("Source path is not specified")

        return loaders[fmt](path, components, *args, **kwargs)

    def _assemble_load(self, results, *args, **kwargs):
        """Concatenate results of different workers and update ``self``.

        Parameters
        ----------
        results : list
            Workers' results.

        Returns
        -------
        batch : RadialBatch
            Assembled batch. Changes components inplace.
        """
        _ = args, kwargs
        self._reraise_exceptions(results)
        components = kwargs.get("components", None)
        if components is None:
            components = self.components
        for comp, data in zip(components, zip(*results)):
            data = np.array(data + (None,))[:-1]
            if comp in ['rig_type', 'target']:
                data = np.array([d for d in data])
            setattr(self, comp, data)
        return self

    @action
    @inbatch_parallel(init="indices", target="threads")
    def drop_negative(self, index):
        """
        Leaves only positive values.

        Raises
        ------
        ValueError
            If all values in derivative component are negative.
        """
        i = self.get_pos(None, 'time', index)
        time = self.time[i]
        derivative = self.derivative[i]

        mask = derivative > 0

        if sum(mask) == 0:
            raise ValueError("All values in derivative {} are negative!".format(index))

        self.time[i] = time[mask]
        self.derivative[i] = derivative[mask]

        return self

    @action
    @inbatch_parallel(init="indices", target="threads")
    def get_samples(self, index, n_points, n_samples=1,
                    sampler=None, interpolate='linear', seed=None):
        """ Draws samples from the interpolation of time and derivative components.

        Performs interpolation of the `derivative` on time using
        `scipy.interpolate.interp1d`, samples `n_points` within `time` component
        range, and calculates values of `derivative` in sampled points.
        Sampler should return values from [0, 1] interval, which later is
        stretched to the range on `time` component. Size of the sample equals
        `(n_samples, n_points)`.

        Parameters
        ----------
        n_points : int
            Size of a sample.
        n_samples : int, optional
            Numper of samples. Defaults to `1`.
        sampler : function or named expression
            Method to sample points within [0, 1] range.
            If callable, it should only take `size` as an argument.
        interpolate : str, optional
            Specifies the kind of interpolation as a string
            ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
            'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic'
            refer to a spline interpolation of zeroth, first, second or third
            order; 'previous' and 'next' simply return the previous or next value
            of the point) or as an integer specifying the order of the spline
            interpolator to use.
        seed : int, optional
            Numpy seed to be fixed. Defaults to None.
        kwargs : misc
            Named arguments to the sampling
            or interpolation functions.

        Returns
        -------
        batch : RadialBatch
            Batch with changed `time` and `derivative`
            components. Changes components inplace.

        Note
        ----
        Sampler should have following parameters: `tmin`, `tmax`, `size`. Other
        parameters to this function should be passed via kwargs.
        See ``beta_sampler`` method for example.
        """
        if seed:
            np.random.seed(seed)

        i = self.get_pos(None, 'time', index)
        time = self.time[i]
        derivative = self.derivative[i]

        interpolater = sc.interpolate.interp1d(time, derivative, kind=interpolate,
                                               bounds_error=True, assume_sorted=True)

        if  isinstance(sampler, R):
            sampler = R(sampler.name, **sampler.kwargs, size=(n_samples, n_points))
            samples = sampler.get()
        elif callable(sampler):
            samples = sampler(size=(n_samples, n_points))
        else:
            raise ValueError("You should specify sampler function!")

        tmin, tmax = np.min(time), np.max(time)
        sample_times = samples * (tmax - tmin) + tmin
        sample_times.sort(axis=-1)

        sample_derivatives = interpolater(sample_times)

        self.time[i] = np.array(sample_times)
        self.derivative[i] = np.array(sample_derivatives)
        return self

    @action
    def unstack_samples(self):
        """Create a new batch in which each element of `time` and `derivative`
        along axis 0 is considered as a separate signal.

        This method creates a new batch and unstacks components `time` and
        `derivative`. Then the method updates other components by replication if
        they are ndarrays of objects. Other types of components will be discarded.

        Returns
        -------
        batch : same class as self
            Batch with split `time` and `derivative` and replicated other components.

        Examples
        --------
        >>> batch.time
        array([array([[ 0,  1,  2,  3],
                      [ 4,  5,  6,  7],
                      [ 8,  9, 10, 11]])],
              dtype=object)

        >>> batch = batch.unstack_signals()
        >>> batch.time
        array([array([0, 1, 2, 3]),
               array([4, 5, 6, 7]),
               array([ 8,  9, 10, 11])],
              dtype=object)
        """
        n_reps = [sig.shape[0] for sig in self.time]

        # Adding [None] to the list and removing it from resulting
        # array to make ndarray of arrays
        time = np.array([sample for observation in self.time
                         for sample in observation] + [None])[:-1]
        derivative = np.array([sample for observation in self.derivative
                               for sample in observation] + [None])[:-1]

        index = DatasetIndex(len(time))
        batch = type(self)(index)
        batch.time = time
        batch.derivative = derivative

        for component_name in set(self.components).intersection({"rig_type", "target"}):
            component = getattr(self, component_name)
            val = [elem for elem, n in zip(component, n_reps) for _ in range(n)]
            val = np.array(val + [None])[:-1]
            setattr(batch, component_name, val)

        return batch

    @action
    @inbatch_parallel(init="indices", target="threads")
    def drop_outliers(self, index):
        """
        Finds and deletes outliers values.
        """
        i = self.get_pos(None, 'time', index)
        time = self.time[i]
        derivative = self.derivative[i]

        neighbors = np.diff(time)
        mean_elems = np.array([neighbors[i]/np.mean(np.delete(neighbors, i)) for i in range(len(time)-1)])
        outliers = np.where(mean_elems > 70)[0]
        outliers = np.arange(outliers[0], time.shape) if outliers.shape[0] > 0 else np.empty(0)

        self.time[i] = np.delete(time, outliers)
        self.derivative[i] = np.delete(derivative, outliers)
        return self

    @action
    @safe_src_dst_preprocess
    @inbatch_parallel(init='indices')
    def normalize(self, ix, src=None, dst=None, src_range=None, dst_range=None):
        """ Normalizes data element-wise to range [0, 1] by exctracting
        min(data) and dividing by (max(data)-min(data)).
        Parameters
        ----------
        src : list of either str or None
            Contains names of batch component(s) to be normalized
        dst : list of either str or None
            Contains names of batch component(s) where normalized components will be stored.
        dst_range : list of either str or None
            Contains names of batch component(s) where range of normalized componentes will
            be stored.
            Range is saved as a list of 2 elements:
            `[min(component_data), max(component_data) - min(component_data)].`
            if None then no range will be stored.
        src_range : list of either str or None
            Contains names of batch component(s) where range of normalized componentes have
            already been stored. If not None then component will be normalized with
            the given range.
        Returns
        -------
        self

        """
        for i, component in enumerate(src):
            pos = self.get_pos(None, component, ix)
            comp_data = getattr(self, component)[pos]
            if src_range[i]:
                min_value, scale_value = getattr(self, src_range[i])[pos]
            else:
                min_value, scale_value = np.min(comp_data), (np.max(comp_data) - np.min(comp_data))
            new_data = (comp_data - min_value) / scale_value
            getattr(self, dst[i])[pos] = new_data
            if dst_range[i]:
                getattr(self, dst_range[i])[pos] = [min_value, scale_value]
        return self

    @action
    @safe_src_dst_preprocess
    @inbatch_parallel(init='indices')
    def make_grid_data(self, ix, src=None, dst=None, grid_size=500, **kwargs):
        """ Makes grid
        Parameters
        ----------
        src : str or list of str
            Contains names of batch component(s) to make grid data from in format (x, y)
        dst : str or list of str
            Contains names of batch component(s) where (x_grid, y_grid) will be stored.
        grid_size : int
            Number of points in a grid

        Returns
        -------
        self
        """
        _ = kwargs
        if not (len(src) == 2 and len(dst) == 2):
            raise ValueError('Exactly two components must be passed to make grid data')

        pos = self.get_pos(None, src[0], ix)

        x_data = getattr(self, src[0])[pos]
        y_data = getattr(self, src[1])[pos]

        sorted_data = list(zip(*sorted(zip(x_data, y_data), key=lambda x: x[0])))

        x_grid = np.linspace(0, 1, num=grid_size)
        y_grid = np.interp(x_grid, sorted_data[0], sorted_data[1])

        getattr(self, dst[0])[pos] = x_grid
        getattr(self, dst[1])[pos] = y_grid.reshape((-1, 1))
        return self

    @action
    @safe_src_dst_preprocess
    @inbatch_parallel(init='indices')
    def denormalize(self, ix, src=None, dst=None, src_range=None, **kwargs):
        """ Denormalizes component to initial range.

        Parameters
        ----------
        src : list of str
            Contains names of batch component(s) to be denormalized
        dst : list of either str or None
            Contains names of batch component(s) where denormalized components will be stored.
        src_range : list of str
            Contains names of batch component(s) where range data has been stored in format
            `[min(component_data), max(component_data) - min(component_data)].`

        Returns
        -------
        self
        """
        _ = kwargs
        for i, component in enumerate(src):
            pos = self.get_pos(None, component, ix)
            comp_data = getattr(self, component)[pos]
            if src_range[i]:
                min_value, scale_value = getattr(self, src_range[i])[pos]
                new_data = (comp_data - min_value) / scale_value
                new_data = comp_data * scale_value + min_value
                getattr(self, dst[i])[pos] = new_data
            else:
                raise ValueError('Src_range must be provided to denormalize component')
        return self

    @action
    @safe_src_dst_preprocess
    def expand_dims(self, src=None, dst=None, **kwargs):
        """ Expands the shape of an array stored in a given component.
        Parameters
        ----------
        src : list of str
            Contains names of batch component(s) to be modiified
        dst : list of either str or None
            Contains names of batch component(s) where modiified components will be stored
        Returns
        -------
        self
        """
        _ = kwargs
        for i, component in enumerate(src):
            setattr(self, dst[i], getattr(self, component).reshape((-1, 1)))
        return self

    @action
    def hard_negative_sampling(self, statistics_name=None, fraction=0.5):
        """ Recreate batch with new indices corresponding to the largest loss
        value on the previous iterations.
        Parameters
        ----------
        statistics_name : str
            name of the pipeline variable where the dict with the index and
            corresponding loss values is stored.
        fraction : float in [0, 1]
            A fraction of the hard negative examples in the new batch.
        Returns
        -------
        self
        """
        btch_size = len(self.indices)
        if statistics_name and type(self.pipeline.get_variable(statistics_name)) == dict:
            loss_history_dict = self.pipeline.get_variable(statistics_name)

            sorted_by_value = sorted(loss_history_dict.items(), key=lambda kv: kv[1])
            hard_count = int(btch_size * fraction)
            hard_indices = set([x[0] for x in sorted_by_value[:hard_count]]) - set(self.indices)
            new_index = list(self.indices[: btch_size - len(hard_indices)]) + list(hard_indices)

            random.shuffle(new_index)
            batch = RadialBatch(index=self.pipeline.dataset.index.create_subset(new_index))
            return batch
        return self

    @action
    def update_loss_history_dict(self, src='loss_history', dst='loss_history_dict'):
        """ Update pipeline variable dst that is dict with key - element's index and
        corresponding value, i.e. squarred error on that element from src
        Parameters
        ----------
        src : str
            name of the pipeline variable where element-wise statistics
            (e.g. unaggreagated loss) for the batch is stored.
            It can be a list or np.array of length batch_size.
        dst : str
            name of the pipeline variable where the dict with the index and
            corresponding values is stored.
        Returns
        -------
        self
        """
        new_loss_history_dict = dict(zip(self.index.indices, self.pipeline.get_variable(src)))
        if isinstance(self.pipeline.get_variable(dst), dict):
            new_loss_history_dict = {**new_loss_history_dict, **self.pipeline.get_variable(dst)}
        self.pipeline.update_variable(dst, new_loss_history_dict,
                                      mode='w')
        return self

    @action
    @safe_src_dst_preprocess
    def make_array(self, src=None, dst=None, **kwargs):
        """ TODO: Should be rewritten as post function
        """
        _ = kwargs
        for i, component in enumerate(src):
            try:
                setattr(self, dst[i], np.stack(getattr(self, component)))
            except ValueError:
                print('ACHTUNG! ', getattr(self, component))
        return self

    @action
    @safe_src_dst_preprocess
    @inbatch_parallel(init='indices')
    def clip_values(self, ix, src, dst, **kwargs):
        """Clip values from `src` to 0, 1 and save it to `dst`
        """
        _ = kwargs
        if isinstance(src, str):
            src = [src]
            dst = [dst]

        for i, component in enumerate(src):
            pos = self.get_pos(None, component, ix)
            pred = getattr(self, component)[pos]
            getattr(self, dst[i])[pos] = np.clip(pred, 0, 1)
        return self

class RadialImagesBatch(ImagesBatch, RadialBatch):
    """
    Batch class that stores multiple time series along with other parameters
    for radial flow regime regression.

    Parameters
    ----------
    index : DatasetIndex
        Unique identifiers of rig data in the batch.
    preloaded : tuple, optional
        Data to put in the batch if given. Defaults to ``None``.

    Attributes
    ----------
    index : DatasetIndex
        Unique identifiers of rig data in the batch.
    """
    def __init__(self, index, preloaded=None):
        super().__init__(index, preloaded)
        self.time = self.array_of_nones
        self.derivative = self.array_of_nones
        self.rig_type = self.array_of_nones
        self.target = self.array_of_nones
        self.predictions = self.array_of_nones

    @property
    def components(self):
        """tuple of str: Data components names."""
        return "time", "derivative", "rig_type", "target", "predictions", "images"

    @action
    def load(self, fmt=None, components=None, *args, **kwargs):
        """
        Load given components from file.

        Parameters
        ----------
        fmt : str, optional
            Source format.
        components : iterable, optional

        Returns
        -------
        batch : RadialImagesBatchs
            Batch with loaded components. Changes batch data inplace.

        Raises
        ------
        ValueError
            If source path is not specified and batch's ``index`` is not a
            ``FilesIndex``.
        """

        components = [components] if isinstance(components, str) else components

        if fmt == 'npz' or fmt == 'csv':
            RadialBatch.load(self, fmt=fmt, components=components, *args, **kwargs)
        elif fmt == 'image':
            ImagesBatch.load(self, fmt=fmt, components=components, *args, **kwargs)
        return self
