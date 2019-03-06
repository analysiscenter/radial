""" Batch class for radial regime regression. """

import random
import numpy as np
import scipy as sc

from sklearn.ensemble import IsolationForest

from . import radial_batch_tools as bt
from .decorators import init_components
from ..batchflow import Batch, action, inbatch_parallel, any_action_failed, FilesIndex, DatasetIndex, R

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

    components = "time", "derivative", "rig_type", "target"

    @property
    def array_of_nones(self):
        """ 1-D ndarray: ``NumPy`` array with ``None`` values."""
        return np.array([None] * len(self.index))

    def _reraise_exceptions(self, results):
        """ Reraise all exceptions in the ``results`` list.

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
    def load(self, src=None, fmt=None, components=None, *args, **kwargs):
        """ Load given batch components.

        This method supports loading of data from npz format.

        Parameters
        ----------
        src: str, optional
             an array with data
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
        if isinstance(components, str):
            components = list(components)
        if components is None:
            components = self.components

        if fmt is None:
            self.put_into_data(src, components)
            self._sort(components)
        else:
            self._load(fmt, components)

        return self

    @inbatch_parallel(init='indices')
    def _sort(self, index, components):
        i = self.get_pos(None, 'time', index)
        time_mask = np.argsort(getattr(self, 'time')[i])
        for component in components:
            data = getattr(self, component)[i]
            getattr(self, component)[i] = data[time_mask]

    @inbatch_parallel(init="indices", post="_assemble_load", target="threads")
    def _load(self, indice, fmt=None, components=None, *args, **kwargs):
        """ Load given components from file.

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
        """ Concatenate results of different workers and update ``self``.

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
    @init_components
    @inbatch_parallel(init="indices", target="threads")
    def drop_negative(self, index, src=None, dst=None, **kwargs):
        """ Leaves only positive values.

        Raises
        ------
        ValueError
            If all values in derivative component are negative.
        """
        _ = kwargs
        i = self.get_pos(None, src[0], index)
        time = getattr(self, src[0])[i]
        derivative = getattr(self, src[1])[i]
        mask = derivative > 0

        if sum(mask) == 0:
            raise ValueError("All values in derivative {} are negative!".format(index))

        getattr(self, dst[0])[i] = time[mask]
        getattr(self, dst[1])[i] = derivative[mask]
        return self

    @action
    @init_components
    @inbatch_parallel(init="indices", target="threads")
    def drop_outliers(self, index, contam=0.1, src=None, dst=None, **kwargs):
        """Drop outliers using Isolation Forest algorithm.
        Parameters
        ----------
        contam : float (from 0 to 0.5)
            The amount of contamination of the data set
        """
        _ = kwargs
        i = self.get_pos(None, src[0], index)

        derivative = getattr(self, src[1])[i]
        time = getattr(self, src[0])[i]

        x_data = np.array([derivative]).T
        isol = IsolationForest(contamination=contam).fit(x_data)
        pred = isol.predict(x_data)
        getattr(self, dst[0])[i] = time[pred == 1]
        getattr(self, dst[1])[i] = derivative[pred == 1]
        return self

    @action
    @init_components
    @inbatch_parallel(init="indices", target="threads")
    def get_samples(self, index, n_points, n_samples=1, sampler=None, # pylint: disable=too-many-arguments
                    src=None, dst=None, interpolate='linear', seed=None, **kwargs):
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
        _ = kwargs
        if seed:
            np.random.seed(seed)

        i = self.get_pos(None, src[0], index)
        time = getattr(self, src[0])[i]
        derivative = getattr(self, src[1])[i]

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

        getattr(self, dst[0])[i] = np.array(sample_times)
        getattr(self, dst[1])[i] = np.array(sample_derivatives)
        return self

    @action
    def unstack_samples(self):
        """ Create a new batch in which each element of `time` and `derivative`
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
    @init_components
    def make_points(self, src=None, dst=None, **kwargs):
        """ Generated new component with name `dst` with value from `src`."""
        _ = kwargs
        points = []
        for comp in src:
            points.append(np.vstack(getattr(self, comp)))
        if not isinstance(dst, str):
            dst = dst[0]
        setattr(self, dst, np.array(list(zip(*points))).transpose(0, 2, 1))
        return self

    @action
    def make_target(self, src='target', **kwargs):
        """ Reshaped component co vector with shape = (-1, 1)."""
        _ = kwargs
        setattr(self, src, getattr(self, src).reshape(-1, 1))
        return self

    @action
    @init_components
    @inbatch_parallel(init='indices')
    def normalize(self, ix, src=None, dst=None, src_range=None, dst_range=None, **kwargs):
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
        _ = kwargs
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
    @init_components
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
    @init_components
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
    @init_components
    @inbatch_parallel(init='indices')
    def to_log10(self, ix, src=None, dst=None, **kwargs):
        """Takes a decimal logarithm from `src` and saves the resulting value to `dst`

        Parameters
        ----------
        src : src or list
            Name of the component with data
        dst : src or list
            Name of the component to save the result
        """
        _ = kwargs
        if isinstance(src, str):
            src = [src]
            dst = [dst]
        for i, component in enumerate(src):
            pos = self.get_pos(None, component, ix)
            comp_data = getattr(self, component)[pos]
            getattr(self, dst[i])[pos] = np.log10(comp_data)
        return self

    @action
    @init_components
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

    @action
    @init_components
    def make_array(self, src=None, dst=None, **kwargs):
        """ TODO: Should be rewritten as post function
        """
        _ = kwargs
        for i, component in enumerate(src):
            setattr(self, dst[i], np.stack(getattr(self, component)))
        return self

    @action
    @init_components
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
        if statistics_name and isinstance(self.pipeline.get_variable(statistics_name), dict):
            loss_history_dict = self.pipeline.get_variable(statistics_name)

            sorted_by_value = sorted(loss_history_dict.items(), key=lambda kv: kv[1])
            hard_count = int(btch_size * fraction)
            hard_indices = {x[0] for x in sorted_by_value[:hard_count]} - set(self.indices)
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
