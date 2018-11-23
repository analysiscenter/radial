""" Batch class for radial regime regression. """

import numpy as np
import scipy as sc

from .. import batchflow as bf
from . import radial_batch_tools as bt


class RadialBatch(bf.Batch):
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

    @property
    def components(self):
        """tuple of str: Data components names."""
        return "time", "derivative", "rig_type", "target"

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
        if bf.any_action_failed(results):
            all_errors = self.get_errors(results)
            raise RuntimeError("Cannot assemble the batch", all_errors)

    @bf.action
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
        if isinstance(components, str):
            components = list(components)
        if components is None:
            components = self.components

        return self._load(fmt, components)

    @bf.inbatch_parallel(init="indices", post="_assemble_load", target="threads")
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

        if isinstance(self.index, bf.FilesIndex):
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

    @bf.action
    @bf.inbatch_parallel(init="indices", target="threads")
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

    @bf.action
    @bf.inbatch_parallel(init="indices", target="threads")
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

        if  isinstance(sampler, bf.named_expr.R):
            sampler = bf.R(sampler.name, **sampler.kwargs, size=(n_samples, n_points))
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

    @bf.action
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

        index = bf.DatasetIndex(len(time))
        batch = type(self)(index)
        batch.time = time
        batch.derivative = derivative

        for component_name in set(self.components).intersection({"rig_type", "target"}):
            component = getattr(self, component_name)
            val = [elem for elem, n in zip(component, n_reps) for _ in range(n)]
            val = np.array(val + [None])[:-1]
            setattr(batch, component_name, val)

        return batch

    @bf.action
    @bf.inbatch_parallel(init="indices", post='_assemble_load', target="threads")
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