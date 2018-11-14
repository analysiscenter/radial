""" Batch class for radial regime regression. """

import numpy as np

from .. import batchflow as bf


def concatenate_points(batch, model, return_targets=True):
    """Make data function that leads the data to the type required
    for network training.

    Parameters
    ----------
    batch : Dataset batch
        Part of input data.
    model : Dataset model
        The model to train.
    retrun_targets : bool
        If ``True``, targets shape will be changed from ``(1,)`` to ``(-1, 1)``
        targerts won't returned in another case.

    Returns
    -------
    res_dict : dict
        feed dict with inputs and targets (if needed).
    """
    _ = model
    zip_data = zip(batch.time, batch.derivative)
    points = np.array(list(map(lambda d: np.array([d[0], d[1]])\
                .reshape(-1, batch.derivative[0].shape[1]), zip_data)))
    res_dict = {'feed_dict': {'points': points}}
    if return_targets:
        y = batch.target.reshape(-1, 1)
        res_dict['feed_dict']['targets'] = y
    return res_dict

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
        loaders = {'npz': self._load_npz}

        if isinstance(self.index, bf.FilesIndex):
            path = self.index.get_fullpath(indice)  # pylint: disable=no-member
        else:
            raise ValueError("Source path is not specified")

        return loaders[fmt](path, components, *args, **kwargs)

    @staticmethod
    def _load_npz(path=None, components=None, *args, **kwargs):
        """Load given components from npz file.

        Parameters
        ----------
        path : str
            Path to .hea file.
        components : iterable
            Components to load.

        Returns
        -------
        npz_data : list
            List of data components.
        """
        _ = args, kwargs

        data = dict(np.load(path))

        time = data.pop('time')
        derivative = data.pop('derivative')

        order = np.argsort(time)

        data['time'] = time[order]
        data['derivative'] = derivative[order]

        return [data[comp] for comp in components]

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
    def filter_outliers(self, index):
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
