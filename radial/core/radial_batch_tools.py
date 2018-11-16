""" Tools for RadialBatch """

import numpy as np

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

def load_npz(path=None, components=None, *args, **kwargs):
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