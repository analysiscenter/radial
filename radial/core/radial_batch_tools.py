""" Tools for RadialBatch """

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae


def calculate_metrics(target, pred, bins=0, returns=False):
    """Calculate following metrics:
    * MAE
    * MAPE
    * Percentage of error less than 30%

    Parameters
    ----------
    target : list or np.array
        array with answers
    pred : list or np.array
        array with predictions
    bins : int
        number of bins in histogram, if 0 the histogram will not be displayed
    returns : bool
        if True, metrics will be returned
    """
    mape = np.abs(target-pred)/target
    mae_val = mae(target, pred)
    perc = np.mean(mape < 0.3)*100
    print('MAE: {:.4}'.format(mae_val))
    print('MAPE: {:.4}'.format(np.mean(mape)))
    print('Percentage of error less than 30%: {:.4}%'.format(perc))
    if bins:
        plt.figure(figsize=(8, 6))
        sns.distplot(mape, bins=bins)
        plt.title('MAPE hist')
        plt.show()
    if returns:
        return np.mean(mape), mae_val, perc
    return None

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
