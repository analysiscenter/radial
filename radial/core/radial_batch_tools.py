""" Tools for RadialBatch """
import os
from collections import defaultdict

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae

def get_mape(iteration, experiment, pipeline):
    """ Calculate mean absolute percentage error."""
    _ = iteration
    pipeline = experiment[pipeline].pipeline
    y_pred = np.array(pipeline.get_variable('predictions')).reshape(-1)
    y_true = np.array(pipeline.get_variable('targets')).reshape(-1)
    return np.mean(np.abs(y_true-y_pred)/y_true)

def get_mape30(iteration, experiment, pipeline):
    """ Calculate percentage of mean absolute percentage error which less than 30%."""
    _ = iteration
    pipeline = experiment[pipeline].pipeline
    y_pred = np.array(pipeline.get_variable('predictions')).reshape(-1)
    y_true = np.array(pipeline.get_variable('targets')).reshape(-1)
    ape = np.abs(y_true-y_pred)/y_true
    return np.mean(ape < 0.3)*100

def save_model(iteration, experiment, pipeline, model_name, path='./'):
    """ Save model to a path."""
    path = os.path.join(path, experiment[pipeline].config.alias(as_string=True) + '_' + str(iteration))
    pipeline = experiment[pipeline].pipeline
    pipeline.save_model(model_name, path)

def get_ape(iteration, experiment, pipeline):
    """ Calculate absolute percentage error."""
    _ = iteration
    pipeline = experiment[pipeline].pipeline
    y_pred = np.array(pipeline.get_variable('predictions')).reshape(-1)
    y_true = np.array(pipeline.get_variable('targets')).reshape(-1)
    return np.abs(y_true-y_pred)/y_true

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

def draw_predictions(results, names, path=None):
    """Draw a predictions on real data.

    Parameters
    ----------
    results : defaultdict
        Results of testing process.
    names : list
        File names.
    path : str
        Path to data.
    """
    _, ax = plt.subplots(5, 4, figsize=(20, 16))
    ax = ax.reshape(-1)
    for i, name in enumerate(names):
        val = dict(np.load(os.path.join(path, name)))
        ape = np.abs(results[name]['pred']-np.log10(val['target']))/np.log10(val['target'])

        ax[i].scatter(np.log10(val['time']), np.log10(val['derivative']))
        ax[i].axhline(np.log10(val['target']), ls='--', c='g', lw=1, alpha=0.6, label='target')
        ax[i].axhline(np.mean(results[name]['pred']), ls='--', c='b', lw=1, alpha=0.6, label='pred')
        ax[i].set_title('ape: {:.3} name: {}'.format(ape * 100, name))
        ax[i].set_xlabel('log time')
        ax[i].set_ylabel('log derivative')
        ax[i].legend()
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
    plt.show()

def calculate_results(pipeline):
    """Prepare predictions given from `pipeline` that ran many epochs.

    Parameters
    ----------
    pipeline : Pipeline

    Returns
    -------
    results : defaultdict
        dict with predictions and targets for each target
    names : numpy array
        numpy array with names of files that sorted by error value
    """
    results = defaultdict(lambda: defaultdict(list))
    for i in range(len(pipeline.get_variable('ind'))):
        results[pipeline.get_variable('ind')[i]]['pred'].append(np.ravel(pipeline.get_variable('predictions')[i])[0])
        results[pipeline.get_variable('ind')[i]]['true'] = pipeline.get_variable('targets')[i]
    true = []
    pred = []
    for key in results.keys():
        results[key]['pred'] = np.mean(results[key]['pred'])
        true.append(results[key]['true'][0])
        pred.append(results[key]['pred'])

    names = []
    diff = []
    for key, item in results.items():
        diff.append(np.abs(np.mean(item['pred']) - item['true'][0])/item['true'][0])
        names.append(key)

    sorted_ix = np.argsort(diff)
    names = np.array(names)[sorted_ix]
    return results, names
