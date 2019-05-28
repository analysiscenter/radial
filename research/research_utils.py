"Functions for cross validation."
import os
from copy import deepcopy
from itertools import combinations
from collections import defaultdict

import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

from radial.batchflow import Dataset, FilesIndex # pylint: disable=import-error
from radial.batchflow.research import Research # pylint: disable=import-error
from radial.batchflow.research.research import Executable # pylint: disable=import-error

from radial.core import RadialBatch # pylint: disable=import-error


def create_datasets(path, batch, cross_val=None):
    """Generated dataset with following conditions: if `cross_val` is None or equal to 1
    the one dataset will be created. Elsewhere you will receive list of train and test datasets.
    And the lenght of this list will be equal to `cross_val`.

    Parameters
    ----------
    path : str
        Path to data
    batch : Batch

    cross_val : None or int
        Number of bins in cross valudation split

    Returns
    -------
        : Dataset or list of Datasets

    Raises
    ------
    ValueError
        if ```path``` is empty or doesn't exist.
    """
    index = FilesIndex(path=path)
    if len(index.indices) == 0:
        raise ValueError('Wrong path to data')
    if cross_val is None or cross_val == 1:
        dset = Dataset(index, batch)
        dset.split()
        return dset

    split_ix = np.array_split(index.indices, cross_val)
    iterations = zip(combinations(split_ix, cross_val-1), list(combinations(split_ix, 1))[::-1])
    dsets = []
    for train, test in iterations:
        dset_train = Dataset(index=index.create_subset(np.concatenate(train)), batch_class=RadialBatch)
        dset_test = Dataset(index=index.create_subset(np.concatenate(test)), batch_class=RadialBatch)
        dsets.append([dset_train, dset_test])
    return dsets

def _update_research(research, pipeline, name, dataset):
    """Load given the data into research.

    Parameters
    ----------
    research : Research

    pipeline : Pipeline

    name : str
        Pipeline name.
    dataset : Dataset

    Returns
    -------
        :  Research
    """
    for i, ppl in enumerate(pipeline):
        executable = Executable()
        old_exec = research.executables[name[i]]
        executable.add_pipeline(ppl<<dataset[i], name=name[i], variables=old_exec.variables,
                                execute=old_exec.execute, dump=old_exec.dump, run=old_exec.to_run,
                                logging=old_exec.logging, **old_exec.kwargs)
        research.executables[name[i]] = executable
    return research

def execute_research_with_cv(train_pipeline, test_pipeline, res, dataset, n_reps,
                             n_iters, cross_val=None, dir_name='research_cv', research_name='research',
                             train_name='train', test_name='test', **kwargs): # pylint: disable=too-many-arguments, too-many-locals
    """Execute research with given parameters.

    Parameters
    ----------
    train_pipeline : Pipeline

    test_pipeline : Pipeline

    res : Research
        Research object with train and test pipelines.
    dataset : Dataset or list of Datasets
        Dataset to train and test pipelines or list of Dataset if cross validation is used.
    n_reps : int
        Number of repetition Research with current parameters.
    n_iters : int
        Number of iterations in one Research run.
    cross_val : None or int
        Number of bins in cross valudation split.
    dir_name : str
        Name of directory with Researches (availble only when cross validation is used).
    research_name : str
        Name of dir with Research results.
    train_name : str
        Name of train pipeline.
    test_name : str
        Name of test pipeline.

    Returns
    -------
        : batchflow Research or list of Research's
    """
    # simple Research run without cross validation
    research = deepcopy(res)
    workers = kwargs.get('workers', 1)
    gpu = kwargs.get('gpu', None)

    if not isinstance(dataset, list):
        research = _update_research(research, [train_pipeline, test_pipeline],\
                                   [train_name, test_name], [dataset.train, dataset.test])
        research.run(n_reps=n_reps, n_iters=n_iters, name=research_name, bar=True,
                     workers=workers, gpu=gpu)
        return research

    # creation directiory to save cross validation Resutls
    research_list = []
    try:
        os.makedirs(dir_name)
    except FileExistsError as _:
        res_list = os.listdir('./')
        names = np.array(res_list)[list(map(lambda a: dir_name in a, res_list))]
        for name in sorted(names, reverse=True):
            if '.' not in name:
                index = 1 if name[-1] == 'v' else int(name[-1])
                dir_name = dir_name + '_%d'%(index+1)
                os.makedirs(dir_name)
                break

    # Cross validation run
    print('number of bins: ', cross_val)
    for i in range(cross_val):
        research = deepcopy(res)
        research = _update_research(research, [train_pipeline, test_pipeline],\
                                   [train_name, test_name], [dataset[i][0], dataset[i][1]])
        research_name_cv = './%s/' % dir_name + research_name + '_cv_%d' % i
        research.run(n_reps=n_reps, n_iters=n_iters, name=research_name_cv, bar=True,
                     workers=workers, gpu=gpu)
        research_list.append(research)
    return research_list

def split_df_by_name(dataframe, parameters, draw_dict=None):
    """Split the dataframe on the parts with different values in `name` column.

    Parameters
    ----------
    dataframe : pd.Dataframe
        Result of Research.
    parameters : list
        Matched parameters from Research.
    draw_dict : dict
        Only current params from dictionary will be drawn.
        Keys : names of columns.
        Values : The values of visualize params.
    Returns
    -------
        : dict
        key : value of name column
        value : resulted dataframe
    """
    all_names = {}
    if draw_dict is not None:
        for key, value in draw_dict.items():
            dataframe = dataframe[dataframe[key].isin(list([value]))]
            if dataframe.empty:
                raise ValueError("Incorrect column name {} or value {}.".format(key, value))
    for names, name_df in dataframe.groupby('name'):
        if len(parameters) != 0:
            new_df = pd.DataFrame()
            for param_names, values in name_df.groupby(parameters):
                values['parameters'] = str(param_names).replace("'", "")
                new_df = new_df.append(values)
        else:
            new_df = name_df.copy()
            new_df['parameters'] = 'None'
        all_names[names] = new_df
    return all_names

def _load_research(research):
    if not isinstance(research, str):
        return research.load_results()
    return Research().load(research).load_results(use_alias=False)

def _get_parameters(research):
    if isinstance(research, str):
        with open(os.path.join(research, 'description', 'alias.json')) as res:
            alias = json.load(res)
        return list(alias.keys())

    param_set = {}
    for params_list in research.grid_config.alias():
        for params in params_list:
            param_set.update(params)
    return list(param_set)

def _prepare_results(research, hue=None, cross_val=False, aggr=False, iter_start=0, draw_dict=None, names=None):# pylint: disable=too-many-arguments
    if cross_val or cross_val in [0, 1]:
        results = []
        if isinstance(research, str):
            research = list(map(lambda res_name: os.path.join(research, res_name), os.listdir(research)))
        parameters = _get_parameters(research[0])
        for i, res in enumerate(research):
            loaded_res = _load_research(res)
            loaded_res[hue] = 0 if aggr else i
            loaded_res = loaded_res[loaded_res['iteration'] >= iter_start]
            results.append(loaded_res)
        results = pd.concat(results)
    else:
        results = _load_research(research)
        parameters = _get_parameters(research)
    all_names = split_df_by_name(results, parameters, draw_dict)
    correct_names = list(all_names.keys())
    if len(set(correct_names) & set(names)) == 0:
        raise ValueError('One or more names in {} are incorrect,'\
                          ' use the names in the following list {}.'.format(names, correct_names))
    grouped = [all_names.get(name) for name in names]
    return grouped

def draw_history(research, names, types_var, cross_val=None, aggr=False,
                 iter_start=0, draw_dict=None): # pylint: disable=too-many-locals,too-many-arguments
    """Draw plot with history of changes of function named `names` with values from
    column 'types_var'.

    Parameters
    ----------
    research : Research

    names : str or list of str
        Names of functions from Research
    types_var : str or list of str
        Names where the function result saved.
    cross_val : None or int
        Number of cross validation bins.
    aggr : bool, optional
        if True, cross validation reuslts will aggregate in one grap.
        Elsewhere for each cross validation will be drawn own line.
    iter_start : int
        All graph will be drawn from `iter_start` iteration.
    draw_dict : dict
        Only current params from dictionary will be drawn.
        Keys : names of columns.
        Values : The values of visualize params.
    """
    hue = 'number_of_cv' if cross_val is not None and cross_val > 1 else None
    grouped = _prepare_results(research, hue, cross_val, aggr, iter_start, draw_dict, names)

    for name, dframe in zip(names, grouped):
        nan_col = dframe.columns[dframe.isna().any()].tolist()
        dtype = types_var[0]
        for dtp in types_var:
            if dtp not in nan_col:
                dtype = dtp
                break
        dframe[dtype] = dframe[dtype].astype(float)
        graph = sns.FacetGrid(dframe, col='parameters', height=5, hue=hue, col_wrap=5)
        graph.fig.suptitle(dtype + '/' + name, fontsize=16, y=1.05)
        graph.map(sns.lineplot, 'iteration', dtype).add_legend()
        plt.show()

def draw_histogram(research, names, type_var, cross_val=False, draw_dict=None):
    """Draw historgram of research results by given `name` of function and `type_var`.

    Parameters
    ----------
    research : Research

    names : str or list of str
        Names of functions from Research.
    type_var : str or list of str
        Names where the function result saved.
    cross_val : None or int
        Number of cross validation bins.
    draw_dict : dict
        Only current params from dictionary will be drawn.
        Keys : names of columns.
        Values : The values of visualize params.
    """
    data = _prepare_results(research, cross_val=cross_val, draw_dict=draw_dict, names=[names])[0]
    plt.figure(figsize=(10, 7))
    for numb, metric_list in data.groupby('parameters')[type_var]:
        sns.distplot(np.mean(list(metric_list), axis=0), label=str(numb))
    plt.xlabel('Error value')
    plt.ylabel('Probability of error')
    plt.title('Distribution of absolute error')
    plt.legend()
    plt.show()

def print_results_cv(research, names, types_var, cross_val=None, draw_dict=None,
                     n_last=100, none=False, sort_by=None, hue='number_of_cv'): # pylint: disable=too-many-locals,too-many-arguments
    """Print table with mean values of 'names' columns from 'n_last' iterations.

    Parameters
    ----------
    research : Research
    names : str or list of str
        Names of functions from Research.
    type_var : str or list of str
        Names where the function result saved.
    cross_val : None or int
        Number of cross validation bins.
    draw_dict : dict
        Only current params from dictionary will be drawn.
        Keys : names of columns.
        Values : The values of visualize params.
    n_last : int
        Number of iterations(from the end) from which mean values will be calculated.
    none : bool
        If True, than none values will be ignored,
        else if any none in column appeared, than mean value of its column will be none.
    sort_by : str
        Name of the column for sort.
    hue : str or None, optional
        Name of different cross validation.

    Returns
    -------
        : list of DataFrame
        Each DataFrame consists results with given name.
    """
    grouped = _prepare_results(research, hue, cross_val, False, 0, draw_dict, names)
    statistics_df = []
    for name, dframe in zip(names, grouped):
        global_stats = []
        nan_col = dframe.columns[dframe.isna().any()].tolist()
        dtype = types_var[0]
        for dtp in types_var:
            if dtp not in nan_col:
                dtype = dtp
                break
        max_iter = dframe['iteration'].max()
        dframe = dframe[dframe['iteration'] >= max_iter - n_last]
        for param, dfm in dframe.groupby('parameters'):
            par_mean = []
            for cv, cv_df in dfm.groupby(hue):
                cv_mean = []
                for i in range(np.max(cv_df['repetition']) + 1):
                    rep_val = cv_df[cv_df['repetition'] == i][dtype]
                    if len(rep_val.shape) == 1 and len(rep_val) < n_last:
                        n_last = len(rep_val)
                    elif len(rep_val) < n_last:
                        rep_val = np.concatenate(rep_val)
                    cv_mean.append(np.nanmean(np.array(rep_val).astype(np.float32))
                                   if none else np.mean(rep_val))
                par_mean.append(np.mean(cv_mean))
            global_stats.append([*par_mean, np.mean(par_mean), np.std(par_mean)])

        columns = []
        for name in names:
            for cv in range(cross_val):
                columns.append(name + "_" + str(cv))
            columns.extend([name + "_mean", name + "_std"])
        printed = pd.DataFrame(data=global_stats, index=dframe["parameters"].unique(),
                               columns=columns)
        statistics_df.append(printed.sort_values(by=sort_by, ascending=False))
    return statistics_df

def print_results_research(df, name, type_var, average_repetitions=False, sort_by=None, ascending=True,
                           n_last=100):
    """ Show results given by research dataframe.
    Parameters
    ----------
    df : DataFrame
        Research's results
    layout : str
        string where each element consists two parts that splited by /. First part is the type
        of calculated value wrote in the "name" column. Second is name of column  with the parameters
        that will be drawn.
    average_repetitions : bool, optional
        If True, then a separate values will be written
        else one mean value will be written.
    sort_by : str or None, optional
        If not None, column's name to sort.
    ascending : bool, None
        Same as in ```pd.sort_value```.
    n_last : int, optional
        The number of iterations at the end of which the averaging takes place.
    Returns
    -------
        : DataFrame
        Research results in DataFrame, where indices is a config parameters and colums is `layout` values
    """
    columns = []
    data = []
    index = []
    ndf = df[df['name'] == name]
    if average_repetitions:
        columns.extend([name + '_mean', name + '_std'])
    else:
        columns.extend([name + '_' + str(i) for i in [*ndf['repetition'].unique(), 'mean', 'std']])
    for config, cdf in ndf.groupby("config"):
        index.append(config)
        cdf = cdf.drop(['config', 'name'], axis=1).dropna(axis=1).astype('float')
        if average_repetitions:
            idf = cdf.groupby('iteration').mean().drop('repetition', axis=1)
            max_iter = idf.index.max()
            idf = idf[idf.index > max_iter - n_last]
            data.append([idf[type_var].mean(), idf[type_var].std()])
        else:
            rep = []
            for _, rdf in cdf.groupby('repetition'):
                rdf = rdf.drop('repetition', axis=1)
                max_iter = rdf['iteration'].max()
                rdf = rdf[rdf['iteration'] > max_iter - n_last]
                rep.append(rdf[type_var].mean())
            data.append([*rep, np.mean(rep), np.std(rep)])

    res_df = pd.DataFrame(data=data, index=index, columns=columns)
    if sort_by:
        res_df.sort_values(by=sort_by, ascending=ascending, inplace=True)
    return res_df
