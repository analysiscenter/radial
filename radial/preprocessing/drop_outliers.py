"""File with function that deleted outliers. """
import os
import sys

import argparse
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import IsolationForest


def parse_args():
    """Parse arguments from command line and run xls to npz function with given parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load', type=str, nargs='+', help="Path to data with xlsx format. \
                                                                   If there are several path, \
                                                                   write it throught the space.",
                        required=True)
    parser.add_argument('-s', '--save', type=str, nargs='+', help="Path to save npz data. Write paths \
                                                                   by space if needed",
                        required=True)
    parser.add_argument('-r', '--rewrite', type=bool, help='If True - existed directory might be rewritten,\
                                                            else exception will raise.',
                        default=False)
    parser.add_argument('-c', '--contam', type=float, help='Parameter from Isolation Forest algorithm. \
                                                           The amount of contamination of the data set. \
                                                           (Might be from 0 to 0.5)',
                        default=0.1)

    args = parser.parse_args()

    drop_outliers(args.load, args.save, args.rewrite, args.contam)

def load(path):
    """Load npz file with from 'path'.

    Parameters
    ----------
    path : str
        path to file

    Return
    ------
     : tuple
     (time, derivative, target)
    """
    data = dict(np.load(path))
    return data['time'], data['derivative'], data['target']

def drop_outliers(path_from, path_to, rewrite=False, contam=0.1):
    """Delete outliers by Isolation Forest.

    Parameters
    ----------
    path_from : str
        path to data
    path_to : str
        path to files after Isolation Forest
    rewrite : bool
        if true, 'path_to' will be rewritten if exists
        else if 'path_to' exists the exeption will be raised
    contam : float (from 0 to 0.5)
        The amount of contamination of the data set

    Raises
    ------
    ValueError
        if path_from > 1 and path_to != 1.
    """
    if len(path_from) != len(path_to):
        if len(path_to) != 1:
            raise ValueError("`path_to` should be str or contains the number of paths that equal to `path_from`.")

    for path_t in path_to:
        if os.path.isdir(path_t):
            if not rewrite:
                raise FileExistsError("Directory with name `{}` already exists".format(path_to))
        else:
            os.makedirs(path_t)

    path_to = path_to * len(path_from)

    for path_f, path_t in zip(path_from, path_to):
        data_names = os.listdir(path_f)
        for name in tqdm(data_names):
            file_name = os.path.join(path_f, name)
            time, derivative, target = load(file_name)
            x_data = np.array([derivative]).T
            isol = IsolationForest(contamination=contam).fit(x_data)
            pred = isol.predict(x_data)
            np.savez(os.path.join(path_t, name),
                     time=time[pred == 1],
                     derivative=derivative[pred == 1],
                     target=target)
    print("Done!")

if __name__ == "__main__":
    sys.exit(parse_args())
