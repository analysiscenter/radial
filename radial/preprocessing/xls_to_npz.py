"""Preprocess xls data to npz. """
import os
import sys
import argparse

import numpy as np
import xlrd # pylint: disable=import-error

def parse_args():
    """Parse arguments from command line and run xls to npz function with given parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load', type=str, nargs='+', help="Path to data with xlsx format. \
                                                                   If there are several path, \
                                                                   write it throught the space",
                        required=True)
    parser.add_argument('-s', '--save', type=str, nargs='+', help="Path to save npz data. Write paths \
                                                                   by space if needed",
                        required=True)

    args = parser.parse_args()
    xls_to_npz(args.load, args.save)

def xls_to_npz(path_from, path_to):
    """Creation of npz files from xls tables.

    Parameters
    ----------
    path_from : list with len 2 or 1
        if len == 1,
            is the path to xls file
        if len == 2,
            first element is the path to TEST xls file
            second element is the path to TRAIN xls file
    path_to : list with len 2 or 1
        if len == 1,
            npz files from each files will save in 'path_to' directory
        if len == 2,
            TEST files will be saved in 'path_to[0]', TRAIN - 'path_to[1]'
    """
    if len(path_from) == 1:
        if len(path_to) != 1:
            raise TypeError('if the data comes from one file ``Path_to`` should contains one path')

        path_from = path_from[0]
        path_to = path_to[0]

        xls = xlrd.open_workbook(path_from)
        if not os.path.isdir(path_to):
            os.makedirs(path_to)

        for shnum in range(xls.nsheets):
            sheet = xls.sheet_by_index(shnum)
            np.savez(os.path.join(path_to, 'rr_{}'.format(shnum)),
                     time=np.array([val for val in sheet.col_values(0)[2:] if val != '']),
                     derivative=np.array([val for val in sheet.col_values(2)[2:] if val != '']),
                     target=sheet.col_values(6)[2])
    else:
        save_path = []
        path_to = list(path_to) * len(path_from) if len(path_to) == 1 else path_to
        for path in path_to:
            if not os.path.isdir(path):
                os.makedirs(path)
            save_path.append(path)
        i = 0
        for num_p, path_f in enumerate(path_from):
            xls = xlrd.open_workbook(path_f)
            for shnum in range(xls.nsheets):
                sheet = xls.sheet_by_index(shnum)
                np.savez(os.path.join(save_path[num_p], 'rr_{}'.format(i)),
                         time=np.array([val for val in sheet.col_values(0)[2:] if val != '']),
                         derivative=np.array([val for val in sheet.col_values(2)[2:] if val != '']),
                         target=sheet.col_values(6)[2])
                i += 1
    print('Done!')

if __name__ == "__main__":
    sys.exit(parse_args())
