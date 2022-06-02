from os.path import exists
import numpy as np


def check_file_exist(arg):
    if not exists(arg):
        print("File ", arg, " does not exits")
        exit(0)
    return True


def sigmoid_function(array):
    i = 0
    for element in array:
        array[i] = 1/(1 + np.exp(-array[i]))
        i = i + 1
    return array