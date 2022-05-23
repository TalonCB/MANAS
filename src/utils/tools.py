import pandas as pd
import torch
import numpy as np
from utils.constant import *


LOWER_METRIC_LIST = ["rmse", 'mae']


def numpy_to_torch(d):
    """
    numpy array to pytorch tensor, send to gpu if available
    :param d:
    :return:
    """
    t = torch.from_numpy(d)
    if torch.cuda.device_count() > 0:
        t = t.cuda()
    return t


def batch_to_gpu(batch):
    if torch.cuda.device_count() > 0:
        for c in batch:
            if type(batch[c]) is torch.Tensor:
                batch[c] = batch[c].cuda()
    return batch


def torch_to_gpu(t):
    if torch.cuda.device_count() > 0:
        t = t.cuda()
    return t


def best_result(metric, results_list):
    if type(metric) is list or type(metric) is tuple:
        metric = metric[0]
    if metric in LOWER_METRIC_LIST:
        return min(results_list)
    return max(results_list)


def format_metric(metric):
    """
    convert output into string
    :param metric:
    :return:
    """
    # print(metric, type(metric))
    if type(metric) is not tuple and type(metric) is not list:
        metric = [metric]
    format_str = []
    if type(metric) is tuple or type(metric) is list:
        for m in metric:
            # print(type(m))
            if type(m) is float or type(m) is np.float or type(m) is np.float32 or type(m) is np.float64:
                format_str.append('%.4f' % m)
            elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
                format_str.append('%d' % m)
    return ','.join(format_str)

