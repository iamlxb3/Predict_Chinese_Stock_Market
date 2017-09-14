import datetime
import math
import numpy as np
import random
import os
import collections
import matplotlib.pyplot as plt
import re
import itertools
from sklearn.metrics import mean_squared_error
from pjslib.logger import logger1
import sys


# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
path1 = os.path.join(parent_folder, 'strategy')
#path2 = os.path.join(parent_folder, 'general_functions')
sys.path.append(path1)
#sys.path.append(path2)

# ==========================================================================================================


# ==========================================================================================================
# local package import
# ==========================================================================================================
from a_share_strategy import top_n_avg_strategy, top_1_stock_return
#from trade_general_funcs import top_1_stock_return
# ==========================================================================================================




def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def split_list_by_percentage(per_tuple, list1):
    list_len = len(list1)
    split_list = []

    stop_index_list = []
    for i, per in enumerate(per_tuple):

        stop_index = math.ceil(per * list_len)
        if i == 0:
            stop_index_tuple = (0, stop_index)
        elif i == len(per_tuple) - 1:
            stop_index_tuple = (previous_stop_index, len(list1))
        else:
            stop_index_tuple = (previous_stop_index, stop_index)

        stop_index_list.append(stop_index_tuple)
        previous_stop_index = stop_index

    # print (stop_index_list)
    for stop_index_tuple in stop_index_list:
        split_list.append(list1[stop_index_tuple[0]:stop_index_tuple[1]])

    return split_list


def calculate_rmse(actual_value_array, pred_value_array):
    '''root-mean-square error sk learn'''
    # TODO try and except for debug
    try:
        rmse = math.sqrt(mean_squared_error(actual_value_array, pred_value_array))
        return rmse
    except ValueError:
        # print ("actual_value_array: ", actual_value_array)
        # print ("pred_value_array: ", pred_value_array)
        logger1.info("actual_value_array:{}".format(actual_value_array))
        logger1.info("pred_value_array:{}".format(pred_value_array))
        logger1.info("--------------------------------------------")
        return None


def calculate_mrse_PJS(golden_value_array, pred_value_array):
    '''root-mean-square error'''
    if len(golden_value_array) != len(pred_value_array):
        print("golden_value_array len is not equal to pred_value_array len {}".
              format(golden_value_array, pred_value_array))
        return None
    sample_count = len(golden_value_array)
    rmse = golden_value_array - pred_value_array
    rmse = rmse ** 2
    rmse = np.sum(rmse)
    rmse = rmse / sample_count
    rmse = np.sqrt(rmse)
    return rmse


def list_by_index(list1, index_list):
    new_list = [list1[index] for index in index_list]
    return new_list


def create_random_sub_set_list(set1, sub_set_size, random_seed=1):
    sub_set_list = []
    while (len(set1)) >= sub_set_size:
        set1_list = sorted(list(set1))
        random.seed(random_seed)
        sub_set = set(random.sample(set1_list, sub_set_size))
        set1 -= sub_set
        sub_set_list.append(sub_set)
    return sub_set_list


def count_label(folder):
    file_name_list = os.listdir(folder)
    label_dict = collections.defaultdict(lambda: 0)
    for file_name in file_name_list:
        try:
            label = re.findall(r'_([0-9A-Za-z]+)\.', file_name)[0]
        except IndexError:
            print("Check folder path!")
            break
        label_dict[label] += 1
    print("label_dict: {}".format(list(label_dict.items())))


def feature_degradation(features_list, feature_switch_tuple):
    new_feature_list = []
    for i, switch in enumerate(feature_switch_tuple):
        if switch == 1:
            new_feature_list.append(features_list[i])
    return new_feature_list

def compute_average_f1(pred_label_list, gold_label_list):
    # TODO, need unitest, maybe some problem exist if there is only 1 label in gold_label_list
    label_tp_fp_tn_dict = collections.defaultdict(lambda: [0, 0, 0, 0])  # tp,fp,fn,f1,precision,recall
    label_set = set(gold_label_list)

    for i, pred_label in enumerate(pred_label_list):
        gold_label = gold_label_list[i]
        for label in label_set:
            if pred_label == label and gold_label == label:
                label_tp_fp_tn_dict[label][0] += 1  # true positve
            elif pred_label == label and gold_label != label:
                label_tp_fp_tn_dict[label][1] += 1  # false positve
            elif pred_label != label and gold_label == label:
                label_tp_fp_tn_dict[label][2] += 1  # false nagative

    # compute f1
    precision_list = []
    recall_list = []

    count = 0
    for i, label in enumerate(pred_label_list):
        if label == gold_label_list[i]:
            count += 1
    accuracy = count / len(pred_label_list)

    for label, f1_list in label_tp_fp_tn_dict.items():
        tp, fp, fn = f1_list[0:3]
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        recall = tp / (tp + fn)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (2 * precision * recall) / (precision + recall)  # equal weight to precision and recall
        f1_list[3] = f1
        precision_list.append(precision)
        recall_list.append(recall)
        # reference:
        # https://www.quora.com/What-is-meant-by-F-measure-Weighted-F-Measure-and-Average-F-Measure-in-NLP-Evaluation

    # compute the per class average of precision,recall and F1
    avg_precision = np.average(precision_list)
    avg_recall = np.average(recall_list)



    if avg_precision + avg_recall == 0.0:
        avg_F1 = 0.0
    else:
        avg_F1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)  # equal weight to precision and recall



    return label_tp_fp_tn_dict,avg_F1


def generate_feature_switch_list(folder):
    # read feature length
    file_name_list = os.listdir(folder)
    file_path_0 = [os.path.join(folder, x) for x in file_name_list][0]
    with open(file_path_0, 'r', encoding='utf-8') as f:
        feature_name_list = f.readlines()[0].split(',')[::2]
    feature_num = len(feature_name_list)
    feature_switch_list_all = list(itertools.product([0, 1], repeat=feature_num))
    feature_switch_list_all.remove(tuple([0 for x in range(feature_num)]))
    print("Total feature combination: {}".format(len(feature_switch_list_all)))
    return feature_switch_list_all


def get_avg_price_change(pred_value_list, actual_value_list, date_list,
                          stock_id_list, include_top_list=None):
    if not include_top_list:
        include_top_list = [1]
    avg_price_change_list = []
    var_list = []
    std_list = []
    # construct stock_pred_v_dict
    stock_pred_v_dict = collections.defaultdict(lambda: [])
    for i, date in enumerate(date_list):
        stock_pred_v_pair = (stock_id_list[i], pred_value_list[i])
        stock_pred_v_dict[date].append(stock_pred_v_pair)
    #
    stock_actual_v_dict = collections.defaultdict(lambda: 0)
    for i, date in enumerate(date_list):
        date_stock_id_pair = (date, stock_id_list[i])
        stock_actual_v_dict[date_stock_id_pair] = actual_value_list[i]
    #
    for include_top in include_top_list:
        # (0.) avg_price_change
        avg_price_change, var, std = top_n_avg_strategy(stock_actual_v_dict, stock_pred_v_dict,
                                                        include_top=include_top)
        avg_price_change_list.append(avg_price_change)
        var_list.append(var)
        std_list.append(std)

    return tuple(avg_price_change_list), tuple(var_list), tuple(std_list)


def get_full_feature_switch_tuple(folder):
    # read feature length
    file_name_list = os.listdir(folder)
    file_path_0 = [os.path.join(folder, x) for x in file_name_list][0]
    with open(file_path_0, 'r', encoding='utf-8') as f:
        feature_name_list = f.readlines()[0].split(',')[::2]
    full_feature_switch_tuple = tuple([1 for x in feature_name_list])
    return full_feature_switch_tuple


def build_hidden_layer_sizes_list(hidden_layer_config_tuple):
    hidden_layer_node_min, hidden_layer_node_max, hidden_layer_node_step, hidden_layer_depth_min, \
    hidden_layer_depth_max = hidden_layer_config_tuple

    hidden_layer_unit_list = [x for x in range(hidden_layer_node_min, hidden_layer_node_max + 1)]
    hidden_layer_unit_list = hidden_layer_unit_list[::hidden_layer_node_step]
    #

    hidden_layer_layer_list = [x for x in range(hidden_layer_depth_min, hidden_layer_depth_max + 1)]
    #
    hidden_layer_sizes_list = list(itertools.product(hidden_layer_unit_list, hidden_layer_layer_list))
    return hidden_layer_sizes_list

def read_pca_component(folder_path):
    file_name_list = os.listdir(folder_path)
    file_name1 = file_name_list[0]
    file_name1_path = os.path.join(folder_path, file_name1)
    with open (file_name1_path, 'r', encoding = 'utf-8') as f:
        feature_list = f.readlines()[0].strip().split(',')[::2]
        feature_num = len(feature_list)
    print ("Read PCA n-component {}".format(feature_num))
    return feature_num

def compute_f1_accuracy(predict_list, actual_list):
    # (3.) compute the average f-measure
    _,average_f1 = compute_average_f1(predict_list, actual_list)
    # average_f1 = f1_list[0] # using F-measure
    #

    # (4.) compute accuracy
    correct = 0
    for i, pred_label in enumerate(predict_list):
        if pred_label == actual_list[i]:
            correct += 1
    accuracy = correct / len(actual_list)
    #

    # (5.) count the occurrence for each label
    pred_label_dict = collections.defaultdict(lambda: 0)
    for pred_label in predict_list:
        pred_label_dict[pred_label] += 1


    dev_label_dict = collections.defaultdict(lambda: 0)
    for dev_label in actual_list:
        dev_label_dict[dev_label] += 1
    #
    return average_f1, accuracy, pred_label_dict, dev_label_dict


def compute_trade_weekly_clf_result(pred_label_list, actual_label_list, data_list):
    # (3.) get the pred label for each week
    pred_label_dict_by_week = collections.defaultdict(lambda: [])
    golden_label_dict_by_week = collections.defaultdict(lambda: [])

    for i, pred_label in enumerate(pred_label_list):
        date = data_list[i]
        pred_label_dict_by_week[date].append(pred_label)
        golden_label = actual_label_list[i]
        golden_label_dict_by_week[date].append(golden_label)

    week_average_f1_list = []
    week_average_accuracy_list = []
    dev_label_dict = collections.defaultdict(lambda: 0)
    pred_label_dict = collections.defaultdict(lambda: 0)

    # (4.) compute the f1, accuracy for each week in 1 validation set
    for date, pred_label_list_for_1_week in pred_label_dict_by_week.items():
        pred_label_list = pred_label_list_for_1_week
        golden_label_list = golden_label_dict_by_week[date]

        # (3.) compute the average f-measure

        _, average_f1 = compute_average_f1(pred_label_list, golden_label_list)
        week_average_f1_list.append(average_f1)
        # average_f1 = f1_list[0] # using F-measure
        #

        # (4.) compute accuracy
        correct = 0
        for i, pred_label in enumerate(pred_label_list):
            if pred_label == golden_label_list[i]:
                correct += 1
        accuracy = correct / len(golden_label_list)
        week_average_accuracy_list.append(accuracy)
        #

        # (5.) count the occurrence for each label
        for dev_label in golden_label_list:
            dev_label_dict[dev_label] += 1
        for pred_label in pred_label_list:
            pred_label_dict[pred_label] += 1
            #


            # # (6.) save result for 1-fold
            # self.average_f1_list.append(average_f1)
            # self.accuracy_list.append(accuracy)
            # #

    week_average_f1 = np.average(week_average_f1_list)
    week_average_accuracy = np.average(week_average_accuracy_list)
    return week_average_f1, week_average_accuracy, dev_label_dict, pred_label_dict


def get_chosen_stock_return(pred_value_list, actual_value_list, date_list,
                          stock_id_list, include_top_list=None):

    # construct stock_pred_v_dict
    stock_pred_v_dict = collections.defaultdict(lambda: [])
    for i, date in enumerate(date_list):
        stock_pred_v_pair = (stock_id_list[i], pred_value_list[i])
        stock_pred_v_dict[date].append(stock_pred_v_pair)
    #
    stock_actual_v_dict = collections.defaultdict(lambda: 0)
    for i, date in enumerate(date_list):
        date_stock_id_pair = (date, stock_id_list[i])
        stock_actual_v_dict[date_stock_id_pair] = actual_value_list[i]

    include_top = 1
    date_actual_avg_priceChange_list = top_1_stock_return(stock_actual_v_dict, stock_pred_v_dict,
                                                        include_top=include_top)

    return date_actual_avg_priceChange_list

def plot_stock_return(each_week_return_list, date_list, capital = 1, title = '', xlabel = '', save_path = '',
                      is_plot = False,
                      simple_baseline_each_week_return_list = None,
                      random_baseline_each_week_return_list=None,
                      highest_profit_baseline_each_week_return_list=None,
                      highest_profit = None,
                      model_label = ''
                      ):
    capital = 1
    return_list = []
    simple_baseline_return_list = []
    random_baseline_return_list = []
    highest_profit_baseline_return_list = []

    for each_week_return in each_week_return_list:
        capital += capital*each_week_return
        return_list.append(capital)

    # simple baseline
    if simple_baseline_each_week_return_list:
        capital = 1
        for each_week_return in simple_baseline_each_week_return_list:
            capital += capital*each_week_return
            simple_baseline_return_list.append(capital)
    #

    # random baseline
    if random_baseline_each_week_return_list:
        capital = 1
        for each_week_return in random_baseline_each_week_return_list:
            capital += capital*each_week_return
            random_baseline_return_list.append(capital)
    #

    # highest profit baseline
    if highest_profit_baseline_each_week_return_list:
        capital = 1
        for each_week_return in highest_profit_baseline_each_week_return_list:
            capital += capital*each_week_return
            highest_profit_baseline_return_list.append(capital)
    #


    f1, (ax1) = plt.subplots(1, sharex=True, sharey=True)
    # ax1
    gap_length = 5
    my_xticks = date_list
    for i, x_ticks in enumerate(my_xticks):
        if i % gap_length != 0:
            my_xticks[i] = ''

    x = np.array([x for x in range(0, len(date_list))])

    plt.xticks(x, my_xticks)
    ax1.plot(x, return_list, '-', color = '#005d98', label=model_label)
    if simple_baseline_each_week_return_list:
        ax1.plot(x, simple_baseline_return_list, '-',  dashes=[2, 4], color = '#0099cc', label='Simple baseline')
    if random_baseline_each_week_return_list:
        ax1.plot(x, random_baseline_return_list, '-', dashes=[1, 1], color = '#0099cc', label='Random baseline')
    if highest_profit_baseline_each_week_return_list:
        ax1.plot(x, highest_profit_baseline_return_list, 'k-', label='Highest-profit baseline')

    #plt.locator_params(axis='x', nbins=4)
    f1.autofmt_xdate()
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    if highest_profit:
        highest_profit = float("{:.2f}".format(highest_profit))
        ax1.set_ylabel('profit (Theoretical highest profit: {})'.format(highest_profit))
    else:
        ax1.set_ylabel('profit')
    ax1.legend(loc=2)
    #
    if is_plot:
        plt.show()
    if save_path:
        f1.savefig('{}'.format(save_path))
        #plt.savefig('{}'.format(save_path))
