# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# The complete PCA CV test for a share MLP regressor
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general package import
# ==========================================================================================================
import sys
import re
import os
import collections
import random
import numpy as np
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlp_path = os.path.join(parent_folder, 'classifiers', 'mlp')
path2 = os.path.join(parent_folder, 'general_functions')
sys.path.append(mlp_path)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from mlp_trade_regressor import MlpTradeRegressor
from trade_general_funcs import calculate_rmse
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ==========================================================================================================
# Build MLP classifier
# ==========================================================================================================
def stock_prediction_baseline_reg(train_data_folder_name, test_data_folder_name, is_random = False,
                                  is_highest_profit = False, random_seed = None, data_set = 'a_share'):

    print ("Baseline for regression!")

    # (1.) build classifer
    mlp_classifier1 = MlpTradeRegressor()
    #hidden_layer_sizes = (26,6)
    #learning_rate_init = 0.0001
    #print ("hidden_layer_sizes: ", hidden_layer_sizes)
    ##mlp_classifier1.set_regressor(hidden_layer_sizes, learning_rate_init = learning_rate_init)
    #clsfy_name = 'dow_jones_mlp_trade_classifier_window_shift'
    #clf_path = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)

    # (2.) data folder
    train_data_folder = os.path.join(data_set,train_data_folder_name)
    train_data_folder = os.path.join(parent_folder, 'data', train_data_folder)

    test_data_folder = os.path.join(data_set,test_data_folder_name)
    test_data_folder = os.path.join(parent_folder, 'data', test_data_folder)
    #

    # read train data set
    train_data_set = set()
    train_stock_set = set()
    train_baseline_dict = collections.defaultdict(lambda :collections.defaultdict(lambda :collections.defaultdict(lambda :0)))

    file_name_list = os.listdir(train_data_folder)
    for file_name  in file_name_list:
        data_str = re.findall(r'([0-9\-]+)_', file_name)[0]
        stock_name = re.findall(r'[0-9\-]+_([A-Za-z0-9]+)_', file_name)[0]
        price_change = float(re.findall(r'#([0-9\.\-]+)#', file_name)[0])
        train_baseline_dict[data_str][stock_name]['actual'] = price_change
        train_data_set.add(data_str)
        train_stock_set.add(stock_name)
    train_data_list = sorted(list(train_data_set))
    print ("train_data_list: ", train_data_list)
    print ("train_stock_set: ", train_stock_set)

    #



    # read test data set
    test_date_set = set()
    test_stock_set = set()
    NUMBER_OF_PREVIOUS_WEEK = 2
    baseline_dict = collections.defaultdict(lambda :collections.defaultdict(lambda :collections.defaultdict(lambda :0)))
    file_name_list = os.listdir(test_data_folder)
    file_path_list = [os.path.join(test_data_folder, x) for x in file_name_list]
    for file_name  in file_name_list:
        data_str = re.findall(r'([0-9\-]+)_', file_name)[0]
        stock_name = re.findall(r'[0-9\-]+_([A-Za-z0-9]+)_', file_name)[0]
        price_change = float(re.findall(r'#([0-9\.\-]+)#', file_name)[0])
        baseline_dict[data_str][stock_name]['actual'] = price_change
        test_date_set.add(data_str)
        test_stock_set.add(stock_name)
    # baseline algorithm
    train_last_date_list = train_data_list[-NUMBER_OF_PREVIOUS_WEEK:]
    complete_test_date_list = train_last_date_list + sorted(list(test_date_set))
    for last_date in train_last_date_list:
        for test_stock in list(test_stock_set):
            baseline_dict[last_date][test_stock]['actual'] = train_baseline_dict[last_date][test_stock]['actual']
    #
    #test_date_set.update(set(train_last_date_list))

    #

    for test_date in sorted(list(test_date_set)):
        this_week_index = complete_test_date_list.index(test_date)
        p1_week_index = this_week_index - 1
        p2_week_index = this_week_index - 2
        p1_week_date = complete_test_date_list[p1_week_index]
        p2_week_date = complete_test_date_list[p2_week_index]
        for stock in list(test_stock_set):
            p1_pc = baseline_dict[p1_week_date][stock]['actual']
            p2_pc = baseline_dict[p2_week_date][stock]['actual']
            #predicted_price = (p1_pc + p2_pc) / 2 # baseline algorithm1
            predicted_price = p1_pc # baseline algorithm2

            baseline_dict[test_date][stock]['predict'] = predicted_price


    print ("complete_test_date_list: ", complete_test_date_list)
    print ("test_data_set: ", test_date_set)
    print ("test_stock_set: ", test_stock_set)
    #print ("baseline_dict.dates: ", baseline_dict.items())
    #







    # get the predicted value
    if is_random:
        predict_pc_list = []
        for i, test_date in enumerate(sorted(list(test_date_set))):
            if random_seed:
                random.seed(i + random_seed)
            best_stock = random.sample(test_stock_set, 1)[0]
            actual_pc = baseline_dict[test_date][best_stock]['actual']
            predict_pc_list.append(actual_pc)
        print ("predict_pc_list: ", predict_pc_list)

    elif is_highest_profit:
        predict_pc_list = []
        for i, test_date in enumerate(sorted(list(test_date_set))):
            highest_return = float('-inf')
            for stock in test_stock_set:
                stock_return = baseline_dict[test_date][stock]['actual']
                if stock_return > highest_return:
                    highest_return = stock_return
            predict_pc_list.append(highest_return)
        print ("predict_pc_list: ", predict_pc_list)

    else:
        predict_stock_list = []
        predict_pc_list = []
        rmse_list = []
        for test_date in sorted(list(test_date_set)):
            complete_predict_value_list = []
            complete_actual_value_list = []
            highest_pc = float('-inf')
            for stock in list(test_stock_set):
                predict_value = baseline_dict[test_date][stock]['predict']
                actual_value = baseline_dict[test_date][stock]['actual']
                complete_predict_value_list.append(predict_value)
                complete_actual_value_list.append(actual_value)
                if predict_value >  highest_pc:
                    highest_pc = predict_value
                    best_stock = stock
            rmse = calculate_rmse(complete_actual_value_list, complete_predict_value_list)
            rmse_list.append(rmse)
            predict_stock_list.append(best_stock)
            actual_pc = baseline_dict[test_date][best_stock]['actual']
            predict_pc_list.append(actual_pc)
            print ("predict_pc_list: ", predict_pc_list)
            print ("avg_pc: ", np.average(predict_pc_list))
            print ("avg_rmse: ", np.average(rmse_list))

    return predict_pc_list

#stock_prediction_baseline_reg('dow_jones_index_extended_regression', 'dow_jones_index_extended_regression_test2')


