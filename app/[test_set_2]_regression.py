# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Build the MLP regressor.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general package import
# ==========================================================================================================
import sys
import os
import re
import random
import collections
import numpy as np
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
clf_path = os.path.join(parent_folder, 'classifiers', 'mlp')
path2 = os.path.join(parent_folder, 'general_functions')

sys.path.append(clf_path)
sys.path.append(path2)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from mlp_trade_regressor import MlpTradeRegressor
from mlp_trade_ensemble_regressor import MlpTradeEnsembleRegressor
from trade_general_funcs import compute_f1_accuracy, compute_trade_weekly_clf_result, get_chosen_stock_return, \
    plot_stock_return, calculate_rmse
from trade_general_funcs import get_avg_price_change
from _baseline_regression import stock_prediction_baseline_reg
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ------------------------------------------------------------------------------------------------------------
# read the dataset
# ------------------------------------------------------------------------------------------------------------
# train
data_set = 'a_share'
train_folder_name = 'a_share_regression_data'
test1_folder_name = 'a_share_regression_data_test_1'
test2_folder_name = 'a_share_regression_data_test_2'

train_data_folder = os.path.join(data_set, train_folder_name)
train_data_folder = os.path.join(parent_folder, 'data', train_data_folder)
# test1
test1_data_folder = os.path.join(data_set, test1_folder_name)
test1_data_folder = os.path.join(parent_folder, 'data', test1_data_folder)
# test2
test_data_folder = os.path.join(data_set, test2_folder_name)
test_data_folder = os.path.join(parent_folder, 'data', test_data_folder)
# ------------------------------------------------------------------------------------------------------------



# ==========================================================================================================
# Build MLP classifier for a-share data, save the mlp to local
# ==========================================================================================================
HP_FILE_NAME = '4_avg_pc_rank_[3].csv'
print ("Build MLP regressor for dow_jones extended test data!")
mode = 'reg' #'reg'
#model = 'regressor'
#model = 'bagging_regressor'
model = 'adaboost_regressor'

# ------------------------------------------------------------------------------------------------------------
# hyper parameters
# ------------------------------------------------------------------------------------------------------------
true_false_dict = {'True':True,'False':False}


is_read_hyper_parameters_from_file = True
if is_read_hyper_parameters_from_file:
    hp_file_name = HP_FILE_NAME
    hp_folder_path = os.path.join(parent_folder, 'results', 'test_results_with_plot', model)
    hp_file_path = os.path.join(hp_folder_path, hp_file_name)
    hp_dict = collections.defaultdict(lambda :0)
    hp_skip_list = ['shifting_size', 'unique_id','shift_num',
                    'experiment','trail','random_state_total']
    with open (hp_file_path, 'r') as f:
        for i, line in enumerate(f):
            if re.findall('feature_switch_tuple', line):
                print ("line: ", line.strip().split(':'))
                hp, hp_value = line.split(':')
                hp_value = line.split(':')[1].strip()
                if hp_value == 'None':
                    hp_value = None
                else:
                    hp_value = tuple(hp_value[1:-1].split(','))
                    hp_value = tuple([int(x) for x in hp_value])
                hp_dict[hp] = hp_value
            elif re.findall(r'[0-9]+,random_state', line):
                print ("line: ", line)
                random_state = int(re.findall(r'random_state,([0-9]+)', line)[0])
                random_state_ensemble = int(re.findall(r'random_state_ensemble,([0-9]+)', line)[0])
                hp_dict['random_state'] = random_state
                hp_dict['random_state_ensemble'] = random_state_ensemble
                break
            else:
                hp, hp_value = line.split(',')
                if hp in hp_skip_list:
                    continue
                hp_value = hp_value.strip()
                if hp == 'early_stopping' or hp == 'is_standardisation' or hp == 'is_PCA':
                    hp_value = true_false_dict[hp_value]
                if hp == 'validation_fraction' or hp == 'hidden_layer_1' or hp == 'alpha' or hp == 'learning_rate_init':
                    hp_value = float(hp_value)
                if hp == 'hidden_layer_1' or hp == 'hidden_layer_2' or hp == 'training_window_size':
                    hp_value = int(hp_value)
                if hp == 'pca_n_component':
                    if hp_value == 'None':
                        hp_value = None
                    else:
                        hp_value = int(hp_value)

                hp_dict[hp] = hp_value

    # print ("hp_dict: ", hp_dict)
    # sys.exit()
    # read hp dict
    is_standardisation = hp_dict['is_standardisation']
    is_PCA = hp_dict['is_PCA']
    pca_n_component = hp_dict['pca_n_component']
    if hp_dict.get('hidden_layer_2'):
        hidden_layer_sizes = (hp_dict['hidden_layer_1'],hp_dict['hidden_layer_2'])
    else:
        hidden_layer_sizes = (hp_dict['hidden_layer_1'],)
    learning_rate_init = hp_dict['learning_rate_init']
    learning_rate = hp_dict['learning_rate']
    early_stopping = hp_dict['early_stopping']
    activation  = hp_dict['activation_function']
    week_for_predict = hp_dict['training_window_size']
    validation_fraction  = hp_dict['validation_fraction']
    alpha  = hp_dict['alpha']
    feature_switch_tuple = hp_dict['feature_switch_tuple']
    random_state = hp_dict['random_state']
    random_state_ensemble = hp_dict['random_state_ensemble']
    #
else:
    is_standardisation = True
    is_PCA = False
    pca_n_component = None
    hidden_layer_sizes = (245,379)
    learning_rate_init = 0.044017
    learning_rate = 'invscaling'
    early_stopping = False
    activation  = 'relu'
    week_for_predict = 15  # None
    validation_fraction  = 0 # The proportion of training data to set aside as validation set for early stopping.
                               # Must be between 0 and 1. Only used if early_stopping is True.
    alpha  = 0.000445
    feature_switch_tuple = None
    random_state = 77678
    random_state_ensemble = 2
# ------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------
# other parameters
# ------------------------------------------------------------------------------------------------------------
verbose = False
tol = 1e-8
# ------------------------------------------------------------------------------------------------------------
best_f1_list = [0,0,0] # f1, accuracy, random_state
best_accuracy_list = [0,0,0] # accuracy, f1, random_state
best_random_state = -1
best_rmse_list = [float('inf'),0,0]
best_avg_pc_list = [float('-inf'),0,0]
best_return_list = []
best_date_list = []
# ------------------------------------------------------------------------------------------------------------




# (1.) build classifer
# (1.) build classifer
if model == 'regressor':
    mlp1 = MlpTradeRegressor()
elif model == 'bagging_regressor':
    mlp1 = MlpTradeEnsembleRegressor(ensemble_number=3, mode='bagging')
elif model == 'adaboost_regressor':
    mlp1 = MlpTradeEnsembleRegressor(ensemble_number=3, mode='adaboost')

clsfy_name = 'regression_test_{}'.format(model)
clf_path = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)
data_per = 1.0 # the percentage of data using for training and testing
#

# (2.) read window_size and other config
is_moving_window = True
# window size
_1, _2, date_str_list, _3 = mlp1._feed_data(test_data_folder, data_per, feature_switch_tuple=None,
                                            is_random=False, random_seed=1, mode=mode)
date_str_set = set(date_str_list)
test_date_num = len(date_str_set)
if is_moving_window:
    window_size = 1
else:
    window_size = test_date_num
#
# window index
window_index_start = 0
wasted_date_num = test_date_num % window_size
if wasted_date_num != 0:
    print ("Some instances in test set may be wasted!! test_date_num%window_size: {}".format(wasted_date_num))
max_window_index = int(test_date_num/window_size)
#


pred_label_list = []
actual_label_list = []
data_list_for_classification = []
avg_price_change_list = []
rmse_list = []
chosen_stock_return_list = []
plot_date_list = []
var_list = []
std_list = []



for window_index in range(window_index_start, max_window_index):
    print ("===window_index: {}===".format(window_index))
    # (2.) load training data, save standardisation_file and pca_file
    standardisation_file_path = os.path.join(parent_folder, 'data_processor','z_score')
    pca_file_path = os.path.join(parent_folder,'data_processor','pca')
    mlp1.trade_feed_and_separate_data_for_test(train_data_folder, test_data_folder, data_per = data_per,
                                               standardisation_file_path = standardisation_file_path,
                                               pca_file_path = pca_file_path, mode = mode, pca_n_component = pca_n_component
                                               , is_standardisation = is_standardisation, is_PCA = is_PCA,
                                               is_moving_window = is_moving_window, window_size = window_size,
                                               window_index = window_index, week_for_predict = week_for_predict,
                                               test1_data_folder = test1_data_folder,
                                               feature_switch_tuple = feature_switch_tuple)

    # (3.) load hyper parameters and training
    mlp1.set_regressor(hidden_layer_sizes, tol=tol, learning_rate_init=learning_rate_init, random_state=random_state,
                               verbose = verbose, learning_rate = learning_rate, early_stopping =early_stopping,
                                activation  = activation, validation_fraction  = validation_fraction, alpha  = alpha,
                       random_state_ensemble = random_state_ensemble)
    mlp1.regressor_train(save_clsfy_path= clf_path)
    pred_value_list, actual_value_list, date_list, stock_id_list = mlp1.reg_dev_for_moving_window_test(save_clsfy_path= clf_path)
    #
    print ("date_list: ", date_list)
    pred_label_list_temp = ['pos' if x >= 0 else 'neg' for x in pred_value_list]
    actual_label_list_temp = ['pos' if x >= 0 else 'neg' for x in actual_value_list]
    pred_label_list.extend(pred_label_list_temp)
    actual_label_list.extend(actual_label_list_temp)
    data_list_for_classification.extend(date_list)
    #
    rmse = calculate_rmse(actual_value_list, pred_value_list)
    avg_price_change_tuple, var_tuple, std_tuple = get_avg_price_change(pred_value_list, actual_value_list
                                                                        , date_list, stock_id_list)

    chosen_stock_return_list_temp = get_chosen_stock_return(pred_value_list, actual_value_list, date_list, stock_id_list)
    avg_price_change_1 = avg_price_change_tuple[0] # Strategy: choose the top 1 stock each week
    var_1 = var_tuple[0]
    std_1 = std_tuple[0]
    avg_price_change_list.append(avg_price_change_1)
    var_list.append(var_1)
    std_list.append(std_1)
    chosen_stock_return_list.extend(chosen_stock_return_list_temp)
    rmse_list.append(rmse)


    # date set
    date_set = set(date_list)
    sorted_date_list = sorted(list(date_set))
    plot_date_list.extend(sorted_date_list)

    print ("pred_value_list: ", pred_value_list)
    print ("actual_value_list: ", actual_value_list)
    print("avg_price_change_1: ", avg_price_change_1)
    print("plot_date_list: ", plot_date_list)

    #

    print ("Regressor for test trained successfully!")

print ("plot_date_list2: ", plot_date_list)

week_average_f1, week_average_accuracy, dev_label_dict, pred_label_dict = \
    compute_trade_weekly_clf_result(pred_label_list, actual_label_list, data_list_for_classification)

avg_price_change = np.average(avg_price_change_list)
avg_std = np.average(std_list)
avg_rmse = np.average(rmse_list)

print ("---------------------------------------------------")
print ("random_state: ", random_state)
print ("avg_rmse: ", avg_rmse)
print ("avg_f1: ", week_average_f1)
print ("accuracy: ", week_average_accuracy)
print ("avg_price_change: ", avg_price_change)
print ("avg_std: ", avg_std)
print ("window_index_start: {}, max_window_index: {}, window_size: {}".format(window_index_start,max_window_index,
                                                                              window_size))
print ("pred_label_dict: {}".format(pred_label_dict.items()))
print ("dev_label_dict: {}".format(dev_label_dict.items()))

#
if avg_rmse < best_rmse_list[0]:
    best_rmse_list[0] = avg_rmse
    best_rmse_list[1] = avg_price_change
    best_rmse_list[2] = random_state

if avg_price_change > best_avg_pc_list[0]:
    best_avg_pc_list[0] = avg_price_change
    best_avg_pc_list[1] = avg_rmse
    best_avg_pc_list[2] = random_state
    best_return_list = chosen_stock_return_list # to plot stock returns with best avg_price
#

best_date_list = plot_date_list
print ("best_return_list: ", best_return_list)
print ("best_date_list: ", best_date_list)


print ("+++++++++++++++++++++++++++++++++++++++++++")
print ("best_rmse: {}, avg_pc: {}, random_state: {}".format(*best_rmse_list))
print ("best_avg_pc: {}, rmse: {}, random_state: {}".format(*best_avg_pc_list))
print ("+++++++++++++++++++++++++++++++++++++++++++")


# ----------------------------------------------------------------------------------------------------------------------
simple_baseline_best_return_list = stock_prediction_baseline_reg(test1_folder_name, test2_folder_name)
random_baseline_each_week_return_list = stock_prediction_baseline_reg(test1_folder_name, test2_folder_name,
                                                                      is_random=True, random_seed = 1)
highest_profit_baseline_each_week_return_list = stock_prediction_baseline_reg(test1_folder_name, test2_folder_name,
                                                                         is_highest_profit=True)

highest_profit = 1
for profit in highest_profit_baseline_each_week_return_list:
    highest_profit += highest_profit*profit
print ("Highest possible profit: {}".format(highest_profit))

# plot for the stock return
capital = 1
model_label = 'MLP'
title = 'Stock return for ada-boosting ensemble on the 2nd test set'
xlabel = 'Date'
file_name = "{}_{}.png".format('Regressor_stock_return',model)
save_path = os.path.join(parent_folder, 'results', 'test_2_result', file_name)
plot_stock_return(best_return_list, best_date_list, capital = capital,
                  title = title, xlabel = xlabel, save_path = save_path, is_plot = True,
                  simple_baseline_each_week_return_list= simple_baseline_best_return_list,
                  random_baseline_each_week_return_list = random_baseline_each_week_return_list,
                  highest_profit=highest_profit,
                  model_label = model_label
                  )

# ----------------------------------------------------------------------------------------------------------------------
