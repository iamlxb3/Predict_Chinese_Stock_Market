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
current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
clf_path = os.path.join(parent_folder, 'classifiers', 'mlp')
path2 = os.path.join(parent_folder, 'general_functions')

sys.path.append(clf_path)
sys.path.append(path2)
sys.path.append(current_folder)

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
train_folder_name = 'a_share_regression_data'
test_folder_name = 'a_share_regression_data_test_1'


train_data_folder = os.path.join('a_share',train_folder_name)
train_data_folder = os.path.join(parent_folder, 'data', train_data_folder)
# test
test_data_folder = os.path.join('a_share',test_folder_name)
test_data_folder = os.path.join(parent_folder, 'data', test_data_folder)
# ------------------------------------------------------------------------------------------------------------



# ==========================================================================================================
# Build MLP classifier for a-share data, save the mlp to local
# ==========================================================================================================
data_set = 'a_share'
mode = 'reg' #'reg'
#model = 'regressor'
#model = 'bagging_regressor'
model = 'adaboost_regressor'
chosen_metric = 'avg_pc' #avg_pc, avg_f1, accuracy, rmse, return
if chosen_metric == 'return':
    chosen_metric1 = 'avg_pc'
else:
    chosen_metric1 = chosen_metric
model_validation_result_name = '{}_validation_result_[{}].csv'.format(model, chosen_metric1)
model_validation_result_path = os.path.join(parent_folder, 'results', 'model_results', model_validation_result_name)
ENSEMBLE_NUMBER = 3
model_test_range = (0,9)
RANDOM_STATE_TEST_NUM = 5
is_plot = False
RANDOM_SEED = 3 #
week_for_predict = 15  # None
print ("Build MLP {} for {} test data!".format(model, data_set))
# ------------------------------------------------------------------------------------------------------------
# (1.) read the position of hyper-parameters
# ------------------------------------------------------------------------------------------------------------
hpp_file_name = 'hyper_parameter_position.csv'
hpp_file_path = os.path.join(parent_folder, 'results', 'model_results', hpp_file_name)
hyper_parameter_index_dict = {}
with open (hpp_file_path, 'r') as f:
    for line in f:
        line_list = line.split(',')
        hyper_parameter_index_dict[int(line_list[1])] = line_list[0]
# ------------------------------------------------------------------------------------------------------------
hyper_parameter_test_list = []
is_hyper_parameter = False

with open (model_validation_result_path, 'r') as f:
    model_count = 0
    for i, line in enumerate(f):
        if model_count > model_test_range[1]:
            print ("a")
            break
        if model_count < model_test_range[0]:
            model_count = model_test_range[0]
            continue
        if i % 2 == 0:
            print ("1")
            hyper_parameter_list = line.strip().split(',')
            is_hyper_parameter = True
            #print ("hyper_parameter_list: ", hyper_parameter_list)
        else:
            if is_hyper_parameter:
                hyper_parameter_dict_temp = {}
                for index, value in enumerate(hyper_parameter_list):
                    hyper_parameter_name = hyper_parameter_index_dict[index]
                    hyper_parameter_dict_temp[hyper_parameter_name] = value
                hyper_parameter_test_list.append((hyper_parameter_dict_temp, model_count))
                is_hyper_parameter = False
                model_count += 1
            else:
                pass

print ("hyper_parameter_test_list: ", hyper_parameter_test_list)

# str to bool
def str2bool(v):
    if v == 'True':
       return True
    elif v == 'False':
        return False
    else:
        print ("Input: ", v)
        print ("Check your true, false input!")
        sys.exit()
#


for hyper_parameter_dict, rank in hyper_parameter_test_list:
    # ------------------------------------------------------------------------------------------------------------
    tol = 1e-8
    verbose = False
    # ------------------------------------------------------------------------------------------------------------
    # read hyper parameters
    # ------------------------------------------------------------------------------------------------------------
    unique_id = int(hyper_parameter_dict['unique_id'])
    is_standardisation = str2bool(hyper_parameter_dict['is_standardisation'])
    is_PCA = str2bool(hyper_parameter_dict['is_PCA'])
    if hyper_parameter_dict['pca_n_component'] != 'None':
        pca_n_component = int(hyper_parameter_dict['pca_n_component'])
    else:
        pca_n_component = None
    hidden_layer_sizes_list = []
    if hyper_parameter_dict['hidden_layer_1']:
        hidden_layer_sizes_list.append(int(hyper_parameter_dict['hidden_layer_1']))
    if hyper_parameter_dict.get('hidden_layer_2'):
        hidden_layer_sizes_list.append(int(hyper_parameter_dict['hidden_layer_2']))
    hidden_layer_sizes = tuple(hidden_layer_sizes_list)
    learning_rate_init = float(hyper_parameter_dict['learning_rate_init'])
    learning_rate = hyper_parameter_dict['learning_rate']

    early_stopping = str2bool(hyper_parameter_dict['early_stopping'])
    activation_function  = hyper_parameter_dict['activation_function']
    validation_fraction  = float(hyper_parameter_dict['validation_fraction']) # The proportion of training data to set aside as validation set for early stopping.
                               # Must be between 0 and 1. Only used if early_stopping is True.
    alpha  = float(hyper_parameter_dict['alpha'])


    # ------------------------------------------------------------------------------------------------------------
    if is_standardisation and is_PCA:
        data_preprocessing = 'pca_standardization'
    elif is_standardisation and not is_PCA:
        data_preprocessing = 'standardization'
    elif not is_standardisation and is_PCA:
        data_preprocessing = 'pca'
    elif not is_standardisation and not is_PCA:
        data_preprocessing = 'origin'
    else:
        print ("Check data preprocessing switch")
        sys.exit()
    # ------------------------------------------------------------------------------------------------------------

    # feature_list
    if not is_PCA:
        feature_switch_file_path = os.path.join(parent_folder, 'hyper_parameter_test', data_set, model,
                                                data_preprocessing, 'feature_selection.txt')
        with open (feature_switch_file_path, 'r') as f:
            for line in f:
                line_list = line.split(',')
                if unique_id == int(line_list[0]):
                    feature_switch_list = []
                    feature_switch_list_temp = line_list[1:]
                    for switch in feature_switch_list_temp:
                        feature_switch_list.append(int(re.findall(r'[0-9]+', switch)[0]))
                    break
        feature_switch_tuple = tuple(feature_switch_list)
    else:
        feature_switch_tuple = None

    # ------------------------------------------------------------------------------------------------------------
    random.seed(RANDOM_SEED)
    random_state_pool = [random.randint(0,99999) for x in range(0,RANDOM_STATE_TEST_NUM+1)]
    random.seed(RANDOM_SEED+1)
    random_state_ensemble_pool = [random.randint(0,99999) for x in range(0,RANDOM_STATE_TEST_NUM+1)]
    print ("random_state_pool: ", random_state_pool)
    best_f1_list = [0,0,0] # f1, accuracy, random_state
    best_accuracy_list = [0,0,0] # accuracy, f1, random_state
    best_random_state = -1
    best_rmse_list = [float('inf'),0,0,0,0]
    best_avg_pc_list = [float('-inf'),0,0,0,0]
    best_capital_list = [float('-inf'),0,0,0,0]
    best_return_list = []
    best_date_list = []
    # ------------------------------------------------------------------------------------------------------------


    for i, random_state in enumerate(random_state_pool):
        random_state_ensemble = random_state_ensemble_pool[i]
        # (1.) build classifer
        if model == 'regressor':
            mlp1 = MlpTradeRegressor()
        elif model == 'bagging_regressor':
            mlp1 = MlpTradeEnsembleRegressor(ensemble_number = ENSEMBLE_NUMBER, mode='bagging')
        elif model == 'adaboost_regressor':
            mlp1 = MlpTradeEnsembleRegressor(ensemble_number = ENSEMBLE_NUMBER, mode='adaboost')

        clsfy_name = 'dow_jones_extened_test_{}'.format(model)
        clf_path = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)
        data_per = 1.0 # the percentage of data using for training and testing
        #

        # (2.) read window_size and other config
        is_moving_window = True
        # window size
        _1, _2, date_str_list, _3 = mlp1._feed_data(test_data_folder, data_per,
                                                    feature_switch_tuple=feature_switch_tuple,
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
                                              ,is_standardisation = is_standardisation, is_PCA = is_PCA,
                                                       is_moving_window = is_moving_window, window_size = window_size,
                                                       window_index = window_index, week_for_predict = week_for_predict)

            # (3.) load hyper parameters and training
            mlp1.set_regressor(hidden_layer_sizes, tol=tol, learning_rate_init=learning_rate_init, random_state=random_state,
                               verbose = verbose, learning_rate = learning_rate,early_stopping =early_stopping,
                               activation  = activation_function, validation_fraction  = validation_fraction,
                               alpha  = alpha, random_state_ensemble = random_state_ensemble)
            mlp1.regressor_train(save_clsfy_path= clf_path)
            pred_value_list, actual_value_list, date_list, stock_id_list = mlp1.reg_dev_for_moving_window_test(save_clsfy_path= clf_path)
            #
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
            #

            print ("Regressor for test trained successfully!")

        week_average_f1, week_average_accuracy, dev_label_dict, pred_label_dict = \
            compute_trade_weekly_clf_result(pred_label_list, actual_label_list, data_list_for_classification)

        # compute capital
        capital = 1
        for stock_return in chosen_stock_return_list:
            capital += capital*float(stock_return)
        #


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
        print ("final_capital : ", capital)
        print ("window_index_start: {}, max_window_index: {}, window_size: {}".format(window_index_start,max_window_index,
                                                                                      window_size))
        print ("pred_label_dict: {}".format(pred_label_dict.items()))
        print ("dev_label_dict: {}".format(dev_label_dict.items()))

        #
        if avg_rmse < best_rmse_list[0]:
            best_rmse_list[0] = avg_rmse
            best_rmse_list[1] = avg_price_change
            best_rmse_list[2] = capital
            best_rmse_list[3] = random_state
            best_rmse_list[4] = random_state_ensemble
            if chosen_metric == 'rmse':
                best_return_list = chosen_stock_return_list

        if avg_price_change > best_avg_pc_list[0]:
            best_avg_pc_list[0] = avg_price_change
            best_avg_pc_list[1] = avg_rmse
            best_avg_pc_list[2] = capital
            best_avg_pc_list[3] = random_state
            best_avg_pc_list[4] = random_state_ensemble
            if chosen_metric == 'avg_pc':
                best_return_list = chosen_stock_return_list # to plot stock returns with best avg_price

        if capital > best_capital_list[0]:
            best_capital_list[0] = capital
            best_capital_list[1] = avg_rmse
            best_capital_list[2] = avg_price_change
            best_capital_list[3] = random_state
            best_capital_list[4] = random_state_ensemble
            if chosen_metric == 'return':
                best_return_list = chosen_stock_return_list # to plot stock returns with best avg_price
        #

        if not best_return_list:
            print ("please check chosen_metric!")
            sys.exit()

        best_date_list = plot_date_list


    print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print ("best_rmse: {}, avg_pc: {}, capital: {}, random_state: {}, random_state_ensemble: {}".format(*best_rmse_list))
    print ("best_avg_pc: {}, rmse: {}, capital: {}, random_state: {}, random_state_ensemble: {}".format(*best_avg_pc_list))
    print ("capital: {}, rmse: {}, avg_pc: {}, random_state: {}, random_state_ensemble: {}".format(*best_capital_list))
    print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


    # ----------------------------------------------------------------------------------------------------------------------
    # plot for the stock return
    capital = 1
    title = 'Stock return for MLP {} on the 1st test set'.format(model)
    xlabel ='Date'
    # save hyper parameters
    hyper_parameter_file_name = '{}_{}_rank_[{}].csv'.format(unique_id, chosen_metric, rank)
    hyper_parameter_file_path = os.path.join(parent_folder, 'results', 'test_results_with_plot', "{}".format(model),
                                             hyper_parameter_file_name)
    with open (hyper_parameter_file_path, 'w') as f:
        for key, value in hyper_parameter_dict.items():
            f.write('{},{}\n'.format(key, value))
        f.write('feature_switch_tuple:{}\n'.format(feature_switch_tuple))
        if chosen_metric == 'rmse':
            f.write("best_rmse,{},avg_pc,{},capital,{},random_state,{},random_state_ensemble,{}".format(*best_rmse_list))
        elif chosen_metric == 'avg_pc':
            f.write("best_avg_pc,{},rmse,{},capital,{},random_state,{},random_state_ensemble,{}".format(*best_avg_pc_list))
        elif chosen_metric == 'return':
            f.write("capital,{},rmse,{},avg_pc,{},random_state,{},random_state_ensemble,{}".format(*best_capital_list))

    #

    file_name = "{}_rank_{}.png".format(unique_id, rank)
    save_path = os.path.join(parent_folder, 'results', 'test_results_with_plot', "{}".format(model), file_name)

    simple_baseline_each_week_return_list = stock_prediction_baseline_reg(train_folder_name,
                                                                          test_folder_name, data_set = data_set)
    random_baseline_each_week_return_list = stock_prediction_baseline_reg(train_folder_name,
                                                                          test_folder_name,
                                                                          is_random=True, random_seed=1,
                                                                          data_set = data_set)
    highest_profit_baseline_each_week_return_list = stock_prediction_baseline_reg(train_folder_name,
                                                                          test_folder_name, is_highest_profit=True
                                                                                  ,data_set = data_set)
    highest_profit = 1
    model_label = 'MLP regressor'
    for profit in highest_profit_baseline_each_week_return_list:
        highest_profit += highest_profit * profit

    print("Highest possible profit: {}".format(highest_profit))

    plot_stock_return(best_return_list, best_date_list, capital = capital,
                      title = title, xlabel = xlabel, save_path = save_path, is_plot = is_plot
                      , simple_baseline_each_week_return_list= simple_baseline_each_week_return_list,
                      random_baseline_each_week_return_list = random_baseline_each_week_return_list,
                      highest_profit = highest_profit,
                      model_label=model_label
                      )

    # ----------------------------------------------------------------------------------------------------------------------
