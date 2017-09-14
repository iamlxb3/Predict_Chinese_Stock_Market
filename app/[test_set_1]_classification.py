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
from mlp_trade_classifier import MlpTradeClassifier
from mlp_trade_ensemble_classifier import MlpTradeDataEnsembleClassifier
from mlp_trade_regressor import MLPRegressor
from mlp_trade_ensemble_regressor import MlpTradeEnsembleRegressor
from trade_general_funcs import compute_f1_accuracy, compute_trade_weekly_clf_result, get_chosen_stock_return, \
    plot_stock_return, calculate_rmse
from trade_general_funcs import get_avg_price_change
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ------------------------------------------------------------------------------------------------------------
# read the dataset
# ------------------------------------------------------------------------------------------------------------
# train
train_data_folder = os.path.join('a_share','a_share_labeled_data')
train_data_folder = os.path.join(parent_folder, 'data', train_data_folder)
# test
test_data_folder = os.path.join('a_share','a_share_labeled_data_test_1')
test_data_folder = os.path.join(parent_folder, 'data', test_data_folder)
# ------------------------------------------------------------------------------------------------------------



# ==========================================================================================================
# Build MLP classifier for a-share data, save the mlp to local
# ==========================================================================================================
# TODO MISSING ENSEMBLE RANDOM STATE!!!
data_set = 'a_share'
mode = 'clf' #'reg'
model = 'classifier'
#model = 'bagging_classifier'

#chosen_metric = 'avg_f1' #avg_pc, avg_f1, accuracy, rmse
chosen_metric = 'accuracy' #avg_pc, avg_f1, accuracy, rmse

model_validation_result_name = '{}_validation_result_[{}].csv'.format(model, chosen_metric)
model_validation_result_path = os.path.join(parent_folder, 'results', 'model_results', model_validation_result_name)
model_test_range = (0,3)
ENSEMBLE_NUMBER = 3
RANDOM_STATE_TEST_NUM = 5
is_plot = False
RANDOM_SEED = 1
WEEK_FOR_PREDICT = 15
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
with open (model_validation_result_path, 'r') as f:
    model_count = 0
    for i, line in enumerate(f):
        if model_count > model_test_range[1]:
            break
        if model_count < model_test_range[0]:
            continue
        if i % 2 == 0:
            hyper_parameter_list = line.strip().split(',')
            #print ("hyper_parameter_list: ", hyper_parameter_list)
        else:
            model_count += 1
            hyper_parameter_dict_temp = {}
            for index, value in enumerate(hyper_parameter_list):
                hyper_parameter_name = hyper_parameter_index_dict[index]
                hyper_parameter_dict_temp[hyper_parameter_name] = value
            hyper_parameter_test_list.append((hyper_parameter_dict_temp, model_count))
#print ("hyper_parameter_test_list: ", hyper_parameter_test_list)

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
    random_state_pool = [random.randint(0,99999) for x in range(0,RANDOM_STATE_TEST_NUM)]
    best_f1_list = [0,0,0] # f1, accuracy, random_state
    best_accuracy_list = [0,0,0] # accuracy, f1, random_state
    best_random_state = -1

    # ------------------------------------------------------------------------------------------------------------


    for random_state in random_state_pool:

        # (1.) build classifer
        if model == 'classifier':
            mlp1 = MlpTradeClassifier()
        elif model == 'bagging_classifier':
            mlp1 = MlpTradeDataEnsembleClassifier(ensemble_number = ENSEMBLE_NUMBER, mode = 'bagging')

        clsfy_name = 'dow_jones_extened_test_mlp_{}'.format(model)
        clf_path = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)
        data_per = 1.0 # the percentage of data using for training and testing
        #

        # (2.) read window_size and other config
        is_moving_window = True
        week_for_predict = WEEK_FOR_PREDICT  # None
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
        data_list = []



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
            mlp1.set_mlp_clf(hidden_layer_sizes, tol=tol, learning_rate_init=learning_rate_init, random_state=random_state,
                               verbose = verbose, learning_rate = learning_rate,early_stopping =early_stopping,
                               activation  = activation_function, validation_fraction  = validation_fraction, alpha  = alpha)
            mlp1.clf_train(save_clsfy_path=clf_path)
            pred_label_list_temp, actual_label_list_temp, dev_date_list_temp, _ = \
                mlp1.clf_dev_for_moving_window_test(save_clsfy_path=clf_path)
            #
            pred_label_list.extend(pred_label_list_temp)
            actual_label_list.extend(actual_label_list_temp)
            data_list.extend(dev_date_list_temp)


        week_average_f1, week_average_accuracy, dev_label_dict, pred_label_dict = \
            compute_trade_weekly_clf_result(pred_label_list, actual_label_list, data_list)
        #
        print("Classifier for test trained successfully!")

        #


        print ("window_index_start: {}, max_window_index: {}, window_size: {}".format(window_index_start,max_window_index,
                                                                                      window_size))
        print ("week_average_f1: ", week_average_f1)
        print ("week_average_accuracy: ", week_average_accuracy)

        #
        if week_average_f1 > best_f1_list[0]:
            best_f1_list[0] = week_average_f1
            best_f1_list[1] = week_average_accuracy
            best_f1_list[2] = random_state

        if week_average_accuracy > best_accuracy_list[0]:
            best_accuracy_list[0] = week_average_accuracy
            best_accuracy_list[1] = week_average_f1
            best_accuracy_list[2] = random_state
        #

    print ("+++++++++++++++++++++++++++++++++++++++++++")
    print ("best_f1: {}, accuracy: {}, random_state: {}".format(*best_f1_list))
    print ("best_accuracy: {}, f1: {}, random_state: {}".format(*best_accuracy_list))
    print ("+++++++++++++++++++++++++++++++++++++++++++")


    # ----------------------------------------------------------------------------------------------------------------------
    # plot for the stock return
    capital = 1
    title = 'Stock return for MLP regressor'
    xlabel ='Date'
    # save hyper parameters
    hyper_parameter_file_name = '[{}]_{}_rank_{}_{}.csv'.format(rank, chosen_metric, model, unique_id)

    hyper_parameter_file_path_acc = os.path.join(parent_folder, 'results', 'test_results_with_plot', "{}".format(model),
                                                 hyper_parameter_file_name)
    hyper_parameter_file_path = os.path.join(parent_folder, 'results', 'test_results_with_plot', "{}".format(model),
                                             hyper_parameter_file_name)
    if chosen_metric == 'accuracy':
        with open (hyper_parameter_file_path, 'w') as f:
            for key, value in hyper_parameter_dict.items():
                f.write('{},{}\n'.format(key, value))
            f.write('feature_switch_tuple:{}\n'.format(feature_switch_tuple))
            f.write("best_accuracy,{},f1,{},random_state,{}".format(*best_accuracy_list))
    elif chosen_metric == 'avg_f1':
        with open(hyper_parameter_file_path, 'w') as f:
            for key, value in hyper_parameter_dict.items():
                f.write('{},{}\n'.format(key, value))
            f.write('feature_switch_tuple:{}\n'.format(feature_switch_tuple))
            f.write("best_f1,{},accuracy,{},random_state,{}".format(*best_f1_list))
    #

    # ----------------------------------------------------------------------------------------------------------------------
