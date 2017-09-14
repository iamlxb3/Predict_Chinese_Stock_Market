# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Build the MLP regressor.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general package import
# ==========================================================================================================
import sys
import random
import os
import itertools
import numpy as np
import collections
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path1 = os.path.join(parent_folder, 'general_functions')
sys.path.append(path1)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from stock_box_plot2 import data_preprocessing_result_box_plot
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# ----------------------------------------------------------------------------------------------------------------------
# read data into dictionary
# ----------------------------------------------------------------------------------------------------------------------
data_set = 'a_share'
mode = 'clf'
classifier_list = ['classifier','bagging_classifier']
regressor_list = ['regressor','bagging_regressor','adaboost_regressor']
data_preprocessing_list = ['pca','pca_standardization','standardization','origin']


# input path

if mode =='clf':
    model_list = classifier_list
elif mode == 'reg':
    model_list = regressor_list

result_dict = collections.defaultdict(lambda :collections.defaultdict(lambda:collections.defaultdict(lambda:[])))


for model, data_preprocessing in list(itertools.product(model_list, data_preprocessing_list)):
    hyper_parameter_folder = os.path.join(parent_folder, 'hyper_parameter_test', data_set,
                                          model, data_preprocessing)
    file_name_list = os.listdir(hyper_parameter_folder)
    file_path_list = [os.path.join(hyper_parameter_folder, x) for x in file_name_list]

    print ("classifier: {}, data_preprocessing: {}".format(model, data_preprocessing))
    print ("file_path_list_length: ", len(file_path_list))

# save path


    # ----------------------------------------------------------------------------------------------------------------------
    # (1.) push every model's result to dict
    # ----------------------------------------------------------------------------------------------------------------------
    for i, file_path in enumerate(file_path_list):
        with open (file_path, 'r') as f:
            #print ("file_path: ", file_path)
            file_name = file_name_list[i]
            if file_name == '.gitignore' or  file_name == 'feature_selection.txt':
                continue
            for j, line in enumerate(f):
                unique_id = "{}_{}".format(i,j)
                line_list = line.strip().split(',')
                if mode == 'clf':
                    loss = line_list[0]
                    avg_iter_num = line_list[1]
                    avg_f1 = line_list[2]
                    accuracy = line_list[3]
                    result_dict[model][data_preprocessing]['loss_list'].append(float(loss))
                    result_dict[model][data_preprocessing]['avg_iter_num_list'].append(float(avg_iter_num))
                    result_dict[model][data_preprocessing]['avg_f1_list'].append(float(avg_f1))
                    result_dict[model][data_preprocessing]['accuracy_list'].append(float(accuracy))
                elif mode == 'reg':
                    loss = line_list[0]
                    avg_iter_num = line_list[1]
                    rmse = line_list[2]
                    avg_pc = line_list[3]
                    accuracy = line_list[4]
                    avg_f1 = line_list[5]
                    result_dict[model][data_preprocessing]['loss_list'].append(float(loss))
                    result_dict[model][data_preprocessing]['avg_iter_num_list'].append(float(avg_iter_num))
                    result_dict[model][data_preprocessing]['rmse_list'].append(float(rmse))
                    result_dict[model][data_preprocessing]['avg_pc_list'].append(float(avg_pc))
                    result_dict[model][data_preprocessing]['accuracy_list'].append(float(accuracy))
                    result_dict[model][data_preprocessing]['avg_f1_list'].append(float(avg_f1))
                else:
                    print ("Error! Please print the right mode")
                    sys.exit()
    # ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# (3.) box_plot
# ----------------------------------------------------------------------------------------------------------------------
#title = 'MLP {} performance under different number of trails'.format(classifier)
data_preprocessing_list = ['pca','pca_standardization','standardization','origin']
data_preprocessing_show_list =  ['pca','pca&z-score','z-score','none']
#data_preprocessing = 'pca_standardization'
#data_preprocessing_name = 'PCA and standardisation'

ylim_range = (0.0, 0.6)
xlim_range = (0, 11)
#reg_metric = 'rmse' # avg_pc
reg_metric = 'avg_pc' # avg_pc


if mode == 'clf':
    metrics_name_list = ['avg_f1_list','accuracy_list']
    metrics_show_list = ['AverageFmeasure', 'Accuracy']
    model_list = ['classifier', 'bagging_classifier','regressor','bagging_regressor','adaboost_regressor']
    baseline_colour_tuple = ('r', 'b')
    baseline_value_tuple = (0.479, 0.506)
    baseline_legend_tuple = ('Baseline Accuracy ', 'Baseline Average F-measure')
    mode_name = 'Classification'
elif mode == 'reg':

    # # rmse
    if reg_metric == 'rmse':
        metrics_name_list = ['rmse_list']
        metrics_show_list = ['Rmse']
        xlim_range = (0,9)
        baseline_colour_tuple = ('b',)
        baseline_value_tuple = (0.03898,)
        baseline_legend_tuple = ('Baseline rmse',)
        #
    elif reg_metric == 'avg_pc':
        # avg_pc
        metrics_name_list = ['avg_pc_list']
        metrics_show_list = ['Average return']
        xlim_range = (0,9)
        ylim_range = (-0.01, 0.015)
        baseline_colour_tuple = ('b',)
        baseline_value_tuple = (-0.001,)
        baseline_legend_tuple = ('Baseline average return',)
    #




    # metrics_name_list = ['avg_pc_list']
    # metrics_show_list = ['AverageReturn']
    model_list = ['regressor','bagging_regressor','adaboost_regressor']
    mode_name = 'Regression'

model_name = 'MLP classifier'
#model = 'classifier' #
#model_list = ['bagging_classifier']
#model_list = ['classifier', 'bagging_classifier']

#model = 'regressor' #

title = "{} result of different data pre-processing methods".format(mode_name)
#title = "{} result on the UCI repository".format(mode_name)

data_preprocessing_result_box_plot(result_dict, model_list, data_preprocessing_list, metrics_name_list,
                                   data_preprocessing_show_list,metrics_show_list,
                                   title =title, x_label = '', ylim_range = ylim_range, xlim_range=xlim_range,
baseline_colour_tuple = baseline_colour_tuple,
baseline_value_tuple = baseline_value_tuple,
baseline_legend_tuple = baseline_legend_tuple
)


# ----------------------------------------------------------------------------------------------------------------------