# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# MLP for general functions
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# General import
# ==========================================================================================================
import os
import sys
import random
import numpy as np
import math
import re
import collections
# ==========================================================================================================


# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
path1 = os.path.join(parent_folder, 'general_functions')
path2 = os.path.join(parent_folder, 'strategy')
path3 = os.path.join(parent_folder, 'data_processor')
sys.path.append(path1)
sys.path.append(path2)
sys.path.append(path3)
# ==========================================================================================================


# ==========================================================================================================
# local package import
# ==========================================================================================================
from trade_general_funcs import feature_degradation
from data_preprocessing import DataPp
# ==========================================================================================================


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


class MultilayerPerceptron:

    def __init__(self):

        # hidden_layer
        self.mlp_hidden_layer_sizes_list = []
        self.hidden_size_list = []
        #

        # feature switch
        self.feature_switch_list = []
        self.feature_selected_list = []
        #

        # iteration_loss
        self.iteration_loss_list = []
        self.tp_cv_iteration_loss_list = []
        self.tp_cv_pca_n_component_list = []
        #


    def read_selected_feature_list(self, folder, feature_switch_list):
        '''Used in topology test, for initializing self.feature_selected_list'''
        file_name_list = os.listdir(folder)
        file_path_0 = [os.path.join(folder, x) for x in file_name_list][0]
        with open(file_path_0, 'r', encoding='utf-8') as f:
            feature_name_list = f.readlines()[0].split(',')[::2]
            selected_feature_list = feature_degradation(feature_name_list, feature_switch_list)
        self.feature_selected_list.append(selected_feature_list)



    def _update_feature_switch_list(self, i):
        if i != 0:
            # --------------------------------------------------------------------------
            # update feature_switch_list and feature_selected list for easy output
            # --------------------------------------------------------------------------
            self.feature_switch_list.append(self.feature_switch_list[-1])
            self.feature_selected_list.append(self.feature_selected_list[-1])
            # --------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # [C.1] read, feed and load data for general data set, mainly for testing
    # ------------------------------------------------------------------------------------------------------------------
    def _general_feed_data(self, folder, data_per, feature_switch_tuple=None, is_random=False, random_seed=1, mode='reg'):
        # feature_switch_tuple : (0,1,1,0,1,1,0,...) ?
        if feature_switch_tuple:
            self.feature_switch_list.append(feature_switch_tuple)
        # ::: _feed_data :::
        # TODO test the folder exists
        file_name_list = os.listdir(folder)
        file_path_list = [os.path.join(folder, x) for x in file_name_list]
        file_total_number = len(file_name_list)
        file_used_number = math.floor(data_per * file_total_number)  # restrict the number of training sample
        file_path_list = file_path_list[0:file_used_number]
        samples_feature_list = []
        samples_value_list = []

        for f_path in file_path_list:
            f_name = os.path.basename(f_path)
            if mode == 'reg':
                regression_value = float(re.findall(r'#([0-9\.\+\-e]+)#', f_name)[0])
            elif mode == 'clf':
                regression_value = re.findall(r'_([A-Za-z\-0-9]+).txt', f_name)[0]
            else:
                print("Please enter the correct mode!")
                sys.exit()
            with open(f_path, 'r') as f:
                features_list = f.readlines()[0].split(',')
                features_list = features_list[1::2]
                features_list = [float(x) for x in features_list]
                if feature_switch_tuple:
                    features_list = feature_degradation(features_list, feature_switch_tuple)
                features_array = np.array(features_list)
                # features_array = features_array.reshape(-1,1)
                samples_feature_list.append(features_array)
                samples_value_list.append(regression_value)
        print("read feature list and {}_value list for {} successful!".format(mode, folder))

        return samples_feature_list, samples_value_list


    def load_train_dev_general_data_for_1_validation(self, samples_feature_list, samples_value_list, data_per = 1.0,
                                                     dev_per = 0.1, random_seed = 1, n_fold_index = 0, is_print = True):
        '''for normal data, regardless of date restrictions'''
        # (0.) tailor the data set if necessary
        if data_per != 1.0:
            total_sample_number = len(samples_feature_list)
            tailored_sample_number = math.floor(data_per*total_sample_number)
            samples_feature_list = samples_feature_list[0:tailored_sample_number]
            samples_value_list = samples_value_list[0:tailored_sample_number]
        #

        # (1.) do the random for feature and value at the same order
        random.seed(random_seed)
        random_zip = list(zip(samples_feature_list, samples_value_list))
        random.shuffle(random_zip)
        random_samples_feature_list, random_samples_value_list = zip(*random_zip)
        #

        # (2.) get the dev start, end index
        total_sample_number = len(samples_feature_list)
        dev_sample_number = math.floor(total_sample_number*dev_per)
        sample_end_index = total_sample_number
        dev_start_index = n_fold_index*dev_sample_number
        if dev_start_index >= total_sample_number - 1:
            print ("Please check dev_per or n_fold_index, too big!")
            sys.exit()
        dev_end_index = dev_start_index + dev_sample_number
        if dev_end_index > sample_end_index:
            dev_end_index = dev_end_index
        #

        # (3.) push the value to training and dev index
        self.dev_set = random_samples_feature_list[dev_start_index:dev_end_index]
        self.dev_value_set = random_samples_value_list[dev_start_index:dev_end_index]
        self.training_set = random_samples_feature_list[0:dev_start_index] + \
                       random_samples_feature_list[dev_end_index:sample_end_index]
        self.training_value_set = random_samples_value_list[0:dev_start_index] + \
                                  random_samples_value_list[dev_end_index:sample_end_index]
        #


    def mlp_data_pre_processing(self, fit_data, obj_data, is_standardisation, is_PCA, pca_n_component = None,
                                standardisation_file_path = '', pca_file_path = ''):
        trans_fit, trans_obj = fit_data, obj_data
        data_dp = DataPp()
        if is_standardisation:
            trans_fit, trans_obj = data_dp.standardisation_fit_transfrom(trans_fit, trans_obj,
                                                                         standardisation_file_path = standardisation_file_path)
        if is_PCA:
            trans_fit, trans_obj = data_dp.PCA_fit_transfrom(trans_fit, trans_obj, pca_n_component = pca_n_component,
                                                             pca_file_path = pca_file_path)
        print ("Data pre-processing done! is_standardisation: {}, is_PCA: {}".format(is_standardisation, is_PCA))
        return trans_fit, trans_obj

    def _update_train_dev_value_set(self, updated_train, updated_dev):
        self.training_set = updated_train
        self.dev_set = updated_dev


    def general_feed_and_separate_data_1_fold(self, folder, dev_per=0.1, data_per=1.0, feature_switch_tuple=None,
                                              random_seed = 1, mode='reg', is_production = False, is_standardisation = True,
                                              is_PCA = True):
        # (1.) read all the data, feature customizable
        samples_feature_list, samples_value_list = self._general_feed_data(folder, data_per=data_per,
                                                       feature_switch_tuple=feature_switch_tuple,
                                                       is_random=False, mode=mode)

        # (2.) load_train_dev_data_for_1_validation
        self.load_train_dev_general_data_for_1_validation(samples_feature_list, samples_value_list, data_per = data_per,
                                                     dev_per = dev_per, random_seed = random_seed, n_fold_index = 0)

        # (3.) standardisation, PCA
        trans_fit, trans_obj = self.mlp_data_pre_processing(self.training_set, self.dev_set, is_standardisation
                                                            , is_PCA)
        self._update_train_dev_value_set(trans_fit, trans_obj)

        print ("Split training and testing by 1 fold complete!")
        print ("Traning data size: {}, testing data size: {}".format(len(self.training_value_set), len(self.dev_value_set)))

    # def general_feed_and_separate_data_n_fold(self, folder, dev_per=0.1, data_per=1.0, feature_switch_tuple=None,
    #                                           random_seed = 1, mode='reg', is_production = False,
    #                                           is_standardisation = True, is_PCA = True):
    #     # (1.) read all the data, feature customizable
    #     samples_feature_list, samples_value_list = self._general_feed_data(folder, data_per=data_per,
    #                                                    feature_switch_tuple=feature_switch_tuple,
    #                                                    is_random=False, mode=mode)
    #
    #     # (2.) load_train_dev_data_for_1_validation
    #     self.load_train_dev_general_data_for_1_validation(samples_feature_list, samples_value_list, data_per = data_per,
    #                                                  dev_per = dev_per, random_seed = random_seed, n_fold_index = 0)
    #
    #     # (3.) standardisation, PCA
    #     trans_fit, trans_obj = self.mlp_data_pre_processing(self.training_set, self.dev_set, is_standardisation
    #                                                         , is_PCA)
    #     self._update_train_dev_value_set(trans_fit, trans_obj)



    def create_train_dev_vdict_general(self, samples_feature_list, samples_value_list,random_seed, is_cv=True,
                                       data_per = 1.0, dev_per = 0.1,
                                     is_standardisation = True, is_PCA = True):
        print ("Total data: {}".format(len(samples_feature_list)))

        n_fold_range = int(math.floor(1 / dev_per))

        # (0.) reset validation_dict
        self.validation_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: {}))

        # get the date set and count the number of unique dates
        for i in range(n_fold_range):
            self.load_train_dev_general_data_for_1_validation(samples_feature_list, samples_value_list,
                                                              data_per=data_per, dev_per=dev_per,
                                                              random_seed=random_seed, n_fold_index=i,
                                                              is_print=False)

            # (.) standardisation, PCA
            training_set, dev_set = self.mlp_data_pre_processing(self.training_set, self.dev_set,
                                                                is_standardisation
                                                                , is_PCA)
            training_value_set = self.training_value_set
            dev_value_set = self.dev_value_set
            print ("training set size: {}".format(len(training_value_set)))
            print ("dev set size: {}".format(len(dev_value_set)))


            self.validation_dict[random_seed][i]['training_set'] = training_set
            self.validation_dict[random_seed][i]['training_value_set'] = training_value_set
            self.validation_dict[random_seed][i]['dev_set'] = dev_set
            self.validation_dict[random_seed][i]['dev_value_set'] = dev_value_set

        validation_num = len(list(self.validation_dict[random_seed].keys()))
        print("Create validation_dict sucessfully! {}-fold cross validation".format(validation_num))

    def rs_cv_load_train_dev_data(self, random_seed, cv_index):
        self.training_set = self.validation_dict[random_seed][cv_index]['training_set']
        self.training_value_set = self.validation_dict[random_seed][cv_index]['training_value_set']
        self.dev_set = self.validation_dict[random_seed][cv_index]['dev_set']
        self.dev_value_set = self.validation_dict[random_seed][cv_index]['dev_value_set']



    def save_data_image_PCA(self, target_feature_set, target_value_set, title, save_path, is_show = False):
        '''1st and 2nd component of PCA'''
        import matplotlib.pyplot as plt
        label_set = set(target_value_set)

        f1, ax1 = plt.subplots(1)
        ax1.set_xlabel('1st-component')
        ax1.set_ylabel('2nd-component')

        for label in label_set:
            x_list = []
            y_list = []
            for i, sample in enumerate(target_feature_set):
                if target_value_set[i] == label:
                    first_component = sample[0]
                    second_component = sample[1]
                    x_list.append(first_component)
                    y_list.append(second_component)
                else:
                    continue

            ax1.plot(x_list, y_list,'x',label = label)
            ax1.set_title(title)

        ax1.legend()
        if is_show:
            plt.show()
            sys.exit()

        # save
        f1.savefig(save_path)
        print ("save figure to {} successfully!".format(save_path))
        # save