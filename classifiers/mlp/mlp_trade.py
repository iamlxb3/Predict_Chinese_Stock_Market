# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# MLP only for trading, currently for a share stock and forex
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general import
# ==========================================================================================================
import collections
import datetime
import time
import pickle
import os
import re
import sys
import math
import random
import numpy as np

# ==========================================================================================================


# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
path1 = os.path.join(parent_folder, 'general_functions')
path2 = os.path.join(parent_folder, 'strategy')
sys.path.append(path1)
sys.path.append(path2)
# ==========================================================================================================


# ==========================================================================================================
# local package import
# ==========================================================================================================
from mlp_general import MultilayerPerceptron
from trade_general_funcs import feature_degradation
from trade_general_funcs import list_by_index


# ==========================================================================================================


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class MlpTrade(MultilayerPerceptron):
    def __init__(self):
        super().__init__()

        # create validation_dict, store all the cross validation data
        self.validation_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: {}))

        # --------------------------------------------------------------------------------------------------------------
        # container for 1-fold validation training and dev data for both classifier and regressor
        # --------------------------------------------------------------------------------------------------------------
        self.training_set = []
        self.training_value_set = []
        self.training_date_set = []
        self.training_stock_id_set = []
        self.dev_set = []
        self.dev_value_set = []
        self.dev_date_set = []
        self.dev_stock_id_set = []
        # --------------------------------------------------------------------------------------------------------------

    def weekly_predict(self, input_folder, classifier_path, prediction_save_path,
                       standardisation_file_path='', pca_file_path= ''):
        '''weekly predict could be based on regression or classification
        '''
        mlp = pickle.load(open(classifier_path, "rb"))

        if standardisation_file_path:
            z_score = pickle.load(open(standardisation_file_path, "rb"))
        if pca_file_path:
            pca = pickle.load(open(pca_file_path, "rb"))

        file_name_list = os.listdir(input_folder)
        prediction_set = []

        # find the nearest date
        date_set = set()
        for file_name in file_name_list:
            date = re.findall(r'([0-9]+-[0-9]+-[0-9]+)_', file_name)[0]
            date_set.add(date)
        nearest_date = sorted(list(date_set), reverse=True)[0]
        nearest_date_temp = time.strptime(nearest_date, '%Y-%m-%d')
        nearest_date_obj = datetime.datetime(*nearest_date_temp[:3])

        print("==============================================================================")
        print("Prediction complete! This prediction is based on the data of DATE:[{}]".format(nearest_date))
        print("==============================================================================")

        if nearest_date_obj.weekday() != 4:
            print("WARNING! The nearest date for prediction is not friday!")

        # accumulate the prediction result
        for file_name in file_name_list:

            stock_id = re.findall(r'_([0-9]+).txt', file_name)[0]
            date = re.findall(r'([0-9]+-[0-9]+-[0-9]+)_', file_name)[0]
            if date != nearest_date:
                continue
            file_path = os.path.join(input_folder, file_name)

            with open(file_path, 'r') as f:
                feature_value_list = f.readlines()[0].strip().split(',')[1::2]
                feature_value_list = [float(x) for x in feature_value_list]
                feature_array = np.array(feature_value_list).reshape(1, -1)

                # standardisation and PCA
                if standardisation_file_path:
                    feature_array = z_score.transform(feature_array)
                if pca_file_path:
                    feature_array = pca.transform(feature_array)

                # =================================================================================================
                # construct features_set and predict
                # =================================================================================================
                pred_value = float(mlp.predict(feature_array)[0])
                prediction_set.append((stock_id, pred_value))
                # =================================================================================================

        # write the prediciton result to file
        prediction_set = sorted(prediction_set, key=lambda x: x[1], reverse=True)
        with open(prediction_save_path, 'w', encoding='utf-8') as f:
            for stock_id, pred_value in prediction_set:
                f.write(stock_id + ' ' + str(pred_value) + '\n')
        #
        print("Prediction result save to {} successful!".format(prediction_save_path))

    # ------------------------------------------------------------------------------------------------------------------
    # [C.1] read, feed and load data for 1 validation
    # ------------------------------------------------------------------------------------------------------------------
    def _feed_data(self, folder, data_per, feature_switch_tuple=None, is_random=False, random_seed=1, mode='reg'):
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
        date_str_list = []
        stock_id_list = []
        for f_path in file_path_list:
            f_name = os.path.basename(f_path)
            if mode == 'reg':
                regression_value = float(re.findall(r'#([0-9\.\+\-e]+)#', f_name)[0])
            elif mode == 'clf':
                regression_value = re.findall(r'_([A-Za-z\-0-9]+).txt', f_name)[0]
            else:
                print("Please enter the correct mode!")
                sys.exit()
            date_str = re.findall(r'([0-9]+-[0-9]+-[0-9]+)_', f_name)[0]
            stock_id = re.findall(r'_([A-Za-z0-9]{1,6})_', f_name)[0]
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
                date_str_list.append(date_str)
                stock_id_list.append(stock_id)
        print("read feature list and {}_value list for {} successful!".format(mode, folder))

        # random by random seed
        if is_random:
            print("Start shuffling the data...")
            import random
            combind_list = list(zip(samples_feature_list, samples_value_list, date_str_list, stock_id_list))
            random_seed = random_seed
            random.seed(random_seed)
            random.shuffle(combind_list)
            samples_feature_list, samples_value_list, date_str_list, stock_id_list = zip(*combind_list)
            print("Data set shuffling complete! Random Seed: {}".format(random_seed))
        #

        return samples_feature_list, samples_value_list, date_str_list, stock_id_list

    def load_train_dev_trade_data_for_1_validation(self, samples_feature_list, samples_value_list,
                                                   date_str_list, stock_id_list, dev_date_set, is_production = False
                                                   , is_time_series = False):
        all_date_set = set(date_str_list)
        if is_production:
            training_date_set = all_date_set
        else:
            training_date_set = all_date_set - dev_date_set

        # get the dev index
        dev_index_list = []
        for j, date_str in enumerate(date_str_list):
            if date_str in dev_date_set:
                dev_index_list.append(j)
        #

        # get the training index
        training_index_list = []
        for k, date_str in enumerate(date_str_list):
            if date_str in training_date_set:
                training_index_list.append(k)
        #


        self.training_set = list_by_index(samples_feature_list, training_index_list)
        self.training_value_set = list_by_index(samples_value_list, training_index_list)
        if is_time_series:
            self.training_date_set = list_by_index(date_str_list, training_index_list)
            self.training_stock_id_set = list_by_index(stock_id_list, training_index_list)
        self.dev_set = list_by_index(samples_feature_list, dev_index_list)
        self.dev_value_set = list_by_index(samples_value_list, dev_index_list)
        self.dev_date_set = list_by_index(date_str_list, dev_index_list)
        self.dev_stock_id_set = list_by_index(stock_id_list, dev_index_list)

        print("Load train, dev data complete! Train size: {}, dev size: {}".
              format(len(self.training_value_set), len(self.dev_value_set)))
    # ------------------------------------------------------------------------------------------------------------------






    # ------------------------------------------------------------------------------------------------------------------
    # [C.2] read, feed and load data for cross validation and random seed
    # ------------------------------------------------------------------------------------------------------------------
    def trade_feed_and_separate_data(self, folder, dev_per=0.1, data_per=1.0, feature_switch_tuple=None,
                                     random_seed='normal', mode='reg', is_production = False,
                                     is_standardisation = True, is_PCA = True, is_test_folder = False,
                                     standardisation_file_path = '', pca_file_path = '', pca_n_component=None):
        '''feed and seperate data in the normal order
        '''
        # (1.) read all the data, feature customizable
        samples_feature_list, samples_value_list, \
        date_str_list, stock_id_list = self._feed_data(folder, data_per=data_per,
                                                       feature_switch_tuple=feature_switch_tuple,
                                                       is_random=False, mode=mode)

        # (2.) compute the dev part index
        dev_date_num = math.floor(len(set(date_str_list)) * dev_per)
        if dev_date_num == 0.0:
            dev_date_set = set()
            print ("WARNING!! dev_date_num = 0, data set or dev_per may be too small!")
        else:
            dev_date_set = set(sorted(list(set(date_str_list)))[-1 * dev_date_num:])
        print("dev_date_set: ", dev_date_set)

        # (3.) load_train_dev_data_for_1_validation
        self.load_train_dev_trade_data_for_1_validation(samples_feature_list, samples_value_list,
                                                        date_str_list, stock_id_list, dev_date_set,
                                                        is_production = is_production)


        if not is_test_folder:
            # (4.) data pre_processing
            self.training_set = np.array(self.training_set)
            self.dev_set = np.array(self.dev_set)

            trans_fit, trans_obj = self.mlp_data_pre_processing(self.training_set, self.dev_set, is_standardisation
                                                                , is_PCA,
                                                                standardisation_file_path = standardisation_file_path,
                                                                pca_file_path = pca_file_path,
                                                                pca_n_component=pca_n_component)
            self._update_train_dev_value_set(trans_fit, trans_obj)
            
        else:
            if standardisation_file_path:
                standardisation = pickle.load(open(standardisation_file_path, "rb"))
                self.dev_set = standardisation.transform(self.dev_set)
            if pca_file_path:
                pca = pickle.load(open(pca_file_path, "rb"))
                self.dev_set = pca.transform(self.dev_set)


    def create_train_dev_vdict_stock(self, samples_feature_list, samples_value_list,
                                     date_str_list, stock_id_list, date_random_subset_list, random_seed, is_cv=True,
                                     is_standardisation = True, is_PCA = True):
        # (0.) reset validation_dict
        self.validation_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: {}))

        # get the date set and count the number of unique dates

        for i, dev_date_set in enumerate(date_random_subset_list):
            all_date_set = set(date_str_list)
            training_date_set = all_date_set - dev_date_set

            # get the dev index
            dev_index_list = []
            for j, date_str in enumerate(date_str_list):
                if date_str in dev_date_set:
                    dev_index_list.append(j)
            #

            # get the training index
            training_index_list = []
            for k, date_str in enumerate(date_str_list):
                if date_str in training_date_set:
                    training_index_list.append(k)
            #

            training_set = list_by_index(samples_feature_list, training_index_list)
            training_value_set = list_by_index(samples_value_list, training_index_list)
            dev_set = list_by_index(samples_feature_list, dev_index_list)
            dev_value_set = list_by_index(samples_value_list, dev_index_list)
            dev_date_set = list_by_index(date_str_list, dev_index_list)
            dev_stock_id_set = list_by_index(stock_id_list, dev_index_list)

            # data pre-processing
            # (.) standardisation, PCA
            training_set, dev_set = self.mlp_data_pre_processing(training_set, dev_set, is_standardisation
                                                                 , is_PCA)
            #

            self.validation_dict[random_seed][i]['training_set'] = training_set
            self.validation_dict[random_seed][i]['training_value_set'] = training_value_set
            self.validation_dict[random_seed][i]['dev_set'] = dev_set
            self.validation_dict[random_seed][i]['dev_value_set'] = dev_value_set
            self.validation_dict[random_seed][i]['dev_date_set'] = dev_date_set
            self.validation_dict[random_seed][i]['dev_stock_id_set'] = dev_stock_id_set

        validation_num = len(date_random_subset_list)
        print("Create validation_dict sucessfully! {}-fold cross validation".format(validation_num))

    def create_train_dev_vdict_window_shift(self, samples_feature_list, samples_value_list,
                                            date_str_list, stock_id_list, shifting_size = 1,training_window_size = 22,
                                            is_cv=True, shifting_size_percent = 0.1,
                                            shift_num = 5, is_standardisation = True, is_PCA = True,
                                            pca_n_component = None, training_set_percent = 1.0, is_time_series = False):

        # (0.) reset validation_dict
        self.validation_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: {}))

        # (.) window size
        random_seed = 'window_shift'
        sorted_date_list = sorted(list(set(date_str_list)))
        date_num = len(sorted_date_list)

        shifting_size_floor = math.floor(date_num * shifting_size_percent)
        shifting_size_ceil = math.ceil(date_num * shifting_size_percent)




        # print ("shifting_size: ", shifting_size)
        # print ("shift_num: ", shift_num)
        # print ("date_num: ", date_num)
        # sys.exit()

        if shifting_size*shift_num >= date_num:
            print ("TOO BIG shift_num or shifting_size!")
            sys.exit()


        window_size_max = date_num - shifting_size*shift_num
        if not training_window_size:
            training_window_size = window_size_max
        else:
            pass


        if shifting_size >= (1/2)*training_window_size:
            print("Training set too small!! Training set should be at least 2 times as big as testing set")
            print ("training_window_size: {}, testing set size: {}".format(training_window_size, shifting_size))
            sys.exit()
        #


        # # TODO important print
        # print("date_num: ", date_num)
        # print("shift_num: ", shift_num)
        # print("window_size_max: ", window_size_max)
        # print("training_window_size: ", training_window_size)
        # print("shifting_size: ", shifting_size)
        # print("Validation size: ", shifting_size*shift_num)

        for shift in range(shift_num):

            # (1.) get the training and dev date
            if training_window_size:
                training_date_start_index = shift * shifting_size + (window_size_max - training_window_size)
            else:
                training_date_start_index = shift*shifting_size
            if training_window_size:
                training_date_end_index = shift * shifting_size + window_size_max
            else:
                training_date_end_index = training_date_start_index + window_size_max

            dev_date_end_index = training_date_end_index + shifting_size

            # print ("training_date_start_index: ", training_date_start_index)
            # print ("training_date_end_index: ", training_date_end_index)
            # print ("dev_date_end_index: ", dev_date_end_index)
            # sys.exit()

            if dev_date_end_index > len(sorted_date_list) - 1:
                print ("WARNING! dev_date_end_index exceed! Dev set of different shift may have different size!")
            training_date_list = sorted_date_list[training_date_start_index:training_date_end_index]
            dev_date_list = sorted_date_list[training_date_end_index:dev_date_end_index]



            # only for testing the influence of the size of the training set on the validation error for stock market
            training_date_list_index_start = math.floor((1 - training_set_percent / 1.0) * len(training_date_list))
            training_date_list = training_date_list[training_date_list_index_start:]
            #
            print ("training_date_list: ", training_date_list)
            print ("training_date_list: ", training_date_list)

            # # TODO important print
            # print ("---------------------------------------------------------------------")
            # print ("shift_index: {}".format(shift))
            # print ("training_date_list: ", training_date_list)
            # print ("dev_date_list: ", dev_date_list)
            # #

            # (2.) get the dev index
            dev_index_list = []
            for j, date_str in enumerate(date_str_list):
                if date_str in dev_date_list:
                    dev_index_list.append(j)
            #

            # (3.) get the training index
            training_index_list = []
            for k, date_str in enumerate(date_str_list):
                if date_str in training_date_list:
                    training_index_list.append(k)
            #

            # (4.) load the training and dev data
            training_set = list_by_index(samples_feature_list, training_index_list)
            training_value_set = list_by_index(samples_value_list, training_index_list)
            if is_time_series:
                training_date_set = list_by_index(date_str_list, training_index_list)
                training_stock_id_set = list_by_index(stock_id_list, training_index_list)


            dev_set = list_by_index(samples_feature_list, dev_index_list)
            dev_value_set = list_by_index(samples_value_list, dev_index_list)
            dev_date_set = list_by_index(date_str_list, dev_index_list)
            dev_stock_id_set = list_by_index(stock_id_list, dev_index_list)


            # data pre-processing
            # (.) standardisation, PCA
            training_set, dev_set = self.mlp_data_pre_processing(training_set, dev_set, is_standardisation
                                                                 , is_PCA, pca_n_component = pca_n_component)
            #

            print ("Training_set_size: {}".format(len(training_set)))
            print ("Dev_set_size: {}".format(len(dev_set)))

            self.validation_dict[random_seed][shift]['training_set'] = training_set
            self.validation_dict[random_seed][shift]['training_value_set'] = training_value_set
            if is_time_series:
                self.validation_dict[random_seed][shift]['training_date_set'] = training_date_set
                self.validation_dict[random_seed][shift]['training_stock_id_set'] = training_stock_id_set
            self.validation_dict[random_seed][shift]['dev_set'] = dev_set
            self.validation_dict[random_seed][shift]['dev_value_set'] = dev_value_set
            self.validation_dict[random_seed][shift]['dev_date_set'] = dev_date_set
            self.validation_dict[random_seed][shift]['dev_stock_id_set'] = dev_stock_id_set
            #

        print("Create window-shifting validation_dict sucessfully! {}-fold window shifting".format(shift_num))

    def trade_feed_and_separate_data_window_shift(self, folder, data_per=1.0, feature_switch_tuple=None,
                                                  shift_num = 5, mode = 'reg', training_window_size = 74,
                                                  shifting_size = 13,
                                     is_standardisation = True, is_PCA = True, pca_n_component = None):
        samples_feature_list, samples_value_list, \
        date_str_list, stock_id_list = self._feed_data(folder, data_per=data_per,
                                                       feature_switch_tuple=feature_switch_tuple,
                                                       is_random=False, mode = mode)
        #print ("date_str_list: ", set(date_str_list))
        self.create_train_dev_vdict_window_shift(samples_feature_list, samples_value_list,
                                                 date_str_list, stock_id_list, is_cv=True,
                                                 is_standardisation=is_standardisation, is_PCA=is_PCA,
                                                 pca_n_component=pca_n_component, shifting_size = shifting_size,
                                                 training_window_size = training_window_size, shift_num = shift_num)



    def trade_rs_cv_load_train_dev_data(self, random_seed, cv_index):
        self.training_set = self.validation_dict[random_seed][cv_index]['training_set']
        self.training_value_set = self.validation_dict[random_seed][cv_index]['training_value_set']
        self.dev_set = self.validation_dict[random_seed][cv_index]['dev_set']
        self.dev_value_set = self.validation_dict[random_seed][cv_index]['dev_value_set']
        self.dev_date_set = self.validation_dict[random_seed][cv_index]['dev_date_set']
        self.dev_stock_id_set = self.validation_dict[random_seed][cv_index]['dev_stock_id_set']
    # ------------------------------------------------------------------------------------------------------------------

    def trade_feed_and_separate_data_for_test(self, training_folder, test_folder, data_per=1.0, feature_switch_tuple=None,
                                              mode='reg', is_production=False,
                                              is_standardisation=True, is_PCA=True, is_test_folder=False,
                                              standardisation_file_path='', pca_file_path='', pca_n_component=None
                                              , days_to_predict = None, is_moving_window = False,
                                              window_size = 1, window_index =0, week_for_predict = None,
                                              test1_data_folder = None, is_time_series = False):
        '''feed and seperate data in the normal order
        '''


        # (1.a) read training data

        t_samples_feature_list, t_samples_value_list, \
        t_date_str_list, t_stock_id_list = self._feed_data(training_folder, data_per=data_per,
                                                       feature_switch_tuple=feature_switch_tuple,
                                                       is_random=False, mode=mode)

        # (1.b) read test1 data
        if test1_data_folder:
            test1_samples_feature_list, test1_samples_value_list, \
            test1_date_str_list, test1_stock_id_list = self._feed_data(test1_data_folder, data_per=data_per,
                                                                     feature_switch_tuple=feature_switch_tuple,
                                                                     is_random=False, mode=mode)
            # merge train and test1
            t_samples_feature_list.extend(test1_samples_feature_list)
            t_samples_value_list.extend(test1_samples_value_list)
            t_date_str_list.extend(test1_date_str_list)
            t_stock_id_list.extend(test1_stock_id_list)
            #

        # (1.c) read test data
        test_samples_feature_list, test_samples_value_list, \
        test_date_str_list, test_stock_id_list = self._feed_data(test_folder, data_per=data_per,
                                                                 feature_switch_tuple=feature_switch_tuple,
                                                                 is_random=False, mode=mode)

        if is_moving_window:
            total_week_in_training = len(set(t_date_str_list))
            if week_for_predict > total_week_in_training:
                print ("week for predict: {} too big!!, week total in training: {}".
                       format(week_for_predict,total_week_in_training))
                sys.exit()

            # get the test set for prediction
            test_date_str_set = set(test_date_str_list)
            test_date_str_sorted = sorted(list(test_date_str_set))
            test_date_move_window_str_list = test_date_str_sorted[window_index*window_size:window_index*window_size+window_size]
            #

            # get the part of the test set which should be moved to training, include the test set for testing
            move_samples_feature_list = []
            move_samples_value_list = []
            move_date_str_list = []
            move_stock_id_list = []
            date_str_threshold = test_date_move_window_str_list[-1]

            for i, date_str in enumerate(test_date_str_list):

                if date_str > date_str_threshold:
                    continue
                else:
                    move_samples_feature_list.append(test_samples_feature_list[i])
                    move_samples_value_list.append(test_samples_value_list[i])
                    move_date_str_list.append(test_date_str_list[i])
                    move_stock_id_list.append(test_stock_id_list[i])

            #total_move_week = ((len(set(test_date_str_list)) / window_size) - 1)*window_size
            # if total_week_in_training - total_move_week <= week_for_predict:
            #     print ("Not enough week for training!!"
            #            "total_week_in_training: {}, total_move_week: {}, week_for_predict: {}".
            #            format(total_week_in_training,total_move_week,week_for_predict))
            #     sys.exit()

            # sort the training set
            train_date_sort_list  = sorted(list(zip(t_samples_feature_list,t_samples_value_list,t_date_str_list,
                                                    t_stock_id_list)),key = lambda x:x[2])
            t_samples_feature_list, t_samples_value_list, t_date_str_list, t_stock_id_list =\
                list(zip(*train_date_sort_list))

            # sort the move set
            move_date_sort_list = sorted(list(zip(move_samples_feature_list,move_samples_value_list,
                                                   move_date_str_list,move_stock_id_list))
                                     ,key = lambda x:x[2])

            move_samples_feature_list, move_samples_value_list, move_date_str_list, move_stock_id_list = \
                list(zip(*move_date_sort_list))

            samples_feature_list_temp = t_samples_feature_list + move_samples_feature_list
            samples_value_list_temp = t_samples_value_list + move_samples_value_list
            date_str_list_temp = t_date_str_list + move_date_str_list
            stock_id_list_temp = t_stock_id_list + move_stock_id_list

            # get the right number of week for prediction
            if week_for_predict:
                samples_feature_list = []
                samples_value_list = []
                date_str_list = []
                stock_id_list = []

                week_for_training_test_set = set(date_str_list_temp)
                week_for_training_test_list = sorted(list(week_for_training_test_set))[-week_for_predict-window_size:]
                week_begin = week_for_training_test_list[0]
                week_end = week_for_training_test_list[-1]


                for i, date_str in enumerate(date_str_list_temp):
                    if date_str >= week_begin and date_str <= week_end:
                        dev_date_set = set(test_date_move_window_str_list)
                        samples_feature_list.append(samples_feature_list_temp[i])
                        samples_value_list.append(samples_value_list_temp[i])
                        date_str_list.append(date_str)
                        stock_id_list.append(stock_id_list_temp[i])
                    else:
                        continue

            else:
                dev_date_set = set(test_date_move_window_str_list)
                samples_feature_list = samples_feature_list_temp
                samples_value_list = samples_value_list_temp
                date_str_list = date_str_list_temp
                stock_id_list = stock_id_list_temp


        else:
            dev_date_set = set(test_date_str_list)

            if week_for_predict:
                t_samples_feature_list_shortened = []
                t_samples_value_list_shortened = []
                t_date_str_list_shortened = []
                t_stock_id_list_shortened = []
                t_date_str_list_sorted = sorted(list(set(t_date_str_list)))
                t_date_str_for_training = t_date_str_list_sorted[-week_for_predict:]
                t_date_threshold = t_date_str_for_training[0]

                for i, date_str in enumerate(t_date_str_list):
                    if date_str >= t_date_threshold:
                        t_samples_feature_list_shortened.append(t_samples_feature_list[i])
                        t_samples_value_list_shortened.append(t_samples_value_list[i])
                        t_date_str_list_shortened.append(date_str)
                        t_stock_id_list_shortened.append(t_stock_id_list[i])
                samples_feature_list = t_samples_feature_list_shortened + test_samples_feature_list
                samples_value_list = t_samples_value_list_shortened + test_samples_value_list
                date_str_list = t_date_str_list_shortened + test_date_str_list
                stock_id_list = t_stock_id_list_shortened + test_stock_id_list
            else:
                samples_feature_list = t_samples_feature_list + test_samples_feature_list
                samples_value_list = t_samples_value_list + test_samples_value_list
                date_str_list = t_date_str_list + test_date_str_list
                stock_id_list = t_stock_id_list + test_stock_id_list



        # (3.) load_train_dev_data_for_1_validation
        self.load_train_dev_trade_data_for_1_validation(samples_feature_list, samples_value_list,
                                                        date_str_list, stock_id_list, dev_date_set,
                                                        is_production=is_production, is_time_series = is_time_series)


        print ("Training and test date set {}".format(sorted(list(set(date_str_list)))))
        print ("Test date set {}".format(sorted(list(set(dev_date_set)))))


        # (4.) data pre_processing
        self.training_set = np.array(self.training_set)
        self.dev_set = np.array(self.dev_set)

        trans_fit, trans_obj = self.mlp_data_pre_processing(self.training_set, self.dev_set, is_standardisation
                                                            , is_PCA,
                                                            standardisation_file_path=standardisation_file_path,
                                                            pca_file_path=pca_file_path,
                                                            pca_n_component=pca_n_component)

        self._update_train_dev_value_set(trans_fit, trans_obj)

