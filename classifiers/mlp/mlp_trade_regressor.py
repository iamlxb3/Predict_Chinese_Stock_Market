# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# MLP regressor only for trading, currently for a share stock and forex
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general import
# ==========================================================================================================
import sys
import os
import re
import math
import numpy as np
import pickle
import random
import collections
from sklearn.neural_network import MLPRegressor
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
from mlp_trade import MlpTrade
from mlp_regressor import MlpRegressor_P
from trade_general_funcs import calculate_rmse
from trade_general_funcs import get_avg_price_change
from trade_general_funcs import create_random_sub_set_list
from trade_general_funcs import build_hidden_layer_sizes_list
from trade_general_funcs import compute_average_f1
from trade_general_funcs import get_chosen_stock_return
# ==========================================================================================================




# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class MlpTradeRegressor(MlpTrade, MlpRegressor_P):

    def __init__(self):
        super().__init__()

        # --------------------------------------------------------------------------------------------------------------
        # container for evaluation config and result
        # --------------------------------------------------------------------------------------------------------------
        # config
        self.include_top_list = []
        # eva
        self.var_std_list = []
        self.avg_price_change_list = []
        self.polar_accuracy_list = []

        self.rsavg_iteration_loss_list = []
        self.rsavg_mres_list = []
        self.rsavg_avg_price_change_list = []
        self.rsavg_polar_accuracy_list = []
        self.rsavg_iteration_list = []
        self.rsavg_loss_list = []
        # --------------------------------------------------------------------------------------------------------------


        # --------------------------------------------------------------------------------------------------------------
        # container for n-fold validation for different hidden layer, tp is topology, cv is cross-validation
        # --------------------------------------------------------------------------------------------------------------
        # (#) cv_avg_price_change_list eg. [[0.1,0.2,0.3,..0.99], ...](each list represent one topology, each element
        # (#) is 1-fold validation avg. List could be 10 or 30 long, based on random seed list).
        self.tp_cv_avg_price_change_list = []
        self.tp_cv_polar_accuracy_list = []
        self.tp_cv_pc_pos_percent_list = [] # how many positive price change values for one topology
        # --------------------------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # [C.1]  Dev
    # ------------------------------------------------------------------------------------------------------------------

    def regressor_dev(self, save_clsfy_path="mlp_trade_regressor", is_cv=False, include_top_list = None):
        # test mode


        if not include_top_list:
            include_top_list = [1]
        mlp_regressor = pickle.load(open(save_clsfy_path, "rb"))
        pred_value_list = np.array(mlp_regressor.predict(self.dev_set))
        actual_value_list = np.array(self.dev_value_set)
        date_list = self.dev_date_set
        stock_id_list = self.dev_stock_id_set
        avg_price_change_tuple, var_tuple, std_tuple = get_avg_price_change(pred_value_list, actual_value_list,
                                                                                  date_list, stock_id_list,
                                                                                  include_top_list=
                                                                                  include_top_list)

        # compute accuracy in terms of positive and negative

        # (3.) get the pred label for each week
        pred_label_dict_by_week = collections.defaultdict(lambda :[])
        golden_label_dict_by_week = collections.defaultdict(lambda :[])
        pred_value_dict_by_week = collections.defaultdict(lambda :[])
        golden_value_dict_by_week = collections.defaultdict(lambda :[])


        pred_label_list = ['pos' if x >= 0 else 'neg' for x in pred_value_list ]
        actual_label_list = ['pos' if x >= 0 else 'neg' for x in actual_value_list ]

        for i, pred_label in enumerate(pred_label_list):
            date = self.dev_date_set[i]
            # classification
            pred_label_dict_by_week[date].append(pred_label)
            golden_label = actual_label_list[i]
            golden_label_dict_by_week[date].append(golden_label)
            #
            # regression
            predict_value = pred_value_list[i]
            golden_value = actual_value_list[i]
            pred_value_dict_by_week[date].append(predict_value)
            golden_value_dict_by_week[date].append(golden_value)

        week_average_f1_list = []
        week_average_accuracy_list = []
        week_average_rmse = []
        dev_label_dict = collections.defaultdict(lambda: 0)
        pred_label_dict = collections.defaultdict(lambda: 0)
        label_f1_list_all = []

        # (4.) compute the f1, accuracy for each week in 1 validation set
        for date, pred_label_list_for_1_week in pred_label_dict_by_week.items():
            pred_label_list = pred_label_list_for_1_week
            golden_label_list = golden_label_dict_by_week[date]

            # (3.) compute the average f-measure

            _,average_f1  = compute_average_f1(pred_label_list, golden_label_list)
            week_average_f1_list.append(average_f1)
            #average_f1 = f1_list[0] # using F-measure
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

            # # (6.) save rmse

            pred_value_list1 = pred_value_dict_by_week[date]
            actual_value_list1 = golden_value_dict_by_week[date]
            rmse = calculate_rmse(actual_value_list1, pred_value_list1)
            week_average_rmse.append(rmse)


        week_average_f1 = np.average(week_average_f1_list)
        week_average_accuracy = np.average(week_average_accuracy_list)
        week_average_rmse = np.average(week_average_rmse)


        # <uncomment for debugging>
        if not is_cv:
            print("----------------------------------------------------------------------------------------")
            print("actual_value_list, ", actual_value_list)
            print("pred_value_list, ", pred_value_list)
            print("week_average_accuracy: {}".format(week_average_accuracy))
            print("week_average_f1: {}".format(week_average_f1))
            print("week_average_rmse: {}".format(week_average_rmse))
            print("week_average_price_change: {}".format(avg_price_change_tuple))
            print("----------------------------------------------------------------------------------------")
        else:
            pass
            # print("Testing complete! Testing Set size: {}".format(len(self.r_dev_value_set)))
            # <uncomment for debugging>
    # ------------------------------------------------------------------------------------------------------------------
        return week_average_rmse, avg_price_change_tuple, week_average_accuracy, week_average_f1

    def regressor_dev_test(self, dev_set, dev_value_set, dev_date_set, dev_stock_id_set, save_clsfy_path="mlp_trade_regressor", is_cv=False, include_top_list = None):
        # test mode


        if not include_top_list:
            include_top_list = [1]
        mlp_regressor = self.mlp_regressor
        pred_value_list = np.array(mlp_regressor.predict(dev_set))
        actual_value_list = np.array(dev_value_set)
        mrse = calculate_rmse(actual_value_list, pred_value_list)
        date_list = dev_date_set
        stock_id_list = dev_stock_id_set

        avg_price_change_tuple, var_tuple, std_tuple = get_avg_price_change(pred_value_list, actual_value_list,
                                                                                  date_list, stock_id_list,
                                                                                  include_top_list=
                                                                                  include_top_list)

        # count how many predicted value has the same polarity as actual value
        polar_list = [1 for x, y in zip(pred_value_list, actual_value_list) if x * y >= 0]
        polar_count = len(polar_list)
        polar_percent = polar_count / len(pred_value_list)
        #

        # <uncomment for debugging>
        if not is_cv:
            print("----------------------------------------------------------------------------------------")
            print("actual_value_list, ", actual_value_list)
            print("pred_value_list, ", pred_value_list)
            print("polarity: {}".format(polar_percent))
            print("mrse: {}".format(mrse))
            print("avg_price_change: {}".format(avg_price_change_tuple))
            print("----------------------------------------------------------------------------------------")
        else:
            pass
            # print("Testing complete! Testing Set size: {}".format(len(self.r_dev_value_set)))
            # <uncomment for debugging>
    # ------------------------------------------------------------------------------------------------------------------
        return mrse, avg_price_change_tuple[0], polar_percent


    # ------------------------------------------------------------------------------------------------------------------
    # [C.2] Topology Test
    # ------------------------------------------------------------------------------------------------------------------
    def cv_r_topology_test(self, input_folder, feature_switch_tuple, other_config_dict,
                           hidden_layer_config_tuple, is_window_shift = False):
        '''10 cross validation test for mlp regressor'''


        # :::topology_test:::


        # ==============================================================================================================
        # Cross Validation Train And Test
        # ==============================================================================================================

        # (1.) read the whole data set
        # cut the number of training sample
        # create list under different random seed
        data_per = other_config_dict['data_per']
        samples_feature_list, samples_value_list, \
        date_str_list, stock_id_list = self._feed_data(input_folder, data_per=data_per,
                                                       feature_switch_tuple=feature_switch_tuple,
                                                       is_random=False)
        # --------------------------------------------------------------------------------------------------------------

        # (2.) construct hidden layer size list
        hidden_layer_sizes_list = build_hidden_layer_sizes_list(hidden_layer_config_tuple)
        hidden_layer_sizes_combination = len(hidden_layer_sizes_list)
        print("Total {} hidden layer size combination to test".format(hidden_layer_sizes_combination))
        # --------------------------------------------------------------------------------------------------------------

        # (3.) set MLP parameters
        learning_rate_init = other_config_dict['learning_rate_init']
        clf_path = other_config_dict['clf_path']
        tol = other_config_dict['tol']
        include_top_list = other_config_dict['include_top_list']
        self.include_top_list = include_top_list

        # (4.) Data Pre-processing
        is_standardisation = other_config_dict['is_standardisation']
        is_PCA = other_config_dict['is_PCA']
        pca_n_component = other_config_dict['pca_n_component']



        # (5.) data split mode
        if not is_window_shift:
            random_seed_list = other_config_dict['random_seed_list']
        else:
            shifting_size_percent = other_config_dict['shifting_size_percent']
            shift_num = other_config_dict['shift_num']


        # (6.) random state
        random_state_num = other_config_dict['random_state_num']

        # --------------------------------------------------------------------------------------------------------------


        # --------------------------------------------------------------------------------------------------------------
        # (4.) create validation_dict
        # --------------------------------------------------------------------------------------------------------------
        if is_window_shift:
            self.create_train_dev_vdict_window_shift(samples_feature_list, samples_value_list,
                           date_str_list, stock_id_list, is_cv=True, shifting_size_percent = shifting_size_percent,
                                                     shift_num = shift_num,
                                                     is_standardisation = is_standardisation, is_PCA = is_PCA,
                                                     pca_n_component = pca_n_component)
        else:
            dev_per = other_config_dict['dev_per']
            for random_seed in random_seed_list:
                # create the random sub set list
                dev_date_num = math.floor(len(set(date_str_list)) * dev_per)
                date_random_subset_list = \
                    create_random_sub_set_list(set(date_str_list), dev_date_num, random_seed=random_seed)
                print("-----------------------------------------------------------------------------")
                print("random_seed: {}, date_random_subset_list: {}".format(random_seed, date_random_subset_list))
                self.create_train_dev_vdict_stock(samples_feature_list, samples_value_list, date_str_list, stock_id_list,
                                                  date_random_subset_list, random_seed,
                                                  is_standardisation = is_standardisation, is_PCA = is_PCA,
                                                  pca_n_component = pca_n_component)
            # --------------------------------------------------------------------------------------------------------------


        # (5.) test the performance of different topology of MLP by 10-cross validation
        for i, hidden_layer_sizes in enumerate(hidden_layer_sizes_list):
            print("====================================================================")
            print("Topology: {} starts training and testing".format(hidden_layer_sizes))
            print("====================================================================")

            self._update_feature_switch_list(i)

            # (a.) clear the evaluation list for one hidden layer topology
            self.iteration_loss_list = []
            self.mres_list = []
            self.avg_price_change_list = []
            self.polar_accuracy_list = []
            self.var_std_list = []

            # (b.) 10-cross-validation train and test


            # random inside, make sure each date has all the
            if is_window_shift:
                # (b.1) [window-shifting-n-fold]
                random_seed = 'window_shift'
                for shift in self.validation_dict[random_seed].keys():
                    self.trade_rs_cv_load_train_dev_data(random_seed, shift)

                    random_pool = [x for x in range(9999)]
                    random.seed(1)
                    random_list = random.sample(random_pool, random_state_num)

                    # clear
                    self.rsavg_avg_price_change_list = []
                    self.rsavg_iteration_list = []
                    self.rsavg_loss_list = []
                    self.rsavg_mres_list = []
                    self.rsavg_polar_accuracy_list = []

                    # Run MLP with different initial weight for several times!
                    for rs_i, random_state in enumerate(random_list):
                        print ("shift:{}, random_state: {}".format(shift, random_state))
                        self.set_regressor(hidden_layer_sizes, learning_rate_init=learning_rate_init, tol=tol,
                                           random_state=random_state)
                        n_iter, loss = self.regressor_train(save_clsfy_path=clf_path)
                        mrse, avg_price_change_tuple, polar_percent = self.regressor_dev(save_clsfy_path=clf_path, is_cv=True, include_top_list=include_top_list)
                        self.save_evaluate_value_per_random_state(rs_i, mrse, avg_price_change_tuple, polar_percent, n_iter, loss)
                    #

                    self.save_average_evaluate_value()
            else:
                for random_seed in random_seed_list:
                    for cv_index in self.validation_dict[random_seed].keys():
                        self.trade_rs_cv_load_train_dev_data(random_seed, cv_index)
                        self.regressor_train(save_clsfy_path=clf_path)
                        self.regressor_dev(save_clsfy_path=clf_path, is_cv=True, include_top_list=include_top_list)
            #

        # (c.) save the 10-cross-valiation evaluate result for each topology
            self.tp_cv_iteration_loss_list.append(self.iteration_loss_list)
            self.tp_cv_mres_list.append(self.mres_list)
            self.tp_cv_avg_price_change_list.append(self.avg_price_change_list)
            self.tp_cv_polar_accuracy_list.append(self.polar_accuracy_list)
            self.tp_cv_pca_n_component_list.append(pca_n_component)

            # **********************************************************************************************************
            # self.avg_price_change_list: [(vaset1_top1_pc, vaset1_top2_pc ,..., vaset1_topn_pc),
            # (vaset2_top1_pc, vaset2_top2_pc ,..., vaset2_topn_pc), ..., (vasetm_top1_pc, vasetm_top2_pc ,...,
            # vasetm_topn_pc)]
            # length(m) is equal to the number of validation set
            # **********************************************************************************************************

            # tp_cv_pc_pos_percent_list
            pos_percent_list = []
            for j, top in enumerate(include_top_list):
                n_top_pc_list = [x[j] for x in self.avg_price_change_list]
                pos_count = list((np.array(n_top_pc_list) > 0)).count(True)
                all_count = len(n_top_pc_list)
                pos_percent = float("{:.5f}".format(pos_count/all_count))
                pos_percent_list.append(pos_percent)
            self.tp_cv_pc_pos_percent_list.append(tuple(pos_percent_list))
            #

            # TODO ignore var and std for a while
            # self.cv_var_std_list.append(self.var_std_list)
            #


            # (d.) real-time print
            print("====================================================================")
            print("Feature selected: {}, Total number: {}".format(self.feature_selected_list[-1],
                                                                  self.feature_switch_list[-1].count(1)))
            print("PCA-n-component: {}".format(pca_n_component))
            print("Average mres: {} len: {}".format(np.average(self.mres_list), len(self.mres_list)))
            for j, top in enumerate(include_top_list):
                n_top_list = [x[j] for x in self.avg_price_change_list]
                print("Top: {} Average price change: {}".format(top, np.average(n_top_list)))
                print("Positive percent: {}".format(pos_percent_list[j]))

            # TODO ignore var,std for a while
            # print ("Average var: {}, Average std: {}".format(np.average([x[0] for x in self.var_std_list]),
            #                                  np.average([x[1] for x in self.var_std_list])))

            print("Average polarity: {}".format(np.average(self.polar_accuracy_list)))
            print("Average iteration_loss: {}".format(np.average([x[1] for x in self.iteration_loss_list])))
            print("====================================================================")
            print("Completeness: {:.5f}".format((i + 1) / hidden_layer_sizes_combination))
            print("====================================================================")

            if i != 0 and i % 10 == 0:
                self._r_print_real_time_best_result()


    def cv_r_save_feature_topology_result(self, path, key='rmse'):
        # compute the average for each list
        cv_iteration_loss_list = [[y[1] for y in x] for x in self.tp_cv_iteration_loss_list]
        cv_iteration_loss_list = [np.average(x) for x in cv_iteration_loss_list]
        cv_polar_accuracy_list = [np.average(x) for x in self.tp_cv_polar_accuracy_list]

        # get the avg_price_change for top_n
        cv_avg_price_change_list = []
        for y in self.tp_cv_avg_price_change_list:
            y_avg_price_change_list = []
            for i, include_top in enumerate(self.include_top_list):
                top_i_avg_price_change_list = [x[i] for x in y]
                top_i_avg_price_change = np.average(top_i_avg_price_change_list)
                y_avg_price_change_list.append(top_i_avg_price_change)
            cv_avg_price_change_list.append(tuple(y_avg_price_change_list))
        #

        cv_mres_list = [np.average(x) for x in self.tp_cv_mres_list]
        pca_n_component_list = [x for x in self.tp_cv_pca_n_component_list]

        topology_list = list(zip(self.feature_switch_list, self.feature_selected_list,
                                 self.hidden_size_list, cv_iteration_loss_list,
                                 cv_polar_accuracy_list, cv_mres_list, pca_n_component_list, cv_avg_price_change_list))


        if key == 'avg_pc':
            # write the best result under different trading strategy
            for i, include_top in enumerate(self.include_top_list):

                # get the cv_avg_price_change_list for a particular strategy
                top_n_pc_list = [[y[i] for y in x] for x in self.tp_cv_avg_price_change_list]
                cv_avg_price_change_list = [np.average(x) for x in top_n_pc_list]
                #

                #
                cv_top_n_pp_list = [x[i] for x in self.tp_cv_pc_pos_percent_list]
                #

                #
                topology_list = list(zip(self.feature_switch_list, self.feature_selected_list,
                                         self.hidden_size_list, cv_iteration_loss_list, cv_polar_accuracy_list,
                                         cv_avg_price_change_list, cv_mres_list, cv_top_n_pp_list, pca_n_component_list))
                topology_list = sorted(topology_list,
                                       key=lambda x: x[-4], reverse=True)
                #

                # modify the path according to how many top N stocks are traded each week

                upper_folder = os.path.dirname(path)
                path_base_name = os.path.basename(path)[:-4]
                new_name = path_base_name + "_top_{}.txt".format(include_top)
                new_path = os.path.join(upper_folder, new_name)
                #

                # save file
                with open(new_path, 'w', encoding='utf-8') as f:
                    for i,tuple1 in enumerate(topology_list):
                        feature_switch = str(tuple1[0])
                        feature_selected = str(tuple1[1])
                        hidden_size = str(tuple1[2])
                        iteration_loss = str(tuple1[3])
                        polar_accuracy = str(tuple1[4])
                        avg_price_change = str(tuple1[5])
                        mres = str(tuple1[6])
                        pos_percent = str(tuple1[7])
                        pca_n_component = str(tuple1[8])
                        # TODO ignore var and std for a while
                        # var = str(tuple1[7])
                        # std = str(tuple1[8])
                        f.write('----------------------------------------------------\n')
                        f.write('id: {}\n'.format(i))
                        f.write('feature_switch: {}\n'.format(feature_switch))
                        f.write('feature_selected: {}\n'.format(feature_selected))
                        f.write('hidden_size: {}\n'.format(hidden_size))
                        f.write('average_iteration_loss: {}\n'.format(iteration_loss))
                        f.write('average_polar_accuracy: {}\n'.format(polar_accuracy))
                        f.write('average_avg_price_change: {}\n'.format(avg_price_change))
                        f.write('average_mres: {}\n'.format(mres))
                        f.write('pos_percent: {}\n'.format(pos_percent))
                        f.write('pca_n_component: {}\n'.format(pca_n_component))
                        # # TODO ignore var and std for a while
                print(
                    "Regression! Save 10-cross-validation topology test result by {} to {} sucessfully!".format(
                        key, new_path))
                # save file

        elif key == 'rmse':
            topology_list = sorted(topology_list,
                                   key=lambda x: x[-3])
            # write to file
            with open(path, 'w', encoding='utf-8') as f:
                for i, tuple1 in enumerate(topology_list):
                    feature_switch = str(tuple1[0])
                    feature_selected = str(tuple1[1])
                    hidden_size = str(tuple1[2])
                    iteration_loss = str(tuple1[3])
                    polar_accuracy = str(tuple1[4])
                    mres = str(tuple1[5])
                    pca_n_component = str(tuple1[6])
                    avg_price_change = str(tuple1[7])
                    # TODO ignore var and std for a while
                    # var = str(tuple1[7])
                    # std = str(tuple1[8])
                    f.write('----------------------------------------------------\n')
                    f.write('id: {}\n'.format(i))
                    f.write('feature_switch: {}\n'.format(feature_switch))
                    f.write('feature_selected: {}\n'.format(feature_selected))
                    f.write('hidden_size: {}\n'.format(hidden_size))
                    f.write('average_iteration_loss: {}\n'.format(iteration_loss))
                    f.write('average_polar_accuracy: {}\n'.format(polar_accuracy))
                    f.write('average_mres: {}\n'.format(mres))
                    f.write('pca_n_component: {}\n'.format(pca_n_component))
                    f.write('avg_price_change: {}\n'.format(avg_price_change))

            print("Regression! Save 10-cross-validation topology test result by {} to {} sucessfully!".format(
                key, path))

        elif key == 'polar':
            topology_list = sorted(topology_list,
                               key=lambda x: x[-4], reverse=True)
            # write to file
            with open(path, 'w', encoding='utf-8') as f:
                for i, tuple1 in enumerate(topology_list):
                    feature_switch = str(tuple1[0])
                    feature_selected = str(tuple1[1])
                    hidden_size = str(tuple1[2])
                    iteration_loss = str(tuple1[3])
                    polar_accuracy = str(tuple1[4])
                    mres = str(tuple1[5])
                    pca_n_component = str(tuple1[6])
                    avg_price_change = str(tuple1[7])
                    # TODO ignore var and std for a while
                    # var = str(tuple1[7])
                    # std = str(tuple1[8])
                    f.write('----------------------------------------------------\n')
                    f.write('id: {}\n'.format(i))
                    f.write('feature_switch: {}\n'.format(feature_switch))
                    f.write('feature_selected: {}\n'.format(feature_selected))
                    f.write('hidden_size: {}\n'.format(hidden_size))
                    f.write('average_iteration_loss: {}\n'.format(iteration_loss))
                    f.write('average_polar_accuracy: {}\n'.format(polar_accuracy))
                    f.write('average_mres: {}\n'.format(mres))
                    f.write('pca_n_component: {}\n'.format(pca_n_component))
                    f.write('avg_price_change: {}\n'.format(avg_price_change))
            print("Regression! Save 10-cross-validation topology test result by {} to {} sucessfully!".format(
                key, path))
                    #

        else:
            print("Key should be mres or avg_pc, key: {}".format(key))

            # ==============================================================================================================
            # avg_pc
            # ==============================================================================================================



            # # TODO ignore var and std for a while
            # _var_list = [[y[0] for y in x] for x in self.cv_var_std_list] # cv_var_std_list : [[(0.1,0.2), ...], [(0.3,0.4), ...], ...]
            # _var_list = [np.average(x) for x in _var_list]
            # _std_list = [[y[1] for y in x] for x in self.cv_var_std_list] # cv_var_std_list : [[(0.1,0.2), ...], [(0.3,0.4), ...], ...]
            # _std_list = [np.average(x) for x in _std_list]
            #

            # ==============================================================================================================


    def _r_print_real_time_best_result(self):
        for i, include_top in enumerate(self.include_top_list):
            # get the cv_avg_price_change_list for a particular strategy
            top_n_pc_list = [[y[i] for y in x] for x in self.tp_cv_avg_price_change_list]
            cv_avg_price_change_list = [np.average(x) for x in top_n_pc_list]
            pca_n_component_list = [x for x in self.tp_cv_pca_n_component_list]
            #
            cv_top_n_pp_list = [x[i] for x in self.tp_cv_pc_pos_percent_list]
            #

            topology_list = list(zip(self.feature_switch_list, self.feature_selected_list,
                                     self.hidden_size_list, cv_avg_price_change_list, pca_n_component_list,
                                     cv_top_n_pp_list))
            sorted_topology_list = sorted(topology_list,
                                          key=lambda x: x[-3], reverse=True)

            top_feature_switch = sorted_topology_list[0][0]
            top_hidden_size = sorted_topology_list[0][2]
            top_cv_avg_price_change = sorted_topology_list[0][3]
            top_pca_n_component = sorted_topology_list[0][4]
            top_n_pp = sorted_topology_list[0][5]

            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("{}-TOP-BEST".format(include_top))
            print("cv_avg_price_change: ", top_cv_avg_price_change)
            print("top_n_pp: ", top_n_pp)
            print("feature_switch: ", top_feature_switch)
            print("hidden_size: ", top_hidden_size)
            print("pca_n_component: ", top_pca_n_component)


    # ------------------------------------------------------------------------------------------------------------------

    def save_evaluate_value_per_random_state(self, i, mrse, avg_price_change_tuple, polar_percent, n_iter, loss):
        if i == 0:
            if self.rsavg_avg_price_change_list or self.rsavg_iteration_list or \
                    self.rsavg_loss_list or self.rsavg_mres_list or self.rsavg_polar_accuracy_list:
                print ("rs list should be empty!")
                sys.exit()

        self.rsavg_avg_price_change_list.append(avg_price_change_tuple)
        self.rsavg_iteration_list.append(n_iter)
        self.rsavg_loss_list.append(loss)
        self.rsavg_mres_list.append(mrse)
        self.rsavg_polar_accuracy_list.append(polar_percent)


    def save_average_evaluate_value(self):

        # --------------------------------------------------------------------------------------------------------------
        # sort by loss, keep N results of different initial weight
        # --------------------------------------------------------------------------------------------------------------
        all = list(zip(self.rsavg_loss_list, self.rsavg_iteration_list, self.rsavg_polar_accuracy_list,
                       self.rsavg_avg_price_change_list, self.rsavg_mres_list))
        sorted_all = sorted(all, key = lambda x:x[0])
        unzip_all = list(zip(*sorted_all))
        keep_num = 3
        self.rsavg_loss_list = unzip_all[0][0:keep_num]
        self.rsavg_iteration_list = unzip_all[1][0:keep_num]
        self.rsavg_polar_accuracy_list = unzip_all[2][0:keep_num]
        self.rsavg_avg_price_change_list = unzip_all[3][0:keep_num]
        self.rsavg_mres_list = unzip_all[4][0:keep_num]
        # --------------------------------------------------------------------------------------------------------------

        self.mres_list.append(np.average(self.rsavg_mres_list))
        avg_price_change_temp = tuple([np.average(x) for x in list(zip(*self.rsavg_avg_price_change_list))])
        self.avg_price_change_list.append(avg_price_change_temp)
        self.polar_accuracy_list.append(np.average(self.rsavg_polar_accuracy_list))
        self.iteration_loss_list.append((np.average(self.rsavg_iteration_list), np.average(self.rsavg_loss_list)))


    def baseline_reg_dev(self, target_folder):
        file1 = os.listdir(target_folder)[0]
        file1_path = os.path.join(target_folder, file1)
        with open(file1_path, 'r') as f:
            feature_list = f.readlines()[0].strip().split(',')[::2]

        key_index = feature_list.index('percent_change_price')
        print ("key_index: ", key_index)

        pred_value_list = []
        for dev_sample in self.dev_set:
            percent_change_price = float(dev_sample[key_index])*0.01
            pred_value_list.append(percent_change_price)

        actual_value_list = np.array(self.dev_value_set)

        date_list = self.dev_date_set
        stock_id_list = self.dev_stock_id_set
        include_top_list = [1]

        avg_price_change_tuple, var_tuple, std_tuple = get_avg_price_change(pred_value_list, actual_value_list,
                                                                                  date_list, stock_id_list,
                                                                                  include_top_list=
                                                                                  include_top_list)

        date_actual_avg_priceChange_list = get_chosen_stock_return(pred_value_list, actual_value_list, date_list,
                          stock_id_list, include_top_list=None)

        # compute accuracy in terms of positive and negative

        # (3.) get the pred label for each week
        pred_label_dict_by_week = collections.defaultdict(lambda :[])
        golden_label_dict_by_week = collections.defaultdict(lambda :[])
        pred_value_dict_by_week = collections.defaultdict(lambda :[])
        golden_value_dict_by_week = collections.defaultdict(lambda :[])


        pred_label_list = ['pos' if x >= 0 else 'neg' for x in pred_value_list ]
        actual_label_list = ['pos' if x >= 0 else 'neg' for x in actual_value_list ]

        for i, pred_label in enumerate(pred_label_list):
            date = self.dev_date_set[i]
            # classification
            pred_label_dict_by_week[date].append(pred_label)
            golden_label = actual_label_list[i]
            golden_label_dict_by_week[date].append(golden_label)
            #
            # regression
            predict_value = pred_value_list[i]
            golden_value = actual_value_list[i]
            pred_value_dict_by_week[date].append(predict_value)
            golden_value_dict_by_week[date].append(golden_value)

        week_average_f1_list = []
        week_average_accuracy_list = []
        week_average_rmse = []
        dev_label_dict = collections.defaultdict(lambda: 0)
        pred_label_dict = collections.defaultdict(lambda: 0)
        label_f1_list_all = []

        # (4.) compute the f1, accuracy for each week in 1 validation set
        for date, pred_label_list_for_1_week in pred_label_dict_by_week.items():
            pred_label_list = pred_label_list_for_1_week
            golden_label_list = golden_label_dict_by_week[date]

            # (3.) compute the average f-measure

            _,average_f1  = compute_average_f1(pred_label_list, golden_label_list)
            week_average_f1_list.append(average_f1)
            #average_f1 = f1_list[0] # using F-measure
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

            # # (6.) save rmse

            pred_value_list1 = pred_value_dict_by_week[date]
            actual_value_list1 = golden_value_dict_by_week[date]
            rmse = calculate_rmse(actual_value_list1, pred_value_list1)
            week_average_rmse.append(rmse)


        week_average_f1 = np.average(week_average_f1_list)
        week_average_accuracy = np.average(week_average_accuracy_list)
        week_average_rmse = np.average(week_average_rmse)


        # # <uncomment for debugging>
        # print("----------------------------------------------------------------------------------------")
        # print("actual_value_list, ", actual_value_list)
        # print("pred_value_list, ", pred_value_list)
        # print("week_average_accuracy: {}".format(week_average_accuracy))
        # print("week_average_f1: {}".format(week_average_f1))
        # print("week_average_rmse: {}".format(week_average_rmse))
        # print("week_average_price_change: {}".format(avg_price_change_tuple))
        # print("----------------------------------------------------------------------------------------")

    # ------------------------------------------------------------------------------------------------------------------
        return week_average_rmse, avg_price_change_tuple, week_average_accuracy, week_average_f1, \
               date_actual_avg_priceChange_list

    def reg_dev_for_moving_window_test(self, save_clsfy_path="mlp_trade_classifier"):

        # (1.) read classifier
        mlp = pickle.load(open(save_clsfy_path, "rb"))
        #
        # (2.) get pred label list
        pred_value_list = mlp.predict(self.dev_set)
        #
        return list(pred_value_list), list(self.dev_value_set), self.dev_date_set, self.dev_stock_id_set
