# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# MLP classifier only for trading, currently for a share stock and forex
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general import
# ==========================================================================================================
import collections
import pickle
import numpy as np
import sys
import os
import math
from sklearn.neural_network import MLPClassifier
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
from trade_general_funcs import build_hidden_layer_sizes_list
from trade_general_funcs import compute_average_f1
from trade_general_funcs import compute_trade_weekly_clf_result
# ==========================================================================================================




# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class MlpClassifier_P(MultilayerPerceptron):

    def __init__(self):
        super().__init__()

        # --------------------------------------------------------------------------------------------------------------
        # container for evaluation
        # --------------------------------------------------------------------------------------------------------------
        self.accuracy_list = []
        self.average_f1_list = []
        self.label_tp_fp_tn_dict = {}
        # --------------------------------------------------------------------------------------------------------------


        # --------------------------------------------------------------------------------------------------------------
        # container for n-fold validation for different hidden layer, tp is topology, cv is cross-validation
        # --------------------------------------------------------------------------------------------------------------
        self.tp_cv_average_average_f1_list = []
        self.tp_cv_average_accuracy_list = []
        # --------------------------------------------------------------------------------------------------------------

    def set_mlp_clf(self, hidden_layer_sizes, tol=1e-8, learning_rate_init=0.0001, random_state=1,
                           verbose = False, learning_rate = 'constant', early_stopping =False, activation  = 'relu',
                           validation_fraction  = 0.1, alpha  = 0.0001):
        self.hidden_size_list.append(hidden_layer_sizes)
        self.mlp_hidden_layer_sizes_list.append(hidden_layer_sizes)
        self.mlp_clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                     tol=tol, learning_rate_init=learning_rate_init,
                                     max_iter=10000, random_state=random_state, verbose=verbose, learning_rate =
                                     learning_rate, early_stopping =early_stopping, alpha= alpha,
                                              validation_fraction = validation_fraction, activation = activation)



    def clf_train(self, save_clsfy_path="mlp_trade_classifier", is_production=False):
        self.mlp_clf.fit(self.training_set, self.training_value_set)
        #self.iteration_loss_list.append((self.mlp_clf.n_iter_, self.mlp_clf.loss_))

        # try:
        #     self.mlp_clf.fit(self.training_set, self.training_label)
        # except ValueError:
        #     print ("feature_switch_list: ", self.feature_switch_list)
        #     #logger2.error("training_set: {}".format(self.training_set))
        #     sys.exit()

        pickle.dump(self.mlp_clf, open(save_clsfy_path, "wb"))
        return self.mlp_clf.n_iter_, self.mlp_clf.loss_




    def clf_dev(self, save_clsfy_path = "mlp_trade_classifier", is_cv = False, is_return = False):

        # (1.) read classifier
        mlp = pickle.load(open(save_clsfy_path, "rb"))
        #

        # (2.) get pred label list
        pred_label_list = mlp.predict(self.dev_set)
        actual_label_list = self.dev_value_set
        data_list = self.dev_date_set
        #

        week_average_f1, week_average_accuracy, dev_label_dict, pred_label_dict = \
            compute_trade_weekly_clf_result(pred_label_list,actual_label_list,data_list)


        # print
        if not is_cv:
            print("\n=================================================================")
            print("Dev set result!")
            print("=================================================================")
            print("dev_label_dict: {}".format(list(dev_label_dict.items())))
            print("pred_label_dict: {}".format(list(pred_label_dict.items())))
            print("week_average_f1: ", week_average_f1)
            print("week_average_accuracy: ", week_average_accuracy)
            print("=================================================================")
            return week_average_f1, week_average_accuracy
        #
        if is_return:
            return week_average_f1, week_average_accuracy





    def baseline_clf_dev(self, target_folder):
        file1 = os.listdir(target_folder)[0]
        file1_path = os.path.join(target_folder, file1)
        with open(file1_path, 'r') as f:
            feature_list = f.readlines()[0].strip().split(',')[::2]

        key_index = feature_list.index('percent_change_price')
        print ("key_index: ", key_index)

        pred_label_list = []
        for dev_sample in self.dev_set:
            percent_change_price = float(dev_sample[key_index])
            if percent_change_price >= 0:
                pred_label_list.append('pos')
            else:
                pred_label_list.append('neg')


        # (3.) compute the average f-measure
        pred_label_dict = collections.defaultdict(lambda: 0)
        for pred_label in pred_label_list:
            pred_label_dict[pred_label] += 1
        _, average_f1= compute_average_f1(pred_label_list, self.dev_value_set)
        #

        # (4.) compute accuracy
        correct = 0
        for i, pred_label in enumerate(pred_label_list):
            if pred_label == self.dev_value_set[i]:
                correct += 1
        accuracy = correct / len(self.dev_value_set)
        #

        # (5.) count the occurrence for each label
        dev_label_dict = collections.defaultdict(lambda: 0)
        for dev_label in self.dev_value_set:
            dev_label_dict[dev_label] += 1
        #


        print("\n=================================================================")
        print("Dev set result!")
        print("=================================================================")
        print("dev_label_dict: {}".format(list(dev_label_dict.items())))
        print("pred_label_dict: {}".format(list(pred_label_dict.items())))
        #print("label_f1_list: {}".format(label_f1_list))
        print("average_f1: ", average_f1)
        print("accuracy: ", accuracy)
        print("=================================================================")
        #
        return average_f1, accuracy

    def general_cv_cls_topology_test(self, input_folder, feature_switch_tuple, other_config_dict,
                             hidden_layer_config_tuple, is_random=False):
        '''10 cross validation test for mlp classifier'''


        # :::topology_test:::


        # ==============================================================================================================
        # Cross Validation Train And Test
        # ==============================================================================================================

        # feature switch tuple
        if feature_switch_tuple:
            self.feature_switch_list.append(feature_switch_tuple)

        # (1.) read the whole data set
        # cut the number of training sample
        data_per = other_config_dict['data_per']
        dev_per = other_config_dict['dev_per']
        samples_feature_list, samples_value_list= self._general_feed_data(input_folder, data_per=data_per,
                                                       feature_switch_tuple=feature_switch_tuple,
                                                       is_random = False, mode = 'clf')
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
        random_seed_list = other_config_dict['random_seed_list']

        # (4.) data pre-processing
        is_standardisation = other_config_dict['is_standardisation']
        is_PCA = other_config_dict['is_PCA']

        print ("Random_seed_list: {}".format(random_seed_list))
        # --------------------------------------------------------------------------------------------------------------


        # --------------------------------------------------------------------------------------------------------------
        # (4.) create validation_dict
        # --------------------------------------------------------------------------------------------------------------
        random_seed_list = other_config_dict['random_seed_list']
        for random_seed in random_seed_list:
            self.create_train_dev_vdict_general(samples_feature_list, samples_value_list, random_seed, is_cv=True,
                                           data_per = data_per, dev_per = dev_per,
                                         is_standardisation = is_standardisation, is_PCA = is_PCA)
        # --------------------------------------------------------------------------------------------------------------



        # (4.) test the performance of different topology of MLP by 10-cross validation
        for i, hidden_layer_sizes in enumerate(hidden_layer_sizes_list):
            print("====================================================================")
            print("Topology: {} starts training and testing".format(hidden_layer_sizes))
            print("====================================================================")

            self._update_feature_switch_list(i)

            # (a.) clear the evaluation list for one hidden layer topology
            self.iteration_loss_list = []
            self.average_f1_list = []
            self.accuracy_list = []
            #

            # (b.) train and dev
            self.set_mlp_clf(hidden_layer_sizes, learning_rate_init=learning_rate_init, tol=tol)

            # random inside, make sure each date has all the

            for random_seed in random_seed_list:
                for cv_index in self.validation_dict[random_seed].keys():
                    self.rs_cv_load_train_dev_data(random_seed, cv_index)
                    self.clf_train(save_clsfy_path=clf_path)
                    self.clf_dev(save_clsfy_path=clf_path, is_cv=True)
            #

            # (c.) save the 10-cross-valiation evaluate result for each topology

            self.tp_cv_iteration_loss_list.append([x[1] for x in self.iteration_loss_list])
            self.tp_cv_average_average_f1_list.append(self.average_f1_list)
            self.tp_cv_average_accuracy_list.append(self.accuracy_list)

            # (d.) real-time print
            print("====================================================================")
            print("Feature selected: {}, Total number: {}".format(self.feature_selected_list[-1],
                                                                  self.feature_switch_list[-1].count(1)))
            print("Average avg f1: {}".format(np.average(self.average_f1_list)))
            print("Average accuracy: {}".format(np.average(self.accuracy_list)))
            print("Average iteration_loss: {}".format(np.average([x[1] for x in self.iteration_loss_list])))
            print("====================================================================")
            print("Completeness: {:.5f}".format((i + 1) / hidden_layer_sizes_combination))
            print("====================================================================")
            #
