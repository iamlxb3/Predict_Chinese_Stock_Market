# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# MLP classifier only for trading, currently for a share stock and forex
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general import
# ==========================================================================================================
import pickle
import collections
import numpy as np
import sys
import math
import os
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
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
from mlp_trade_classifier import MlpTradeClassifier
from trade_general_funcs import compute_average_f1
# ==========================================================================================================




# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class MlpTradeDataEnsembleClassifier(MlpTradeClassifier):
    '''Ensemble classifier of different data'''

    def __init__(self, ensemble_number, mode):
        super().__init__()
        self.ensemble_number = ensemble_number
        self.mode = mode

    def set_mlp_clf(self, hidden_layer_sizes, tol=1e-8, learning_rate_init=0.0001, random_state=1,
                           verbose = False, learning_rate = 'constant', early_stopping =False, activation  = 'relu',
                           validation_fraction  = 0.1, alpha  = 0.0001):

        self.hidden_size_list.append(hidden_layer_sizes)
        self.mlp_hidden_layer_sizes_list.append(hidden_layer_sizes)

        temp_clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                     tol=tol, learning_rate_init=learning_rate_init,
                                     max_iter=10000, random_state=random_state, verbose=verbose, learning_rate =
                                     learning_rate, early_stopping =early_stopping, alpha= alpha,
                                              validation_fraction = validation_fraction, activation = activation)

        if self.mode == 'bagging':
            print("Set bagging EnsembleClassifier!")
            self.clf = BaggingClassifier(base_estimator=temp_clf,  verbose=0, n_estimators=self.ensemble_number)
        elif self.mode == 'adaboost':
            print("Set adaboost EnsembleClassifier!")
            self.clf = AdaBoostClassifier(base_estimator=temp_clf, n_estimators=self.ensemble_number)
        else:
            print ("Please type the right mode!!!")
            sys.exit()
        print ("ensemble_number: ", self.ensemble_number)


    def clf_train(self, save_clsfy_path="mlp_trade_classifier", is_production=False):

        # split data and train

        self.clf.fit(self.training_set, self.training_value_set)
        n_iter_list = [x.n_iter_ for x in self.clf.estimators_]
        loss_list = [x.loss_ for x in self.clf.estimators_]

        #self.iteration_loss_list.append((np.average(n_iter_list), np.average(loss_list)))

        pickle.dump(self.clf, open(save_clsfy_path, "wb"))
        return np.average(n_iter_list), np.average(loss_list)

    # def clf_dev(self, save_clsfy_path="mlp_trade_classifier", is_cv=False, is_return=False):
    #
    #     save_clsfy_path = save_clsfy_path + '_data_ensemble'
    #
    #     # (1.) read classifier
    #     clf = pickle.load(open(save_clsfy_path, "rb"))
    #     #
    #     pred_label_list = clf.predict(self.dev_set)
    #
    #     # (3.) compute the average f-measure
    #     pred_label_dict = collections.defaultdict(lambda: 0)
    #     for pred_label in pred_label_list:
    #         pred_label_dict[pred_label] += 1
    #     label_tp_fp_tn_dict = compute_average_f1(pred_label_list, self.dev_value_set)
    #     label_f1_list = sorted([(key, x[3]) for key, x in label_tp_fp_tn_dict.items()])
    #     f1_list = [x[1] for x in label_f1_list]
    #     average_f1 = np.average(f1_list)
    #     #
    #
    #     # (4.) compute accuracy
    #     correct = 0
    #     for i, pred_label in enumerate(pred_label_list):
    #         if pred_label == self.dev_value_set[i]:
    #             correct += 1
    #     accuracy = correct / len(self.dev_value_set)
    #     #
    #
    #     # (5.) count the occurrence for each label
    #     dev_label_dict = collections.defaultdict(lambda: 0)
    #     for dev_label in self.dev_value_set:
    #         dev_label_dict[dev_label] += 1
    #     #
    #
    #     # (6.) save result for 1-fold
    #     self.average_f1_list.append(average_f1)
    #     self.accuracy_list.append(accuracy)
    #     #
    #
    #     # print
    #     if not is_cv:
    #         print("\n=================================================================")
    #         print("Dev set result!")
    #         print("=================================================================")
    #         print("dev_label_dict: {}".format(list(dev_label_dict.items())))
    #         print("pred_label_dict: {}".format(list(pred_label_dict.items())))
    #         print("label_f1_list: {}".format(label_f1_list))
    #         print("average_f1: ", average_f1)
    #         print("accuracy: ", accuracy)
    #         print("=================================================================")
    #     #
    #
    #     if is_return:
    #         return average_f1, accuracy
    #


