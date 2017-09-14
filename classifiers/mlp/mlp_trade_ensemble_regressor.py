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
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
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
from mlp_trade_regressor import MlpTradeRegressor
from trade_general_funcs import calculate_rmse
from trade_general_funcs import get_avg_price_change
from trade_general_funcs import create_random_sub_set_list
from trade_general_funcs import build_hidden_layer_sizes_list
# ==========================================================================================================




# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class MlpTradeEnsembleRegressor(MlpTradeRegressor):

    def __init__(self, ensemble_number, mode):
        super().__init__()
        self.ensemble_number =  ensemble_number
        self.mode = mode

    def set_regressor(self, hidden_layer_sizes, tol=1e-8, learning_rate_init=0.001, random_state = 1,
                           verbose = False, learning_rate = 'constant', early_stopping =False, activation  = 'relu',
                           validation_fraction  = 0.1, alpha  = 0.0001, random_state_ensemble = 1):
        self.hidden_size_list.append(hidden_layer_sizes)
        self.mlp_hidden_layer_sizes_list.append(hidden_layer_sizes)
        temp_regressor = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                          tol=tol, learning_rate_init=learning_rate_init,
                                          max_iter=10000, random_state=random_state, verbose=verbose,
                                          learning_rate = learning_rate, early_stopping=early_stopping,
                                          activation = activation, validation_fraction = validation_fraction,
                                          alpha = alpha)
        if self.mode == 'bagging':
            print ("Set bagging EnsembleRegressor!")
            self.mlp_regressor = BaggingRegressor(base_estimator=temp_regressor, n_estimators=self.ensemble_number,
                                                  random_state = random_state_ensemble)
        elif self.mode == 'adaboost':
            print("Set adaboost EnsembleRegressor!")
            self.mlp_regressor = AdaBoostRegressor(base_estimator=temp_regressor, n_estimators=self.ensemble_number,
                                                   random_state = random_state_ensemble)
        else:
            print ("Please type the right mode!!!")
            sys.exit()
        print ("ensemble_number: ", self.ensemble_number)
    # ------------------------------------------------------------------------------------------------------------------


    def regressor_train(self, save_clsfy_path="mlp_trade_ensemble_regressor", is_production=False):
        self.mlp_regressor.fit(self.training_set, self.training_value_set)
        n_iter_list = [x.n_iter_ for x in self.mlp_regressor.estimators_]
        loss_list = [x.loss_ for x in self.mlp_regressor.estimators_]
        #self.iteration_loss_list.append((np.average(n_iter_list), np.average(loss_list)))
        pickle.dump(self.mlp_regressor, open(save_clsfy_path, "wb"))

        if is_production:
            print("classifier for production saved to {} successfully!".format(save_clsfy_path))
        return np.average(n_iter_list), np.average(loss_list)

    # ------------------------------------------------------------------------------------------------------------------
    # [C.1]  Dev
    # ------------------------------------------------------------------------------------------------------------------

    # def regressor_dev(self, save_clsfy_path="mlp_trade_ensemble_regressor", is_cv=False, include_top_list = None):
    #     if not include_top_list:
    #         include_top_list = [1]
    #     mlp_regressor = pickle.load(open(save_clsfy_path, "rb"))
    #     pred_value_list = np.array(mlp_regressor.predict(self.dev_set))
    #     actual_value_list = np.array(self.dev_value_set)
    #     mrse = calculate_rmse(actual_value_list, pred_value_list)
    #     date_list = self.dev_date_set
    #     stock_id_list = self.dev_stock_id_set
    #
    #     avg_price_change_tuple, var_tuple, std_tuple = get_avg_price_change(pred_value_list, actual_value_list,
    #                                                                               date_list, stock_id_list,
    #                                                                               include_top_list=
    #                                                                               include_top_list)
    #
    #     # count how many predicted value has the same polarity as actual value
    #     polar_list = [1 for x, y in zip(pred_value_list, actual_value_list) if x * y >= 0]
    #     polar_count = len(polar_list)
    #     polar_percent = polar_count / len(pred_value_list)
    #     #
    #
    #     # self.mres_list.append(mrse)
    #     # self.avg_price_change_list.append(avg_price_change_tuple)
    #     # self.polar_accuracy_list.append(polar_percent)
    #     # self.var_std_list.append((var_tuple, std_tuple))
    #
    #     # <uncomment for debugging>
    #     if not is_cv:
    #         print("----------------------------------------------------------------------------------------")
    #         print("actual_value_list, ", actual_value_list)
    #         print("pred_value_list, ", pred_value_list)
    #         print("polarity: {}".format(polar_percent))
    #         print("mrse: {}".format(mrse))
    #         print("avg_price_change: {}".format(avg_price_change_tuple))
    #         print("----------------------------------------------------------------------------------------")
    #     else:
    #         pass
    #         # print("Testing complete! Testing Set size: {}".format(len(self.r_dev_value_set)))
    #         # <uncomment for debugging>
    #
    #     return mrse, avg_price_change_tuple, polar_percent
    # # ------------------------------------------------------------------------------------------------------------------

