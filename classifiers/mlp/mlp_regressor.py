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
# ==========================================================================================================




# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class MlpRegressor_P(MultilayerPerceptron):

    def __init__(self):
        super().__init__()

        # --------------------------------------------------------------------------------------------------------------
        # container for evaluation config and result
        # --------------------------------------------------------------------------------------------------------------
        self.mres_list = []
        # --------------------------------------------------------------------------------------------------------------


        # --------------------------------------------------------------------------------------------------------------
        # container for n-fold validation for different hidden layer, tp is topology, cv is cross-validation
        # --------------------------------------------------------------------------------------------------------------
        self.tp_cv_mres_list = []
        # --------------------------------------------------------------------------------------------------------------



    def set_regressor(self, hidden_layer_sizes, tol=1e-8, learning_rate_init=0.0001, random_state=1,
                           verbose = False, learning_rate = 'constant', early_stopping =False, activation  = 'relu',
                           validation_fraction  = 0.1, alpha  = 0.0001):
        self.hidden_size_list.append(hidden_layer_sizes)
        self.mlp_hidden_layer_sizes_list.append(hidden_layer_sizes)
        self.mlp_regressor = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                          tol=tol, learning_rate_init=learning_rate_init,
                                          max_iter=10000, random_state=random_state, verbose=verbose,
                                          learning_rate = learning_rate, early_stopping=early_stopping,
                                          activation = activation, validation_fraction = validation_fraction,
                                          alpha = alpha)

    # ------------------------------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------------------------------
    # [C.1] Train and Dev
    # ------------------------------------------------------------------------------------------------------------------
    def regressor_train(self, save_clsfy_path="mlp_trade_regressor", is_production=False):
        self.mlp_regressor.fit(self.training_set, self.training_value_set)
        self.iteration_loss_list.append((self.mlp_regressor.n_iter_, self.mlp_regressor.loss_))
        pickle.dump(self.mlp_regressor, open(save_clsfy_path, "wb"))

        if is_production:
            print("classifier for production saved to {} successfully!".format(save_clsfy_path))
        return self.mlp_regressor.n_iter_, self.mlp_regressor.loss_


    def regressor_train_test(self, training_set, training_value_set, save_clsfy_path="mlp_trade_regressor", is_production=False):
        self.mlp_regressor.fit(training_set, training_value_set)
        return self.mlp_regressor.n_iter_, self.mlp_regressor.loss_