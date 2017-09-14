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
from sklearn.ensemble import RandomForestClassifier

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
from trade_general_funcs import compute_trade_weekly_clf_result
# ==========================================================================================================

from mlp_trade_classifier import MlpTrade


class RandomForestClassifier_P(MlpTrade):

    def __init__(self):
        super().__init__()
        pass

    def set_mlp_clf(self, n_estimators = 10,
                          max_features = 'auto',
                          min_samples_split = 2,
                          min_samples_leaf =2,
                          random_state = 0,
                          n_jobs = 1,
                          oob_score = False):
        self.mlp_clf = RandomForestClassifier(n_estimators = n_estimators,
                                              max_features = max_features,
                                              min_samples_split = min_samples_split,
                                              min_samples_leaf = min_samples_leaf,
                                              random_state=random_state,
                                              n_jobs = n_jobs,
                                              oob_score = oob_score)

    def clf_train(self, save_clsfy_path="rf_trade_classifier"):
        self.mlp_clf.fit(self.training_set, self.training_value_set)
        pickle.dump(self.mlp_clf, open(save_clsfy_path, "wb"))
        return None

    def clf_dev(self, save_clsfy_path="rf_trade_classifier"):
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

        return week_average_f1, week_average_accuracy