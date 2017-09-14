# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Build the MLP regressor. Train the classifier based on a share history data and test on development set.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general package import
# ==========================================================================================================
import sys
import os
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlp_path = os.path.join(parent_folder, 'classifiers','mlp')
sys.path.append(mlp_path)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from mlp_trade_regressor import MlpTradeRegressor
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



# ==========================================================================================================
# Build MLP classifier for a-share data, save the mlp to local
# ==========================================================================================================

print("====================================================================")
print("Building classifier for production...")
print("====================================================================")

# (1.) build classifer
mlp_regressor1 = MlpTradeRegressor()
hidden_layer_sizes = (99, 8)
learning_rate_init = 0.0001
mlp_regressor1.set_regressor(hidden_layer_sizes, learning_rate_init = learning_rate_init)


# (2.) feed data
data_per = 1.0  # the percentage of data using for training and testing
dev_per = 0.2  # the percentage of data using for training and testing
data_folder = os.path.join('a_share','a_share_regression_PCA_data')
data_folder = os.path.join(parent_folder, 'data', data_folder)
mlp_regressor1.trade_feed_and_separate_data(data_folder, data_per = data_per, dev_per = dev_per, is_production = True)


# (3.) train and test
clsfy_name = 'a_share_mlp_regressor_production_ver1_0'
mlp_path = os.path.join(parent_folder, 'production_classifiers', clsfy_name)
mlp_regressor1.regressor_train(save_clsfy_path= mlp_path, is_production = True)
# only to test , not necessary
mlp_regressor1.regressor_dev(save_clsfy_path= mlp_path)
# ==========================================================================================================