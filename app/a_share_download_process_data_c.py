# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# [1.] Download the a-share raw data.
# [2.] Manually add features to a-share data.
# [3.] Clean data. <a> get rid of nan feature value.
# [4.1] Data pre-processing and label the data. <a> PCA <b> scaling
# [4.2] label the data directly
# [4.3] write the regression value of the data
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
data_processor_path = os.path.join(parent_folder, 'data_processor')
data_generator_path = os.path.join(parent_folder, 'data_generator')
sys.path.append(data_processor_path)
sys.path.append(data_generator_path)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from data_preprocessing import DataPp
from stock_pca import StockPca
from a_share import Ashare
# ==========================================================================================================



# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



# ==========================================================================================================
# [1.] Download the raw data
# ==========================================================================================================
a_share1 = Ashare(is_stock_set = True)
start_date = '2016-07-01'
save_folder = os.path.join(parent_folder, 'data', 'a_share', 'a_share_raw_data')
a_share1.read_a_share_history_date(save_folder, start_date = start_date, is_filter_new_stock = True)
# ==========================================================================================================



# ==========================================================================================================
# [2.] Manually add features to a-share data.
# ==========================================================================================================
a_share1 = Ashare()
input_folder = os.path.join(parent_folder, 'data', 'a_share', 'a_share_raw_data')
save_folder = os.path.join(parent_folder, 'data', 'a_share', 'a_share_f_engineered_data')
a_share1.feature_engineering(input_folder , save_folder)
# ==========================================================================================================


# ==========================================================================================================
# [3.] Clean data. <a> get rid of nan feature value.
# ==========================================================================================================
data_cleaner = DataPp()
input_folder = 'a_share_f_engineered_data'
input_folder = os.path.join(parent_folder, 'data', 'a_share', input_folder)
save_folder = 'a_share_processed_data'
save_folder = os.path.join(parent_folder, 'data', 'a_share', save_folder)
data_cleaner.correct_non_float_feature(input_folder, save_folder)
#data_cleaner.examine_data(input_folder)  # examine the feature to see whether it is float

# ==========================================================================================================


# # # ==========================================================================================================
# # # [4.b] scaling, min-max
# # # ==========================================================================================================
# data_cleaner = DataPp()
# input_folder = 'a_share_processed_data'
# input_folder = os.path.join(parent_folder, 'data', 'a_share', input_folder)
# save_folder = 'a_share_scaled_data'
# save_folder = os.path.join(parent_folder, 'data', 'a_share', save_folder)
# features_scale_list = []
#
# # features_scale_list
# features_scale1 = [('bvps', 'esp', 'fixedAssets', 'gpr',
#                     'holders', 'liquidAssets', 'npr',
#                     'outstanding', 'pb', 'pe', 'perundp',
#                     'profit', 'reserved', 'reservedPerShare'
#                     ,'rev','totalAssets', 'totals', 'undp',
#                     'volume'),(0,1)]
# features_scale_list = [features_scale1]
#
# # scale data and save scaler
# trained_classifiers_folder = os.path.join(parent_folder, 'trained_data_processor')
# scaler_name = 'a_share_scaler'
#
# data_cleaner.scale_data(input_folder, save_folder,  features_scale_list, trained_classifiers_folder, scaler_name)
# # # ==========================================================================================================


# # ==========================================================================================================
# # [4.c] scaling, z-score
# # ==========================================================================================================
data_cleaner = DataPp()
input_folder = 'a_share_processed_data'
input_folder = os.path.join(parent_folder, 'data', 'a_share', input_folder)
save_folder = 'a_share_scaled_data'
save_folder = os.path.join(parent_folder, 'data', 'a_share', save_folder)


features_scale_list = ''

# scale data and save scaler
trained_classifiers_folder = os.path.join(parent_folder, 'trained_data_processor')
scaler_name = 'a_share_z_score_scaler'

data_cleaner.scale_data(input_folder, save_folder,  features_scale_list,
                        trained_classifiers_folder, scaler_name, mode = 'z_score')
# # ==========================================================================================================


# # ==========================================================================================================
# # [4.] Label the data without PCA, etc.
# # ==========================================================================================================
a_share1 = Ashare()
input_folder = os.path.join(parent_folder, 'data', 'a_share', 'a_share_processed_data')
save_folder = os.path.join(parent_folder, 'data', 'a_share', 'a_share_labeled_data')
a_share1.label_data(input_folder, save_folder)
# # ==========================================================================================================


# # ==========================================================================================================
# # [4.a] PCA-clf
# # ==========================================================================================================
# total n_components for a share: 29, without price change 28
stock_pca1 = StockPca(n_components = 17)
input_folder = os.path.join(parent_folder, 'data', 'a_share', 'a_share_labeled_data')
save_folder = os.path.join(parent_folder, 'data', 'a_share', 'a_share_labeled_PCA_data')
stock_pca1.transfrom_data_by_pca(input_folder, save_folder)
# # ==========================================================================================================
