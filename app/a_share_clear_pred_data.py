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

raw = os.path.join(parent_folder, 'data', 'a_share', '[pred]_a_share_raw_data')
f_engineered = os.path.join(parent_folder, 'data', 'a_share', '[pred]_a_share_f_engineered_data')
processed = os.path.join(parent_folder, 'data', 'a_share', '[pred]_a_share_processed_data')
prediction = os.path.join(parent_folder, 'data', 'a_share', '[pred]_a_share_prediction_data')
scaled = os.path.join(parent_folder, 'data', 'a_share', '[pred]_a_share_scaled_data')

delete_folder_list = [raw, f_engineered, processed, prediction, scaled]

for folder in delete_folder_list:
    remove_count = 0
    delete_file_names = os.listdir(folder)
    for file_name in delete_file_names:
        file_path = os.path.join(folder, file_name)
        os.remove(file_path)
        remove_count += 1
    print ("Totally remove {} files in folder {}".format(remove_count, os.path.basename(folder)))

