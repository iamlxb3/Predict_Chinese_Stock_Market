# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# clear all the a share data
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# ==========================================================================================================
# general package import
# ==========================================================================================================
import os
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
# ==========================================================================================================



# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


delete_folder_name_list = [
                           'a_share_raw_data',
                           'a_share_f_engineered_data',
                           'a_share_processed_data',
                           'a_share_scaled_data',
                           'a_share_regression_PCA_data',
                           'a_share_regression_data',
                           'a_share_labeled_data',
                           'a_share_labeled_PCA_data',
                          ]

for delete_folder_name in delete_folder_name_list:
    delete_folder_path = os.path.join(parent_folder, 'data', 'a_share', delete_folder_name)
    remove_count = 0
    delete_file_names = os.listdir(delete_folder_path)
    for file_name in delete_file_names:
        file_path = os.path.join(delete_folder_path, file_name)
        os.remove(file_path)
        remove_count += 1
    print ("Totally remove {} files in folder {}".format(remove_count, os.path.basename(delete_folder_path)))

