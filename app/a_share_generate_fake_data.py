# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# The complete PCA CV test for a share MLP regressor
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
g_d_path = os.path.join(parent_folder, 'data_generator')
sys.path.append(g_d_path)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from gaussian_data import RandomGaussianData
# ==========================================================================================================


reg_data_folder_path = os.path.join(parent_folder, 'data', 'a_share', 'a_share_regression_PCA_data')
clf_data_folder_path = os.path.join(parent_folder, 'data', 'a_share', 'a_share_labeled_PCA_data')

# save folder
reg_data_save_folder_path = os.path.join(parent_folder, 'data', 'a_share', 'a_share_regression_PCA_data_fake')
clf_data_save_folder_path = os.path.join(parent_folder, 'data', 'a_share', 'a_share_labeled_PCA_data_fake')

# gaussian
rgd_1 = RandomGaussianData()

# # ----------------------------------------------------------------------------------------------------------------------
# # write the fake data for classification
# # ----------------------------------------------------------------------------------------------------------------------
# clf_file_name_list = os.listdir(clf_data_folder_path)
# clf_file_path_list = [os.path.join(clf_data_folder_path, x) for x in clf_file_name_list]
# clf_file_save_path_list = [os.path.join(clf_data_save_folder_path, x) for x in clf_file_name_list]
#
# for i, clf_file_path in enumerate(clf_file_path_list):
#     input_path = clf_file_path
#     output_path = clf_file_save_path_list[i]
#     rgd_1.generate_fake_clf_data_for_stock(input_path, output_path)
#     if i % 1000 == 0:
#         print ("Generate fake data {} for clf successfully!".format(i))
# # ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# write the fake data for regression
# ----------------------------------------------------------------------------------------------------------------------

# (0.) delete reg files
old_reg_file_name = os.listdir(reg_data_save_folder_path)
remove_count = 0
for file in old_reg_file_name:
    file = os.path.join(reg_data_save_folder_path, file)
    os.remove(file)
    remove_count += 1
print ("Remove {} files from {}".format(remove_count, reg_data_save_folder_path))


reg_file_name_list = os.listdir(reg_data_folder_path)
reg_file_path_list = [os.path.join(reg_data_folder_path, x) for x in reg_file_name_list]
reg_file_save_path_list = [os.path.join(reg_data_save_folder_path, x) for x in reg_file_name_list]

for i, reg_file_path in enumerate(reg_file_path_list):
    input_path = reg_file_path
    output_path = reg_file_save_path_list[i]
    rgd_1.generate_fake_reg_data_for_stock(input_path, output_path)
    if i % 1000 == 0:
        print ("Generate fake data {} for reg successfully!".format(i))
# ----------------------------------------------------------------------------------------------------------------------

















