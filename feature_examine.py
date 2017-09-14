import re
import sys
import os
import numpy as np
import collections


class FeatureExamine:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        # feature dict contains all the values from the dataset(not one sample!) for each features
        self.feature_dict = collections.defaultdict(lambda:[])
        self.feature_std_dict = collections.defaultdict(lambda: 0)
        self.feature_duplicate_count_dict = collections.defaultdict(lambda: 0)
        self.zero_count_dict = collections.defaultdict(lambda: 0)
        self._construct_feature_dict()
        
    def _construct_file_path_list(self):
        file_name_list = os.listdir(self.input_folder)
        file_path_list = [os.path.join(self.input_folder, x) for x in file_name_list]
        return file_path_list
        
    def _construct_feature_dict(self):
        file_path_list = self._construct_file_path_list()
        for file_path in file_path_list:
            with open (file_path, 'r', encoding = 'utf-8') as f:
                feature_n_v_list = f.readlines()[0].strip().split(',')
                feature_name_list = feature_n_v_list[::2]
                feature_value_list = feature_n_v_list[1::2]
                for i, f_n in enumerate(feature_name_list):
                    f_v = float(feature_value_list[i])
                    self.feature_dict[f_n].append(f_v)
        print ("Construct feature dict successfully! Features: {}".format(self.feature_dict.keys()))
        
    def compute_std(self):
        for feature, f_value_list in self.feature_dict.items():
            feature_var = np.std(f_value_list)
            self.feature_std_dict[feature] = feature_var
            
        self._print_result()
            
    def count_duplicate(self):
        from itertools import groupby
        for feature, f_value_list in self.feature_dict.items():
            count_dict = dict(collections.Counter(f_value_list))
            count_sum = 0
            for value, value_count in count_dict.items():
                if value_count == 1:
                    continue
                else:
                    count_sum += value_count
            self.feature_duplicate_count_dict[feature] = count_sum
        self._print_result(mode = 'd_count')
            
    def count_zero(self):
        from itertools import groupby
        for feature, f_value_list in self.feature_dict.items():
            zero_count = f_value_list.count(0)
            self.zero_count_dict[feature] = zero_count
        self._print_result(mode = 'z_count')
            
    def _print_result(self, mode = 'std'):
        print ("-------------------------------------------------------------------")
        if mode == 'std':
            feature_std_sorted_list = sorted(list(self.feature_std_dict.items()), key = lambda x:x[1])
            for feature, f_std in feature_std_sorted_list:
                print ("Feature: {}, Std: {}".format(feature, f_std))
        elif mode == 'd_count':
            s_fnvcl = sorted(list(self.feature_duplicate_count_dict.items()), key = lambda x:x[1], reverse = True)
            for feature, u_v_c in s_fnvcl:
                print ("Feature: {}, duplicate count: {}".format(feature, u_v_c))
        elif mode == 'z_count':
            s_zcl = sorted(list(self.zero_count_dict.items()), key = lambda x:x[1], reverse = True)
            for feature, z_c in s_zcl:
                print ("Feature: {}, zero count: {}".format(feature, z_c))
                
                
# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.abspath(__file__))
regression_folder = os.path.join(parent_folder, 'data', 'a_share', 'a_share_scaled_data')
# ==========================================================================================================
    
print ("Examine Folder: {}".format(regression_folder))

f_e_1 = FeatureExamine(regression_folder)
# (1.) compute std
f_e_1.compute_std()
# (2.) compute unique count
f_e_1.count_duplicate()
# (3.) compute zero count
f_e_1.count_zero()