# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Sk learn PCA
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general package import
# ==========================================================================================================
import sys
import os
from sklearn.decomposition import PCA
import numpy as np
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


class StockPca:
    def __init__(self, n_components):
        self.pca = PCA(n_components=n_components)
        self.n_components = n_components
        print ("Build stock PCA complete! N components: {}".format(self.n_components))

    def fit_data(self,data):
        # data -> ndarray #np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        self.pca.fit(data)

    def transfrom_data(self, data):
        trans_data = self.pca.transform(data)
        return trans_data

    def fit_transform(self, data):
        trans_data = self.pca.fit_transform(data)
        return trans_data

    def transfrom_data_by_pca(self, input_folder, save_folder):
        print ("Start transforming data from {}".format(input_folder))
        file_name_list = os.listdir(input_folder)
        file_path_list = [os.path.join(input_folder, x) for x in file_name_list]
        save_file_path_list = [os.path.join(save_folder, x) for x in file_name_list]
        data_feature_set = []
        for file_path in file_path_list:
            with open (file_path, 'r', encoding = 'utf-8') as f:
                feature_value_list = f.readlines()[0].split(',')[1::2]
                data_feature_set.append(feature_value_list)
        print ("push all sample's feature to data_feature_set succesful! Total: {}"
               .format(len(data_feature_set)))

        id_list = ["[{}]".format(i) for i in range(self.n_components)]
        trans_feature_set = self.fit_transform(data_feature_set) #type: ndarray [[1][2]]
        for i, pca_feature in enumerate(trans_feature_set):
            pca_feature_list = list(pca_feature)
            pca_feature_list = [str(x) for x in pca_feature_list]
            save_list = [j for i in zip(id_list,pca_feature_list) for j in i]
            save_str = ','.join(save_list)
            save_file_path = save_file_path_list[i]
            with open (save_file_path, 'w', encoding = 'utf-8') as f:
                f.write(save_str)
        print ("PCA done! Output folder: {}".format(save_folder))



