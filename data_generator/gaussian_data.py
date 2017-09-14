import numpy as np
import os
import re
import sys
import math

class RandomGaussianData:
    def __init__(self):
        pass


    def generate_fake_clf_data_for_stock(self, input_path, output_path):

        with open(input_path, 'r', encoding = 'utf-8') as f:
            polarity = re.findall(r'_([A-Za-z]+).txt', input_path)[0]
            if polarity == 'pos':
                mu_list = [1,3,5,-3,-1,0.5,0.4,0.3,0.2,0.1,0]
                sigma_list = [0.2,0.2,0.1,0.5,0.3,0.01,0.01,0.01,0.01,0.01,0.01]
            elif polarity == 'neg':
                mu_list = [-1,1,3,-1,0,0.6,0.5,0.4,0.3,0.2,0.1]
                sigma_list = [0.1,0.1,0.2,0.2,0.2,0.02,0.02,0.02,0.02,0.02,0.02]
            else:
                print ("Error! Polarity should be pos or neg, input: {}".format(polarity))
                sys.exit()

            feature_value_list = f.readlines()[0].strip().split(',')
            feature_list = feature_value_list[::2]
            value_list = feature_value_list[1::2]
            feature_num = len(feature_list)

            # gaussian
            for i, (mu, sigma) in enumerate(zip(mu_list, sigma_list)):
                value = np.random.normal(mu, sigma)
                value_list[i] = value
            #

            # add noise to the rest features
            rest_feature_number = feature_num - len(mu_list)
            for i in range(rest_feature_number):
                feature_index = len(mu_list) + i
                value_list[feature_index] = np.random.random()
            #

            if feature_index != feature_num - 1:
                print ("Something is wrong")

            value_list = [str(x) for x in value_list]
            new_feature_value_list = [j for i in zip(feature_list, value_list) for j in i]

        with open(output_path, 'w', encoding = 'utf-8') as f:
            new_feature_value_str = ','.join(new_feature_value_list)
            f.write(new_feature_value_str)


    def generate_fake_reg_data_for_stock(self, input_path, output_path):


        with open(input_path, 'r', encoding = 'utf-8') as f:
            price_change = re.findall(r'#([0-9.\-e]+)#', input_path)[0]
            orginal_f_str = f.readlines()[0]
            feature_value_list = orginal_f_str.strip().split(',')
            feature_list = feature_value_list[::2]
            value_list = feature_value_list[1::2]
            value_list = [float(x) for x in value_list]
            feature_num = len(feature_list)

            new_price_change = 0
            for i in range(feature_num):
                if i == 0:
                    new_price_change += value_list[i] ** 2
                elif i == 1:
                    new_price_change += value_list[i] * 9
                elif i == 2:
                    new_price_change += math.log(abs(value_list[i]))
                elif i == 3:
                    new_price_change += math.log(abs(value_list[i])) * 2
                elif i == 4:
                    new_price_change += value_list[i] / 2
                elif i == 5:
                    new_price_change += math.log(abs(((value_list[i] +10) / 2))) ** 3
                else:
                    new_price_change += value_list[i]

            # standard Sigmoid function
            #new_price_change = 1 / (1 + math.e ** -new_price_change)


            new_output_path = output_path.replace(price_change, "{:.5f}".format(new_price_change))

        with open(new_output_path, 'w', encoding = 'utf-8') as f:
            f.write(orginal_f_str)


    def generate_data(self, save_folder):

        sample_num = 5000
        sample_count = 0
        feature_num = 20
        label_list = ['dog1', 'dog2', 'cat1', 'cat2']
        mu_list = [0, 1, 2, 4]
        sigma_list = [0.1, 0.2, 0.5, 0.4]

        # dog1

        while sample_count <= sample_num:
            for i, label in enumerate(label_list):
                sample_count += 1
                mu = mu_list[i]
                sigma = sigma_list[i]
                feature_name = ["[{}]".format(x) for x in range(feature_num)]
                feature_value_list = list(np.random.normal(mu, sigma, feature_num))
                feature_list = [j for i in zip(feature_name, feature_value_list) for j in i]
                feature_list = [str(x) for x in feature_list]
                feature_str = ','.join(feature_list)

                # save_file
                file_name = "{}_{}.txt".format(sample_count, label)
                file_path = os.path.join(save_folder, file_name)
                with open (file_path, 'w', encoding = 'utf-8') as f:
                    f.write(feature_str)

        print ("Generated {} samples with {} features complete!".format(sample_num, feature_num))
