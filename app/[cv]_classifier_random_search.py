# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Build the MLP classifier.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# (c) 2017 PJS, University of Sheffield, iamlxb3@gmail.com
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# ==========================================================================================================
# general package import
# ==========================================================================================================
import sys
import random
import time
import os
import itertools
import re
import numpy as np
import math
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
clf_path = os.path.join(parent_folder, 'classifiers', 'mlp')
path2 = os.path.join(parent_folder, 'general_functions')
sys.path.append(clf_path)
sys.path.append(path2)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from mlp_trade_classifier import MlpTradeClassifier
from trade_general_funcs import get_full_feature_switch_tuple
from trade_general_funcs import read_pca_component

# ==========================================================================================================

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT IMPORT I
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
unique_id = 0
unique_start = 0
unique_end = float('inf')

#
data_set = 'a_share'
input_folder = os.path.join(data_set, 'a_share_labeled_data')
input_folder = os.path.join(parent_folder, 'data', input_folder)
#
EXPERIMENTS = 3
is_standardisation_list = [True, False]
is_PCA_list = [True, False]
TRAILS= 64
PCA_MIN_COMPONENT = 8
RANDOM_SEED_OFFSET = 54385438
EXPERIMENT_RANDOM_SEED_OFFSET = 38453845
random_state_total = 5
tol = 1e-10
classifier = 'classifier'
training_window_min = 5 # weeks
training_window_max = 15 # weeks, make sure shift_size*shift_num is fixed
is_training_window_flexible = False # if false, training_window will be training_window_max
# set the shift size to 1 week
shifting_size_min = 1
shifting_size_max = 1
#
# read total_date_num for training and validation
file_name_list = os.listdir(input_folder)
file_path_list = [os.path.join(input_folder,x) for x in file_name_list]
date_str_list = []
for f_path in file_path_list:
    f_name = os.path.basename(f_path)
    date_str = re.findall(r'([0-9]+-[0-9]+-[0-9]+)_', f_name)[0]
    date_str_list.append(date_str)
date_str_set = set(date_str_list)
total_date_num = len(date_str_set) # weeks
print ("total_date_num: ", total_date_num)
#








# ==========================================================================================================
# Build MLP classifier for a-share data, save the mlp to local
# ==========================================================================================================
is_standardisation = True
is_PCA = True

for is_standardisation, is_PCA in list(itertools.product(is_standardisation_list, is_PCA_list)):

    # (1.) build classifer
    mlp_regressor1 = MlpTradeClassifier()
    clsfy_name = '{}_hyper_parameter_{}'.format(data_set, classifier)
    clf_path = os.path.join(parent_folder, 'trained_classifiers', clsfy_name)


    # (2.) GET TRANINING SET
    data_per = 1.0 # the percentage of data using for training, validation and testing
    #dev_per = 0.0 # the percentage of data using for developing


    # (3.)
    if is_standardisation and is_PCA:
        data_preprocessing = 'pca_standardization'
    elif is_standardisation and not is_PCA:
        data_preprocessing = 'standardization'
    elif not is_standardisation and is_PCA:
        data_preprocessing = 'pca'
    elif not is_standardisation and not is_PCA:
        data_preprocessing = 'origin'
    else:
        print ("Check data preprocessing switch")
        sys.exit()
    pca_n_component_max = read_pca_component(input_folder)

    # (4.)
    feature_switch_tuple_all_1 = get_full_feature_switch_tuple(input_folder)
    feature_switch_tuple = feature_switch_tuple_all_1
    samples_feature_list, samples_value_list, date_str_list, stock_id_list = mlp_regressor1._feed_data(input_folder,
                                                                      data_per=data_per,
                                                                      feature_switch_tuple=feature_switch_tuple,
                                                                    is_random=False, mode='clf')



    # ----------------------------------------------------------------------------------------------------------------------
    # generator builder
    # ----------------------------------------------------------------------------------------------------------------------
    def build_generator_from_pool(random_pool, trails, experiment_count):
        for i in range(trails):
            random.seed(i+experiment_count*EXPERIMENT_RANDOM_SEED_OFFSET+RANDOM_SEED_OFFSET)
            random_sample = random.sample(random_pool, 1)[0]
            yield random_sample

    def build_generator_for_shift_flexible_t_window(shift_size_pool, training_window_size_pool, training_window_max, total_date_num,
                                                    trails, experiment_count):
        validation_window_total = total_date_num - training_window_max
        possible_max_training_window = total_date_num - shifting_size_max*1
        if training_window_max > possible_max_training_window:
            print ("training_window_max: {} too big!! MAX: {}".format(training_window_max, possible_max_training_window))
            sys.exit()
        for i in range(trails):
            best_shift_found = False
            j = 0
            while not best_shift_found:
                j += 1
                random.seed(j+i+experiment_count*EXPERIMENT_RANDOM_SEED_OFFSET+RANDOM_SEED_OFFSET)
                shift_size = random.sample(shift_size_pool, 1)[0]
                random.seed(j+i+experiment_count*EXPERIMENT_RANDOM_SEED_OFFSET+RANDOM_SEED_OFFSET)
                training_window_size = random.sample(training_window_size_pool, 1)[0]
                print ("training_window_size---", training_window_size)
                print ("shift_size: ", shift_size)
                if not training_window_size >= 2.5*shift_size:
                    print ("training_window_size:{} too small!!".format(training_window_size))
                    continue
                if validation_window_total%shift_size != 0:
                    print ("shift_number is float:{}"
                           .format(validation_window_total/shift_size))
                else:
                    shift_number = int(validation_window_total/shift_size)
                    best_shift_found = True
                # random.seed(j+i+experiment_count*EXPERIMENT_RANDOM_SEED_OFFSET+RANDOM_SEED_OFFSET)
                # shift_number_pool = [x for x in range(shift_number_pool[0], max_shift_number)]
                # shift_number = random.sample(shift_number_pool, 1)[0]
                # training_date_num = max_num-shift_number*shift_size
            yield shift_size, shift_number,training_window_size


    def build_generator_from_range(target_range, trails, experiment_count):
        for i in range(trails):
            random.seed(i+experiment_count*EXPERIMENT_RANDOM_SEED_OFFSET+RANDOM_SEED_OFFSET)
            random_value = random.uniform(*target_range)
            yield random_value

    def build_generator_for_feature_selection(trails, experiment_count):
        feature_num_max = len(feature_switch_tuple_all_1)
        feature_num_min = math.floor((1 / 2) * len(feature_switch_tuple_all_1))
        for i in range(trails):
            random.seed(i+experiment_count*EXPERIMENT_RANDOM_SEED_OFFSET+RANDOM_SEED_OFFSET)
            feature_num = random.randint(feature_num_min, feature_num_max)
            random.seed(i+experiment_count*EXPERIMENT_RANDOM_SEED_OFFSET+RANDOM_SEED_OFFSET)
            feature_index = random.sample([x for x in range(0, feature_num_max)], feature_num)
            feature_random_switch_tuple = [0 for x in feature_switch_tuple_all_1]
            for index in feature_index:
                feature_random_switch_tuple[index] = 1
            yield feature_random_switch_tuple



    # ----------------------------------------------------------------------------------------------------------------------


    for experiment_count, experiment in enumerate(range(EXPERIMENTS)):


        # (1.) activation function
        activation_list = ['identity', 'logistic', 'tanh', 'relu']
        activation_random_sample_generator = build_generator_from_pool(activation_list, TRAILS, experiment_count)

        # (2.) L2 penalty (regularization term) parameter.
        alpha_range = (0.00001, 0.001)
        alpha_random_sample_generator = build_generator_from_range(alpha_range, TRAILS, experiment_count)

        # (3.) learning_rate
        learning_rate_list = ['constant', 'invscaling']
        learning_rate_random_sample_generator = build_generator_from_pool(learning_rate_list, TRAILS, experiment_count)

        # (4.) learning_rate_init
        learning_rate_init_range = (0.0001, 0.1)
        learning_rate_init_random_sample_generator = build_generator_from_range(learning_rate_init_range, TRAILS,
                                                                                experiment_count)

        # (5.) early-stopping
        early_stopping_list = [True, False]
        early_stopping_random_sample_generator = build_generator_from_pool(early_stopping_list, TRAILS, experiment_count)

        # (6.) early-stopping validation_fraction
        validation_fraction_range = (0.1, 0.3)
        validation_fraction_random_sample_generator = build_generator_from_range(validation_fraction_range, TRAILS,
                                                                                 experiment_count)
        # (6.) shift size
        shifting_size_pool = [x for x in range(shifting_size_min, shifting_size_max+1)]  # 5,50
        if is_training_window_flexible:
            training_window_pool = [x for x in range(training_window_min,training_window_max+1)]  # 1,20
        else:
            training_window_pool = [x for x in range(training_window_max,training_window_max+1)]  # 1,20
        shifting_random_sample_generator = build_generator_for_shift_flexible_t_window(shifting_size_pool, training_window_pool,
                                                                                       training_window_max, total_date_num,
                                                                                       TRAILS, experiment_count)

        # (7.) HIDDEN LAYERS
        hidden_layer_depth = (1,2)
        hidden_layer_node  = (20,400)

        # (8.) Feature selection
        feature_random_switch_tuple = build_generator_for_feature_selection(TRAILS, experiment_count)

        def hidden_layer_generator(hidden_layer_depth, hidden_layer_node, experiment_count):
            for i in range(TRAILS):
                hidden_layer_sizes = []
                random.seed(i + experiment_count*EXPERIMENT_RANDOM_SEED_OFFSET + RANDOM_SEED_OFFSET)
                layer_depth = random.randint(*hidden_layer_depth)
                for j in range(layer_depth):
                    random.seed(j + i + experiment_count*EXPERIMENT_RANDOM_SEED_OFFSET + RANDOM_SEED_OFFSET)
                    layer_node = random.randint(*hidden_layer_node)
                    hidden_layer_sizes.append(layer_node)
                hidden_layer_sizes_tuple = tuple(hidden_layer_sizes)
                yield hidden_layer_sizes_tuple


        hidden_layer_size= hidden_layer_generator(hidden_layer_depth, hidden_layer_node, experiment_count)



        hyper_parameter_trail_zip = zip(activation_random_sample_generator, alpha_random_sample_generator,
                                              learning_rate_random_sample_generator, learning_rate_init_random_sample_generator,
                                              early_stopping_random_sample_generator, validation_fraction_random_sample_generator,
                                        hidden_layer_size,shifting_random_sample_generator,feature_random_switch_tuple)

        # hyper_parameter_size = len(hyper_parameter_trail_list)
        # print ("hyper_parameter_size: ", hyper_parameter_size)
        # print ("hyper_parameter_trail_list: ", hyper_parameter_trail_list)



        for i, hyper_paramter_tuple in enumerate(hyper_parameter_trail_zip):
            if unique_id > unique_end:
                break

            if unique_id < unique_start:
                unique_id += 1
                continue

            # feature selection if not PCA
            if data_preprocessing == 'origin' or data_preprocessing == 'standardization':
                feature_switch_tuple = hyper_paramter_tuple[8]
                samples_feature_list, samples_value_list, date_str_list, stock_id_list = mlp_regressor1._feed_data(
                    input_folder,
                    data_per=data_per,
                    feature_switch_tuple=feature_switch_tuple,
                    is_random=False,
                    mode='clf')

            # (0.) PCA n component
            if data_preprocessing == 'pca' or data_preprocessing == 'pca_standardization':
                random.seed(i + experiment_count*EXPERIMENT_RANDOM_SEED_OFFSET + RANDOM_SEED_OFFSET)
                pca_n_component = random.randint(PCA_MIN_COMPONENT, pca_n_component_max)
            else:
                pca_n_component = None


            shifting_size, shift_num, training_window_size = hyper_paramter_tuple[7]

            mlp_regressor1.create_train_dev_vdict_window_shift(samples_feature_list, samples_value_list,
                                                               date_str_list, stock_id_list, is_cv=True,
                                                               shifting_size=shifting_size,
                                                               shift_num=shift_num,
                                                               is_standardisation=is_standardisation, is_PCA=is_PCA,
                                                               pca_n_component=pca_n_component,
                                                               training_window_size =training_window_size)

            activation_function = hyper_paramter_tuple[0]
            alpha = float("{:7f}".format(hyper_paramter_tuple[1]))
            learning_rate = hyper_paramter_tuple[2]
            learning_rate_init = float("{:7f}".format(hyper_paramter_tuple[3]))
            early_stopping = hyper_paramter_tuple[4]
            if early_stopping:
                validation_fraction = float("{:7f}".format(hyper_paramter_tuple[5]))
            else:
                validation_fraction = 0.0
            hidden_layer_sizes = hyper_paramter_tuple[6]

            trail = i
            n_iter_list = []
            loss_list = []
            f1_list = []
            accuracy_list = []
            random_state_list = []


            for random_state in range(random_state_total):
                random_seed = 'window_shift'

                shift_n_iter_list = []
                shift_loss_list = []
                shift_f1_list = []
                shift_accuracy_list = []


                mlp_regressor1.set_mlp_clf(hidden_layer_sizes, tol=tol, learning_rate_init=learning_rate_init,
                                                  random_state=random_state,
                                   verbose = False, learning_rate = learning_rate, early_stopping =early_stopping, alpha= alpha,
                                                  validation_fraction = validation_fraction, activation = activation_function)

                for shift in mlp_regressor1.validation_dict[random_seed].keys():
                    mlp_regressor1.trade_rs_cv_load_train_dev_data(random_seed, shift)
                    n_iter, loss = mlp_regressor1.clf_train(save_clsfy_path=clf_path)
                    average_f1, accuracy = mlp_regressor1.clf_dev(save_clsfy_path=clf_path, is_cv=True, is_return=True)
                    shift_n_iter_list.append(n_iter)
                    shift_loss_list.append(loss)
                    shift_f1_list.append(average_f1)
                    shift_accuracy_list.append(accuracy)




                avg_n_iter = np.average(shift_n_iter_list)
                avg_loss= np.average(shift_loss_list)
                avg_f1 = np.average(shift_f1_list)
                avg_accuracy = np.average(shift_accuracy_list)


                random_state_list.append(random_state)
                n_iter_list.append(avg_n_iter)
                loss_list.append(avg_loss)
                f1_list.append(avg_f1)
                accuracy_list.append(avg_accuracy)

                print ("-----------------------------------------------------------------------------------")
                print ("unique_id: ", unique_id)
                print ("feature_switch_tuple: ", feature_switch_tuple)
                print ("training_window_size: ", training_window_size)
                print ("shifting_size: ", shifting_size)
                print ("shift_num: ", shift_num)
                print ("is_PCA", is_PCA)
                print ("is_standardisation", is_standardisation)
                print ("pca_n_component: ", pca_n_component)
                print ("random_state: {}|{}".format(random_state, random_state_total))
                print ("activation_function: ", activation_function)
                print ("alpha: ", alpha)
                print ("learning_rate: ", learning_rate)
                print ("learning_rate_init: ", learning_rate_init)
                print ("early_stopping: ", early_stopping)
                print ("validation_fraction: ", validation_fraction)
                print ("hidden_layer_sizes: ", hidden_layer_sizes)
                print ("avg_n_iter: ", avg_n_iter)
                print ("avg_loss: ", avg_loss)
                print ("avg_f1: ", avg_f1)
                print ("avg_accuracy: ", avg_accuracy)
                print ("Testing percent: {:.7f}%".format(100*i/TRAILS))
                print ("experiment: {}/{}".format(experiment, EXPERIMENTS))

            # ==========================================================================================================
            write_list = list(zip(loss_list, n_iter_list, f1_list, accuracy_list, random_state_list))
            write_list = sorted(write_list, key = lambda x:x[0])


            hidden_layer_write_str = '_'.join([str(x) for x in hidden_layer_sizes])

            write_tuple = (unique_id, experiment, trail, random_state_total, pca_n_component, activation_function, alpha,
                           learning_rate, learning_rate_init, early_stopping,validation_fraction,
                           shifting_size, shift_num, training_window_size, hidden_layer_write_str)
            save_folder = os.path.join(parent_folder, 'hyper_parameter_test', data_set, classifier, data_preprocessing)
            csv_file_path = os.path.join(save_folder, '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(*write_tuple))
            txt_file_path = os.path.join(save_folder, '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.txt'.format(*write_tuple))



            with open(csv_file_path, 'w') as f:
                for i, tuple1 in enumerate(write_list):
                    list1 = list(tuple1)
                    list1 = [str(x) for x in list1]
                    tuple_str = ','.join(list1)
                    f.write(tuple_str + '\n')
            print ("Save csv to {}".format(csv_file_path))

            if data_preprocessing == 'origin' or data_preprocessing == 'standardization':
                feature_selection_save_path = os.path.join(save_folder, 'feature_selection.txt')
                with open (feature_selection_save_path, 'a') as f:
                    f.write('{},{}\n'.format(unique_id,feature_switch_tuple))
            # ------------------------------------------------------------------------------------------------------------------
            # Write txt file
            # ------------------------------------------------------------------------------------------------------------------
            # with open (txt_file_path, 'w') as f:
            #     for i,tuple in enumerate(write_list):
            #         id = str(i)
            #         loss = str(tuple[0])
            #         n_iter = str(tuple[1])
            #         rmse = str(tuple[2])
            #         avg_pc = str(tuple[3])
            #         polar_percent = str(tuple[4])
            #         random_state = str(tuple[5])
            #
            #         f.write("--------------------------------------------\n".format(id))
            #         f.write("id: {}\n".format(id))
            #         f.write("loss: {}\n".format(loss))
            #         f.write("n_iter: {}\n".format(n_iter))
            #         f.write("rmse: {}\n".format(rmse))
            #         f.write("avg_pc: {}\n".format(avg_pc))
            #         f.write("polar_percent: {}\n".format(polar_percent))
            #         f.write("random_state: {}\n".format(random_state))
            # print ("Save txt to {}".format(txt_file_path))
            # ------------------------------------------------------------------------------------------------------------------

            unique_id += 1