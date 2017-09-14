# ==========================================================================================================
# general package import
# ==========================================================================================================
import sys
import os
import os
import datetime
import time
import tushare as ts
import collections
import re
import numpy as np
import urllib
import math
import shutil
# ==========================================================================================================

# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path1 = os.path.join(parent_folder, 'general_functions')
sys.path.append(path1)
# ==========================================================================================================

# ==========================================================================================================
# local package import
# ==========================================================================================================
from trade_general_funcs import daterange, split_list_by_percentage
from pjslib.logger import logger1
# ==========================================================================================================





class Ashare:
    def __init__(self, is_stock_set = False):
        self.a_share_samples_f_dict = collections.defaultdict(lambda :0)
        self.a_share_samples_t_dict = collections.defaultdict(lambda: 0)
        self.a_share_samples_dict = collections.defaultdict(lambda: 0)
        if is_stock_set:
            self.stock_set =  set(ts.get_stock_basics().to_dict()['profit'].keys())
            print ("Got stock set!")
        self.t_attributors = []
        self.f_attributors = []

    def read_a_share_history_date(self,save_folder, start_date, is_prediction = False, is_filter_new_stock = False):
        self.read_fundamental_data(start_date, is_filter_new_stock = is_filter_new_stock)
        self.read_tech_history_data(start_date, is_prediction = is_prediction)
        self.save_raw_data(save_folder, is_prediction = is_prediction)

    def read_tech_history_data(self, start_date, is_prediction = False):
        # clear
        self.a_share_samples_t_dict = collections.defaultdict(lambda: 0)
        #

        start_date_temp = time.strptime(start_date, '%Y-%m-%d')
        start_date_obj = datetime.datetime(*start_date_temp[:3]).date()
        today_obj = datetime.datetime.today().date()
        today = datetime.datetime.today().strftime("%Y-%m-%d")
        t_attributors_set = set(ts.get_k_data("600883", start="2017-05-09", ktype='W').keys())
        t_attributors_set -= {'code', 'date'}
        t_attributors_set.add('priceChange')
        t_attributors_set.add('candleLength')
        t_attributors_set.add('candlePos')
        t_attributors = sorted(list(t_attributors_set))
        self.t_attributors = t_attributors

        stock_list = list(self.stock_set)[:]
        is_close_price_exist = True

        for stock_id in stock_list:
            fund_dict = ts.get_k_data(stock_id, start=start_date, ktype='W').to_dict()
            # date_list: ['2017-05-05', '2017-05-12', '2017-05-19']
            try:
                # date_items: [(29, '2016-08-05'), (30, '2016-08-12'), (31, '2016-08-19'), ...]
                date_items = sorted(list(fund_dict['date'].items()), key = lambda x:x[0])
                #print ("date_items: ", date_items)
            except KeyError:
                logger1.error("{} stock has no key data".format(stock_id))
                continue

            for i, (id, date_str) in enumerate(date_items):
                if i > len(date_items) - 3 and is_prediction is False:
                    print("Skip {} on {} because of reaching the end. The data of the rest date "
                          "cannot be fully presented".format(id, date_str))
                    continue

                # # ======================================================================================================
                # # TODO
                # # DATA CHECK FOR NEXT WEEK AND NEXT NEXT WEEK, BUT IT'S HARD TO ACHIEVE BECAUSE THE VARITIES OF HOLIDAIES
                # # ======================================================================================================
                # # get the date_str for next next week
                # date_temp = time.strptime(date_str, '%Y-%m-%d')
                # date_obj = datetime.datetime(*date_temp[:3])
                # delta_14 = datetime.timedelta(days=14)
                # delta_7 = datetime.timedelta(days=7)
                # date_obj_nw = date_obj + delta_7
                # date_nw_str = date_obj_nw.strftime("%Y-%m-%d")
                # date_obj_nnw = date_obj + delta_14
                # date_nnw_str = date_obj_nnw.strftime("%Y-%m-%d")
                # #
                #
                # # check next week's data
                # if date_nw_str != date_items[i + 1][1]:
                #     # print ("date_nw_str: ", date_nw_str)
                #     # print ("date_items[i + 1][1]: ", date_items[i + 1][1])
                #     # sys.exit()
                #     logger1.error("{} stock has no tech data on {} for next week".format(stock_id, date_nnw_str))
                #     continue
                # #
                #
                # # check next next week's data
                # if date_nnw_str != date_items[i + 2][1]:
                #     logger1.error("{} stock has no tech data on {} for next next week".format(stock_id, date_nnw_str))
                #     continue
                # #
                # # ======================================================================================================

                feature_list = []

                for attributor in t_attributors:
                    # for pricechange
                    if attributor == 'priceChange' and is_prediction is False:

                        nw_open = fund_dict['open'][date_items[i + 1][0]]
                        nnw_open = fund_dict['open'][date_items[i + 2][0]]
                        priceChange = "{:.5f}".format((nnw_open - nw_open) / nw_open)

                        # # price change for the next week's close
                        # close_price = fund_dict['close'][id]
                        # close_price_next_week = fund_dict['close'][date_items[i + 1][0]]
                        # priceChange = "{:.5f}".format((close_price_next_week - close_price) / close_price)
                        # #
                        feature_list.append(priceChange)

                    elif attributor == 'priceChange' and is_prediction is True:
                        priceChange = "nan"
                        feature_list.append(priceChange)

                    elif attributor == 'candleLength':
                        close_price = fund_dict['close'][id]
                        open_price = fund_dict['open'][id]
                        high_price = fund_dict['high'][id]
                        low_price = fund_dict['low'][id]
                        candle_length = "{:.5f}".format(abs((close_price- open_price)/(high_price-low_price)))
                        feature_list.append(candle_length)

                    elif attributor == 'candlePos':
                        close_price = fund_dict['close'][id]
                        open_price = fund_dict['open'][id]
                        high_price = fund_dict['high'][id]
                        low_price = fund_dict['low'][id]
                        price = max(close_price, open_price)
                        candle_pos = "{:.5f}".format(abs((high_price- price)/(high_price-low_price)))
                        feature_list.append(candle_pos)

                    else:
                        # for other attributors
                        feature_list.append(fund_dict[attributor][id])

                feature_array = np.array(feature_list)
                sample_name = date_str + '_' + stock_id
                self.a_share_samples_t_dict[sample_name] = feature_array
            print ("saving {} stock t features".format(stock_id))

        print ("t_attributors: {}".format(t_attributors))
        print ("a_share_samples_t_dict: {}".format(self.a_share_samples_t_dict.values()))
        print ("a_share_samples_t_dict_value: {}".format(list(self.a_share_samples_t_dict.values())[0]))


    def read_fundamental_data(self, start_date, is_filter_new_stock = False):
        # clear
        self.a_share_samples_f_dict = collections.defaultdict(lambda: 0)
        #
        start_date_temp = time.strptime(start_date, '%Y-%m-%d')
        start_date_obj = datetime.datetime(*start_date_temp[:3]).date()
        today_obj = datetime.datetime.today().date()
        today = datetime.datetime.today().strftime("%Y-%m-%d")

        # manually type
        f_attributors_set = {'holders', 'undp', 'gpr', 'pb', 'industry', 'bvps', 'timeToMarket',
                             'rev', 'perundp', 'fixedAssets', 'name', 'reservedPerShare', 'totals',
                             'outstanding', 'liquidAssets', 'profit', 'pe', 'reserved', 'npr', 'area',
                             'totalAssets', 'esp'}
        #
        # change the date if time out
        #f_attributors_set = set(ts.get_stock_basics(date = "2017-05-26").to_dict().keys())
        #

        filter_set = {'name', 'industry', 'area'}
        f_attributors_set = f_attributors_set - filter_set
        f_attributors = sorted(list(f_attributors_set))
        self.f_attributors = f_attributors

        for single_date in daterange(start_date_obj, today_obj):
            temp_stock_feature_dict = collections.defaultdict(lambda :[])
            temp_stock_feature_dict_key_pop_set = set() # for filtering the new stocks
            # if it is not friday, skip!
            if single_date.weekday() != 4:
                continue
            date_str = single_date.strftime("%Y-%m-%d")

            try:
                print ("date_str: ", date_str)
                ts_temp = ts.get_stock_basics(date = date_str)
                if ts_temp is None:
                    logger1.error("{} not found any data!".format(date_str))
                    continue
                fund_dict = ts_temp.to_dict()
            except urllib.error.HTTPError:
                logger1.error("{} not found any data!".format(date_str))
                continue



            for key, stock_key_value_dict in sorted(fund_dict.items()):

                # filter name,industry,,area,
                if key in filter_set:
                    continue
                #


                for stock_id, value in stock_key_value_dict.items():

                    if is_filter_new_stock:
                        if key == "timeToMarket":
                            timeToMarket = str(value)
                            try:
                                date_temp = time.strptime(timeToMarket, '%Y%m%d')
                            except ValueError:
                                logger1.error("{} has invalid timeToMarket value!".format(stock_id))
                                temp_stock_feature_dict[stock_id].append((key, value))
                                continue

                            date_obj = datetime.datetime(*date_temp[:3]).date()

                            # set the threshold for new stock
                            delta = datetime.timedelta(days=28)
                            #

                            date_gap = single_date - date_obj

                            if date_gap <= delta:
                                print ("stock_id: {} is new stock for {}, release date: {}".format(stock_id, single_date, timeToMarket))
                                temp_stock_feature_dict_key_pop_set.add(stock_id)

                    temp_stock_feature_dict[stock_id].append((key, value))

            # filter new stocks
            if is_filter_new_stock:
                for stock_id in temp_stock_feature_dict_key_pop_set:
                    temp_stock_feature_dict.pop(stock_id, 'None')
                #

            for stock_id, feature_list in temp_stock_feature_dict.items():
                feature_list = sorted(feature_list, key = lambda x: x[0])
                feature_value_list = [x[1] for x in feature_list]
                feature_array = np.array(feature_value_list)
                sample_name = date_str + '_' + stock_id

                # save samples
                self.a_share_samples_f_dict[sample_name] = feature_array
            print ("saving {}'s stock feature to a_share_samples_f_dict".format(single_date))

        print ("f_attributors: {}".format(f_attributors))
        print ("a_share_samples_f_dict_value: {}".format(list(self.a_share_samples_f_dict.values())[0]))


    def integrate_tech_fundamental_feature(self, feature1, feature2):
        new_feature = np.concatenate((feature1, feature2))
        return new_feature

    def save_raw_data(self, save_folder, is_f = True, is_prediction = False):
        save_count = 0
        for sample, t_feature_array in self.a_share_samples_t_dict.items():
            feature_array_list = []
            # (0.) add technical features
            feature_array_list.append(t_feature_array)
            # (1.) add fundamental features
            if is_f:
                is_sample_exist = self.a_share_samples_f_dict.get(sample)
                if is_sample_exist is None:
                    logger1.error('sample {} does not have any fundamental data'.format(sample))
                    continue
                f_feature_array = self.a_share_samples_f_dict[sample]
                feature_array_list.append(f_feature_array)

            # concatenate all features
            feature_array_final = np.array([])
            for feature_array in feature_array_list:
                feature_array_final = self.integrate_tech_fundamental_feature(feature_array_final, feature_array)

            # convert every feature to float
            feature_array_final = feature_array_final.astype(float)
            #
            feature_list_final = list(feature_array_final)

            attribitors = self.t_attributors + self.f_attributors

            if len(attribitors) != len(feature_list_final):
                logger1.error('sample: {}, feature_list_final and attribitors are not the same length! {}, {}'
                              .format(sample, len(attribitors), len(feature_list_final)))
                continue

            save_zip = zip(attribitors, feature_list_final)
            # save file
            save_name = sample + '.csv'
            save_path = os.path.join(save_folder, save_name)
            with open(save_path, 'w', encoding = 'utf-8') as f:
                for attribitor, feature_value in save_zip:
                    f.write(str(attribitor) + ',' + str(feature_value) + '\n')
                save_count += 1

        print ("Save {} samples to {} succesfully!".format(save_count, save_folder))


    def label_data(self, input_folder, save_folder, split_test = False, test_1_folder ='', test_set_1_percent = 0.1,
                   test_2_folder='', test_set_2_percent=0.5):

        samples_list = []
        raw_data_file_name_list = os.listdir(input_folder)
        for raw_data_file_name in raw_data_file_name_list:

            sample_id = raw_data_file_name[0:-4]

            raw_data_file_path = os.path.join(input_folder, raw_data_file_name)
            with open(raw_data_file_path, 'r', encoding = 'utf-8') as f:
                sample_feature_list = f.readlines()[0].split(',')

                price_change_index = sample_feature_list.index('priceChange')
                sample_price_change = float(sample_feature_list[price_change_index + 1])

                del sample_feature_list[price_change_index: price_change_index+2]
                # feature_name_list = feature_name_tuple_list[::2]
                # price_change_index = feature_name_list.index('priceChange')
                # sample_feature_list = feature_name_tuple_list[1::2]
                # sample_price_change = float(sample_feature_list[price_change_index])
                # del sample_feature_list[price_change_index]

            samples_list.append([sample_id, sample_feature_list, sample_price_change])

        # sort by pricechange
        samples_list = sorted(samples_list, key = lambda x:x[2], reverse = True)
        neg_samples_list = [x for x in samples_list if x[2] < 0]
        pos_samples_list = [x for x in samples_list if x[2] >= 0]

        # # for multiple tags
        # per_tuple = (0.05, 0.3, 1)
        # pos_label_tuple = ('top','good','pos')
        # neg_label_tuple = ('bottom', 'bad', 'neg')
        # #

        per_tuple = (1,)
        pos_label_tuple = ('pos',)
        neg_label_tuple = ('neg',)


        pos_samples_split_list = split_list_by_percentage(per_tuple, pos_samples_list)
        neg_samples_split_list = split_list_by_percentage(per_tuple, neg_samples_list)

        # label postive the data and output
        for i, small_pos_samples_list in enumerate(pos_samples_split_list):
            label = pos_label_tuple[i]
            for pos_sample in small_pos_samples_list:
                pos_sample[2] = label

        # label negative the data and output
        for i, small_neg_samples_list in enumerate(neg_samples_split_list):
            label = neg_label_tuple[i]
            for pos_sample in small_neg_samples_list:
                pos_sample[2] = label

        #print (neg_samples_list)

        # save labeled data to local
        samples_list = pos_samples_list + neg_samples_list

        # save the file
        for sample_list in samples_list:
            file_name = sample_list[0] + '_' +sample_list[2] + '.txt'
            file_path = os.path.join(save_folder, file_name)
            feature_list = sample_list[1]
            feature_list = [str(x) for x in feature_list]
            feature_str = ','.join(feature_list)
            with open (file_path, 'w', encoding = 'utf-8') as f:
                f.write(feature_str)

        print ("Label {} samples succesfully! Pos: {} Neg: {}".format(len(samples_list),
                                                                       len(pos_samples_list),
                                                                       len(neg_samples_list),
                                                                       ))
        print ("All files have been saved to {}".format(save_folder))


        if split_test:
            self.seperate_test_set(save_folder, test_1_folder, test_set_percent = test_set_1_percent)
            self.seperate_test_set(test_1_folder, test_2_folder, test_set_percent = test_set_2_percent)


    def regression(self, input_folder, save_folder, split_test = False, test_1_folder ='', test_2_folder ='',
                   test_set_1_percent = 0.2, test_set_2_percent = 0.5):

        file_name_list = os.listdir(input_folder)
        file_path_list = [os.path.join(input_folder, file_name) for file_name in file_name_list]
        key = 'priceChange'

        for i, file_path in enumerate(file_path_list):
            with open(file_path, 'r', encoding = 'utf-8') as f:
                feature_name_value_list = f.readlines()[0].split(',')
                key_index = feature_name_value_list.index(key)
                key_value = feature_name_value_list[key_index + 1]
                del feature_name_value_list[key_index: key_index+2]
                feature_name_value_str = ','.join(feature_name_value_list)
                file_name = file_name_list[i][:-4] + '_' + '#' + key_value + '#' + '.txt'
                file_save_path = os.path.join(save_folder, file_name)
                with open (file_save_path, 'w', encoding = 'utf-8') as f:
                    f.write(feature_name_value_str)

        print ("key: ", key)
        print ("Write the regression value for {} files".format(len(file_name_list)))

        if split_test:
            self.seperate_test_set(save_folder, test_1_folder, test_set_percent = test_set_1_percent)
            self.seperate_test_set(test_1_folder, test_2_folder, test_set_percent = test_set_2_percent)


    def get_stocks_feature_this_week(self):
        nearest_friday = datetime.datetime.today().date()
        delta = datetime.timedelta(days=1)
        while nearest_friday.weekday() != 4:
            nearest_friday -= delta

        start_date = nearest_friday.strftime("%Y-%m-%d")

        self.read_fundamental_data(start_date = start_date)
        self.read_tech_history_data(start_date = start_date, is_prediction = True)
        self.save_raw_data(is_prediction = True)

    def feature_engineering(self, input_folder, save_folder, keep_stock_ids_path = None):

        if keep_stock_ids_path:
            keep_stock_ids_list = []
            with open (keep_stock_ids_path, 'r') as f:
                for line in f:
                    keep_stock = line.strip()
                    keep_stock_ids_list.append(keep_stock)


        file_name_list = os.listdir(input_folder)
        file_path_list = [os.path.join(input_folder, file_name) for file_name in file_name_list]

        successful_save_count = 0
        original_data_count = len(file_name_list)

        for i, file_path in enumerate(file_path_list):
            file_name = file_name_list[i]
            stock_id = re.findall(r'_([0-9]+).csv', file_name)[0]
            # filter stock ids
            if keep_stock_ids_path:
                if stock_id not in keep_stock_ids_list:
                    continue
            #

            date = re.findall(r'([0-9]+-[0-9]+-[0-9]+)_', file_name)[0]
            # find the data of the previous friday
            date_obj_temp = time.strptime(date, '%Y-%m-%d')
            date_obj = datetime.datetime(*date_obj_temp[:3])


            # find the file for the previous week for the calculation of certain features,
            # the day gap does not necessary to be 7 days
            previous_week_date_full_path = ''
            pre_f_day_range = (7, 13)
            for days in range(pre_f_day_range[0], pre_f_day_range[1]):
                previous_friday_obj = date_obj - datetime.timedelta(days = days)
                previous_friday_str = previous_friday_obj.strftime("%Y-%m-%d")
                previous_friday_full_path = previous_friday_str + '_' + stock_id + '.csv'
                previous_friday_full_path = os.path.join(input_folder, previous_friday_full_path)
                try:
                    open (previous_friday_full_path, 'r', encoding = 'utf-8')
                    previous_week_date_full_path = previous_friday_full_path
                    break
                except FileNotFoundError:
                    continue

            if not previous_week_date_full_path:
                logger1.error("{} cannot find the previous week's data within 13 days".format(file_name))
                continue
            else:
                with open(previous_week_date_full_path, 'r', encoding='utf-8') as f:
                    previous_f_feature_pair_dict = {}
                    for line in f:
                        line_list = line.split(',')
                        feature_name = line_list[0]
                        feature_value = float(line_list[1].strip())
                        previous_f_feature_pair_dict[feature_name] = feature_value
            #

            feature_pair_dict = {}
            with open(file_path, 'r', encoding = 'utf-8') as f:
                for line in f:
                    line_list = line.split(',')
                    feature_name = line_list[0]
                    feature_value = float(line_list[1].strip())
                    feature_pair_dict[feature_name] = feature_value

            # ===================================================================================
            # add features
            # ===================================================================================
            # (1.) open change
            pre_f = previous_f_feature_pair_dict['open']
            f = feature_pair_dict['open']
            feature_pair_dict['openChange'] = "{:.5f}".format((f - pre_f) / pre_f)
            # -----------------------------------------------------------------------------------
            # (2.) close change
            pre_f = previous_f_feature_pair_dict['close']
            f = feature_pair_dict['close']
            feature_pair_dict['closeChange'] = "{:.5f}".format((f - pre_f) / pre_f)
            # -----------------------------------------------------------------------------------
            # (3.) high change
            pre_f = previous_f_feature_pair_dict['high']
            f = feature_pair_dict['high']
            feature_pair_dict['highChange'] = "{:.5f}".format((f - pre_f) / pre_f)
            # -----------------------------------------------------------------------------------
            # (4.) low change
            pre_f = previous_f_feature_pair_dict['low']
            f = feature_pair_dict['low']
            feature_pair_dict['lowChange'] = "{:.5f}".format((f - pre_f) / pre_f)
            # -----------------------------------------------------------------------------------
            # (5.) volume change
            pre_f = previous_f_feature_pair_dict['volume']
            f = feature_pair_dict['volume']
            feature_pair_dict['volumeChange'] = "{:.5f}".format((f - pre_f) / pre_f)
            # -----------------------------------------------------------------------------------
            # (6.) open close change
            open_price = feature_pair_dict['open']
            close_price = feature_pair_dict['close']
            open_close_change = (close_price - open_price) / open_price
            feature_pair_dict['openCloseChange'] = "{:.5f}".format(open_close_change)
            # -----------------------------------------------------------------------------------
            # (7.) low high change
            low_price = feature_pair_dict['low']
            high_price = feature_pair_dict['high']
            low_high_change = (high_price - low_price) / low_price
            feature_pair_dict['lowHighChange'] = "{:.5f}".format(low_high_change)


            # **********************************************************************************************
            # FUNDAMENTALS
            # **********************************************************************************************


            FUNDAMENTAL_ATTRIBUTOR_SET = {'pb','pe'}
            for attritubtor in FUNDAMENTAL_ATTRIBUTOR_SET:
                pre = previous_f_feature_pair_dict[attritubtor]
                this_week = feature_pair_dict[attritubtor]
                new_attributor_name = attritubtor + 'Change'
                try:
                    feature_pair_dict[new_attributor_name] = "{:.6f}".format((this_week - pre) / pre)
                except ZeroDivisionError:
                    set_value = "1.0"
                    feature_pair_dict[new_attributor_name] = set_value
                    logger1.error("New attributor {} has ZeroDivisionError! attritubtor: {}, temporal set value: {}"
                                  .format(os.path.basename(previous_week_date_full_path), new_attributor_name, set_value))

            # **********************************************************************************************




            # ===================================================================================

            # ===================================================================================
            # delete features: close, high, low, open
            # ===================================================================================
            # delete_features_set = {'close', 'high', 'low', 'open', 'timeToMarket'}
            delete_features_set = {'close', 'high', 'low', 'open', 'timeToMarket', 'liquidAssets',
                                   'fixedAssets','reserved', 'reservedPerShare', 'esp', 'bvps', 'pb',
                                   'undp','perundp', 'holders', 'totals', 'totalAssets', 'outstanding'}

            for feature_name in delete_features_set:
                feature_pair_dict.pop(feature_name)
            # ===================================================================================

            # write the feature engineered file to folder
            file_name = file_name.replace('csv','txt')
            save_file_path = os.path.join(save_folder, file_name)
            with open(save_file_path, 'w', encoding = 'utf-8') as f:
                feature_pair_list = []
                feature_pair_tuple_list = sorted(list(feature_pair_dict.items()), key = lambda x:x[0])
                for feature_pair in feature_pair_tuple_list:
                    feature_pair_list.append(feature_pair[0])
                    feature_pair_list.append(feature_pair[1])

                feature_pair_list = [str(x) for x in feature_pair_list]
                feature_pair_str = ','.join(feature_pair_list)

                f.write(feature_pair_str)
                successful_save_count += 1
        print ("Succesfully engineered {} raw data! original count: {}, delete {} files"
               .format(successful_save_count, original_data_count, original_data_count - successful_save_count))


    def prediction_transfrom(self, input_folder, save_folder):

        file_name_list = os.listdir(input_folder)
        file_path_list = [os.path.join(input_folder, file_name) for file_name in file_name_list]
        file_save_path_list = [os.path.join(save_folder, file_name) for file_name in file_name_list]
        delete_key = 'priceChange'

        for i, file_path in enumerate(file_path_list):
            with open(file_path, 'r', encoding = 'utf-8') as f:
                feature_name_value_list = f.readlines()[0].split(',')
                delete_key_index = feature_name_value_list.index(delete_key)
                del feature_name_value_list[delete_key_index: delete_key_index+2]
                feature_name_value_str = ','.join(feature_name_value_list)
                with open(file_save_path_list[i], 'w', encoding = 'utf-8') as f:
                    f.write(feature_name_value_str)

        print ("All files are ready for prediction! Total: {} files".format(len(file_name_list)))

    def delete_all_prediction_folder(self, prediction_folder_list):
        for prediction_foler in prediction_folder_list:
            file_name_list = os.listdir(prediction_foler)
            file_count = len(file_name_list)
            file_path_list = [os.path.join(prediction_foler, x) for x in file_name_list]
            for file_path in file_path_list:
                os.remove(file_path)
            print ("Succefully remove {} files in {}".format(file_count, prediction_foler))

    def seperate_test_set(self, all_data_path, test_set_path, test_set_percent = 0.1):
        file_name_list = os.listdir(all_data_path)
        data_str_list = []
        for file_name in file_name_list:
            data_str = re.findall(r'([0-9]{4}-[0-9]{2}-[0-9]{2})_', file_name)[0]
            data_str_list.append(data_str)

        # (1.) get test set threshold index, get test data list
        data_str_set = set(data_str_list)
        unique_data_list = sorted(list(data_str_set))
        date_number = len(data_str_set)
        test_set_number = math.floor(test_set_percent*date_number)
        test_unique_data_list = unique_data_list[-test_set_number:]


        # (2.) cut and paste
        test_file_name_list = []
        for file_name in file_name_list:
            data_str = re.findall(r'([0-9]{4}-[0-9]{2}-[0-9]{2})_', file_name)[0]
            if data_str in test_unique_data_list:
                test_file_name_list.append(file_name)
            else:
                continue

        delete_file_path_list =  [os.path.join(all_data_path, x) for x in test_file_name_list]
        test_save_file_path_list = [os.path.join(test_set_path, x) for x in test_file_name_list]

        move_count = 0
        for i, delete_path in enumerate(delete_file_path_list):
            test_file_path = test_save_file_path_list[i]
            shutil.move(delete_path, test_file_path)
            move_count += 1

        print ("Move {} files from {} to {} succesfully for separating test set[a_share].".
               format(move_count, all_data_path, test_set_path))

    def filter_stocks(self, input_folder, output_folder, filter_stock_ids_path):
        keep_stock_id_list = []
        with open (filter_stock_ids_path, 'r') as f:
            for line in f:
                keep_stock = line.strip()
                keep_stock_id_list.append(keep_stock)
