import sys
import os
import collections
import re
import datetime
import time


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



class DowJonesIndexExtended:
    """
    quarter:  the yearly quarter (1 = Jan-Mar; 2 = Apr=Jun).
    stock: the stock symbol (see above)
    date: the last business day of the work (this is typically a Friday)
    open: the price of the stock at the beginning of the week
    high: the highest price of the stock during the week
    low: the lowest price of the stock during the week
    close: the price of the stock at the end of the week
    volume: the number of shares of stock that traded hands in the week
    percent_change_price: the percentage change in price throughout the week
    percent_chagne_volume_over_last_wek: the percentage change in the number of shares of
    stock that traded hands for this week compared to the previous week
    previous_weeks_volume: the number of shares of stock that traded hands in the previous week
    next_weeks_open: the opening price of the stock in the following week
    next_weeks_close: the closing price of the stock in the following week
    percent_change_next_weeks_price: the percentage change in price of the stock in the following week
    days_to_next_dividend: the number of days until the next dividend
    percent_return_next_dividend: the percentage of return on the next dividend
    """
    def __init__(self):
        pass

    def format_raw_data(self, input_folder, save_folder):
        PREVIOUS_WEEK_NUM = 4  # the number of week from which the feature is extracted

        dividend_folder = os.path.join(input_folder, 'dividend')
        file_name_list = os.listdir(input_folder)
        file_name_list.remove('dividend')
        file_name_list.remove('BACKUP')
        file_path_list = [os.path.join(input_folder, x) for x in file_name_list]

        for j, file_path in enumerate(file_path_list):
            print ("Start formatting {}".format(file_path))
            dividend_file_path = os.path.join(dividend_folder, 'EOD-' + file_name_list[j])
            with open(file_path, 'r', encoding = 'utf-8') as f:
                f_readlines1 = f.readlines()
                for i, line in enumerate(f_readlines1):
                    if i <= PREVIOUS_WEEK_NUM or i == len(f_readlines1) - 1:
                        continue
                    #print ("line: ", line)
                    line_split = line.split(',')
                    date_str = line_split[0]
                    date_str_next = f_readlines1[i+1].split(',')[0]
                    if date_str >= date_str_next:
                        print ("ERROR! check raw data date order!")
                        sys.exit()


                    # --------------------------------------------------------------------------------------------------
                    # read dividend data
                    # --------------------------------------------------------------------------------------------------
                    date_temp = time.strptime(date_str, '%Y-%m-%d')
                    date_obj = datetime.datetime(*date_temp[:3])
                    percent_return_next_dividend = 0
                    days_to_next_dividend = 0
                    with open (dividend_file_path, 'r') as f:
                        f_readlines2 = f.readlines()[::-1]
                        for k, line in enumerate(f_readlines2):
                            if k == len(f_readlines2) - 1:
                                continue
                            if not line:
                                continue
                            line_list = line.strip().split(',')
                            check_date_str = line_list[0]
                            check_date_temp = time.strptime(check_date_str, '%Y-%m-%d')
                            check_date_obj = datetime.datetime(*check_date_temp[:3])
                            if check_date_obj < date_obj:
                                continue
                            else:
                                percent_return_next_dividend = float(line_list[6])
                                if percent_return_next_dividend == 0:
                                    continue
                                else:
                                    days_to_next_dividend = float((check_date_obj - date_obj).days)
                                    break

                    if percent_return_next_dividend == 0:
                        print ("Error! Please check days_to_next_dividend or dividend_value, "
                               "path: {}".format(dividend_file_path))
                        sys.exit()

                    # --------------------------------------------------------------------------------------------------




                    save_file_name = date_str + '_' + file_name_list[j][:-4] + '.txt'
                    save_file_path = os.path.join(save_folder, save_file_name)

                    # this week data
                    open1 = line_split[1]
                    high = line_split[2]
                    low = line_split[3]
                    close = line_split[4]
                    volume = line_split[6]
                    #

                    # previous 1 week data
                    previous_line_split = f_readlines1[i-1].split(',')
                    previous_open1 = previous_line_split[1]
                    previous_high = previous_line_split[2]
                    previous_low = previous_line_split[3]
                    previous_close = previous_line_split[4]
                    previous_volume = previous_line_split[6]
                    #

                    # previous 2 week data
                    previous_line_split = f_readlines1[i-2].split(',')
                    previous_2_close = previous_line_split[4]
                    previous_2_volume = previous_line_split[6]
                    previous_2_close_change = (float(close)-float(previous_2_close)) / float(previous_2_close)
                    previous_2_volume_change = (float(volume)-float(previous_2_volume)) / float(previous_2_volume)
                    #

                    # previous 3 week data
                    previous_line_split = f_readlines1[i-3].split(',')
                    previous_3_close = previous_line_split[4]
                    previous_3_volume = previous_line_split[6]
                    previous_3_close_change = (float(close) - float(previous_3_close)) / float(previous_3_close)
                    previous_3_volume_change = (float(volume) - float(previous_3_volume)) / float(previous_3_volume)
                    #

                    # previous 4 week data
                    previous_line_split = f_readlines1[i-4].split(',')
                    previous_4_close = previous_line_split[4]
                    previous_4_volume = previous_line_split[6]
                    previous_4_close_change = (float(close) - float(previous_4_close)) / float(previous_4_close)
                    previous_4_volume_change = (float(volume) - float(previous_4_volume)) / float(previous_4_volume)
                    #



                    # next week data
                    next_line_split = f_readlines1[i+1].split(',')
                    next_open1 = next_line_split[1]
                    next_high = next_line_split[2]
                    next_low = next_line_split[3]
                    next_close = next_line_split[4]
                    next_volume = next_line_split[6]
                    #

                    #


                    percent_change_price = 100*(float(close)-float(previous_close)) / float(previous_close)
                    percent_change_volume_over_last_wk = 100*(float(volume)-float(previous_volume)) / float(previous_volume)
                    percent_change_next_weeks_price = 100*(float(next_close)-float(close)) / float(close)

                    write_str = 'open,{},high,{},low,{},close,{},volume,{},' \
                                'percent_change_price,{},percent_change_volume_over_last_wk,{},' \
                                'previous_weeks_volume,{},next_weeks_open,{},next_weeks_close,{},' \
                                'percent_change_next_weeks_price,{},days_to_next_dividend,{},' \
                                'percent_return_next_dividend,{},previous_2_close_change,{},' \
                                'previous_3_close_change,{},previous_4_close_change,{},previous_2_volume_change,{},' \
                                'previous_3_volume_change,{},previous_4_volume_change,{}'.format(open1, high, low, close, volume,
                                                                            percent_change_price,
                                                                            percent_change_volume_over_last_wk,
                                                                            previous_volume,next_open1,next_close,
                                                                            percent_change_next_weeks_price,
                                                                            days_to_next_dividend,
                                                                            percent_return_next_dividend,
                                                                                   previous_2_close_change,
                                                                                   previous_3_close_change,
                                                                                   previous_4_close_change,
                                                                                   previous_2_volume_change,
                                                                                   previous_3_volume_change,
                                                                                   previous_4_volume_change)
                    with open (save_file_path, 'w', encoding='utf-8') as f:
                        f.write(write_str)

    def feature_engineering(self, input_folder, save_folder):
        file_name_list = os.listdir(input_folder)
        file_path_list = [os.path.join(input_folder, file_name) for file_name in file_name_list]

        successful_save_count = 0
        original_data_count = len(file_name_list)

        for i, file_path in enumerate(file_path_list):
            file_name = file_name_list[i]
            date = re.findall(r'([0-9]+-[0-9]+-[0-9]+)_', file_name)[0]
            stock_id = re.findall(r'_([0-9A-Za-z]+).txt', file_name)[0]
            date_obj_temp = time.strptime(date, '%Y-%m-%d')
            date_obj = datetime.datetime(*date_obj_temp[:3])

            feature_pair_dict = {}
            with open(file_path, 'r', encoding = 'utf-8') as f:

                line_list = ''
                for line in f.readlines():
                    line_list += line.strip()
                line_list = line_list.split(',')

                feature_name_list = line_list[::2]
                feature_value_list = line_list[1::2]
                for i,f_n in enumerate(feature_name_list):
                    feature_pair_dict[f_n] = feature_value_list[i]

            # ===================================================================================
            # add features
            # ===================================================================================
            #print ("feature_pair_dict: ", feature_pair_dict.keys())
            open_value = float(feature_pair_dict['open'])
            close_value = float(feature_pair_dict['close'])
            high_value = float(feature_pair_dict['high'])
            low_value = float(feature_pair_dict['low'])
            previous_weeks_volume = float(feature_pair_dict['previous_weeks_volume'])
            volume = float(feature_pair_dict['volume'])

            # (1.) candleLength
            feature_pair_dict['candleLength'] = "{:.5f}".format(abs((close_value - open_value) / (high_value - low_value)))
            # (2.) candlePos
            feature_pair_dict['candlePos'] = "{:.5f}".format(abs((high_value - open_value) / (high_value - low_value)))
            # (3.) highlow_change
            feature_pair_dict['highLowChange'] = "{:.5f}".format(abs((high_value - low_value) / (low_value)))
            # (4.) volumeChangePreviousWeek
            feature_pair_dict['volumeChangePreviousWeek'] = "{:.5f}".format((volume - previous_weeks_volume) / (previous_weeks_volume))
            # (5.) close open change
            feature_pair_dict['closeOpenChange'] = "{:.5f}".format((close_value - open_value) / (open_value))
            # -----------------------------------------------------------------------------------



            # ===================================================================================
            # delete features:
            # ===================================================================================
            delete_features_set = {'next_weeks_open',
                                   'next_weeks_close',
                                   'percent_change_volume_over_last_wk',
                                   'previous_weeks_volume',
                                   'volume',
                                   'open',
                                   'close',
                                   'high',
                                   'low'}
            for feature_name in delete_features_set:
                feature_pair_dict.pop(feature_name)
            # ===================================================================================

            # write the feature engineered file to folder
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


    def label_data(self, input_folder, save_folder, key = 'percent_change_next_weeks_price'):

        samples_list = []
        raw_data_file_name_list = os.listdir(input_folder)
        for raw_data_file_name in raw_data_file_name_list:

            sample_id = raw_data_file_name[0:-4]

            raw_data_file_path = os.path.join(input_folder, raw_data_file_name)
            with open(raw_data_file_path, 'r', encoding = 'utf-8') as f:
                sample_feature_list = f.readlines()[0].split(',')

                price_change_index = sample_feature_list.index(key)


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
        per_tuple = (1,)
        pos_label_tuple = ('pos',)
        neg_label_tuple = ('neg',)
        pos_samples_split_list = split_list_by_percentage(per_tuple, pos_samples_list)
        neg_samples_split_list = split_list_by_percentage(per_tuple, neg_samples_list)

        label_dict = collections.defaultdict(lambda: 0)
        # label postive the data and output
        for i, small_pos_samples_list in enumerate(pos_samples_split_list):
            label = pos_label_tuple[i]
            for pos_sample in small_pos_samples_list:
                label_dict[label] += 1 # count the label
                pos_sample[2] = label

        # label negative the data and output
        for i, small_neg_samples_list in enumerate(neg_samples_split_list):
            label = neg_label_tuple[i]
            for pos_sample in small_neg_samples_list:
                label_dict[label] += 1 # count the label
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
        print("label data successfully")
    #print ("label data successfully, label_dict: {}".format(list(label_dict.items()))

    def f_engineering_add_1_week_data(self, input_folder, output_folder):
        file_name_list = os.listdir(input_folder)
        file_path_list = [os.path.join(input_folder, file_name) for file_name in file_name_list]

        successful_save_count = 0
        original_data_count = len(file_name_list)

        for i, file_path in enumerate(file_path_list):
            file_name = file_name_list[i]
            date = re.findall(r'([0-9]+-[0-9]+-[0-9]+)_', file_name)[0]
            stock_id = re.findall(r'_([0-9A-Za-z]+).txt', file_name)[0]

            date_obj_temp = time.strptime(date, '%Y-%m-%d')
            date_obj = datetime.datetime(*date_obj_temp[:3])

            # find the data of the previous friday
            delta1 = datetime.timedelta(days=7)
            delta2 = datetime.timedelta(days=6)
            delta3 = datetime.timedelta(days=8)
            delta_list = [delta1, delta2, delta3]
            previous_week_stock_path_list = []
            for delta in delta_list:
                previous_week_data_obj = date_obj - delta
                previous_week_data_str = previous_week_data_obj.strftime("%Y-%m-%d")
                previous_week_stock_path = previous_week_data_str + file_name[10:]
                previous_week_stock_path = os.path.join(input_folder, previous_week_stock_path)
                previous_week_stock_path_list.append(previous_week_stock_path)

            #print ("previous_week_stock_path: ", previous_week_stock_path)
            if os.path.exists(previous_week_stock_path_list[0]):
                previous_week_stock_path = previous_week_stock_path_list[0]
            elif os.path.exists(previous_week_stock_path_list[1]):
                previous_week_stock_path = previous_week_stock_path_list[1]
            elif os.path.exists(previous_week_stock_path_list[2]):
                previous_week_stock_path = previous_week_stock_path_list[2]
            else:
                print("{}-{} has no previous week's data".format(stock_id, date))
                continue

            # previous week data
            with open (previous_week_stock_path, 'r', encoding='utf-8') as f:
                f_readlines = f.readlines()
                pre_feature_name_list = f_readlines[0].strip().split(',')[::2]
                pre_feature_value_list = f_readlines[0].strip().split(',')[1::2]

                delete_feature_list = ['days_to_next_dividend',
                                       'percent_return_next_dividend',
                                       'previous_2_close_change',
                                       'previous_3_close_change',
                                       'previous_4_close_change',
                                       'previous_2_volume_change',
                                       'previous_3_volume_change',
                                       'previous_4_volume_change',
                                       'percent_change_next_weeks_price',
                                       ]

                #print ("pre_feature_name_list: ", pre_feature_name_list)
                for delete_feature in delete_feature_list:
                    delete_index = pre_feature_name_list.index(delete_feature)
                    #print ("delete_index: ", delete_index)
                    pre_feature_name_list.remove(pre_feature_name_list[delete_index])
                    pre_feature_value_list.remove(pre_feature_value_list[delete_index])


                pre_feature_name_list = ['previous_week_' + feature_n for feature_n in pre_feature_name_list]
            # this week data
            with open (file_path, 'r', encoding='utf-8') as f:
                f_readlines = f.readlines()
                feature_name_list = f_readlines[0].strip().split(',')[::2]
                feature_value_list = f_readlines[0].strip().split(',')[1::2]


            combined_feature_name_list = pre_feature_name_list + feature_name_list
            combined_feature_value_list = pre_feature_value_list + feature_value_list
            feature_list = [j for i in zip(combined_feature_name_list, combined_feature_value_list) for j in i]
            feature_list_str = ','.join(feature_list)
            save_path = os.path.join(output_folder, file_name)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(feature_list_str)

            successful_save_count += 1

        print ("Save {}/{} files to {}".format(successful_save_count, original_data_count, output_folder))

    def price_change_regression(self, input_folder, save_folder, key = 'percent_change_next_weeks_price'):

        raw_data_file_name_list = os.listdir(input_folder)
        for raw_data_file_name in raw_data_file_name_list:

            # read percent_change_next_weeks_price
            short_file_name = raw_data_file_name[0:-4]
            raw_data_file_path = os.path.join(input_folder, raw_data_file_name)
            with open(raw_data_file_path, 'r', encoding = 'utf-8') as f:
                sample_feature_list = f.readlines()[0].split(',')
                price_change_index = sample_feature_list.index(key)
                sample_price_change = float(sample_feature_list[price_change_index + 1]) * 0.01
                del sample_feature_list[price_change_index: price_change_index+2]
            #

            # modify file name and save
            short_file_name = short_file_name.replace('_1', '')
            short_file_name = short_file_name.replace('_2', '')
            save_file_name = short_file_name + "_#{:.5f}#".format(sample_price_change) + '.txt'
            save_file_folder = os.path.join(save_folder, save_file_name)
            with open (save_file_folder, 'w', encoding='utf-8') as f:
                f.write(','.join(sample_feature_list))
            #

        print ("save regression data of dow_jones, total: {}, save_folder: {}\n".
               format(len(raw_data_file_name_list), save_folder))