import numpy as np
import sys

# top n average
def top_n_avg_strategy(actual_id_value_dict, pred_v_dict, include_top=1):
    '''Buy top n stocks evenly, and compute their average profit
    actual_id_value_dict[key], key = (date, best_stock_id) tuple
    pred_v_dict[date], date = '2015-09-09', value: pred_v_pair_list: [(stock1, value1), (stock2, value2), ...]
    '''
    date_actual_avg_priceChange_list = []

    for date, pred_v_pair_list in pred_v_dict.items():
        sorted_pred_v_pair_list = sorted(pred_v_pair_list, key = lambda x:x[1], reverse = True)
        chosen_pred_v_pair = sorted_pred_v_pair_list[0:include_top]
        top_n_actual_priceChange_list = []
        # chosen_pred_v_pair:  [('VZ', 0.033974090949209501)] for top = 1
        for pred_v_pair in chosen_pred_v_pair:
            id = pred_v_pair[0]
            date_id_pair = (date, id)
            actual_priceChange = actual_id_value_dict[date_id_pair]
            top_n_actual_priceChange_list.append(actual_priceChange)
        top_n_price_average = np.average(top_n_actual_priceChange_list)
        date_actual_avg_priceChange_list.append(top_n_price_average)
    average_profit = np.average(date_actual_avg_priceChange_list)
    var = np.var(date_actual_avg_priceChange_list)
    std = np.std(date_actual_avg_priceChange_list)

    return average_profit, var, std


# top n average
def top_1_stock_return(actual_id_value_dict, pred_v_dict, include_top=1):
    '''Buy top n stocks evenly, and compute their average profit
    actual_id_value_dict[key], key = (date, best_stock_id) tuple
    pred_v_dict[date], date = '2015-09-09', value: pred_v_pair_list: [(stock1, value1), (stock2, value2), ...]
    '''
    date_actual_avg_priceChange_list = []

    for date, pred_v_pair_list in pred_v_dict.items():
        sorted_pred_v_pair_list = sorted(pred_v_pair_list, key = lambda x:x[1], reverse = True)
        chosen_pred_v_pair = sorted_pred_v_pair_list[0:include_top]
        top_n_actual_priceChange_list = []
        for pred_v_pair in chosen_pred_v_pair:
            id = pred_v_pair[0]
            date_id_pair = (date, id)
            actual_priceChange = actual_id_value_dict[date_id_pair]
            top_n_actual_priceChange_list.append(actual_priceChange)
        top_n_price_average = np.average(top_n_actual_priceChange_list)
        date_actual_avg_priceChange_list.append(top_n_price_average)

    return date_actual_avg_priceChange_list
