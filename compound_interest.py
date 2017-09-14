import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import collections
import re
import os


# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname(os.path.abspath(__file__))
regression_folder = os.path.join(parent_folder, 'data', 'a_share', 'a_share_regression_data')
# ==========================================================================================================


def compute_sigma(date_folder):
    file_name_list = os.listdir(date_folder)
    priceChange_list = []
    for file_name in file_name_list:
        priceChange = float(re.findall(r'#([0-9e\.\-]+)#', file_name)[0])
        priceChange_list.append(priceChange)
        
    priceChange_std = np.std(priceChange_list)
    print ("all samples priceChange_std: {}".format(priceChange_std))
    return priceChange_std

# compute the sigma for all samples
#compute_sigma(regression_folder)


initial_capital = 50000
trading_percent = 0.004
avg_profit = 0.029
avg_profit -= trading_percent

mu = avg_profit
sigma = 0.075 # mean and standard deviation
week_num = 20
run_loop = 1000
capital_all_list = []

def calculate_capital(capital, mu, sigma, week_num, is_plot = False, multiple_run = True):
    profit_range_count_list = []
    week_profit_list = np.random.normal(mu, sigma, week_num)
    profit_range_list = np.arange(-1.1, 1.1, 0.01)


    profit_range_tuple_list  = [(x, profit_range_list[i+1]) for i, x in enumerate(profit_range_list) if i <= len(profit_range_list) - 2]
    for profit_range_tuple in profit_range_tuple_list:
        low_limit = profit_range_tuple[0]
        up_limit = profit_range_tuple[1]
        range_count = ((week_profit_list >= low_limit) & (week_profit_list < up_limit)).sum()
        profit_range_count_list.append(range_count)



    
    week_list = [x for x in range(week_num)]
    
    if is_plot:
        # plot
        plt.bar([x for x in range(len(profit_range_count_list))], profit_range_count_list, align='center')
        plt.show()
        #

    for week_profit in week_profit_list:
        capital = capital*(1+week_profit)
        
        
    if not multiple_run:
        print ("profit_range_list: ", profit_range_list)
        print ("week_profit_list: ", week_profit_list)
        print ("capital: ", capital)
        
    return capital


for i in range(run_loop):
    
    capital = calculate_capital(initial_capital, mu, sigma, week_num,  multiple_run = True)
    capital_all_list.append(capital)
    
lose_money_count = ((np.array(capital_all_list) < initial_capital)).sum()
lose_money_precent = "{:.5f}".format(100*lose_money_count / run_loop)


print ("Run loop: {}".format(run_loop))
print ("lose_money_precent {}%, avg_capital {}, min_capital {}, max_capital {} in {} weeks, avg_profit_per_week: {}, sigma: {}, initial_capital: {}"
.format(lose_money_precent, np.average(capital_all_list), min(capital_all_list), max(capital_all_list), week_num, avg_profit, sigma, initial_capital))
    
    
    