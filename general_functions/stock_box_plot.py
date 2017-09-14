import matplotlib.pyplot as plt
import numpy as np
import random

def setBoxColors(bp,metrics_name_list):
    colour_list = ['blue', 'red', 'black', 'grey', 'green']
    box_count = 0
    caps_count = 0
    whiskers_count = 0
    medians_count = 0

    for i, metric in enumerate(metrics_name_list):
        plt.setp(bp['boxes'][box_count], color=colour_list[i])
        box_count += 1
        plt.setp(bp['caps'][caps_count], color=colour_list[i])
        caps_count += 1
        plt.setp(bp['caps'][caps_count], color=colour_list[i])
        caps_count += 1
        plt.setp(bp['whiskers'][whiskers_count], color=colour_list[i])
        whiskers_count += 1
        plt.setp(bp['whiskers'][whiskers_count], color=colour_list[i])
        whiskers_count += 1
        plt.setp(bp['medians'][medians_count], color=colour_list[i])
        medians_count += 1




def stock_metrics_result_box_plot(trail_number_list, metrics_name_list):
    data = {}
    for x in trail_number_list:
        data[x] = {}
        for type in metrics_name_list:
                data[x][type] = np.random.uniform(0, 800, size=50)
    fig, axes = plt.subplots(ncols=len(trail_number_list), sharey=True)
    #fig.subplots_adjust(wspace=0)

    for ax, name in zip(axes, trail_number_list):
        bp = ax.boxplot([data[name][item] for item in metrics_name_list],widths = 0.5)
        setBoxColors(bp, metrics_name_list)
        #ax.set(xticklabels=metrics_name_list, xlabel=name)
        ax.get_xaxis().set_ticks([])
        ax.set_xlabel('Number of trails')
        ax.set( xlabel=name)
        fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

        ax.margins(0.2) # Optional
    plt.show()

