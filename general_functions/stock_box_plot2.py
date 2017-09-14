import matplotlib.pyplot as plt
import numpy as np
import random
from pylab import plot, show, savefig, xlim, figure, \
    hold, ylim, legend, boxplot, setp, axes

def setBoxColors(bp,metrics_name_list):
    colour_list = ['blue', 'red', 'green', 'cyan', 'yellow', 'magenta']
    box_count = 0
    caps_count = 0
    whiskers_count = 0
    medians_count = 0
    fliers_count = 0

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
        plt.setp(bp['fliers'][fliers_count], markeredgecolor=colour_list[i])
        fliers_count += 1


def get_positions(metrics_name_list, position_now, box_gap):
    positions_list = []
    for i,_ in enumerate(metrics_name_list):
        positions_list.append(position_now)
        position_now += box_gap
    return positions_list, position_now

def stock_metrics_result_box_plot(metrics_result_dict, trail_number_list, metrics_name_list, title = ''):
    box_widths = 0.3
    box_gap = 0.5
    category_gap = 0.7
    position_now = 0
    category_pos_list = []
    fig = figure()
    ax = axes()
    hold(True)

    # Some fake data to plot
    for trail_number in trail_number_list:
        X = []
        for metrics in metrics_name_list:
            X.append(metrics_result_dict[trail_number][metrics])

        # first boxplot pair
        position_now += category_gap
        positions_list,position_now = get_positions(metrics_name_list, position_now, box_gap)
        category_pos_list.append(np.average(positions_list))
        bp = boxplot(X, positions=positions_list, widths=box_widths, sym='+')
        setBoxColors(bp,metrics_name_list)


    # set axes limits and labels
    xlim(0, 13)
    ylim(0.28, 0.6)
    ax.set_xticklabels(trail_number_list)
    ax.set_xticks(category_pos_list)
    ax.set_xlabel('Number of trails in 1 experiment')
    ax.set_title(title)

    # draw temporary red and blue lines and use them to create a legend
    h_list = []
    shape = '-'
    legend_list = ['b{}'.format(shape),
                   'r{}'.format(shape),
                   'g{}'.format(shape),
                   'c{}'.format(shape),
                   'y{}'.format(shape),
                   'm{}'.format(shape)]

    for i,_ in enumerate(metrics_name_list):
        h, = plot([1, 1], legend_list[i])
        h_list.append(h)
        #h.set_visible(False)

    # hB, = plot([1, 1], 'b-')
    # hR, = plot([1, 1], 'r-')
    legend(h_list, metrics_name_list)
    for h in h_list:
        h.set_visible(False)
    # hB.set_visible(False)
    # hR.set_visible(False)
    show()


def model_result_box_plot(result_dict, model_list, data_preprocessing_list, metrics_name_list,
                          title ='', x_label = '', xlim_range = (0,15), ylim_range = (0.2, 0.6)
                          ,metrics_print_list = '', plot_baseline = False, baseline_value_tuple = None,
                          baseline_legend_tuple = None, baseline_colour_tuple = None):
    box_widths = 0.3
    box_gap = 0.5
    category_gap =1.5
    position_now = 0
    category_pos_list = []
    fig = figure()
    ax = axes()
    hold(True)

    # Some fake data to plot
    for model in model_list:
        X = []
        model_result_dict = result_dict[model]
        for metric in metrics_name_list:
            metric_value_list = []
            for data_preprocessing in data_preprocessing_list:
                value_list = model_result_dict[data_preprocessing][metric]
                metric_value_list.extend(value_list)
            X.append(metric_value_list)

        # print the highest result
        for i, metrics_list in enumerate(X):
            if metrics_list:
                metric = metrics_name_list[i]
                if metric == 'rmse_list':
                    max_value = sorted(metrics_list)[0]
                else:
                    max_value = sorted(metrics_list, reverse=True)[0]

                print ("{}-best {}: {}".format(model, metric, max_value))
        #

        # first boxplot pair
        position_now += category_gap
        positions_list,position_now = get_positions(metrics_name_list, position_now, box_gap)
        category_pos_list.append(np.average(positions_list))
        bp = boxplot(X, positions=positions_list, widths=box_widths, sym='+')
        setBoxColors(bp,metrics_name_list)


    # set axes limits and labels
    xlim(*xlim_range)
    ylim(*ylim_range)
    ax.set_xticklabels(x_label)
    ax.set_xticks(category_pos_list)
    #ax.set_xlabel(x_label)
    ax.set_title(title)

    # draw temporary red and blue lines and use them to create a legend
    h_list = []
    shape = '-'
    legend_list = ['b{}'.format(shape),
                   'r{}'.format(shape),
                   'g{}'.format(shape),
                   'c{}'.format(shape),
                   'y{}'.format(shape),
                   'm{}'.format(shape)]
    
    if not metrics_print_list:
        metrics_print_list = metrics_name_list

    for i,_ in enumerate(metrics_print_list):
        h = plot([1, 1], legend_list[i])[0]
        h_list.append(h)
        #h.set_visible(False)
    # hB, = plot([1, 1], 'b-')
    # hR, = plot([1, 1], 'r-')
    if plot_baseline and baseline_value_tuple:
        for i, baseline_value in enumerate(baseline_value_tuple):
            baseline_plot = plt.plot((0, 99), (baseline_value, baseline_value),
                                     '{}-'.format(baseline_colour_tuple[i]), dashes=[2, 5])[0]
            h_list.append(baseline_plot)
            metrics_print_list.append('{}'.format(baseline_legend_tuple[i]))

        #legend(baseline_plot, 'baseline'

    legend(h_list, metrics_print_list)
    for i, h in enumerate(h_list):
        if i >= len(baseline_legend_tuple):
            pass
        else:
            h.set_visible(False)
    # hB.set_visible(False)
    # hR.set_visible(False)

    else:
        print ("Check plot baseline and baseline_value_tuple")



    show()

def data_preprocessing_result_box_plot(result_dict, model_list, data_preprocessing_list, metrics_name_list,
                                       data_preprocessing_show_list, metrics_show_list, title='',
                                       x_label='', xlim_range = (0,15), ylim_range = (0.2, 0.6), plot_baseline= True
                                       ,baseline_value_tuple = None,
                          baseline_legend_tuple = None, baseline_colour_tuple = None):
    box_widths = 0.3
    box_gap = 0.5
    category_gap = 1.5
    position_now = 0
    category_pos_list = []
    fig = figure()
    ax = axes()
    hold(True)

    # Some fake data to plot
    # model_result_dict = result_dict[model_list]
    for data_preprocessing in data_preprocessing_list:
        X = []
        for metric in metrics_name_list:
            model_metric_list = []
            for model in model_list:
                model_metric_list.extend(result_dict[model][data_preprocessing][metric])
            X.append(model_metric_list)
        # first boxplot pair
        position_now += category_gap
        positions_list, position_now = get_positions(metrics_name_list, position_now, box_gap)
        category_pos_list.append(np.average(positions_list))
        bp = boxplot(X, positions=positions_list, widths=box_widths, sym='+')
        setBoxColors(bp, metrics_name_list)

    # set axes limits and labels
    xlim(*xlim_range)
    ylim(*ylim_range)
    ax.set_xticklabels(data_preprocessing_show_list)
    ax.set_xticks(category_pos_list)
    ax.set_xlabel(x_label)
    ax.set_title(title)

    # draw temporary red and blue lines and use them to create a legend
    h_list = []
    shape = '-'
    legend_list = ['b{}'.format(shape),
                   'r{}'.format(shape),
                   'g{}'.format(shape),
                   'c{}'.format(shape),
                   'y{}'.format(shape),
                   'm{}'.format(shape)]

    for i, _ in enumerate(metrics_show_list):
        h, = plot([1, 1], legend_list[i])
        h_list.append(h)
        # h.set_visible(False)

    # hB, = plot([1, 1], 'b-')
    # hR, = plot([1, 1], 'r-')



    if plot_baseline and baseline_value_tuple:
        for i, baseline_value in enumerate(baseline_value_tuple):
            baseline_plot = plt.plot((0, 99), (baseline_value, baseline_value),
                                     '{}-'.format(baseline_colour_tuple[i]), dashes=[2, 5])[0]
            h_list.append(baseline_plot)
            metrics_show_list.append('{}'.format(baseline_legend_tuple[i]))


    legend(h_list, metrics_show_list)
    for i, h in enumerate(h_list):
        if i >= len(baseline_legend_tuple):
            pass
        else:
            h.set_visible(False)
    show()