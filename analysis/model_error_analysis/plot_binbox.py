import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['ytick.direction'] = 'in'
import numpy as np
from collections import OrderedDict



def plot_binbox(datadict:OrderedDict, title, logy=False, size=None):
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    data_list = [v['data'] for k, v in datadict.items()]
    tick_label_list = [v['label'] for k, v in datadict.items()]
    num_boxes = len(data_list)
    fig = plt.figure(figsize=size)
    bp = plt.boxplot(data_list, 0, '', showmeans=False, patch_artist=True)

    # set box attributes
    boxatrr = ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']
    for att in boxatrr:
        cnt = 0
        for k, v in datadict.items():
            if v.get(att) is not None:
                setting = v.get(att)
                plt.setp(bp[att][cnt], **setting)
                pass
            cnt += 1

    ax = plt.gca()
    if logy:
        ax.set_yscale('log')

    # show median/mean number
    medians = [bp['medians'][i].get_ydata()[0] for i in range(num_boxes)]
    # means = [bp['means'][i].get_ydata()[0] for i in range(num_boxes)]

    pos = np.arange(num_boxes) + 1
    upper_labels_median = ['{:.4f}'.format(s) for s in medians]
    # upper_labels_mean = ['{:.4f}'.format(s) for s in means]


    for tick, label in zip(range(num_boxes), ax.get_xticklabels()):
        k = tick % 2
        t = plt.text(pos[tick], 1.2, upper_labels_median[tick],
                 transform=ax.get_xaxis_transform(),
                 horizontalalignment='center', size=8,
                 color='C1')
        t.set_in_layout(True)
        # t = plt.text(pos[tick], 1.1, upper_labels_mean[tick],
        #          transform=ax.get_xaxis_transform(),
        #          horizontalalignment='center', size=8,
        #         color='C2')
        # t.set_in_layout(True)


    ax.tick_params(axis=u'x', which=u'both', length=0)
    ax.set_xticklabels(tick_label_list)
    for tick in ax.get_xticklabels():
        tick.set_rotation(40)
    plt.title(title)
    # plt.xlim(0,8)


    # set fig size
    fig.tight_layout()

    return fig