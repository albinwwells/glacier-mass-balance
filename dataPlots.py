import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm

def elevationBinPlot(xVal1, yVal1, xVal2, yVal2, xLabel1, yLabel1, xLabel2, title1,
                     elevationBinWidth, minVal, buff, alpha):
    '''
    Plot y (elevation bin) vs two x data sources: one primary one (xVal1) and a secondary bar graph (xVal2)
    :param xVal1: Primary x values (scatter plot)
    :param yVal1: Primary values
    :param xVal2: Secondary x values (bar graph)
    :param yVal2: Secondary y values
    :param xLabel1, xlabel2, yLabel1, title: Chart axis labels and titles (xLabel2 is above plot)
    :param elevationBinWidth: Elevation bin width
    :param minVal: Minimum x-value of plot: if 0, plot begins at 0. Otherwise it begins at the minimum value in
    xVal1 minus buff.
    :param buff: x-axis minimum and maximum value buffer
    :param alpha: opacity of bar plot
    :return: No return: plots the figure
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(131, label="2", frame_on=False)

    ax.scatter(xVal1, yVal1)
    ax.set_xlabel(xLabel1)
    ax.set_ylabel(yLabel1)
    ax.tick_params(axis='x')
    ax.barh(xVal2, yVal2, elevationBinWidth, alpha=0)
    if minVal == 0:
        ax.set_xlim([0, max(xVal1)+buff])
    else:
        ax.set_xlim([min(xVal1) - buff, max(xVal1) + buff])

    ax2.barh(xVal2, yVal2, elevationBinWidth, alpha=alpha, color="C1")
    ax2.xaxis.tick_top()
    ax2.set_xlabel(xLabel2, color="C1")
    ax2.xaxis.set_label_position('top')
    ax2.tick_params(axis='x', colors="C1")

    fig.suptitle(title1, weight='bold')
    fig.tight_layout()
    plt.show()
    figName = title1.replace(' ', '_') + '.png'
    fig.savefig('Figures/' + figName)


def elevationBinPlot2(xVal1, labx1, xVal1_2, labx1_2, yVal1, xVal2, yVal2, xLabel1, yLabel1, xLabel2, title1,
                     elevationBinWidth, minVal, buff, alpha):
    '''
    Plot y (elevation bin) vs two x data sources: one primary one (xVal1) and a secondary bar graph (xVal2)
    :param xVal1: Primary x values (scatter plot)
    :param labx1: scatter plot legend label for xVal1
    :param xVal1_2: Second x values (scatter plot)
    :param labx1_2: scatter plot legend label for xVal1_2
    :param yVal1: Primary values
    :param xVal2: Secondary x values (bar graph)
    :param yVal2: Secondary y values
    :param xLabel1, xlabel2, yLabel1, title: Chart axis labels and titles (xLabel2 is above plot)
    :param elevationBinWidth: Elevation bin width
    :param minVal: Minimum x-value of plot: if 0, plot begins at 0. Otherwise it begins at the minimum value in
    xVal1 minus buff.
    :param buff: x-axis minimum and maximum value buffer
    :param alpha: opacity of bar plot
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(131, label="2", frame_on=False)

    ax.scatter(xVal1, yVal1, label=labx1)
    ax.set_xlabel(xLabel1)
    ax.set_ylabel(yLabel1)
    ax.tick_params(axis='x')
    ax.barh(xVal2, yVal2, elevationBinWidth, alpha=0)
    if minVal == 0:
        ax.set_xlim([0, max(xVal1)+buff])
    else:
        ax.set_xlim([min(xVal1) - buff, max(xVal1) + buff])

    ax.scatter(xVal1_2, yVal1, label=labx1_2)
    ax.legend()

    ax2.barh(xVal2, yVal2, elevationBinWidth, alpha=alpha, color="C1")
    ax2.xaxis.tick_top()
    ax2.set_xlabel(xLabel2, color="C1")
    ax2.xaxis.set_label_position('top')
    ax2.tick_params(axis='x', colors="C1")

    fig.suptitle(title1, weight='bold')
    fig.tight_layout()
    plt.show()
    figName = title1.replace(' ', '_') + '.png'
    fig.savefig('Figures/' + figName)


def elevationBinPlot2Subfigs(x_subp1, x_subp1_lab, x_subp1_2, x_subp1_lab_2, y_subp1, x2_subp1, y2_subp1, xLabel1,
                             yLabel1, xLabel2, title1, elevationBinWidth, minVal, buff, alpha, title,
                             subp3, title3, cbar3, subp4, title4, cbar4, subp5, title5, cbar5, cbar4ticks=None):
    '''
    Plot y (elevation bin) vs two x data sources: one primary one (xVal1) and a secondary bar graph (xVal2)
    Includes 3 subplots showing a desired source: emergence velocity, thickness, speed
    :param x_subp1: Primary x values (scatter plot)
    :param x_subp1_lab: scatter plot legend label for xVal1
    :param x_subp1_2: Second x values (scatter plot)
    :param x_subp1_lab_2: scatter plot legend label for xVal1_2
    :param y_subp1: Primary values
    :param x2_subp1: Secondary x values (bar graph)
    :param y2_subp1: Secondary y values
    :param xLabel1, xlabel2, yLabel1, title: Chart axis labels and titles (xLabel2 is above plot)
    :param title1: title of subplot 1
    :param elevationBinWidth: Elevation bin width
    :param minVal: Minimum x-value of plot: if 0, plot begins at 0. Otherwise it begins at the minimum value in
    xVal1 minus buff.
    :param buff: x-axis minimum and maximum value buffer
    :param alpha: opacity of bar plot
    :return: No return: plots the figure
    :param title: overall figure title
    :param subp2, subp3, subp4: array for subplots 2, 3, and 4
    :param title2, title3, title4: title for subplots 2, 3, and 4
    :param cbar2, cbar3, cbar4: colorbar for subplots 2, 3, and 4
    '''
    fig = plt.figure()
    ax = fig.add_subplot(221, label="1")
    ax2 = fig.add_subplot(261, label="2", frame_on=False)

    ax3 = fig.add_subplot(222)
    ax4 = fig.add_subplot(223)
    ax5 = fig.add_subplot(224)

    ax.scatter(x_subp1, y_subp1, label=x_subp1_lab)
    ax.set_xlabel(xLabel1)
    ax.set_ylabel(yLabel1)
    ax.set_title(title1, pad=10)
    ax.tick_params(axis='x')
    ax.barh(x2_subp1, y2_subp1, elevationBinWidth, alpha=0)
    if minVal == 0:
        ax.set_xlim([0, max(x_subp1)+buff])
    else:
        ax.set_xlim([min(x_subp1) - buff, max(x_subp1) + buff])

    ax.scatter(x_subp1_2, y_subp1, label=x_subp1_lab_2)
    ax.legend()

    ax2.barh(x2_subp1, y2_subp1, elevationBinWidth, alpha=alpha, color="C1")
    ax2.xaxis.tick_top()
    ax2.set_xlabel(xLabel2, color="C1")
    ax2.xaxis.set_label_position('top')
    ax2.tick_params(axis='x', colors="C1")

    # add 3 subplots:
    divnorm = TwoSlopeNorm(vmin=subp3.min(), vcenter=0, vmax=max(subp3.max(), 1))               # center colorbar at 0 (white)
    im3 = ax3.imshow(subp3, cmap=plt.cm.get_cmap('RdBu', len(x_subp1)), norm=divnorm)   # plot array
    divider3 = make_axes_locatable(ax3)                         # plot on ax3 (northeast subplot)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)   # specify colorbar axis properties
    fig.colorbar(im3, cax=cax3, label=cbar3)                    # add colorbar
    ax3.set_title(title3, pad=10)                               # add subplot title

    subp4 = subp4.astype(float)
    subp4[subp4 == 0] = np.nan                                  # to remove 0 values for white background in plot
    im4 = ax4.imshow(subp4)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes('right', size='5%', pad=0.05)
    cbar4 = fig.colorbar(im4, cax=cax4, label=cbar4)
    if cbar4ticks != None:              # this would be if we want colorbar ticks/labels different from actual values
        cbar4.ax.locator_params(nbins=len(cbar4ticks)//2)
        cbar4.ax.set_yticklabels(np.linspace(cbar4ticks[0], cbar4ticks[-1], len(cbar4ticks)//2).astype(int))
    ax4.set_title(title4, pad=10)

    subp5 = subp5.astype(float)
    subp5[subp5 == 0] = np.nan
    im5 = ax5.imshow(subp5)
    divider5 = make_axes_locatable(ax5)
    cax5 = divider5.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im5, cax=cax5, label=cbar5)
    ax5.set_title(title5, pad=10)

    fig.set_size_inches(12, 8)
    fig.tight_layout(pad=0.5, w_pad=-3.0, h_pad=2.0)
    fig.suptitle(title, weight='bold')
    plt.show()
    figName = title.replace(' ', '_') + '.png'
    fig.savefig('Figures/' + figName)

