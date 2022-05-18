import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import matplotlib.image as img
from PIL import Image

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

    ax2.barh(xVal2, yVal2, elevationBinWidth, alpha=alpha, color="C1", zorder=0)
    ax2.xaxis.tick_top()
    ax2.set_xlabel(xLabel2, color="C1")
    ax2.xaxis.set_label_position('top')
    ax2.tick_params(axis='x', colors="C1")

    fig.suptitle(title1, weight='bold')
    fig.tight_layout()
    plt.show(block=False)
    figName = title1.replace(' ', '_') + '.png'
    fig.savefig('Figures/' + figName)
    # Image.open('Figures/' + figName).show()

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
    ax.barh(xVal2, yVal2, elevationBinWidth, alpha=0, zorder=0)
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
    plt.show(block=False)
    figName = title1.replace(' ', '_') + '.png'
    fig.savefig('Figures/' + figName)
    # Image.open('Figures/' + figName).show()

def elevationBinPlot3Subfigs(x_subp1, x_subp1_lab, x_subp1_2, x_subp1_lab_2, x_subp1_3, x_subp1_lab_3, y_subp1,
                             x2_subp1, y2_subp1, xLabel1, yLabel1, xLabel2, title1, elBinWidth, buff, alpha, title, res,
                             subp3, title3, cbar3, color3,
                             subp4, title4, cbar4, color4,
                             subp5, title5, cbar5, color5,
                             err_subp1_2=None, err_subp1_3=None,
                             x_subp1_ref=None, x_subp1_lab_ref=None, y_subp1_ref=None, ref_var=None, cbar4ticks=None):
    '''
    Plot y (elevation bin) vs two x data sources: one primary one (xVal1) and a secondary bar graph (xVal2)
    Includes 3 subplots showing a desired source: emergence velocity, thickness, speed
    :param x_subp1: Primary x values (scatter plot)
    :param x_subp1_lab: scatter plot legend label for xVal1
    :param x_subp1_2: Second x values (scatter plot)
    :param x_subp1_lab_2: scatter plot legend label for xVal1_2
    :param x_subp1_3: Third x values (scatter plot)
    :param x_subp1_lab_3: scatter plot legend label for xVal1_3
    :param x_subp1_ref: x-values for the known data
    :param x_subp1_lab_ref: label for the known data
    :param y_subp1: Primary values
    :param y_subp1_ref: y-values for the known data
    :param x2_subp1: Secondary x values (bar graph)
    :param y2_subp1: Secondary y values
    :param xLabel1, xlabel2, yLabel1, title: Chart axis labels and titles (xLabel2 is above plot)
    :param title1: title of subplot 1
    :param elevationBinWidth: Elevation bin width
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

    ax.scatter(x_subp1, y_subp1, label=x_subp1_lab, color='w', edgecolor='dimgray', zorder=2)
    ax.plot(x_subp1, y_subp1, color='dimgray', zorder=1)
    ax.set_xlabel(xLabel1)
    ax.set_ylabel(yLabel1)
    ax.set_title(title1, pad=10)
    ax.tick_params(axis='x')
    ax.barh(x2_subp1, y2_subp1, elBinWidth, alpha=0, zorder=0)
    if err_subp1_2 is not None:
        if x_subp1_ref is None:
            if len(err_subp1_2.shape) > 1:
                ax.set_xlim([min(np.min(x_subp1_2 - err_subp1_2[0]), np.min(x_subp1_3 - err_subp1_3[0])) - 50,
                             max(np.max(x_subp1_2 + err_subp1_2[-1]), np.max(x_subp1_3 + err_subp1_3[-1])) + 50])
            else:
                ax.set_xlim([min(np.min(x_subp1_2 - err_subp1_2), np.min(x_subp1_3 - err_subp1_3)) - 50,
                             max(np.max(x_subp1_2 + err_subp1_2), np.max(x_subp1_3 + err_subp1_3)) + 50])
        else:
            if len(err_subp1_2.shape) > 1:
                ax.set_xlim([min(np.min(x_subp1_2 - err_subp1_2[0]), np.min(x_subp1_3 - err_subp1_3[0]),
                                 np.min(x_subp1_ref)) - 20,
                             max(np.max(x_subp1_2 + err_subp1_2[-1]), np.max(x_subp1_3 + err_subp1_3[-1]),
                                 np.max(x_subp1_ref), 0) + 20])
            else:
                ax.set_xlim(
                    [min(np.min(x_subp1_2 - err_subp1_2), np.min(x_subp1_3 - err_subp1_3), np.min(x_subp1_ref)) - 20,
                     max(np.max(x_subp1_2 + err_subp1_2), np.max(x_subp1_3 + err_subp1_3), np.max(x_subp1_ref),
                         0) + 20])
            ax.plot(x_subp1_ref, y_subp1_ref, color='black', zorder=5)  # plot line to connect reference data
    elif x_subp1_ref is not None:
        ax.set_xlim([min(np.min(x_subp1_ref), np.min(x_subp1))-buff, max(np.max(x_subp1_ref), np.max(x_subp1), 0)+buff])
        ax.plot(x_subp1_ref, y_subp1_ref, color='black', zorder=5)  # plot line to connect reference data
    else:
        ax.set_xlim([min(x_subp1) - buff, max(np.max(x_subp1), 0) + buff])

    ax2.barh(x2_subp1, y2_subp1, elBinWidth, alpha=alpha, color='dimgray')
    ax2.xaxis.tick_top()
    ax2.set_xlabel(xLabel2, color='dimgray')
    ax2.xaxis.set_label_position('top')
    ax2.tick_params(axis='x', colors='dimgray')

    ax.vlines(x_subp1_2, y_subp1 - np.random.uniform(elBinWidth/2, elBinWidth/2, len(y_subp1)),
              y_subp1 + np.random.uniform(elBinWidth/2, elBinWidth/2, len(y_subp1)), label=x_subp1_lab_2,
              lw=2, color='#01665e', zorder=7)
    ax.errorbar(x_subp1_2, y_subp1 - np.random.uniform(elBinWidth/10, elBinWidth/10, len(y_subp1)), xerr=err_subp1_2,
                fmt='', color='#01665e', ls='none', ecolor='#35978f', elinewidth=2, alpha=0.6, zorder=3)
    ax.vlines(x_subp1_3, y_subp1 - np.random.uniform(elBinWidth/2, elBinWidth/2, len(y_subp1)),
              y_subp1 + np.random.uniform(elBinWidth/2, elBinWidth/2, len(y_subp1)), label=x_subp1_lab_3,
              lw=2, color='#8c510a', zorder=6)
    ax.errorbar(x_subp1_3, y_subp1 + np.random.uniform(elBinWidth/10, elBinWidth/10, len(y_subp1)), xerr=err_subp1_3,
                fmt='', color='#8c510a', ls='none', ecolor='#bf812d', elinewidth=2, alpha=0.6, zorder=4)
    ax.scatter(x_subp1_ref, y_subp1_ref, label=x_subp1_lab_ref, color='black', zorder=5)
    ax.errorbar(x_subp1_ref, y_subp1_ref, xerr=ref_var,
                fmt='', color='black', ls='none', ecolor='black', elinewidth=2, alpha=0.6, zorder=3)
    if x_subp1_ref != None:
        m, b = np.polyfit(x_subp1_ref, y_subp1_ref, 1)
        ax.axline((0, b), slope=m, color='black', ls='--', alpha=0.8, zorder=6)
    ax.axvline(x=0, color='black', lw=0.5, dashes=(10, 3))
    ax.legend()

    # add 3 subplots
    # top right plot:
    divnorm3 = TwoSlopeNorm(vmin=min(subp3.min(), -1), vcenter=0, vmax=max(subp3.max(), 1))       # center colorbar at 0
    im3 = ax3.imshow(subp3, cmap=plt.cm.get_cmap(color3, len(x_subp1)), norm=divnorm3)   # plot array
    divider3 = make_axes_locatable(ax3)                         # plot on ax3 (northeast subplot)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)   # specify colorbar axis properties
    fig.colorbar(im3, cax=cax3, label=cbar3)                    # add colorbar
    ax3.set_title(title3, pad=10)                               # add subplot title
    scalebar3 = AnchoredSizeBar(ax3.transData, 1000 / res, '1 km', 'lower left', pad=0.1,
                                color='k', frameon=False, size_vertical=1, fontproperties=fm.FontProperties(size=10))
    ax3.add_artist(scalebar3)

    # bottom left plot:
    divnorm4 = TwoSlopeNorm(vmin=min(subp4.min(), -1), vcenter=0, vmax=max(subp4.max(), 1))  # center colorbar at 0
    # subp4 = subp4.astype(float)
    # subp4[subp4 == 0] = np.nan                                  # to remove 0 values for white background in plot
    im4 = ax4.imshow(subp4, cmap=plt.cm.get_cmap(color4, len(x_subp1)), norm=divnorm4)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes('right', size='5%', pad=0.05)
    cbar4 = fig.colorbar(im4, cax=cax4, label=cbar4)
    if cbar4ticks != None:              # this would be if we want colorbar ticks/labels different from actual values
        cbar4.ax.locator_params(nbins=len(cbar4ticks)//2)
        cbar4.ax.set_yticklabels(np.linspace(cbar4ticks[0], cbar4ticks[-1], len(cbar4ticks)//2).astype(int))
    ax4.set_title(title4, pad=10)
    scalebar4 = AnchoredSizeBar(ax4.transData, 1000 / res, '1 km', 'lower left', pad=0.1,
                                color='k', frameon=False, size_vertical=1, fontproperties=fm.FontProperties(size=10))
    ax4.add_artist(scalebar4)

    # bottom right plot:
    divnorm5 = TwoSlopeNorm(vmin=min(subp5.min(), -1), vcenter=0, vmax=max(subp5.max(), 1))  # center colorbar at 0
    im5 = ax5.imshow(subp5, cmap=plt.cm.get_cmap(color5, len(x_subp1)), norm=divnorm5)
    divider5 = make_axes_locatable(ax5)
    cax5 = divider5.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im5, cax=cax5, label=cbar5)
    ax5.set_title(title5, pad=10)
    scalebar5 = AnchoredSizeBar(ax5.transData, 1000 / res, '1 km', 'lower left', pad=0.1,
                                color='k', frameon=False, size_vertical=1, fontproperties=fm.FontProperties(size=10))
    ax5.add_artist(scalebar5)

    fig.set_size_inches(12, 8)
    fig.tight_layout(pad=0.5, w_pad=-3.0, h_pad=2.0)
    fig.suptitle(title, weight='bold')
    plt.show(block=False)
    figName = title.replace(' ', '_') + '.png'
    fig.savefig('Figures/' + figName)
    # Image.open('Figures/' + figName).show()

def plotData(dataVals, cbarTitle, color, plotTitle, cluster=None, quiver=None):
    '''
    Plots a map from input values (array-like)
    :param dataVals: values to be plotted
    :param cbarTitle: title of the colorbar
    :param color: colorbar scale (https://matplotlib.org/stable/tutorials/colors/colormaps.html)
    :param plotTitle: title of the plot
    :param cluster: number of discrete colorbar clusters. None by default, which has a continuous colorbar
    :param quiver: for quiver plot of arrows, list with 6 inputs from velPlot function below
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")

    divnorm = TwoSlopeNorm(vmin=dataVals.min(), vcenter=dataVals.mean(), vmax=dataVals.max())
    dataVals = dataVals.astype(float)
    dataVals[dataVals == 0] = np.nan                                  # to remove 0 values for white background in plot
    im = ax.imshow(dataVals, cmap=plt.cm.get_cmap(color, cluster), norm=divnorm) #check len datavals
    if quiver != None:
        ax.quiver(quiver[0], quiver[1], quiver[2], quiver[3], quiver[4],
                   cmap=quiver[5], scale=quiver[6], width=.003)                   # velocity arrows
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, label=cbarTitle)
    ax.set_title(plotTitle, weight='bold', pad=10)
    fig.tight_layout(pad=3, w_pad=-3.0, h_pad=2.0)

    plt.show(block=False)
    figName = plotTitle.replace(' ', '_') + '.png'
    fig.savefig('Figures/' + figName)
    # Image.open('Figures/' + figName).show()

def plotData3(dataVals1, cbarTitle1, color1, title1, dataVals2, cbarTitle2, color2, title2,
              dataVals3, cbarTitle3, color3, title3, plotTitle, res, quiver2=None, quiver3=None):
    fig = plt.figure()
    ax1 = fig.add_subplot(131, label="1")
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    divnorm1 = TwoSlopeNorm(vmin=dataVals1.min(), vcenter=dataVals1.mean(), vmax=dataVals1.max())
    # divnorm1 = TwoSlopeNorm(vmin=750, vcenter=1000, vmax=1450)
    dataVals1 = dataVals1.astype(float)
    dataVals1[dataVals1 == 0] = np.nan  # to remove 0 values for white background in plot
    im1 = ax1.imshow(dataVals1, cmap=plt.cm.get_cmap(color1), norm=divnorm1)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax1, label=cbarTitle1)
    ax1.set_title(title1)
    scalebar1 = AnchoredSizeBar(ax1.transData, 1000 / res, '1 km', 'lower left', pad=0.1,
                                color='k', frameon=False, size_vertical=1, fontproperties=fm.FontProperties(size=10))
    ax1.add_artist(scalebar1)

    divnorm2 = TwoSlopeNorm(vmin=dataVals2.min(), vcenter=dataVals2.mean(), vmax=dataVals2.max())
    dataVals2 = dataVals2.astype(float)
    dataVals2[dataVals2 == 0] = np.nan  # to remove 0 values for white background in plot
    im2 = ax2.imshow(dataVals2, cmap=plt.cm.get_cmap(color2), norm=divnorm2)
    if quiver2 != None:
        ax2.quiver(quiver2[0], quiver2[1], quiver2[2], quiver2[3], quiver2[4],
                   cmap=quiver2[5], scale=quiver2[6], width=.003)                   # velocity arrows
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax2, label=cbarTitle2)
    ax2.set_title(title2)
    scalebar2 = AnchoredSizeBar(ax2.transData, 1000 / res, '1 km', 'lower left', pad=0.1,
                                color='k', frameon=False, size_vertical=1, fontproperties=fm.FontProperties(size=10))
    ax2.add_artist(scalebar2)

    divnorm3 = TwoSlopeNorm(vmin=dataVals3.min(), vcenter=dataVals3.mean(), vmax=dataVals3.max())
    # divnorm3 = TwoSlopeNorm(vmin=dataVals3.min(), vcenter=dataVals3.mean(), vmax=50)
    dataVals3 = dataVals3.astype(float)
    dataVals3[dataVals3 == 0] = np.nan  # to remove 0 values for white background in plot
    im3 = ax3.imshow(dataVals3, cmap=plt.cm.get_cmap(color3), norm=divnorm3)
    if quiver3 != None:
        ax3.quiver(quiver3[0], quiver3[1], quiver3[2], quiver3[3], quiver3[4],
                   cmap=quiver3[5], scale=quiver3[6], width=.003)                   # velocity arrows
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax3, label=cbarTitle3)
    ax3.set_title(title3)
    scalebar3 = AnchoredSizeBar(ax3.transData, 1000 / res, '1 km', 'lower left', pad=0.1,
                                color='k', frameon=False, size_vertical=1, fontproperties=fm.FontProperties(size=10))
    ax3.add_artist(scalebar3)

    fig.set_size_inches(14, 4)
    fig.suptitle(plotTitle, weight='bold')
    fig.tight_layout(pad=3, w_pad=2.0, h_pad=0.0)
    plt.show(block=False)
    figName = plotTitle.replace(' ', '_') + '.png'
    fig.savefig('Figures/' + figName)
    # Image.open('Figures/' + figName).show()

def plotData6(dataVals1, cbarTitle1, color1, title1, dataVals2, cbarTitle2, color2, title2,
              dataVals3, cbarTitle3, color3, title3, dataVals4, cbarTitle4, color4, title4,
              dataVals5, cbarTitle5, color5, title5, dataVals6, cbarTitle6, color6, title6, plotTitle, res):
    fig = plt.figure()
    ax1 = fig.add_subplot(231, label="1")
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    divnorm1 = TwoSlopeNorm(vmin=dataVals1.min(), vcenter=dataVals1.mean(), vmax=dataVals1.max())
    # divnorm1 = TwoSlopeNorm(vmin=dataVals1.min(), vcenter=dataVals1.mean(), vmax=350)
    dataVals1 = dataVals1.astype(float)
    dataVals1[dataVals1 == 0] = np.nan  # to remove 0 values for white background in plot
    im1 = ax1.imshow(dataVals1, cmap=plt.cm.get_cmap(color1), norm=divnorm1)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax1, label=cbarTitle1)
    ax1.set_title(title1)
    scalebar1 = AnchoredSizeBar(ax1.transData, 1000 / res, '1 km', 'lower left', pad=0.1,
                                color='k', frameon=False, size_vertical=1, fontproperties=fm.FontProperties(size=10))
    ax1.add_artist(scalebar1)

    divnorm2 = TwoSlopeNorm(vmin=dataVals2.min(), vcenter=dataVals2.mean(), vmax=dataVals2.max())
    dataVals2 = dataVals2.astype(float)
    dataVals2[dataVals2 == 0] = np.nan  # to remove 0 values for white background in plot
    im2 = ax2.imshow(dataVals2, cmap=plt.cm.get_cmap(color2), norm=divnorm2)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax2, label=cbarTitle2)
    ax2.set_title(title2)
    scalebar2 = AnchoredSizeBar(ax2.transData, 1000 / res, '1 km', 'lower left', pad=0.1,
                                color='k', frameon=False, size_vertical=1, fontproperties=fm.FontProperties(size=10))
    ax2.add_artist(scalebar2)

    divnorm3 = TwoSlopeNorm(vmin=dataVals3.min(), vcenter=dataVals3.mean(), vmax=dataVals3.max())
    dataVals3 = dataVals3.astype(float)
    dataVals3[dataVals3 == 0] = np.nan  # to remove 0 values for white background in plot
    im3 = ax3.imshow(dataVals3, cmap=plt.cm.get_cmap(color3), norm=divnorm3)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax3, label=cbarTitle3)
    ax3.set_title(title3)
    scalebar3 = AnchoredSizeBar(ax3.transData, 1000 / res, '1 km', 'lower left', pad=0.1,
                                color='k', frameon=False, size_vertical=1, fontproperties=fm.FontProperties(size=10))
    ax3.add_artist(scalebar3)

    divnorm4 = TwoSlopeNorm(vmin=dataVals4.min(), vcenter=dataVals4.mean(), vmax=dataVals4.max())
    # divnorm4 = TwoSlopeNorm(vmin=dataVals4.min(), vcenter=dataVals4.mean(), vmax=300)
    dataVals4 = dataVals4.astype(float)
    dataVals4[dataVals4 == 0] = np.nan  # to remove 0 values for white background in plot
    im4 = ax4.imshow(dataVals4, cmap=plt.cm.get_cmap(color4), norm=divnorm4)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im4, cax=cax4, label=cbarTitle4)
    ax4.set_title(title4)
    scalebar4 = AnchoredSizeBar(ax4.transData, 1000 / res, '1 km', 'lower left', pad=0.1,
                                color='k', frameon=False, size_vertical=1, fontproperties=fm.FontProperties(size=10))
    ax4.add_artist(scalebar4)

    divnorm5 = TwoSlopeNorm(vmin=dataVals5.min(), vcenter=dataVals5.mean(), vmax=dataVals5.max())
    dataVals5 = dataVals5.astype(float)
    dataVals5[dataVals5 == 0] = np.nan  # to remove 0 values for white background in plot
    im5 = ax5.imshow(dataVals5, cmap=plt.cm.get_cmap(color5), norm=divnorm5)
    divider5 = make_axes_locatable(ax5)
    cax5 = divider5.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im5, cax=cax5, label=cbarTitle5)
    ax5.set_title(title5)
    scalebar5 = AnchoredSizeBar(ax5.transData, 1000 / res, '1 km', 'lower left', pad=0.1,
                                color='k', frameon=False, size_vertical=1, fontproperties=fm.FontProperties(size=10))
    ax5.add_artist(scalebar5)

    divnorm6 = TwoSlopeNorm(vmin=dataVals6.min(), vcenter=dataVals6.mean(), vmax=dataVals6.max())
    dataVals6 = dataVals6.astype(float)
    dataVals6[dataVals6 == 0] = np.nan  # to remove 0 values for white background in plot
    im6 = ax6.imshow(dataVals6, cmap=plt.cm.get_cmap(color6), norm=divnorm6)
    divider6 = make_axes_locatable(ax6)
    cax6 = divider6.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im6, cax=cax6, label=cbarTitle6)
    ax6.set_title(title6)
    scalebar6 = AnchoredSizeBar(ax6.transData, 1000 / res, '1 km', 'lower left', pad=0.1,
                                color='k', frameon=False, size_vertical=1, fontproperties=fm.FontProperties(size=10))
    ax6.add_artist(scalebar6)

    fig.set_size_inches(14, 8)
    fig.suptitle(plotTitle, weight='bold')
    fig.tight_layout(pad=3, w_pad=2.0, h_pad=0.0)
    plt.show(block=False)
    figName = plotTitle.replace(' ', '_') + '.png'
    fig.savefig('Figures/' + figName)
    # Image.open('Figures/' + figName).show()

def plotDataPoints(dataVals1, cbarTitle1, color1, dataVals2, cbarTitle2, color2,
                   plotTitle, pointx, pointy, pointlabels):
    '''
    Plots a map from input values (array-like) with labelled point locations from reference data
    Plots 2 side by side maps: identical but with difference basemaps (e.g. elevation and pixel mass balance)
    :param dataVals: values to be plotted (array)
    :param cbarTitle: title of the colorbar
    :param color: colorbar scale (https://matplotlib.org/stable/tutorials/colors/colormaps.html)
    :param plotTitle: title of the plot
    :param pointx: x-values of points to plot (raster/array column)
    :param pointy: y-values of points to plot (raster/array row)
    :param pointlabels: text labels for the reference points
    :return:
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(121, label="1")
    ax2 = fig.add_subplot(122, label="1")

    divnorm1 = TwoSlopeNorm(vmin=dataVals1.min(), vcenter=dataVals1.mean(), vmax=dataVals1.max())
    dataVals1 = dataVals1.astype(float)
    dataVals1[dataVals1 == 0] = np.nan                                  # to remove 0 values for white background in plot
    im1 = ax1.imshow(dataVals1, cmap=plt.cm.get_cmap(color1), norm=divnorm1)  # check len datavals
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax1, label=cbarTitle1)
    ax1.scatter(pointx, pointy, s=3, c='Red', marker='o')        # add points where reference data exists
    for i, txt in enumerate(pointlabels):                       # add labels to each point
        ax1.annotate(text=txt, xy=(pointx[i][0]+3, pointy[i][0]-3), c='k')

    divnorm2 = TwoSlopeNorm(vmin=dataVals2.min(), vcenter=dataVals2.mean(), vmax=dataVals2.max())
    dataVals2 = dataVals2.astype(float)
    dataVals2[dataVals2 == 0] = np.nan                                  # to remove 0 values for white background in plot
    im2 = ax2.imshow(dataVals2, cmap=plt.cm.get_cmap(color2), norm=divnorm2)  # check len datavals
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax2, label=cbarTitle2)
    ax2.scatter(pointx, pointy, s=3, c='Red', marker='o')        # add points where reference data exists
    for i, txt in enumerate(pointlabels):                       # add labels to each point
        ax2.annotate(text=txt, xy=(pointx[i][0]+3, pointy[i][0]-3), c='k')

    fig.suptitle(plotTitle, weight='bold')
    fig.tight_layout(pad=3, w_pad=0.0, h_pad=3.0)
    plt.show(block=False)
    figName = plotTitle.replace(' ', '_') + '.png'
    fig.savefig('Figures/' + figName)
    # Image.open('Figures/' + figName).show()

def velPlot(vx, vy, v_tot, area, threshold):
    '''
    Show velocity vector arrows
    :param vx1: x-direction velocity
    :param vy1: y-direction velocity
    :param v_tot: magnitude of velocity
    :param pixel_size: pixel size (for density of vectors)
    :param threshold: velocity magnitude threshold for showing arrows
    :return:
    '''
    # fig, ax = plt.subplots()                                  # uncomment lines to plot this graph separately
    if area/1000000 > 10:
        freq = 20
    else:
        freq = 10
    xx = np.arange(0, vx.shape[1], freq)                          # last number represents arrow frequency
    yy = np.arange(0, vy.shape[0], freq)
    points = np.ix_(yy, xx)
    px, py = np.meshgrid(xx, yy)

    vx_norm = np.divide(vx[points], v_tot[points], out=np.zeros_like(vx[points]), where=v_tot[points] > threshold)
    vy_norm = np.divide(vy[points], v_tot[points], out=np.zeros_like(vx[points]), where=v_tot[points] > threshold)
    vx_norm[np.isnan(vx_norm)] = 0
    vy_norm[np.isnan(vy_norm)] = 0

    mask = np.logical_or(vx_norm != 0, vy_norm != 0)                # remove 0 points
    quiverInput = [px[mask], py[mask], vx_norm[mask], vy_norm[mask], 1, 'gist_gray', 20]
    # ax.quiver(px[mask], py[mask], vx_norm[mask], vy_norm[mask], 1, cmap='gist_gray', 20)
    #
    # v_tot = v_tot.astype(float)
    # v_tot[v_tot == 0] = np.nan                    # to remove 0 values for white background in plot
    # im = ax.imshow(v_tot, color)
    # fig.colorbar(im, label='Velocity (m/a)')
    # fig.tight_layout(pad=3, w_pad=0, h_pad=0)
    # plt.show(block=False)
    return quiverInput

def elevationBinPlot3data3Subfigs(x_subp1, x_subp1_lab, x_subp1_2, x_subp1_lab_2, x_subp1_3, x_subp1_lab_3, x_subp1_4,
                                  x_subp1_lab_4, y_subp1, x2_subp1, y2_subp1, xLabel1, yLabel1, xLabel2, title1,
                                  elBinWidth, buff, alpha, title, res,
                                  subp3, title3, cbar3, color3,
                                  subp4, title4, cbar4, color4,
                                  subp5, title5, cbar5, color5,
                                  err_subp1_2=None, err_subp1_3=None, err_subp1_4=None,
                                  x_subp1_ref=None, x_subp1_lab_ref=None, y_subp1_ref=None, ref_var=None, cbar4ticks=None):
    '''
    Same as elevationBinPlot3Subfigs (above) but includes comparison of 3 datasets in NW plot
    '''
    fig = plt.figure()
    ax = fig.add_subplot(221, label="1")
    ax2 = fig.add_subplot(261, label="2", frame_on=False)

    ax3 = fig.add_subplot(222)
    ax4 = fig.add_subplot(223)
    ax5 = fig.add_subplot(224)

    ax.scatter(x_subp1, y_subp1, label=x_subp1_lab, color='w', edgecolor='dimgray', zorder=2)
    ax.plot(x_subp1, y_subp1, color='dimgray', zorder=1)
    ax.set_xlabel(xLabel1)
    ax.set_ylabel(yLabel1)
    ax.set_title(title1, pad=10)
    ax.tick_params(axis='x')
    ax.barh(x2_subp1, y2_subp1, elBinWidth, alpha=0, zorder=0)
    if err_subp1_2 is not None:
        if x_subp1_ref is None:
            if len(err_subp1_2.shape) > 1:
                ax.set_xlim([min(np.min(x_subp1_2 - err_subp1_2[0]), np.min(x_subp1_3 - err_subp1_3[0]),
                                 np.min(x_subp1_4 - err_subp1_4[0])) - 50,
                             max(np.max(x_subp1_2 + err_subp1_2[-1]), np.max(x_subp1_3 + err_subp1_3[-1]),
                                 np.max(x_subp1_4 + err_subp1_4[-1])) + 50])
            else:
                ax.set_xlim([min(np.min(x_subp1_2 - err_subp1_2), np.min(x_subp1_3 - err_subp1_3),
                                 np.min(x_subp1_4 - err_subp1_4)) - 50,
                             max(np.max(x_subp1_2 + err_subp1_2), np.max(x_subp1_3 + err_subp1_3),
                                 np.max(x_subp1_4 + err_subp1_4)) + 50])
        else:
            if len(err_subp1_2.shape) > 1:
                ax.set_xlim([min(np.min(x_subp1_2 - err_subp1_2[0]), np.min(x_subp1_3 - err_subp1_3[0]),
                                 np.min(x_subp1_4 - err_subp1_4[0]), np.min(x_subp1_ref)) - 20,
                             max(np.max(x_subp1_2 + err_subp1_2[-1]), np.max(x_subp1_3 + err_subp1_3[-1]),
                                 np.max(x_subp1_4 + err_subp1_4[-1]), np.max(x_subp1_ref), 0) + 20])
            else:
                ax.set_xlim(
                    [min(np.min(x_subp1_2 - err_subp1_2), np.min(x_subp1_3 - err_subp1_3),
                         np.min(x_subp1_4 - err_subp1_4), np.min(x_subp1_ref)) - 20,
                     max(np.max(x_subp1_2 + err_subp1_2), np.max(x_subp1_3 + err_subp1_3),
                         np.max(x_subp1_4 + err_subp1_4), np.max(x_subp1_ref), 0) + 20])
            ax.plot(x_subp1_ref, y_subp1_ref, color='black', zorder=5)  # plot line to connect reference data
    elif x_subp1_ref is not None:
        ax.set_xlim([min(np.min(x_subp1_ref), np.min(x_subp1))-buff, max(np.max(x_subp1_ref), np.max(x_subp1), 0)+buff])
        ax.plot(x_subp1_ref, y_subp1_ref, color='black', zorder=5)  # plot line to connect reference data
    else:
        ax.set_xlim([min(x_subp1) - buff, max(np.max(x_subp1), 0) + buff])

    ax2.barh(x2_subp1, y2_subp1, elBinWidth, alpha=alpha, color='dimgray')
    ax2.xaxis.tick_top()
    ax2.set_xlabel(xLabel2, color='dimgray')
    ax2.xaxis.set_label_position('top')
    ax2.tick_params(axis='x', colors='dimgray')

    ax.vlines(x_subp1_2, y_subp1 - np.random.uniform(elBinWidth/2, elBinWidth/2, len(y_subp1)),
              y_subp1 + np.random.uniform(elBinWidth/2, elBinWidth/2, len(y_subp1)), label=x_subp1_lab_2,
              lw=2, color='#01665e', zorder=7)
    ax.errorbar(x_subp1_2, y_subp1 - np.random.uniform(elBinWidth/5, elBinWidth/5, len(y_subp1)), xerr=err_subp1_2,
                fmt='', color='#01665e', ls='none', ecolor='#35978f', elinewidth=2, alpha=0.6, zorder=3)
    ax.vlines(x_subp1_3, y_subp1 - np.random.uniform(elBinWidth/2, elBinWidth/2, len(y_subp1)),
              y_subp1 + np.random.uniform(elBinWidth/2, elBinWidth/2, len(y_subp1)), label=x_subp1_lab_3,
              lw=2, color='#8c510a', zorder=6)
    ax.errorbar(x_subp1_3, y_subp1, xerr=err_subp1_3,
                fmt='', color='#8c510a', ls='none', ecolor='#bf812d', elinewidth=2, alpha=0.6, zorder=4)
    ax.vlines(x_subp1_4, y_subp1 - np.random.uniform(elBinWidth/2, elBinWidth/2, len(y_subp1)),
              y_subp1 + np.random.uniform(elBinWidth/2, elBinWidth/2, len(y_subp1)), label=x_subp1_lab_4,
              lw=2, color='#67000d', zorder=6)
    ax.errorbar(x_subp1_4, y_subp1 + np.random.uniform(elBinWidth/5, elBinWidth/5, len(y_subp1)), xerr=err_subp1_4,
                fmt='', color='#67000d', ls='none', ecolor='#a50f15', elinewidth=2, alpha=0.6, zorder=4)
    ax.scatter(x_subp1_ref, y_subp1_ref, label=x_subp1_lab_ref, color='black', zorder=5)
    ax.errorbar(x_subp1_ref, y_subp1_ref, xerr=ref_var,
                fmt='', color='black', ls='none', ecolor='black', elinewidth=2, alpha=0.6, zorder=3)
    if x_subp1_ref != None:
        m, b = np.polyfit(x_subp1_ref, y_subp1_ref, 1)
        ax.axline((0, b), slope=m, color='black', ls='--', alpha=0.8, zorder=6)
    ax.axvline(x=0, color='black', lw=0.5, dashes=(10, 3))
    ax.legend(fontsize=8)   # bbox_to_anchor=(1.02, 1), loc='upper left', 'best'

    # add 3 subplots
    # top right plot:
    divnorm3 = TwoSlopeNorm(vmin=min(subp3.min(), -1), vcenter=0, vmax=max(subp3.max(), 1))       # center colorbar at 0
    im3 = ax3.imshow(subp3, cmap=plt.cm.get_cmap(color3, len(x_subp1)), norm=divnorm3)   # plot array
    divider3 = make_axes_locatable(ax3)                         # plot on ax3 (northeast subplot)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)   # specify colorbar axis properties
    fig.colorbar(im3, cax=cax3, label=cbar3)                    # add colorbar
    ax3.set_title(title3, pad=10)                               # add subplot title
    scalebar3 = AnchoredSizeBar(ax3.transData, 1000 / res, '1 km', 'lower left', pad=0.1,
                                color='k', frameon=False, size_vertical=1, fontproperties=fm.FontProperties(size=10))
    ax3.add_artist(scalebar3)

    # bottom left plot:
    divnorm4 = TwoSlopeNorm(vmin=min(subp4.min(), -1), vcenter=0, vmax=max(subp4.max(), 1))  # center colorbar at 0
    # subp4 = subp4.astype(float)
    # subp4[subp4 == 0] = np.nan                                  # to remove 0 values for white background in plot
    im4 = ax4.imshow(subp4, cmap=plt.cm.get_cmap(color4, len(x_subp1)), norm=divnorm4)
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes('right', size='5%', pad=0.05)
    cbar4 = fig.colorbar(im4, cax=cax4, label=cbar4)
    if cbar4ticks != None:              # this would be if we want colorbar ticks/labels different from actual values
        cbar4.ax.locator_params(nbins=len(cbar4ticks)//2)
        cbar4.ax.set_yticklabels(np.linspace(cbar4ticks[0], cbar4ticks[-1], len(cbar4ticks)//2).astype(int))
    ax4.set_title(title4, pad=10)
    scalebar4 = AnchoredSizeBar(ax4.transData, 1000 / res, '1 km', 'lower left', pad=0.1,
                                color='k', frameon=False, size_vertical=1, fontproperties=fm.FontProperties(size=10))
    ax4.add_artist(scalebar4)

    # bottom right plot:
    divnorm5 = TwoSlopeNorm(vmin=min(subp5.min(), -1), vcenter=0, vmax=max(subp5.max(), 1))  # center colorbar at 0
    im5 = ax5.imshow(subp5, cmap=plt.cm.get_cmap(color5, len(x_subp1)), norm=divnorm5)
    divider5 = make_axes_locatable(ax5)
    cax5 = divider5.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im5, cax=cax5, label=cbar5)
    ax5.set_title(title5, pad=10)
    scalebar5 = AnchoredSizeBar(ax5.transData, 1000 / res, '1 km', 'lower left', pad=0.1,
                                color='k', frameon=False, size_vertical=1, fontproperties=fm.FontProperties(size=10))
    ax5.add_artist(scalebar5)

    fig.set_size_inches(12, 8)
    fig.tight_layout(pad=0.5, w_pad=-3.0, h_pad=2.0)
    fig.suptitle(title, weight='bold')
    plt.show(block=False)
    figName = title.replace(' ', '_') + '.png'
    fig.savefig('Figures/' + figName)
    # Image.open('Figures/' + figName).show()
