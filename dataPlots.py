import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm
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
    Image.open('Figures/' + figName).show()

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
    Image.open('Figures/' + figName).show()

def elevationBinPlot3Subfigs(x_subp1, x_subp1_lab, x_subp1_2, x_subp1_lab_2, x_subp1_3, x_subp1_lab_3, y_subp1, x2_subp1, y2_subp1, xLabel1,
                             yLabel1, xLabel2, title1, elevationBinWidth, buff, alpha, title,
                             subp3, title3, cbar3, color3,
                             subp4, title4, cbar4, color4,
                             subp5, title5, cbar5, color5,
                             x_subp1_ref=None, x_subp1_lab_ref=None, y_subp1_ref=None, cbar4ticks=None):
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

    ax.scatter(x_subp1, y_subp1, label=x_subp1_lab, color='dimgray')
    ax.plot(x_subp1, y_subp1, color='dimgray')
    ax.set_xlabel(xLabel1)
    ax.set_ylabel(yLabel1)
    ax.set_title(title1, pad=10)
    ax.tick_params(axis='x')
    ax.barh(x2_subp1, y2_subp1, elevationBinWidth, alpha=0, zorder=0)
    if x_subp1_ref != None:
        ax.set_xlim([min(x_subp1_ref) - buff, max(x_subp1_ref)+buff])
        ax.plot(x_subp1_ref, y_subp1_ref, color='black')  # plot line to connect reference data
    else:
        ax.set_xlim([min(x_subp1) - buff, max(x_subp1) + buff])

    ax2.barh(x2_subp1, y2_subp1, elevationBinWidth, alpha=alpha, color="C1")
    ax2.xaxis.tick_top()
    ax2.set_xlabel(xLabel2, color="C1")
    ax2.xaxis.set_label_position('top')
    ax2.tick_params(axis='x', colors="C1")

    ax.scatter(x_subp1_2, y_subp1, label=x_subp1_lab_2, color='steelblue')
    ax.scatter(x_subp1_3, y_subp1, label=x_subp1_lab_3, color='seagreen')
    ax.scatter(x_subp1_ref, y_subp1_ref, label=x_subp1_lab_ref, color='black')      # otherwise maroon (colorblind tho)
    ax.legend()

    # add 3 subplots
    # top right plot:
    divnorm3 = TwoSlopeNorm(vmin=min(subp3.min(), -1), vcenter=0, vmax=max(subp3.max(), 1))       # center colorbar at 0
    im3 = ax3.imshow(subp3, cmap=plt.cm.get_cmap(color3, len(x_subp1)), norm=divnorm3)   # plot array
    divider3 = make_axes_locatable(ax3)                         # plot on ax3 (northeast subplot)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)   # specify colorbar axis properties
    fig.colorbar(im3, cax=cax3, label=cbar3)                    # add colorbar
    ax3.set_title(title3, pad=10)                               # add subplot title

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

    # bottom right plot:
    divnorm5 = TwoSlopeNorm(vmin=min(subp5.min(), -1), vcenter=0, vmax=max(subp5.max(), 1))  # center colorbar at 0
    im5 = ax5.imshow(subp5, cmap=plt.cm.get_cmap(color5, len(x_subp1)), norm=divnorm5)
    divider5 = make_axes_locatable(ax5)
    cax5 = divider5.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im5, cax=cax5, label=cbar5)
    ax5.set_title(title5, pad=10)

    fig.set_size_inches(12, 8)
    fig.tight_layout(pad=0.5, w_pad=-3.0, h_pad=2.0)
    fig.suptitle(title, weight='bold')
    plt.show(block=False)
    figName = title.replace(' ', '_') + '.png'
    fig.savefig('Figures/' + figName)
    Image.open('Figures/' + figName).show()

def plotData(dataVals, cbarTitle, color, plotTitle):
    '''
    Plots a map from input values (array-like)
    :param dataVals: values to be plotted
    :param cbarTitle: title of the colorbar
    :param color: colorbar scale (https://matplotlib.org/stable/tutorials/colors/colormaps.html)
    :param plotTitle: title of the plot
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(131, label="1")

    divnorm = TwoSlopeNorm(vmin=dataVals.min(), vcenter=dataVals.mean(), vmax=dataVals.max())
    dataVals = dataVals.astype(float)
    dataVals[dataVals == 0] = np.nan                                  # to remove 0 values for white background in plot
    im = ax.imshow(dataVals, cmap=plt.cm.get_cmap(color), norm=divnorm) #check len datavals
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, label=cbarTitle)
    ax.set_title(plotTitle, weight='bold', pad=10)
    fig.tight_layout(pad=3, w_pad=-3.0, h_pad=2.0)

    plt.show(block=False)
    figName = plotTitle.replace(' ', '_') + '.png'
    fig.savefig('Figures/' + figName)
    Image.open('Figures/' + figName).show()

def plotData3(dataVals1, cbarTitle1, color1, title1, dataVals2, cbarTitle2, color2, title2,
              dataVals3, cbarTitle3, color3, title3, quiver3, plotTitle):
    fig = plt.figure()
    ax1 = fig.add_subplot(131, label="1")
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    divnorm1 = TwoSlopeNorm(vmin=dataVals1.min(), vcenter=dataVals1.mean(), vmax=dataVals1.max())
    dataVals1 = dataVals1.astype(float)
    dataVals1[dataVals1 == 0] = np.nan  # to remove 0 values for white background in plot
    im1 = ax1.imshow(dataVals1, cmap=plt.cm.get_cmap(color1), norm=divnorm1)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax1, label=cbarTitle1)
    ax1.set_title(title1)

    divnorm2 = TwoSlopeNorm(vmin=dataVals2.min(), vcenter=dataVals2.mean(), vmax=dataVals2.max())
    dataVals2 = dataVals2.astype(float)
    dataVals2[dataVals2 == 0] = np.nan  # to remove 0 values for white background in plot
    im2 = ax2.imshow(dataVals2, cmap=plt.cm.get_cmap(color2), norm=divnorm2)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax2, label=cbarTitle2)
    ax2.set_title(title2)

    divnorm3 = TwoSlopeNorm(vmin=dataVals3.min(), vcenter=dataVals3.mean(), vmax=dataVals3.max())
    dataVals3 = dataVals3.astype(float)
    dataVals3[dataVals3 == 0] = np.nan  # to remove 0 values for white background in plot
    im3 = ax3.imshow(dataVals3, cmap=plt.cm.get_cmap(color3), norm=divnorm3)
    ax3.quiver(quiver3[0], quiver3[1], quiver3[2], quiver3[3], quiver3[4],
               cmap=quiver3[5], scale=quiver3[6], width=.003)                   # velocity arrows
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im3, cax=cax3, label=cbarTitle3)
    ax3.set_title(title3)

    fig.set_size_inches(14, 4)
    fig.suptitle(plotTitle, weight='bold')
    fig.tight_layout(pad=3, w_pad=2.0, h_pad=0.0)
    plt.show(block=False)
    figName = plotTitle.replace(' ', '_') + '.png'
    fig.savefig('Figures/' + figName)
    Image.open('Figures/' + figName).show()

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

