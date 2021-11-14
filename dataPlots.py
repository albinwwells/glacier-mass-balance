import matplotlib.pyplot as plt

def elevationBinPlot(xVal1, yVal1, xVal2, yVal2, xLabel1, yLabel1, xLabel2, title,
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
    ax.set_title(title, pad=30)
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

    plt.show()
    figName = title.replace(' ', '_') + '.png'
    fig.savefig('Figures/' + figName)

