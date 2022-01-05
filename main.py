'''
awwells, 12780, final project code
All the necessary modules to import are listed at the top of this file and the other files.
The necessary raster files are in their respective folder
No other files or code is required to run this project
The code will automatically create and save updated raster files and plots in the respective folders
'''
# Import necessary modules
from dataPlots import elevationBinPlot, elevationBinPlot2, elevationBinPlot3Subfigs, plotData, plotData3, velPlot
from emergenceFunction import emergence_pixels
from smoothingFunctions import sgolay2d, gaussianFilter, dynamicSmoothing
from glacierDatabase import glacierInfo
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import rasterio.plot
from rasterio.plot import show
from tkinter import *
from fpdf import FPDF

# -----------------------------------------------------GENERAL---------------------------------------------------------#
res = 20                    # Resampling resolution (x, y) tuple
avgDensity = 850            # Average glacier mass density (kg/m3)
elevationBinWidth = 50      # Elevation bin width, for elevation binning calculations (m)
vCol = 0.8                  # surface column average velocity scaling

g1 = 'Gulkana'
g2 = 'Wolverine'
g3 = 'Eklutna'
g4 = 'LemonCreek'
g5 = 'SouthCascade'
g6 = 'Rainbow'
g7 = 'Conrad'
g8 = 'Illecillewaet'
g9 = 'Nordic'
g10 = 'Zillmer'
g11 = 'RikhaSamba'

# chose glacier to perform calculations with GUI
# select one of multiple glaciers in the set
# double click, press the return key, or click the 'SELECT' button to run calculations
def handler(event=None):
    global glac
    glac = lb.curselection()
    glac = [lb.get(int(x)) for x in glac]
    t.destroy()

t = Tk()
t.title('Glacier Selection')
glaciers = (g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11)
lb = Listbox(t, height=15)
lb.config(selectmode=EXTENDED)      # select multiple items with CTRL and ranges of items with SHIFT
for i in range(len(glaciers)):
       lb.insert(i+1, glaciers[i])
lb.select_set(0)                    # default beginning selection in GUI
lb.pack(side=LEFT)
t.bind('<Return>', handler)
t.bind('<Double-1>', handler)
b = Button(command=handler, text='SELECT', fg='black')
b.pack(side=LEFT)

w = 250                             # tk window width
h = 200                             # tk window height
ws = t.winfo_screenwidth()          # width of the screen
hs = t.winfo_screenheight()         # height of the screen
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)
t.geometry('%dx%d+%d+%d' % (w, h, x, y))
t.mainloop()

# ---------------------------------------------------------------------------------------------------------------------#
for glacier in glac:
    glacierCalc = glacierInfo(glacier, res, avgDensity, elevationBinWidth, vCol)

    if os.path.exists(os.getcwd() + '/' + glacierCalc.name) == True:
        os.chdir(os.getcwd() + '/' + glacierCalc.name)      # navigate to working directory
    else:
        main_directory = os.path.dirname(os.getcwd())
        os.chdir(main_directory)
        path = os.getcwd() + '/' + glacierCalc.name
        os.chdir(path)

    print()
    print('-------------------------------', glacierCalc.name, 'Glacier Calculations -------------------------------')
    # STEP 1: CONVERT COORDINATE SYSTEMS AND RESAMPLE
    glacierCalc.tifReprojectionResample()
    glacierCalc.velReprojectionResample()
    glacierCalc.shpReprojection()               # reproject shapefile

    # STEP 2: CLIP RASTERS TO SAME EXTENT
    glacierCalc.tifClip()
    glacierCalc.fillHole()
    glacierCalc.fillVel()

    glacierCalc.shpClip()           # clip rasters by glacier RGI shape
    glacierCalc.glacierOutline()

    # CHECK TO SEE IF THINGS WORKED
    check = 0
    if check == True:
        checkFile = glacierCalc.dem     # change this to check different files (.dem, .h, .dhdt, .vx, .vy)
        src = rasterio.open(checkFile)
        dst = rasterio.open(checkFile[:-4]+'_reproj.tif')
        dst_clip = rasterio.open(checkFile[:-4] + '_clip.tif')

        print('Source coordinate system of', checkFile, 'is: ', src.crs)
        print('Destination coordinate system of', checkFile, 'is: ', dst.crs)

        print('Source shapefile coordinate system is:', gpd.read_file(glacierCalc.shape).crs)
        print('Destination shapefile coordinate system is:', gpd.read_file(glacierCalc.shape[:-4] + '_reproj.shp').crs)

        ps_src = src.transform
        ps_dst = dst.transform
        print(checkFile, 'Pixel size (x, y):', ps_src[0], -ps_src[4])
        print(checkFile, 'Reprojected Pixel size (x, y):', ps_dst[0], -ps_dst[4])

        print(checkFile, 'Reprojected (SOURCE) has bounds:', dst.bounds)
        print(checkFile, 'CLIPPED has bounds: ', dst_clip.bounds)

        print(checkFile, 'Reprojected (SOURCE) has width and height:', dst.width, dst.height)
        print(checkFile, 'CLIPPED has width and height: ', dst_clip.width, dst_clip.height)


    # ----------------------------------------------FIX VELOCITY DIRECTIONS----------------------------------------------#
    outlineArray = rasterio.open(glacierCalc.name + 'Outline.tif').read(1)
    demArray = rasterio.open(glacierCalc.dem[:-4] + '_clip.tif').read(1) * outlineArray
    hArray = rasterio.open(glacierCalc.h[:-4] + '_clip.tif').read(1) * outlineArray
    vxArray = rasterio.open(glacierCalc.vx[:-4] + '_clip.tif').read(1) * outlineArray
    vyArray = rasterio.open(glacierCalc.vy[:-4] + '_clip.tif').read(1) * outlineArray
    vArray = np.power((np.square(vxArray) + np.square(vyArray)), 0.5)

    '''
    # TO FIX VELOCITY DIRECTIONS
    maxvalh = np.max(hArray)            # Obtain max glacier thickness and array index of this point
    maxindexh = np.where(hArray == maxvalh)
    indexrowh = maxindexh[0][0]
    indexcolh = maxindexh[1][0]
    
    vxmaxh = vxArray[indexrowh, indexcolh]          # Obtain x- and y- velocity at max thickness point
    vymaxh = vyArray[indexrowh, indexcolh]
    if abs(vxmaxh) > abs(vymaxh):                   # Define primary and secondary velocity based on velocity magnitudes
        vprimarydir = [vxArray, 'vx']
        vsecondarydir = [vyArray, 'vy']
    else:
        vprimarydir = [vyArray, 'vy']
        vsecondarydir = [vxArray, 'vx']
    
    drc = [0, 0, 0]                         # Obtain change in rows (drc) and cols (dcc) elevation at maximum thickness
    dcc = [0, 0, 0]
    for i in [0, 1, 2]:
        drc[i] = demArray[indexrowh + ((i-1)*2), indexcolh]
        dcc[i] = demArray[indexrowh, indexcolh + ((i-1)*2)]
    
    if abs(drc[0]-drc[2]) > abs(dcc[0]-dcc[2]):
        # if glacier is primarily north-south (change in drc greater than change in dcc)
        if drc[0] > drc[1] > drc[2]:
            print('Primary direction sloping south (row in sequence decreasing from top to bottom)')
            if vprimarydir[0][indexrowh, indexcolh] < 0:
                y_vel_array_dir = vprimarydir[0]
                print(' Primary direction velocity vector: ' + vprimarydir[1])
            else:
                y_vel_array_dir = -vprimarydir[0]
                print(' Primary direction velocity vector: -' + vprimarydir[1])
        if drc[0] < drc[1] < drc[2]:
            print('Primary direction sloping north (row in sequence increasing from top to bottom)')
            if vprimarydir[0][indexrowh, indexcolh] > 0:
                y_vel_array_dir = vprimarydir[0]
                print(' Primary direction velocity vector: ' + vprimarydir[1])
            else:
                y_vel_array_dir = -vprimarydir[0]
                print(' Primary direction velocity vector: -' + vprimarydir[1])
        if dcc[0] > dcc[2]:
            print('Secondary direction sloping east (col in sequence decreasing from left to right)')
            if vsecondarydir[0][indexrowh, indexcolh] > 0:
                x_vel_array_dir = vsecondarydir[0]
                print(' Secondary direction velocity vector: ' + vsecondarydir[1])
            else:
                x_vel_array_dir = -vsecondarydir[0]
                print(' Secondary direction velocity vector: -' + vsecondarydir[1])
        if dcc[0] < dcc[2]:
            print('Secondary direction sloping west (col in sequence increasing from left to right)')
            if vsecondarydir[0][indexrowh, indexcolh] < 0:
                x_vel_array_dir = vsecondarydir[0]
                print(' Secondary direction velocity vector: ' + vsecondarydir[1])
            else:
                x_vel_array_dir = -vsecondarydir[0]
                print(' Secondary direction velocity vector: -' + vsecondarydir[1])
    
    if abs(drc[0]-drc[2]) < abs(dcc[0]-dcc[2]):
        # if glacier is primarily east-west (change in dcc greater than change in drc)
        if drc[0] > drc[2]:
            print('Secondary direction sloping south (row in sequence decreasing from top to bottom)')
            if vsecondarydir[0][indexrowh, indexcolh] < 0:
                y_vel_array_dir = vsecondarydir[0]
                print(' Secondary direction velocity vector: ' + vsecondarydir[1])
            else:
                y_vel_array_dir = -vsecondarydir[0]
                print(' Secondary direction velocity vector: -' + vsecondarydir[1])
        if drc[0] < drc[2]:
            print('Secondary direction sloping north (row in sequence increasing from top to bottom)')
            if vsecondarydir[0][indexrowh, indexcolh] > 0:
                y_vel_array_dir = vsecondarydir[0]
                print(' Secondary direction velocity vector: ' + vsecondarydir[1])
            else:
                y_vel_array_dir = -vsecondarydir[0]
                print(' Secondary direction velocity vector: -' + vsecondarydir[1])
        if dcc[0] > dcc[1] > dcc[2]:
            print('Primary direction sloping east (col in sequence decreasing from left to right)')
            if vprimarydir[0][indexrowh, indexcolh] > 0:
                x_vel_array_dir = vprimarydir[0]
                print(' Primary direction velocity vector: ' + vprimarydir[1])
            else:
                x_vel_array_dir = -vprimarydir[0]
                print(' Primary direction velocity vector: -' + vprimarydir[1])
        if dcc[0] < dcc[1] < dcc[2]:
            print('Primary direction sloping west (col in sequence increasing from right to left)')
            if vprimarydir[0][indexrowh, indexcolh] < 0:
                x_vel_array_dir = vprimarydir[0]
                print(' Primary direction velocity vector: ' + vprimarydir[1])
            else:
                x_vel_array_dir = -vprimarydir[0]
                print(' Primary direction velocity vector: -' + vprimarydir[1])
    '''

    x_vel_array_dir = vxArray * glacierCalc.vCor[0]
    y_vel_array_dir = vyArray * glacierCalc.vCor[1]
    # Overwrite velocity files with new velocity files: new directions
    kwargs = rasterio.open(glacierCalc.vx[:-4] + '_clip.tif').meta.copy()
    dst_file_vx = glacierCalc.vx[:-4] + '_clip.tif'
    with rasterio.open(dst_file_vx, 'w', **kwargs) as dst:
        dst.write(x_vel_array_dir, 1)

    kwargs = rasterio.open(glacierCalc.vy[:-4] + '_clip.tif').meta.copy()
    dst_file_vy = glacierCalc.vy[:-4] + '_clip.tif'
    with rasterio.open(dst_file_vy, 'w', **kwargs) as dst:
        dst.write(y_vel_array_dir, 1)

    vel_x = x_vel_array_dir
    vel_y = y_vel_array_dir

    # ------------------------------------------STEP 3: CALCULATE AREA-----------------------------------------#
    showArea = 1                            # to print glacier area

    glacierCalc.totalMassBalance()
    area = glacierCalc.glacierArea()                # calculate and print glacier area
    if showArea == True:
        print(glacierCalc.name, 'Area (sq. km):', area/1000000)

    # ------------------------------------------PART 4 DISPLAY OPTIONS-----------------------------------------#
    use_filter = 'dynamic'      # 'dynamic' for dynamic filter, 'sg' for savitzky-golay filter, 'gauss' for gaussian filter
    showSmoothFilter = 0                    # to show the difference in raw data and smoothed data
    showSmoothClimMassBalance = 0           # to show the difference in climatic mass balance using vel vs smoothed vel
    printMassLoss = 1                       # to print the mass loss values
    showMassLossElevation = 0               # to plot of the mean mass balance per elevation bin (np.gradient)
    # ---------------------------------------------------------------------------------------------------------#
    # STEP 4: SMOOTH VELOCITY DATA WITH LOW PASS MOVING WINDOW FILTER, CLIMATIC MASS BALANCE FOR SMOOTH VELOCITY DATA
    raw_data_smooth_factor = 0
    divQ_smooth_factor = 0
    if use_filter == 'dynamic':
        smoothhArray = dynamicSmoothing(hArray, hArray, glacierCalc.res, raw_data_smooth_factor)
        smoothvxArray = dynamicSmoothing(vel_x, hArray, glacierCalc.res, raw_data_smooth_factor)
        smoothvyArray = dynamicSmoothing(vel_y, hArray, glacierCalc.res, raw_data_smooth_factor)
    elif use_filter == 'sg':
        smoothhArray = sgolay2d(hArray, window_size=17, order=1)
        smoothvxArray = sgolay2d(vel_x, window_size=17, order=1)
        smoothvyArray = sgolay2d(vel_y, window_size=17, order=1)
    elif use_filter == 'gauss':
        smoothhArray = gaussianFilter(hArray, st_dev=4, truncate=3)
        smoothvxArray = gaussianFilter(vel_x, st_dev=4, truncate=3)
        smoothvyArray = gaussianFilter(vel_y, st_dev=4, truncate=3)

    divQRaster = (dynamicSmoothing((np.gradient(smoothvxArray, glacierCalc.res)[1] * smoothhArray), hArray, glacierCalc.res, 0) +
                  dynamicSmoothing((np.gradient(smoothhArray, glacierCalc.res)[1] * smoothvxArray), hArray, glacierCalc.res, 0) +
                  dynamicSmoothing((np.gradient(smoothvyArray * -1, glacierCalc.res)[0] * smoothhArray), hArray, glacierCalc.res, 0) +
                  dynamicSmoothing((np.gradient(smoothhArray * -1, glacierCalc.res)[0] * smoothvyArray), hArray, glacierCalc.res, 0)) * glacierCalc.vCol

    if use_filter == 'dynamic':
        smoothdivQRaster = dynamicSmoothing(divQRaster, hArray, glacierCalc.res, divQ_smooth_factor)
    elif use_filter == 'sg':
        smoothdivQRaster = sgolay2d(divQRaster, window_size=9, order=1)
    elif use_filter == 'gauss':
        smoothdivQRaster = gaussianFilter(divQRaster, st_dev=2, truncate=3)

    # total and climatic SMB calculations
    totalMassBalanceValue = glacierCalc.totalMassBalanceValue()         # calculate glacier mass balance
    clim_smb_pixel = (rasterio.open(glacierCalc.dhdt[:-4] + '_clip.tif').read(1) + smoothdivQRaster)*glacierCalc.rho
    climMassBalanceValue = np.sum(clim_smb_pixel * glacierCalc.res * glacierCalc.res)/area

    bindivQMean, binElevations, binCounts, binNumber = glacierCalc.altitudeAggregation(smoothdivQRaster, stat='mean')
    dhdtFile = glacierCalc.dhdt[:-4] + '_clip.tif'
    binMeandhdt, binElevations, binCounts, binNumber = glacierCalc.altitudeAggregation(dhdtFile, stat='mean')

    # binned total and climatic mass balance calculations
    binTotalChange = binMeandhdt*glacierCalc.rho
    binClimChangeDiv = (binMeandhdt + bindivQMean)*glacierCalc.rho

    demTotBinTotalMass = binCounts * binTotalChange * glacierCalc.res * glacierCalc.res  # this is in kg/yr
    demClimBinTotalMass = binCounts * binClimChangeDiv * glacierCalc.res * glacierCalc.res
    altResTotMassBalance = np.sum(demTotBinTotalMass)/area      # divide by total area for alt-resolved mass balance
    altResClimMassBalance = np.sum(demClimBinTotalMass)/area

    if showSmoothFilter == True:
        show(rasterio.open(glacierCalc.h).read(1), title='original thickness file')
        show(sgolay2d(rasterio.open(glacierCalc.h).read(1), 7, 1), title='savitzky-golay filter: thickness')
        show(gaussianFilter(rasterio.open(glacierCalc.h).read(1), 5, 3), title='gaussian filter: thickness')
        show(dynamicSmoothing(hArray, hArray, glacierCalc.res, 4), title='moving gaussian filter: thickness')

    if showSmoothClimMassBalance == True:
        show(rasterio.open(glacierCalc.name+'TotalMassBalance.tif').read(1), title='Total Mass Balance Result')
        show(rasterio.open(glacierCalc.name+'ClimaticMassBalance.tif').read(1), title='Climatic Mass Balance Result')
        show(binClimChangeDiv, title='Smoothed Climatic Mass Balance Result')

    if printMassLoss == True:
        # to print the values for mass balance (kg/m2-yr or mm w.e.)
        print('The TOTAL mass balance is: ', totalMassBalanceValue, '(mm w.e.)')
        print('The altitudinally resolved TOTAL mass balance is:', altResTotMassBalance, '(mm w.e.)')
        print('The CLIMATIC (SMOOTHED, np.gradient) mass balance is: ', climMassBalanceValue, '(mm w.e.)')
        print('The altitudinally resolved CLIMATIC (SMOOTHED, np.gradient) mass balance is:', altResClimMassBalance, '(mm w.e.)')

    if showMassLossElevation == True:
        # to show a scatter plot of elevation vs mean mass loss at each elevation bin
        binElevationsList = binElevations.tolist()
        for i in range(len(binElevationsList)):
            binElevationsList[i] = binElevationsList[i] + (elevationBinWidth*0.5)
        binTotalChangeList = binTotalChange.tolist()
        binClimChangeDivList = binClimChangeDiv.tolist()
        binAreasList = (binCounts*glacierCalc.res*glacierCalc.res).tolist()

        x_values1 = binTotalChangeList
        x_values1_lab = 'Total Mass Balance'
        x_values1_2 = binClimChangeDivList
        x_values1_2_lab = 'Climatic Mass Balance (smooth)'
        y_values1 = binElevationsList[0:len(binElevationsList)-1]
        x_values2 = binElevationsList[0:len(binElevationsList)-1]
        y_values2 = np.divide(binAreasList, 1000000)    # sq.km area
        xLabel1 = 'Bin Mean Mass Balance (mm w.e.)'
        yLabel1 = 'Bin Mean Elevation (m)'
        xLabel2 = 'Bin Area ($km^2$)'
        title = 'Glacier mean mass balance per elevation bin: np.gradient method'
        min = 'default'
        buff = 1500
        alpha = 0.3

        elevationBinPlot2(x_values1, x_values1_lab, x_values1_2, x_values1_2_lab, y_values1, x_values2, y_values2, xLabel1,
                          yLabel1, xLabel2, title, elevationBinWidth, min, buff, alpha)

    # ------------------------------------------PART 6 DISPLAY OPTIONS-----------------------------------------#
    showSmoothClimMassBalanceEmg = 1    # to print alt-resolved climatic MB from emergence method
    livePlots = 0                       # to show the plots as live plots (not just reopened saved files)
    create_pdf = 1                      # to create pdf containing glacier mass balance information

    # STEP 6: EMERGENCE METHOD
    vel_total = np.power((np.square(vel_x) + np.square(vel_y)), 0.5)
    icethickness_fn = glacierCalc.h[:-4] + '_clip.tif'
    icethickness = rasterio.open(icethickness_fn).read(1)

    max_velocity = 200
    vmin = 0

    emergence_velocity = emergence_pixels(vel_x, vel_y, vel_total, vmin, max_velocity, vCol, icethickness, glacierCalc.res)
    binEmergenceMean, binElevations, binCounts, binNumber = glacierCalc.altitudeAggregation(emergence_velocity, stat='mean')
    binClimChangeEm = (binMeandhdt - binEmergenceMean) * glacierCalc.rho

    # checking the number of non zero values in each product
    # print(np.count_nonzero(smoothdivQRaster))
    # print(np.count_nonzero(rasterio.open(dhdtFile).read(1)))
    # print(np.count_nonzero(emergence_velocity*outlineArray))
    # print(np.count_nonzero(emergence_velocity))

    demClimBinTotalMassEm = binCounts * binClimChangeEm * glacierCalc.res * glacierCalc.res
    altResClimMassBalanceEm = np.sum(demClimBinTotalMassEm)/area
    clim_smb_pixel_em = ((rasterio.open(dhdtFile).read(1) - emergence_velocity)*glacierCalc.rho)*outlineArray
    climMassBalanceValueEm = np.sum(clim_smb_pixel_em * glacierCalc.res * glacierCalc.res)/area
    climMassBalanceValueEmLoss = np.sum((((rasterio.open(dhdtFile).read(1) - emergence_velocity)*glacierCalc.rho) -
                                         clim_smb_pixel_em) * glacierCalc.res * glacierCalc.res) / area
    if showSmoothClimMassBalanceEmg == True:
        print('The CLIMATIC (emergence) mass balance is:', climMassBalanceValueEm)
        print('The altitudinally resolved CLIMATIC (emergence) mass balance is:', altResClimMassBalanceEm)
        print("     CLIMATIC MB (emergence) mass loss from 'out-of-bounds' pixels is:", climMassBalanceValueEmLoss,
              "(mm w.e.). Any other discrepancy is from missing or depleted 'in-bounds' pixels from emergence method")

    # apply bin SMB to each value in that bin: FOR PLOTTING ONLY
    binnedTotalMassBalance = np.zeros(binNumber.shape)
    binnedClimaticMassBalanceEm = np.zeros(binNumber.shape)
    binnedClimaticMassBalanceDiv = np.zeros(binNumber.shape)
    binnedEmergence = np.zeros(binNumber.shape)
    for i in range(len(binNumber)):
        binnedTotalMassBalance[i] = binTotalChange[binNumber[i]-1]
        binnedClimaticMassBalanceEm[i] = binClimChangeEm[binNumber[i]-1]
        binnedClimaticMassBalanceDiv[i] = binClimChangeDiv[binNumber[i]-1]
        binnedEmergence[i] = binEmergenceMean[binNumber[i]-1]

    outlineArrayShape = outlineArray.shape
    binnedTotalMassBalance = np.array(binnedTotalMassBalance).reshape(outlineArrayShape) * outlineArray
    binnedClimaticMassBalanceEm = np.array(binnedClimaticMassBalanceEm).reshape(outlineArrayShape) * outlineArray
    binnedClimaticMassBalanceDiv = np.array(binnedClimaticMassBalanceDiv).reshape(outlineArrayShape) * outlineArray
    binnedEmergence = np.array(binnedEmergence).reshape(outlineArrayShape) * outlineArray
    binnedElevationNumber = np.array(binNumber).reshape(outlineArrayShape) * outlineArray

    # for plotting: each subplot has: [data array, title, colorbar label]
    emergence_plot = [binnedEmergence, 'Emergence Map', 'Mean Elevation Bin Emergence (m/a)', 'RdBu']
    h_plot = [icethickness, 'Thickness Map', 'Glacier Thickness (m)', 'BrBG']
    v_plot = [vel_total, 'Velocity Map', 'Velocity (m/a)', 'BrBG']     # BrBG, BuGr, PuBuGr are other good ones
    z_plot = [demArray, 'Elevation Map', 'Elevation (m)', 'gist_earth']             # gist_earth, viridis, jet
    b_tot_plot = [binnedTotalMassBalance, 'Total Mass Balance Map', 'Mean Elevation Bin SMB (mm w.e.)', 'RdBu']
    b_clim_em_plot = [binnedClimaticMassBalanceEm,
                      'Climatic Mass Balance Map (emergence)', 'Mean Elevation Bin SMB (mm w.e.)', 'RdBu']
    b_clim_div_plot = [binnedClimaticMassBalanceDiv,
                       'Climatic Mass Balance Map (np.gradient)', 'Mean Elevation Bin SMB (mm w.e.)', 'RdBu']

    # for pixel-wise SMB and emergence plots (not altitudinally aggregated):
    emergence_plot_pixel = [smoothdivQRaster * outlineArray, 'Emergence Map (np.gradient)', 'Pixel Emergence (m/a)', 'RdBu']
    b_tot_plot_pixel = [rasterio.open(glacierCalc.name + 'TotalMassBalance.tif').read(1) * outlineArray,
                        'Total Mass Balance Map', 'Pixel Mass Balance (mm w.e.)', 'RdBu']
    b_clim_plot_pixel = [clim_smb_pixel * outlineArray,
                         'Climatic Mass Balance Map (np.gradient)', 'Pixel Mass Balance (mm w.e.)', 'RdBu']

    # -----------chose subplots to show:------------
    subplot2 = b_clim_em_plot       # this should be b_tot_plot, b_clim_em_plot, or b_clim_div_plot
    subplot3 = b_tot_plot
    subplot4 = b_clim_div_plot      # emergence_plot, emergence_plot_pixel, b_tot_plot_pixel, b_clim_div_plot
    fig2_1 = z_plot                 # figure 2: other plots
    fig2_2 = h_plot
    fig2_3 = v_plot

    # to show a scatter plot of elevation vs mean mass loss at each elevation bin, as well as 3 other figures
    binElevationsList = binElevations.tolist()
    for i in range(len(binElevationsList)):
        binElevationsList[i] = binElevationsList[i] + (elevationBinWidth * 0.5)
    binClimChangeEmList = binClimChangeEm.tolist()
    binClimChangeDivList = binClimChangeDiv.tolist()
    binTotalChangeList = binTotalChange.tolist()
    binEmergenceList = binEmergenceMean.tolist()
    binAreasList = (binCounts * glacierCalc.res * glacierCalc.res).tolist()

    x_subplot_nw = binTotalChangeList
    x_subplot_nw_lab = 'Total MB'
    x_subplot_nw_2 = binClimChangeEmList
    x_subplot_nw_2_lab = 'Climatic MB (emergence)'
    x_subplot_nw_3 = binClimChangeDivList
    x_subplot_nw_3_lab = 'Climatic MB (np.gradient)'
    if glacierCalc.data is not None:        # add reference data points
        x_subplot_nw_ref = glacierCalc.data[0]
        y_values1_ref = glacierCalc.data[1]
        x_subplot_nw_ref_lab = 'Reference Data'
    else:
        x_subplot_nw_ref = None
        y_values1_ref = None
        x_subplot_nw_ref_lab = None
    y_values1 = binElevationsList[0:len(binElevationsList) - 1]
    x_values2 = binElevationsList[0:len(binElevationsList) - 1]
    y_values2 = np.divide(binAreasList, 1000000)    # sq.km area
    xLabel1 = 'Bin Mean Mass Balance (mm w.e.)'
    yLabel1 = 'Bin Mean Elevation (m)'
    xLabel2 = 'Bin Area ($km^2$)'
    title1 = 'Binned SMB Plot'
    mbPlotsTitle = glacierCalc.name + ' Glacier Plots:'
    buff = 500
    alpha = 0.3

    # 3 subplots:
    subplot_ne = subplot2[0]
    subplot_ne_title = subplot2[1]
    subplot_ne_label = subplot2[2]
    color_ne = subplot2[3]
    subplot_sw = subplot3[0]
    subplot_sw_title = subplot3[1]
    subplot_sw_label = subplot3[2]
    color_sw = subplot3[3]
    subplot_sw_cbar_ticks = None
    subplot_se = subplot4[0]
    subplot_se_title = subplot4[1]
    subplot_se_label = subplot4[2]
    color_se = subplot4[3]

    elevationBinPlot3Subfigs(x_subplot_nw, x_subplot_nw_lab, x_subplot_nw_2, x_subplot_nw_2_lab, x_subplot_nw_3,
                             x_subplot_nw_3_lab, y_values1, x_values2,
                             y_values2, xLabel1, yLabel1, xLabel2, title1, elevationBinWidth, buff, alpha, mbPlotsTitle,
                             subplot_ne, subplot_ne_title, subplot_ne_label, color_ne,
                             subplot_sw, subplot_sw_title, subplot_sw_label, color_sw,
                             subplot_se, subplot_se_title, subplot_se_label, color_se,
                             x_subplot_nw_ref, x_subplot_nw_ref_lab, y_values1_ref, subplot_sw_cbar_ticks)

    # second figure with 3 subplots
    fig2_1Vals = fig2_1[0]
    fig2_1cbar = fig2_1[2]
    fig2_1Title = fig2_1[1]
    fig2_1Color = fig2_1[3]
    fig2_2Vals = fig2_2[0]
    fig2_2cbar = fig2_2[2]
    fig2_2Title = fig2_2[1]
    fig2_2Color = fig2_2[3]
    fig2_3Vals = fig2_3[0]
    fig2_3cbar = fig2_3[2]
    fig2_3Title = fig2_3[1]
    fig2_3Color = fig2_3[3]
    fig2_3Quiver = velPlot(vel_x, vel_y, vel_total, area, threshold=1.0)     # plot velocity arrows
    extraPlotTitle = glacierCalc.name + ' Extra Plots'
    plotData3(fig2_1Vals, fig2_1cbar, fig2_1Color, fig2_1Title,
              fig2_2Vals, fig2_2cbar, fig2_2Color, fig2_2Title,
              fig2_3Vals, fig2_3cbar, fig2_3Color, fig2_3Title, fig2_3Quiver, extraPlotTitle)

    if livePlots == False:
        plt.close('all')

    # to make and save information as pdf file
    if create_pdf == True:
        pdf = FPDF('P', 'mm', 'A4')             # initialize pdf with A4 portrait mode size in mm units
        pdf.add_page()
        pdf.set_font('Times', 'BU', 18)         # set Times font (or 'Arial'), regular text (or 'B', 'I', 'U'), size 12
        epw = pdf.w - 2 * pdf.l_margin  # effective page width
        # pdf.cell(w, h=0, txt='', border=0, ln=0, align='', fill=False, link='')
        pdf.cell(epw, 10, glacierCalc.name + ' Glacier Info', border=0, ln=1, align='C')

        col_width = epw / 2                     # Set column width to 1/2 of effective page width to distribute content
        data = [['DEM file', glacierCalc.dem], ['Change in thickness file', glacierCalc.dhdt],
                ['Thickness file', glacierCalc.h], ['X-velocity file', glacierCalc.vx],
                ['Y-velocity file', glacierCalc.vy]]
        pdf.set_font('Times', 'B', 12)
        pdf.cell(epw, 5, 'Input data files', ln=1, align='C')
        pdf.ln(0.5)
        pdf.set_font('Times', '', 10.0)
        th = pdf.font_size                      # Text height is the same as current font size
        for row in data:
            for datum in row:
                pdf.cell(col_width, th, str(datum), border=1)
            pdf.ln(th)                          # 2 * th is for double padding

        pdf.ln(2 * th)          # Line break equivalent to 2 lines
        pdf.set_font('Times', 'B', 12)
        pdf.cell(epw, 5, 'Input Constants and Assumptions', ln=1, align='C')
        pdf.set_font('Times', 'B', 10)
        pdf.ln(0.5)
        col_width = epw / 5
        data = [['Resolution (m)', 'Density (kg/m3)', 'Vel Col Scaling Factor', 'Elevation Bin Width (m)', 'Coordinate system'],
                [glacierCalc.res, glacierCalc.rho, glacierCalc.vCol, glacierCalc.bin, glacierCalc.crs]]
        pdf.ln(0.5)
        th = pdf.font_size                      # Text height is the same as current font size
        for row in data:
            for datum in row:
                pdf.cell(col_width, th, str(datum), border=1)
            pdf.set_font('Times', '', 10)
            pdf.ln(th)                          # 2 * th is for double padding

        pdf.ln(2 * th)
        pdf.set_font('Times', 'B', 12)
        pdf.cell(epw, 5, 'Input Calculation Settings', ln=1, align='C')
        pdf.set_font('Times', 'B', 10)
        pdf.ln(0.5)
        col_width = epw / 3
        if use_filter == 'dynamic':
            data = [
                ['Smoothing Filter', 'Smoothing Factor (raw data)', 'Smoothing Factor (divQ product)'],
                [use_filter, str(raw_data_smooth_factor) + 'x local thickness', str(divQ_smooth_factor) + 'x local thickness']]
            pdf.ln(0.5)
            th = pdf.font_size  # Text height is the same as current font size
            for row in data:
                for datum in row:
                    pdf.cell(col_width, th, str(datum), border=1)
                pdf.set_font('Times', '', 10)
                pdf.ln(th)  # 2 * th is for double padding
        else:
            pdf.cell(30, 5, 'Smoothing Filter', ln=0)
            pdf.set_font('Times', '', 10)
            pdf.cell(30, 5, use_filter, ln=1)
        pdf.set_font('Times', '', 8)
        pdf.cell(epw, 5, '**Smoothing is only done for velocity and ice thickness data. '
                         'If dynamic smoothing uses 0 for both inputs, there is no smoothing**')

        pdf.ln(2 * th)
        pdf.set_font('Times', 'B', 12)
        pdf.cell(epw, 5, 'Calculation Results', ln=1, align='C')
        pdf.ln(0.5)
        col_width = epw / 2
        data = [['Area', str(round(area/1000000, 2))+' (sq. km)'],
                ['Total MB', str(round(totalMassBalanceValue, 2))+' (mm w.e.)'],
                ['Total MB, alt-resolved', str(round(altResTotMassBalance, 2))+' (mm w.e.)'],
                ['Climatic MB (smoothed np.gradient)', str(round(climMassBalanceValue, 2)) + ' (mm w.e.)'],
                ['Climatic MB, alt-resolved (smoothed np.gradient)', str(round(altResClimMassBalance, 2))+' (mm w.e.)'],
                ['Climatic MB (emergence)', str(round(climMassBalanceValueEm, 2))+' (mm w.e.)'],
                ['Climatic MB, alt-resolved (emergence)', str(round(altResClimMassBalanceEm, 2))+' (mm w.e.)'],
                ["Emergence method 'out-of-bounds' pixel mass", str(round(climMassBalanceValueEmLoss, 2))+' (mm w.e.)']]

        pdf.set_font('Times', '', 10)
        for row in data:
            for datum in row:
                pdf.cell(col_width, th, str(datum), border=1)
            pdf.ln(th)                          # 2 * th is for double padding
        pdf.set_font('Times', '', 8)
        pdf.cell(epw, 5, '**Climatic MB from emergence method does not use smoothed data**')

        pdf.ln(2 * th)
        fig1 = mbPlotsTitle.replace(' ', '_') + '.png'
        pdf.image('Figures/' + fig1, x=None, y=None, w=epw)
        pdf.ln(2 * th)
        fig2 = extraPlotTitle.replace(' ', '_') + '.png'
        pdf.image('Figures/' + fig2, x=None, y=None, w=epw)

        pdfName = glacierCalc.name + '_' + str(glacierCalc.res) + 'res_' + use_filter + 'Smoothing' + \
                  str(raw_data_smooth_factor) + str(divQ_smooth_factor) + '_' + glacierCalc.h[:-4] + '_' + \
                  glacierCalc.vx[:-4] + '.pdf'
        pdf.output('PDFs/' + pdfName, 'F')

# to show all the figures from the calculation (not saved .png files) -- make sure livePlots is set to 1 or True
plt.show()

