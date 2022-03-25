'''
Code for compariong velocity products from each glacier
Automatically obtains glacier data using RGI shapefiles, not specification/prep needed
Outputs pdf with plot comparing result from velocity products
'''

# Import necessary modules
from Modules.dataPlots import elevationBinPlot, elevationBinPlot2, elevationBinPlot3Subfigs, elevationBinPlot3data3Subfigs
from Modules.dataPlots import plotData, plotData3, plotData6, plotDataPoints, velPlot
from Modules.smoothingFunctions import sgolay2d, gaussianFilter, dynamicSmoothing
from glacierDatabase import glacierInfo
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio.plot
import shapefile
from rasterio.plot import show
from numpy import savetxt
from sklearn.metrics import mean_absolute_error
from tkinter import *
from fpdf import FPDF
import os
import sys

vel_dict = {}
# -----------------------------------------------------GENERAL---------------------------------------------------------#
res = 20  # Resampling resolution (x, y) tuple
avgDensity = 850  # Average glacier mass density (kg/m3)
elevationBinWidth = 50  # Elevation bin width, for elevation binning calculations (m)
vCol = 0.8  # surface column average velocity scaling

g1 = 'Gulkana'
g2 = 'Wolverine'
g3 = 'Eklutna'
g4 = 'LemonCreek'
g5 = 'Rainbow'
g6 = 'SouthCascade'
g7 = 'Conrad'
g8 = 'Illecillewaet'
g9 = 'Nordic'
g10 = 'Zillmer'
g11 = 'RikhaSamba'
g12 = 'All Other Glaciers'

h1 = 'FarinottiThickness'
h2 = 'MillanThickness'

v1 = 'ITS_LIVE_20yrComposite'
v2 = 'ITS_LIVE_2017-2018'
v3 = 'MillanVelocity_2017-2018'
v4 = 'RETREAT_2017-2018'
v5 = 'RETREAT_2020'
v6 = 'RETREAT_2015-2020'

time1 = '2000-2020'
time2 = '2015-2020'

glaciers = (g12, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11)
thick_data = (h1, h2)
vel_data = (v1, v2, v3, v4, v5, v6)
time_data = (time1, time2)

# chose glacier to perform calculations with GUI
# select one of multiple glaciers in the set
# double click, press the return key, or click the 'SELECT' button to run calculations
def handler(event=None):
    global glac, thick_select, vel_select
    glac = lb1.curselection()
    glac = [lb1.get(int(x)) for x in glac]
    thick_select = lb2.curselection()
    thick_select = [lb2.get(int(x)) for x in thick_select]
    vel_select = lb3.curselection()
    vel_select = [lb3.get(int(x)) for x in vel_select]
    t.destroy()


t = Tk()
t.title('Glacier and Data Selection')
lb1 = Listbox(t, height=15, exportselection=FALSE)
lb2 = Listbox(t, height=15, exportselection=FALSE)
lb3 = Listbox(t, height=15, exportselection=FALSE)
lb1.config(selectmode=EXTENDED)  # select multiple items with CTRL and ranges of items with SHIFT
lb2.config(selectmode=SINGLE)  # select only one item
lb3.config(selectmode=EXTENDED)
for i in range(len(glaciers)):
    lb1.insert(i + 1, glaciers[i])
for i in range(len(thick_data)):
    lb2.insert(i + 1, thick_data[i])
for i in range(len(vel_data)):
    lb3.insert(i + 1, vel_data[i])
lb1.select_set(0)  # default beginning selection in GUI
lb2.select_set(1)
lb3.select_set(1, 3)
lb1.pack(side=LEFT)
lb2.pack(side=LEFT)
lb3.pack(side=LEFT)
t.bind('<Return>', handler)
t.bind('<Double-1>', handler)
b = Button(command=handler, text='SELECT', fg='black')
b.pack(side=LEFT)

w = 650  # tk window width
h = 220  # tk window height
ws = t.winfo_screenwidth()  # width of the screen
hs = t.winfo_screenheight()  # height of the screen
x = (ws / 2) - (w / 2)
y = (hs / 2) - (h / 2)
t.geometry('%dx%d+%d+%d' % (w, h, x, y))
t.mainloop()

# create a GUI to handle regional analysis for all other glaciers
if glac[0] == 'All Other Glaciers':
    def handler2(event=None):
        global glacier_number_from, glacier_number_to, time_select
        glacier_number_from = ent.get()
        glacier_number_to = ent2.get()
        time_select = lb4.curselection()
        time_select = [lb4.get(int(x)) for x in time_select]
        t2.destroy()

    t2 = Tk()
    t2.title('Glacier RGI Number')
    ent = Entry(t2, exportselection=FALSE, justify='center')
    ent.insert(END, '570')          # gulkana is 570, wolverine is 9162, lemon creek 1104,
    ent.pack(side=TOP)
    ent2 = Entry(t2, exportselection=FALSE, justify='center')
    # ent2.insert(END, '571')
    ent2.pack(side=TOP)

    lb4 = Listbox(t2, height=3, exportselection=FALSE)
    lb4.config(selectmode=SINGLE)
    for i in range(len(time_data)):
        lb4.insert(i + 1, time_data[i])
    lb4.select_set(0)
    lb4.configure(justify=CENTER)
    lb4.pack()

    t2.bind('<Return>', handler2)
    t2.bind('<Double-1>', handler2)
    b2 = Button(command=handler2, text='ENTER', fg='black')
    b2.pack(side=BOTTOM)
    w = 300  # tk window width
    h = 130  # tk window height
    ws = t2.winfo_screenwidth()  # width of the screen
    hs = t2.winfo_screenheight()  # height of the screen
    x = (ws / 2) - (w / 2)
    y = (hs / 2) - (h / 2)
    t2.geometry('%dx%d+%d+%d' % (w, h, x, y))
    t2.mainloop()

    if glacier_number_to == '':
        glacier_number_to = int(glacier_number_from) + 1
    if int(glacier_number_from) >= 27112 or int(glacier_number_from) < 1:
        sys.exit('Value out of range (must be between 1 and 27,112).')
    if int(glacier_number_to) > 27112 or int(glacier_number_to) <= 1:
        sys.exit('Value out of range (must be between 1 and 27,112).')
    if int(glacier_number_from) >= int(glacier_number_to):
        sys.exit('First entry must be smaller than second entry')

    other_glac_list = list(map(str, range(int(glacier_number_from), int(glacier_number_to))))
    glac = glac[1:]
    glac.extend(other_glac_list)

# ---------------------------------------------------------------------------------------------------------------------#
time_span = '2000-2020'     # dhdt files are 2000-2020 for glaciers by default, unless specified otherwise
number_of_glac = len(glac)
# iterate through the total selected glaciers
for glacier in range(number_of_glac):
    glacier_calc_type = 'Reference'
    print()
    print('--------------------------------', glac[glacier], 'Glacier Calculations --------------------------------')

    number_of_vel = len(vel_select)
    # iterate through each velocity product
    if number_of_vel >= 4:
        sys.exit('Too many velocity products selected. Maximum of 3 products allowed')
    for j in range(number_of_vel):
        # create an object to define each glacier that specifies input parameters and files
        glacierCalc = glacierInfo(glac[glacier], res, avgDensity, elevationBinWidth, vCol, thick_select[0], vel_select[j])

        # navigate to appropriate directory (only once per glacier)
        if j == 0:
            if os.path.exists(os.getcwd() + '/' + glacierCalc.name) == True:
                os.chdir(os.getcwd() + '/' + glacierCalc.name)  # navigate to working directory
            elif os.path.exists(os.path.dirname(os.getcwd()) + '/' + glacierCalc.name) == True:
                os.chdir(os.path.dirname(os.getcwd()))
                os.chdir(os.getcwd() + '/' + glacierCalc.name)

            # else:           # create folder of glacier name if needed
            #     if glacier == 0:
            #         os.makedirs(os.getcwd() + '/' + glacierCalc.name)
            #         os.chdir(os.getcwd() + '/' + glacierCalc.name)
            #     else:
            #         os.chdir(os.path.dirname(os.getcwd()))
            #         os.makedirs(os.getcwd() + '/' + glacierCalc.name)
            #         os.chdir(os.getcwd() + '/' + glacierCalc.name)

            # create necessary folders IF they don't already exist: PDF and Figure
            if os.path.exists(os.getcwd() + '/Figures') == False:
                os.makedirs(os.getcwd() + '/Figures')
            if os.path.exists(os.getcwd() + '/PDFs') == False:
                os.makedirs(os.getcwd() + '/PDFs')

        # for regional analysis, obtain the glacier name, dhdt time span, crs, and all files (h, v, dhdt, shp, dem)
        if glacierCalc.name == 'Other':
            glacier_calc_type = 'Other'
            glacierCalc.name = glac[glacier].zfill(5)
            time_span = time_select[0]
            glacierCalc.data, glacierCalc.dataVariability, dataCoords = glacierCalc.getReferenceData(time_span)

            shp_dataframe = gpd.GeoDataFrame.from_file(glacierCalc.shape)
            shp_attributes = shp_dataframe.loc[shp_dataframe['RGIId'] == 'RGI60-01.' + glacierCalc.name]
            shp_attributes.to_file('RGI60-01.' + glacierCalc.name + '.shp')
            glacierCalc.shape = 'RGI60-01.' + glacierCalc.name + '.shp'
            glacierCalc.crs = glacierCalc.getCRS()

            glacierCalc.dem = 'RGI1_DEM/RGI60-01.' + glacierCalc.name + '_dem.tif'
            glacierCalc.dhdt = glacierCalc.getDhdtFile(time_span)
            if thick_select[0] == 'MillanThickness':
                glacierCalc.h = glacierCalc.MillanThicknessRGI1_2()
            elif thick_select[0] == 'FarinottiThickness':
                glacierCalc.h = 'RGI1_Thickness_Farinotti_Composite/RGI60-01.' + glacierCalc.name + '_thickness.tif'
            if vel_select[j] == 'MillanVelocity_2017-2018':
                glacierCalc.vx, glacierCalc.vy = glacierCalc.MillanVelocityRGI1_2()

            glacierCalc.initialClip()       # pre filtering to clip extent of regional files (exclude DEM!)
            glacierCalc.h = glacierCalc.name + '_h.tif'
            glacierCalc.dhdt = glacierCalc.name + '_dhdt.tif'
            glacierCalc.vx = glacierCalc.name + '_vx.tif'
            glacierCalc.vy = glacierCalc.name + '_vy.tif'

        # STEP 1: CONVERT COORDINATE SYSTEMS AND RESAMPLE
        glacierCalc.shpReprojection()  # reproject shapefile
        glacierCalc.tifReprojectionResample()
        glacierCalc.velReprojectionResample()

        # STEP 2: CLIP RASTERS TO SAME EXTENT
        glacierCalc.tifClip()
        glacierCalc.fillHole()
        glacierCalc.fillVel()

        glacierCalc.shpClip()  # clip rasters by glacier RGI shape
        glacierCalc.glacierOutline()
        glacierCalc.missingPixels()  # make sure data does not have large gaps or holes

        # ----------------------------------------------FIX VELOCITY DIRECTIONS----------------------------------------------#
        outlineArray = rasterio.open(glacierCalc.name + 'Outline.tif').read(1)
        demArray = rasterio.open(glacierCalc.dem[:-4] + '_clip.tif').read(1) * outlineArray
        hArray = rasterio.open(glacierCalc.h[:-4] + '_clip.tif').read(1) * outlineArray
        vxArray = rasterio.open(glacierCalc.vx[:-4] + '_clip.tif').read(1) * outlineArray
        vyArray = rasterio.open(glacierCalc.vy[:-4] + '_clip.tif').read(1) * outlineArray
        vArray = np.power((np.square(vxArray) + np.square(vyArray)), 0.5)

        # correct for x and y velocity, if needed (usually needed ITS_LIVE in AK, but not for other velocity products)
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
        vel_total = np.power((np.square(vel_x) + np.square(vel_y)), 0.5)

        # -------------------------------------STEP 3: CALCULATE AREA AND TOTAL MB-------------------------------#
        if j == 0:
            area = glacierCalc.glacierArea()  # calculate and print glacier area
            massBalanceTotal = glacierCalc.totalMassBalance()
            totalMassBalanceValue = glacierCalc.totalMassBalanceValue()
            print(glacierCalc.name, 'Area (sq. km):', area / 1000000)
            print('The TOTAL mass balance is: ', totalMassBalanceValue, '(mm w.e.)')

        # filter out glaciers with small areas (1 sq. km), and remove any created/saved files in the process
        skip = False
        if area / 1000000 < 1:
            skip = True
            print('AREA IS TOO SMALL (' + str(area/1000000) + ' sq. km)')
            os.remove(glacierCalc.vx[:-4] + '_clip.tif')
            os.remove(glacierCalc.vx[:-4] + '_reproj.tif')
            os.remove(glacierCalc.vy[:-4] + '_clip.tif')
            os.remove(glacierCalc.vy[:-4] + '_reproj.tif')
            os.remove(glacierCalc.h)
            os.remove(glacierCalc.dhdt)
            os.remove(glacierCalc.vx)
            os.remove(glacierCalc.vy)
            os.remove('RGI60-01.' + glacierCalc.name + '.shp')
            os.remove('RGI60-01.' + glacierCalc.name + '.shx')
            os.remove('RGI60-01.' + glacierCalc.name + '.dbf')
            os.remove('RGI60-01.' + glacierCalc.name + '.prj')
            os.remove('RGI60-01.' + glacierCalc.name + '.cpg')
            os.remove(glacierCalc.name + 'Outline.tif')
            os.remove(glacierCalc.shape[:-4] + '_reproj.shp')
            os.remove(glacierCalc.shape[:-4] + '_reproj.shx')
            os.remove(glacierCalc.shape[:-4] + '_reproj.dbf')
            os.remove(glacierCalc.shape[:-4] + '_reproj.prj')
            os.remove(glacierCalc.shape[:-4] + '_reproj.cpg')
            os.remove(glacierCalc.dem[:-4] + '_clip.tif')
            os.remove(glacierCalc.dem[:-4] + '_reproj.tif')
            os.remove(glacierCalc.h[:-4] + '_clip.tif')
            os.remove(glacierCalc.h[:-4] + '_reproj.tif')
            os.remove(glacierCalc.dhdt[:-4] + '_clip.tif')
            os.remove(glacierCalc.dhdt[:-4] + '_reproj.tif')
            if os.path.exists('temp_dhdt.tif'):
                os.remove('temp_dhdt.tif')
            if os.path.exists('temp_h.tif'):
                os.remove('temp_h.tif')
            if os.path.exists('temp_vx.tif'):
                os.remove('temp_vx.tif')
            if os.path.exists('temp_vy.tif'):
                os.remove('temp_vy.tif')
            break

        print('----', vel_select[j], 'Calculations')
        # ------------------------------------------PART 4 DISPLAY OPTIONS-----------------------------------------#
        printMassLoss = 1  # to print the mass loss values
        # STEP 4: SMOOTH VELOCITY DATA WITH LOW PASS MOVING WINDOW FILTER, CLIMATIC MASS BALANCE FOR SMOOTH VELOCITY DATA
        raw_data_smooth_factor = 4      # smoothing factor for h and v files (factor*h is window size)
        divQ_smooth_factor = 1          # smoothing factor for divQ file (factor*h is window size)

        if j == 0:
            smoothhArray = dynamicSmoothing(hArray, hArray, glacierCalc.res, raw_data_smooth_factor)
        smoothvxArray = dynamicSmoothing(vel_x, hArray, glacierCalc.res, raw_data_smooth_factor)
        smoothvyArray = dynamicSmoothing(vel_y, hArray, glacierCalc.res, raw_data_smooth_factor)
        smoothvArray = np.power((np.square(smoothvxArray) + np.square(smoothvyArray)), 0.5)

        divQRaster = ((np.gradient(smoothvxArray, glacierCalc.res)[1] * smoothhArray) +
                      (np.gradient(smoothhArray, glacierCalc.res)[1] * smoothvxArray) +
                      (np.gradient(smoothvyArray * -1, glacierCalc.res)[0] * smoothhArray) +
                      (np.gradient(smoothhArray * -1, glacierCalc.res)[0] * smoothvyArray)) * glacierCalc.vCol
        divQRasterSmooth = dynamicSmoothing(divQRaster, hArray, glacierCalc.res, divQ_smooth_factor)

        if j == 0:
            dhdtFile = glacierCalc.dhdt[:-4] + '_clip.tif'
            dhdtArray = rasterio.open(dhdtFile).read(1)
            # total SMB calculations
            total_smb_pixel = rasterio.open(glacierCalc.name + 'TotalMassBalance.tif').read(1)
            binMeandhdt, binElevations1, binCounts1, binNumber1, bin_std_dhdt, bin_min_dhdt, bin_max_dhdt = \
                glacierCalc.altitudeAggregation(dhdtArray, stat='mean')

            binTotalChange = binMeandhdt * glacierCalc.rho
            demTotBinTotalMass = binCounts1 * binTotalChange * glacierCalc.res * glacierCalc.res  # this is in kg/yr
            altResTotMassBalance = np.sum(demTotBinTotalMass) / area  # divide by total area for alt-resolved mb value

        # climatic SMB calculations
        clim_smb_pixel_smooth = (dhdtArray + divQRasterSmooth) * glacierCalc.rho
        climMassBalanceValueSmooth = np.sum(clim_smb_pixel_smooth * glacierCalc.res * glacierCalc.res) / area
        bindivQSmoothMean, binElevations2, binCounts2, binNumber2, bin_std_divQsmooth, bin_min_divQsmooth, bin_max_divQsmooth = \
            glacierCalc.altitudeAggregation(divQRasterSmooth, stat='mean')

        binClimChangeSmooth = (binMeandhdt + bindivQSmoothMean) * glacierCalc.rho
        demClimBinTotalMassSmooth = binCounts2 * binClimChangeSmooth * glacierCalc.res * glacierCalc.res
        altResClimMassBalanceSmooth = np.sum(demClimBinTotalMassSmooth) / area

        # print('The altitudinally resolved TOTAL mass balance is:', altResTotMassBalance, '(mm w.e.)')
        print('The CLIMATIC (SMOOTH) mass balance is: ', climMassBalanceValueSmooth, '(mm w.e.)')
        # print('The altitudinally resolved CLIMATIC (SMOOTH) MB is:', altResClimMassBalanceSmooth, '(mm w.e.)')

        # ---------------------------------------------STEP 6: BIN ERRORS--------------------------------------------#
        errorVal = 'percent'  # 'std', 'se', 'minmax', 'percent'
        p = 25

        binTotalSMBMean, binElevations3, binCounts3, binNumber3, bin_std_totSmb, bin_min_totSmb, bin_max_totSmb = \
            glacierCalc.altitudeAggregation(total_smb_pixel, stat='mean')
        binClimSMBSmoothMean, binElevations4, binCounts4, binNumber4, bin_std_climSmbSmooth, bin_min_climSmbSmooth, \
        bin_max_climSmbSmooth = glacierCalc.altitudeAggregation(clim_smb_pixel_smooth * outlineArray, stat='mean')

        if errorVal == 'std':
            error_name = 'standard deviation'
            totalSMBErr = bin_std_totSmb
            climSMBSmoothErr = bin_std_climSmbSmooth
        if errorVal == 'se':
            error_name = 'standard error'
            totalSMBErr = bin_std_totSmb / (binCounts3 ** 0.5)
            climSMBSmoothErr = bin_std_climSmbSmooth / (binCounts4 ** 0.5)
        if errorVal == 'minmax':
            error_name = 'min and max value'
            totalSMBErr = np.array([binTotalSMBMean - bin_min_totSmb, bin_max_totSmb - binTotalSMBMean])
            climSMBSmoothErr = np.array([binClimSMBSmoothMean - bin_min_climSmbSmooth, bin_max_climSmbSmooth -
                                         binClimSMBSmoothMean])
        if errorVal == 'percent':
            error_name = 'percentile: lower bound at ' + str(p) + '%, upper bound at ' + str(100 - p) + '%'
            binTotalSMB_lower_p, binTotalSMB_upper_p = glacierCalc.binPercentile(total_smb_pixel, p)
            binClimSMBSmooth_lower_p, binClimSMBSmooth_upper_p = glacierCalc.binPercentile(
                clim_smb_pixel_smooth * outlineArray, p)

            totalSMBErr = abs(np.array([binTotalSMB_lower_p - binTotalSMBMean, binTotalSMB_upper_p - binTotalSMBMean]))
            climSMBSmoothErr = abs(np.array([binClimSMBSmooth_lower_p - binClimSMBSmoothMean, binClimSMBSmooth_upper_p -
                                             binClimSMBSmoothMean]))
        if errorVal == 'none':
            error_name = 'no error bars'
            totalSMBErr = None
            climSMBSmoothErr = None

        # ------------------------------------------STEP 7: PLOTTING/RESULTS-----------------------------------------#
        livePlots = 0  # to show the plots as live plots (not just reopened saved files)
        create_pdf = 1  # to create pdf containing glacier mass balance information
        # apply bin SMB to each value in that bin: FOR PLOTTING ONLY
        binnedTotalMassBalance = np.zeros(binNumber1.shape)
        binnedClimaticMassBalanceSmooth = np.zeros(binNumber2.shape)
        binnedEmergenceSmooth = np.zeros(binNumber2.shape)
        for i in range(len(binNumber1)):
            binnedTotalMassBalance[i] = binTotalChange[binNumber1[i] - 1]
            binnedClimaticMassBalanceSmooth[i] = binClimChangeSmooth[binNumber2[i] - 1]
            binnedEmergenceSmooth[i] = bindivQSmoothMean[binNumber2[i] - 1]

        outlineArrayShape = outlineArray.shape
        binnedTotalMassBalance = np.array(binnedTotalMassBalance).reshape(outlineArrayShape) * outlineArray
        binnedClimaticMassBalanceSmooth = np.array(binnedClimaticMassBalanceSmooth).reshape(
            outlineArrayShape) * outlineArray
        binnedEmergenceSmooth = np.array(binnedEmergenceSmooth).reshape(outlineArrayShape) * outlineArray
        binnedElevationNumber = np.array(binNumber1).reshape(outlineArrayShape) * outlineArray

        # for plotting: each subplot has: [data array, title, colorbar label]
        if j == 0:
            h_plot = [hArray, 'Thickness Map', 'Glacier Thickness (m)', 'ocean']
            h_plot_smooth = [smoothhArray, 'Smoothed Thickness Map', 'Glacier Thickness (m)', 'ocean']
            z_plot = [demArray, 'Elevation Map', 'Elevation (m)', 'gist_earth']  # gist_earth, viridis, jet
            b_tot_plot = [binnedTotalMassBalance, 'Total Mass Balance Map', 'Mean Elevation Bin SMB (mm w.e.)', 'RdBu']
            b_tot_plot_pixel = [total_smb_pixel * outlineArray, 'Total Mass Balance Map', 'Pixel Mass Balance (mm w.e.)',
                                'RdBu']
            dhdt_plot_pixel = [dhdtArray, 'Change in Thickness Map', 'dh/dt (m/a)', 'RdBu']

        v_plot = [vel_total, vel_select[j] + ' Velocity Map', ' Velocity (m/a)', 'BrBG']
        v_plot_smooth = [smoothvArray, vel_select[j] + ' Smoothed Velocity Map', 'Velocity (m/a)', 'BrBG']
        # vx_plot = [vel_x, vel_select[j] + ' X-Velocity Map', 'Velocity (m/a)', 'BrBG']
        # vx_plot_smooth = [smoothvxArray, vel_select[j] + ' Smoothed X-Velocity Map', 'Velocity (m/a)', 'BrBG']
        # vy_plot = [vel_y, vel_select[j] + ' Y-Velocity Map', 'Velocity (m/a)', 'BrBG']
        # vy_plot_smooth = [smoothvyArray, vel_select[j] + ' Smoothed Y-Velocity Map', 'Velocity (m/a)', 'BrBG']
        b_clim_plot_smooth = [binnedClimaticMassBalanceSmooth, vel_select[j] + ' Climatic Mass Balance Map (smooth)',
                              'Mean Elevation Bin SMB (mm w.e.)', 'RdBu']
        b_clim_plot_pixel_smooth = [clim_smb_pixel_smooth * outlineArray, vel_select[j] +
                                    ' Climatic Mass Balance Map (smooth)', 'Pixel Mass Balance (mm w.e.)', 'RdBu']
        emergence_plot_smooth = [binnedEmergenceSmooth, vel_select[j] + ' Emergence Map (smooth)',
                                 'Mean Elevation Bin Emergence (m/a)', 'RdBu']
        emergence_plot_pixel_smooth = [divQRasterSmooth * outlineArray, vel_select[j] + ' Smoothed Emergence Map',
                                       'Pixel Emergence (m/a)', 'RdBu']

        # -------QUANTIFY DIFFERENCES IN VELOCITY AND GLACIOLOGICAL MEASUREMENTS--------#
        # Comparing MB to glaciologic measurements for each velocity product (where WGMS data exist)
        if glacierCalc.data is not None:
            # APPROACH 1: mean errors for elevation bins with glaciologic measurements
            binAreasList = (binCounts2 * glacierCalc.res * glacierCalc.res).tolist()
            binElevationsList = binElevations2.tolist()

            mb_model_prediction = []
            mb_model_prediction_avg = []
            for i in range(len(glacierCalc.data[1])):
                # find index value where reference data elevation exceeds elevation bin
                indexVal = next(x[0] for x in enumerate(binElevationsList) if x[1] >= glacierCalc.data[1][i]) - 1
                # value of MB model for elevation bin where reference data exists
                mb_model_prediction.append(binClimChangeSmooth[indexVal])

                # alternative approach: area-weighted mean of 3 adjacent elevation bins
                # find the total area of 3 adjacent elevation bins
                areaAdjacent = binAreasList[indexVal - 1] + binAreasList[indexVal] + binAreasList[indexVal + 1]
                # weighted mean of 3 adjacted elevation bins
                weighted_mean = \
                    np.mean([binClimChangeSmooth[indexVal - 1] * binAreasList[indexVal - 1] / areaAdjacent,
                            binClimChangeSmooth[indexVal] * binAreasList[indexVal] / areaAdjacent,
                            binClimChangeSmooth[indexVal + 1] * binAreasList[indexVal + 1] / areaAdjacent])
                mb_model_prediction_avg.append(weighted_mean)

            # mean absolute error and mean bias error:
            meanAbsErr = mean_absolute_error(glacierCalc.data[0], mb_model_prediction)
            meanBiasErr = np.mean([L1 - L2 for L1, L2 in zip(glacierCalc.data[0], mb_model_prediction)])
            # mean absolute error and mean bias error, area-weighted by 3 adjacent bin averages
            meanAbsErr3BinWeighted = mean_absolute_error(glacierCalc.data[0], mb_model_prediction_avg)
            meanBiasErr3BinWeighted = np.mean([L1 - L2 for L1, L2 in zip(glacierCalc.data[0],
                                                                         mb_model_prediction_avg)])  # (subtract lists)

            # APPROACH 2: comparison of values to line of best fit from glaciological measurements
            # let's only consider bins from the min to max total mass balance (removes initial, low-area bins)
            min_index = binTotalChange.tolist().index(min(binTotalChange))
            # max bin index must be after min bin index and must have an area at least as large as min index
            min_bin_area = binAreasList[min_index]
            max_index = binTotalChange.tolist().index(max(binTotalChange[min_index:]))
            max_bin_area = binAreasList[max_index]
            # make sure max index has a large enough area:
            # if not, find the max index before max_index with a large enough area
            if max_bin_area < min_bin_area:
                max_index = max(i for i, a in enumerate(binAreasList[:max_index]) if a >= min_bin_area)

            validIndexes = list(range(min_index, max_index))
            valid_mb_vals = [binClimChangeSmooth[x] for x in validIndexes]
            # obtain accompanying values from line of best fit (lobf)
            # get elevations associated with the MB points
            for i in range(len(binElevationsList)):
                binElevationsList[i] = binElevationsList[i] + (elevationBinWidth * 0.5)
            binElevationsList[0:len(binElevationsList) - 1]
            valid_elevation_vals = [binElevationsList[x] for x in validIndexes]
            # get the lobf and the lobf values at these elevations
            m, b = np.polyfit(glacierCalc.data[0], glacierCalc.data[1], 1)
            accompanying_lobf_vals = [yb / m for yb in [y - b for y in valid_elevation_vals]]  # (binElevationsList-b)/m
            # mean absolute error and mean bias error:
            meanAbsErrLine = mean_absolute_error(accompanying_lobf_vals, valid_mb_vals)
            meanBiasErrLine = np.mean([L1 - L2 for L1, L2 in zip(accompanying_lobf_vals, valid_mb_vals)])

            # APPROACH 3: clim MB gradient from model to slope of line of best fit (lobf)
            m_mb_model, b_mb_model = np.polyfit(valid_mb_vals, valid_elevation_vals, 1)
            # positive error means model has more mass loss per elevation than reference data
            clim_mb_grad_err = 100 * (m - m_mb_model) / m

            # APPROACH 4: pixel specific points and locations (not entire elevation bins)
            # compare reference points to pixel-secific regions (perhaps weighted average around location)
            # dataCoords = [Lat, Lon, Avg MB, Elevation, Point ID]
            app4_pixel_window_avg = 10  # the number of pixels to average (10 is 200m window size for 20m pixels)
            pixel_location_clim_mb_abs = []
            pixel_location_clim_mb_diff = []
            dataCoords_pixel_row = []    # store the row indices of the stake locations here
            dataCoords_pixel_col = []    # store the col indices of the stake locations here
            dataCoords_labels = []
            for coords_row in dataCoords:
                row, col = glacierCalc.latlonTiffIndex(clim_smb_pixel_smooth, coords_row)
                dataCoords_pixel_col.append([col])
                dataCoords_pixel_row.append([row])
                dataCoords_labels.append(coords_row[4])
                local_avg_mb = np.mean(clim_smb_pixel_smooth[row-app4_pixel_window_avg:row+app4_pixel_window_avg+1,
                                       col-app4_pixel_window_avg:col+app4_pixel_window_avg+1])  # window avg at point
                pixel_location_clim_mb_abs.append(abs(local_avg_mb - coords_row[2]))
                pixel_location_clim_mb_diff.append(local_avg_mb - coords_row[2])
            meanAbsErrPixelCrs = np.mean(pixel_location_clim_mb_abs)
            meanBiasErrPixelCrs = np.mean(pixel_location_clim_mb_diff)

            # compare emergence to Lucas Zeller emergence data: do we need more/less smoothing?
                # quantitative and qualitative analysis: similar to elevation vs MB bias/mean absolute error quantification above
                # map differencing

        # -------PLOTTING FOR EACH VELOCITY PRODUCT--------#
        fig4_1 = emergence_plot_pixel_smooth  # fig4: raw vs smooth velocity v
        fig4_2 = v_plot
        fig4_3 = v_plot_smooth

        # 'fig4' with 3 subplots
        fig4_1Vals = fig4_1[0]
        fig4_1cbar = fig4_1[2]
        fig4_1Title = fig4_1[1]
        fig4_1Color = fig4_1[3]
        fig4_2Vals = fig4_2[0]
        fig4_2cbar = fig4_2[2]
        fig4_2Title = fig4_2[1]
        fig4_2Color = fig4_2[3]
        fig4_2Quiver = velPlot(vel_x, vel_y, vel_total, area, threshold=1.0)  # plot velocity arrow
        fig4_3Vals = fig4_3[0]
        fig4_3cbar = fig4_3[2]
        fig4_3Title = fig4_3[1]
        fig4_3Color = fig4_3[3]
        fig4_3Quiver = velPlot(smoothvxArray, smoothvyArray, smoothvArray, area, threshold=1.0)  # plot velocity arrows
        smoothedVPlotTitle = glacierCalc.name + ' ' + vel_select[j] + ' Velocity Products'
        plotData3(fig4_1Vals, fig4_1cbar, fig4_1Color, fig4_1Title,
                  fig4_2Vals, fig4_2cbar, fig4_2Color, fig4_2Title,
                  fig4_3Vals, fig4_3cbar, fig4_3Color, fig4_3Title, smoothedVPlotTitle, glacierCalc.res,
                  fig4_2Quiver, fig4_3Quiver)

        if livePlots == False:
            plt.close('all')

        # ---------------------------------ASSIGN UNIQUE NAMES TO VARIABLES---------------------------#
        # we need new names for all data that incorporates velocity
        vel_dict['b_clim_plot_smooth%s' % j] = b_clim_plot_smooth
        vel_dict['b_clim_plot_pixel_smooth%s' % j] = b_clim_plot_pixel_smooth
        vel_dict['binClimChangeSmooth%s' % j] = binClimChangeSmooth
        vel_dict['climSMBSmoothErr%s' % j] = climSMBSmoothErr
        vel_dict['smoothedVPlotTitle%s' % j] = smoothedVPlotTitle
        vel_dict['climMassBalanceValueSmooth%s' % j] = climMassBalanceValueSmooth
        vel_dict['velPlot%s' % j] = v_plot
        vel_dict['velPlotSmooth%s' % j] = v_plot_smooth
        if glacierCalc.data is not None:
            vel_dict['MeanAbsoluteError%s' % j] = meanAbsErr
            vel_dict['WeightedMeanAbsoluteError%s' % j] = meanAbsErr3BinWeighted
            vel_dict['MeanBiasError%s' % j] = meanBiasErr
            vel_dict['WeightedMeanBiasError%s' % j] = meanBiasErr3BinWeighted
            vel_dict['MeanAbsoluteErrorLine%s' % j] = meanAbsErrLine
            vel_dict['MeanBiasErrorLine%s' % j] = meanBiasErrLine
            vel_dict['ClimMBGradError%s' % j] = clim_mb_grad_err
            vel_dict['MeanAbsoluteErrorPixel%s' % j] = meanAbsErrPixelCrs
            vel_dict['MeanBiasErrorPixel%s' % j] = meanBiasErrPixelCrs

        # remove saved files
        os.remove(glacierCalc.vx[:-4] + '_clip.tif')
        os.remove(glacierCalc.vx[:-4] + '_reproj.tif')
        os.remove(glacierCalc.vy[:-4] + '_clip.tif')
        os.remove(glacierCalc.vy[:-4] + '_reproj.tif')

        if glacier_calc_type == 'Other' and j == number_of_vel - 1:
            os.remove(glacierCalc.h)
            os.remove(glacierCalc.dhdt)
            os.remove(glacierCalc.vx)
            os.remove(glacierCalc.vy)
            os.remove('RGI60-01.' + glacierCalc.name + '.shp')
            os.remove('RGI60-01.' + glacierCalc.name + '.shx')
            os.remove('RGI60-01.' + glacierCalc.name + '.dbf')
            os.remove('RGI60-01.' + glacierCalc.name + '.prj')
            os.remove('RGI60-01.' + glacierCalc.name + '.cpg')
            os.remove('temp.shp')
            os.remove('temp.shx')
            os.remove('temp.dbf')
            os.remove('temp.prj')
            os.remove('temp.cpg')
            if os.path.exists('temp_dhdt.tif'):
                os.remove('temp_dhdt.tif')
            if os.path.exists('temp_h.tif'):
                os.remove('temp_h.tif')
            if os.path.exists('temp_vx.tif'):
                os.remove('temp_vx.tif')
            if os.path.exists('temp_vy.tif'):
                os.remove('temp_vy.tif')
            if os.path.exists(glacierCalc.name + '_array_temp.tif'):
                os.remove(glacierCalc.name + '_array_temp.tif')

        if j == number_of_vel - 1:
            os.remove(glacierCalc.name + 'TotalMassBalance.tif')
            os.remove(glacierCalc.name + 'Outline.tif')
            os.remove(glacierCalc.shape[:-4] + '_reproj.shp')
            os.remove(glacierCalc.shape[:-4] + '_reproj.shx')
            os.remove(glacierCalc.shape[:-4] + '_reproj.dbf')
            os.remove(glacierCalc.shape[:-4] + '_reproj.prj')
            os.remove(glacierCalc.shape[:-4] + '_reproj.cpg')
            os.remove(glacierCalc.dem[:-4] + '_clip.tif')
            os.remove(glacierCalc.dem[:-4] + '_reproj.tif')
            os.remove(glacierCalc.h[:-4] + '_clip.tif')
            os.remove(glacierCalc.h[:-4] + '_reproj.tif')
            os.remove(glacierCalc.dhdt[:-4] + '_clip.tif')
            os.remove(glacierCalc.dhdt[:-4] + '_reproj.tif')

    if skip == False:
        '''
        # only make plots for glaciers that are large enough
            # only for the last velocity file (once we have all velocity data) assuming there are at least 2 velocity files
            if j == number_of_vel - 1 and number_of_vel > 1:
                # threshold for what is identified as differing velocities (e.g., 0.1 is 10% tolerance between products)
                discrep_threshold = 0.1
                if number_of_vel == 2:
                    # call function with 2 inputs: vel_dict.get('velPlot%s' % i) OR vel_dict.get('velPlotSmooth%s' % i)
                    adj_vels = glacierCalc.velAdjusted(vel_dict.get('velPlot0')[0], vel_dict.get('velPlot1')[0])
                    discrep_locations = glacierCalc.velDiscrepancyRegions(adj_vels, discrep_threshold)
                if number_of_vel == 3:
                    # call function with 3 inputs
                    adj_vels = glacierCalc.velAdjusted(vel_dict.get('velPlot0')[0], vel_dict.get('velPlot1')[0],
                                                       vel_dict.get('velPlot2')[0])
                    discrep_locations = glacierCalc.velDiscrepancyRegions(adj_vels, discrep_threshold)
        '''

        # -----------non velocity plots-----------
        fig2_1 = z_plot  # figure 2: other plots
        fig2_2 = h_plot
        fig2_3 = b_tot_plot
        # fig2_3 = dhdt_plot_pixel

        fig3_1 = z_plot  # figure 3: other plots
        fig3_2 = h_plot_smooth
        fig3_3 = b_tot_plot_pixel

        # -----------chose subplots to show for main plot:------------
        if number_of_vel == 2:
            subplot2 = vel_dict.get('b_clim_plot_smooth0')
            subplot3 = b_tot_plot
            subplot4 = vel_dict.get('b_clim_plot_smooth1')
        elif number_of_vel == 3:
            subplot2 = vel_dict.get('b_clim_plot_smooth0')
            subplot3 = vel_dict.get('b_clim_plot_smooth1')
            subplot4 = vel_dict.get('b_clim_plot_smooth2')
        else:
            subplot2 = vel_dict.get('b_clim_plot_smooth0')
            subplot3 = b_tot_plot
            subplot4 = vel_dict.get('b_clim_plot_pixel_smooth0')


        # to show a scatter plot of elevation vs mean mass loss at each elevation bin, as well as 3 other figures
        binElevationsList = binElevations1.tolist()
        for i in range(len(binElevationsList)):
            binElevationsList[i] = binElevationsList[i] + (elevationBinWidth * 0.5)
        for i in range(number_of_vel):
            vel_dict['binClimChangeSmoothList%s' % i] = vel_dict.get('binClimChangeSmooth%s' % i).tolist()
        binTotalChangeList = binTotalChange.tolist()
        binAreasList = (binCounts1 * glacierCalc.res * glacierCalc.res).tolist()

        x_subplot_nw = binTotalChangeList
        x_subplot_nw_lab = 'Total MB'
        x_subplot_nw_2 = vel_dict.get('binClimChangeSmoothList0')
        x_subplot_nw_2_err = vel_dict.get('climSMBSmoothErr0')
        x_subplot_nw_2_lab = 'Climatic MB (' + vel_select[0] + ')'
        if number_of_vel == 1:
            x_subplot_nw_3 = vel_dict.get('binClimChangeSmoothList0')
            x_subplot_nw_3_err = vel_dict.get('climSMBSmoothErr0')
            x_subplot_nw_3_lab = 'Climatic MB (' + vel_select[0] + ')'
        if number_of_vel >= 2:
            x_subplot_nw_3 = vel_dict.get('binClimChangeSmoothList1')
            x_subplot_nw_3_err = vel_dict.get('climSMBSmoothErr1')
            x_subplot_nw_3_lab = 'Climatic MB (' + vel_select[1] + ')'

        if glacierCalc.data is not None:  # add reference data points
            x_subplot_nw_ref = glacierCalc.data[0]
            y_values1_ref = glacierCalc.data[1]
            x_subplot_nw_ref_lab = 'Reference Data'
        else:
            x_subplot_nw_ref = None
            y_values1_ref = None
            x_subplot_nw_ref_lab = None
        if glacierCalc.dataVariability is not None:
            var_values1_ref = glacierCalc.dataVariability
        else:
            var_values1_ref = None
        y_values1 = binElevationsList[0:len(binElevationsList) - 1]
        x_values2 = binElevationsList[0:len(binElevationsList) - 1]
        y_values2 = np.divide(binAreasList, 1000000)  # sq.km area
        xLabel1 = 'Bin Mean Mass Balance (mm w.e.)'
        yLabel1 = 'Bin Mean Elevation (m)'
        xLabel2 = 'Bin Area ($km^2$)'
        title1 = 'Binned SMB Plot'
        mbPlotsTitle = glacierCalc.name + ' Velocity Product Comparison'
        buff = 500
        alpha = 0.3

        # to save a csv with the data points needed to plot the mass balance vs elevation
        # savetxt('smoothClimChange.csv', binClimChangeSmoothList, delimiter=',')
        # savetxt('smoothClimChangeErrMin.csv', climSMBSmoothErr[0], delimiter=',')
        # savetxt('smoothClimChangeErrMax.csv', climSMBSmoothErr[1], delimiter=',')

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

        if number_of_vel <= 2:
            elevationBinPlot3Subfigs(x_subplot_nw, x_subplot_nw_lab, x_subplot_nw_2, x_subplot_nw_2_lab, x_subplot_nw_3,
                                     x_subplot_nw_3_lab, y_values1, x_values2, y_values2, xLabel1, yLabel1, xLabel2, title1,
                                     elevationBinWidth, buff, alpha, mbPlotsTitle, glacierCalc.res,
                                     subplot_ne, subplot_ne_title, subplot_ne_label, color_ne,
                                     subplot_sw, subplot_sw_title, subplot_sw_label, color_sw,
                                     subplot_se, subplot_se_title, subplot_se_label, color_se,
                                     x_subplot_nw_2_err, x_subplot_nw_3_err,
                                     x_subplot_nw_ref, x_subplot_nw_ref_lab, y_values1_ref, var_values1_ref, subplot_sw_cbar_ticks)
        else:
            x_subplot_nw_4 = vel_dict.get('binClimChangeSmoothList2')
            x_subplot_nw_4_err = vel_dict.get('climSMBSmoothErr2')
            x_subplot_nw_4_lab = 'Climatic MB (' + vel_select[2] + ')'
            elevationBinPlot3data3Subfigs(x_subplot_nw, x_subplot_nw_lab, x_subplot_nw_2, x_subplot_nw_2_lab, x_subplot_nw_3,
                                     x_subplot_nw_3_lab, x_subplot_nw_4,
                                     x_subplot_nw_4_lab, y_values1, x_values2, y_values2, xLabel1, yLabel1, xLabel2, title1,
                                     elevationBinWidth, buff, alpha, mbPlotsTitle, glacierCalc.res,
                                     subplot_ne, subplot_ne_title, subplot_ne_label, color_ne,
                                     subplot_sw, subplot_sw_title, subplot_sw_label, color_sw,
                                     subplot_se, subplot_se_title, subplot_se_label, color_se,
                                     x_subplot_nw_2_err, x_subplot_nw_3_err, x_subplot_nw_4_err,
                                     x_subplot_nw_ref, x_subplot_nw_ref_lab, y_values1_ref, var_values1_ref, subplot_sw_cbar_ticks)

        # second figure with 3 subplots
        fig2_1Vals = fig2_1[0]
        fig2_1cbar = fig2_1[2]
        fig2_1Title = fig2_1[1]
        fig2_1Color = fig2_1[3]
        fig2_2Vals = fig2_2[0]
        fig2_2cbar = fig2_2[2]
        fig2_2Title = fig2_2[1]
        fig2_2Color = fig2_2[3]
        fig2_2Quiver = None
        fig2_3Vals = fig2_3[0]
        fig2_3cbar = fig2_3[2]
        fig2_3Title = fig2_3[1]
        fig2_3Color = fig2_3[3]
        fig2_3Quiver = None
        extraPlotTitle2 = glacierCalc.name + ' Thickness, Elevation, Total MB Plots'
        plotData3(fig2_1Vals, fig2_1cbar, fig2_1Color, fig2_1Title,
                  fig2_2Vals, fig2_2cbar, fig2_2Color, fig2_2Title,
                  fig2_3Vals, fig2_3cbar, fig2_3Color, fig2_3Title, extraPlotTitle2, glacierCalc.res,
                  fig2_2Quiver, fig2_3Quiver)

        # third figure with 3 subplots
        fig3_1Vals = fig3_1[0]
        fig3_1cbar = fig3_1[2]
        fig3_1Title = fig3_1[1]
        fig3_1Color = fig3_1[3]
        fig3_2Vals = fig3_2[0]
        fig3_2cbar = fig3_2[2]
        fig3_2Title = fig3_2[1]
        fig3_2Color = fig3_2[3]
        fig3_2Quiver = None
        fig3_3Vals = fig3_3[0]
        fig3_3cbar = fig3_3[2]
        fig3_3Title = fig3_3[1]
        fig3_3Color = fig3_3[3]
        fig3_3Quiver = None
        extraPlotTitle3 = glacierCalc.name + ' Raw Thickness, Elevation, Total MB Pixel Plots'
        plotData3(fig3_1Vals, fig3_1cbar, fig3_1Color, fig3_1Title,
                  fig3_2Vals, fig3_2cbar, fig3_2Color, fig3_2Title,
                  fig3_3Vals, fig3_3cbar, fig3_3Color, fig3_3Title, extraPlotTitle3, glacierCalc.res,
                  fig3_2Quiver, fig3_3Quiver)

        # plot of stake reference data locations
        if glacierCalc.data is not None:
            reference_plot_background = z_plot
            reference_plot_background2 = b_tot_plot_pixel
            fig_stakes_title = 'Stake Point Plots'
            plotDataPoints(reference_plot_background[0], reference_plot_background[2], reference_plot_background[3],
                           b_tot_plot_pixel[0], b_tot_plot_pixel[2], b_tot_plot_pixel[3],
                           fig_stakes_title, dataCoords_pixel_col, dataCoords_pixel_row, dataCoords_labels)


        if livePlots == False:
            plt.close('all')

        # to make and save information as pdf file
        if create_pdf == True:
            pdf = FPDF('P', 'mm', 'A4')  # initialize pdf with A4 portrait mode size in mm units
            pdf.add_page()
            pdf.set_font('Times', 'BU', 18)  # set Times font (or 'Arial'), regular text (or 'B', 'I', 'U'), size 12
            epw = pdf.w - 2 * pdf.l_margin  # effective page width
            # pdf.cell(w, h=0, txt='', border=0, ln=0, align='', fill=False, link='')
            pdf.cell(epw, 10, glacierCalc.name + ' Glacier Info', border=0, ln=1, align='C')

            col_width = epw / 5  # Set column width to 1/2 of effective page width to distribute content
            data = [['DEM file', glacierCalc.dem],
                    ['Change in thickness file', glacierCalc.dhdt + ' (Hugonnet, ' + time_span + ')'],
                    ['Thickness file', glacierCalc.h + ' (' + thick_select[0] + ')'],
                    ['Velocity file(s)', vel_select[0:number_of_vel]]]
            pdf.set_font('Times', 'B', 12)
            pdf.cell(epw, 5, 'Input data files', ln=1, align='C')
            pdf.ln(0.5)
            pdf.set_font('Times', '', 9)
            th = pdf.font_size  # Text height is the same as current font size
            for row in data:
                i = 0
                for datum in row:
                    i = i + 1
                    pdf.cell(col_width * (i ** 2), th, str(datum), border=1)
                pdf.ln(th)  # 2 * th is for double padding

            pdf.ln(2 * th)  # Line break equivalent to 2 lines
            pdf.set_font('Times', 'B', 12)
            pdf.cell(epw, 5, 'Input Constants and Assumptions', ln=1, align='C')
            pdf.set_font('Times', 'B', 9)
            pdf.ln(0.5)
            col_width = epw / 5
            data = [['Resolution (m)', 'Density (kg/m3)', 'Vel Col Scaling Factor', 'Elevation Bin Width (m)',
                     'Coordinate system'],
                    [glacierCalc.res, glacierCalc.rho, glacierCalc.vCol, glacierCalc.bin, glacierCalc.crs]]
            pdf.ln(0.5)
            th = pdf.font_size  # Text height is the same as current font size
            for row in data:
                for datum in row:
                    pdf.cell(col_width, th, str(datum), border=1)
                pdf.set_font('Times', '', 9)
                pdf.ln(th)  # 2 * th is for double padding

            pdf.ln(2 * th)
            pdf.set_font('Times', 'B', 12)
            pdf.cell(epw, 5, 'Input Calculation Settings', ln=1, align='C')
            pdf.set_font('Times', 'B', 9)
            pdf.ln(0.5)
            col_width = epw / 3

            data = [['Smoothing Filter', 'Smoothing Factor (raw data)', 'Smoothing Factor (divQ product)'],
                    ['Dynamic Window Gaussian Filter', str(raw_data_smooth_factor) + 'x local thickness',
                     str(divQ_smooth_factor) + 'x local thickness']]
            pdf.ln(0.5)
            th = pdf.font_size  # Text height is the same as current font size
            for row in data:
                for datum in row:
                    pdf.cell(col_width, th, str(datum), border=1)
                pdf.set_font('Times', '', 9)
                pdf.ln(th)  # 2 * th is for double padding

            pdf.set_font('Times', '', 8)
            pdf.cell(epw, 5, '**Smoothing is only done for velocity and ice thickness data. '
                             'If dynamic smoothing uses 0 for both inputs, there is no smoothing**')

            pdf.ln(2 * th)
            pdf.set_font('Times', 'B', 12)
            pdf.cell(epw, 5, 'Calculation Results', ln=1, align='C')
            pdf.ln(0.5)
            col_width = epw / 2
            MB_values_to_report = []
            for i in range(number_of_vel):
                MB_values_to_report.append(['Climatic MB: ' + vel_select[i],
                                            str(round(vel_dict.get('climMassBalanceValueSmooth%i' % i), 2)) + ' (mm w.e.)'])

            data = [['Area', str(round(area / 1000000, 2)) + ' (sq. km)'],
                    ['Total MB', str(round(totalMassBalanceValue, 2)) + ' (mm w.e.)'],
                    ['Total MB, alt-resolved', str(round(altResTotMassBalance, 2)) + ' (mm w.e.)'],
                    [MB_values_to_report[0][0], MB_values_to_report[0][1]]]
            if number_of_vel >= 2:
                data.append([MB_values_to_report[1][0], MB_values_to_report[1][1]])
                if number_of_vel == 3:
                    data.append([MB_values_to_report[2][0], MB_values_to_report[2][1]])

            data.append(['Scatter plot error bar value', error_name])
            vel_dict.get('climMassBalanceValueSmooth0')

            pdf.set_font('Times', '', 9)
            for row in data:
                for datum in row:
                    pdf.cell(col_width, th, str(datum), border=1)
                pdf.ln(th)  # 2 * th is for double padding
            pdf.set_font('Times', '', 8)
            pdf.cell(epw, 5, '**Climatic MB from emergence method does not use smoothed data**')

            if glacierCalc.data is not None:
                Error_to_ref = [['', 'Mean Absolute Error', 'Mean Bias Error', 'MAE (3-Bin Avg)', 'MBE (3-Bin Avg)']]
                Error_to_ref_lobf = [['', 'Mean Absolute Error', 'Mean Bias Error', 'Clim MB Gradient Error']]
                Error_to_ref_point = [['', 'Mean Absolute Error', 'Mean Bias Error', 'Pixel Window Size (for average)']]
                for i in range(number_of_vel):
                    Error_to_ref.append([vel_select[i],
                                                 str(round(vel_dict.get('MeanAbsoluteError%s' % i), 2)) + ' (mm w.e.)',
                                                 str(round(vel_dict.get('MeanBiasError%s' % i), 2)) + ' (mm w.e.)',
                                                 str(round(vel_dict.get('WeightedMeanAbsoluteError%s' % i), 2)) + ' (mm w.e.)',
                                                 str(round(vel_dict.get('WeightedMeanBiasError%s' % i), 2)) + ' (mm w.e.)'])
                    Error_to_ref_lobf.append([vel_select[i],
                                              str(round(vel_dict.get('MeanAbsoluteErrorLine%s' % i), 2)) + ' (mm w.e.)',
                                              str(round(vel_dict.get('MeanBiasErrorLine%s' % i), 2)) + ' (mm w.e.)',
                                              str(round(vel_dict.get('ClimMBGradError%s' % i), 2)) + ' (%)'])
                    Error_to_ref_point.append([vel_select[i],
                                               str(round(vel_dict.get('MeanAbsoluteErrorPixel%s' % i), 2)) + ' (mm w.e.)',
                                               str(round(vel_dict.get('MeanBiasErrorPixel%s' % i), 2)) + ' (mm w.e.)',
                                               str(app4_pixel_window_avg) + ' (pixels); ' +
                                               str(glacierCalc.res * app4_pixel_window_avg) + ' (m)'])

                pdf.ln(2 * th)
                pdf.set_font('Times', 'B', 12)
                pdf.cell(epw, 5, 'Error Metrics: to WGMS Stake Measurements', ln=1, align='C')
                pdf.ln(0.5)
                col_width = epw / 5
                pdf.set_font('Times', 'B', 9)
                for row in Error_to_ref:
                    for datum in row:
                        pdf.cell(col_width, th, str(datum), border=1)
                    pdf.set_font('Times', '', 9)
                    pdf.ln(th)  # 2 * th is for double padding
                pdf.set_font('Times', '', 8)
                pdf.cell(epw, 5, '**3-Bin averages are area-weighted**')

                pdf.ln(2 * th)
                pdf.set_font('Times', 'B', 12)
                pdf.cell(epw, 5, 'Error Metrics: to Best-Fit Line from WGMS Stake Measurements', ln=1, align='C')
                pdf.ln(0.5)
                col_width = epw / 4
                pdf.set_font('Times', 'B', 9)
                for row in Error_to_ref_lobf:
                    for datum in row:
                        pdf.cell(col_width, th, str(datum), border=1)
                    pdf.set_font('Times', '', 9)
                    pdf.ln(th)  # 2 * th is for double padding
                pdf.set_font('Times', '', 8)
                pdf.cell(epw, 5, '**Positive error for the climatic mass balance gradient means the calculated product '
                                 'predicts more mass loss per elevation than the stake measurements**')

                pdf.ln(2 * th)
                pdf.set_font('Times', 'B', 12)
                pdf.cell(epw, 5, 'Error Metrics: to Pixel Location from WGMS Stake Location', ln=1, align='C')
                pdf.ln(0.5)
                col_width = epw / 4
                pdf.set_font('Times', 'B', 9)
                for row in Error_to_ref_point:
                    for datum in row:
                        pdf.cell(col_width, th, str(datum), border=1)
                    pdf.set_font('Times', '', 9)
                    pdf.ln(th)  # 2 * th is for double padding

            pdf.ln(2 * th)
            fig1 = mbPlotsTitle.replace(' ', '_') + '.png'
            pdf.image('Figures/' + fig1, x=None, y=None, w=epw)
            os.remove('Figures/' + fig1)
            pdf.ln(2 * th)
            fig2 = extraPlotTitle2.replace(' ', '_') + '.png'
            pdf.image('Figures/' + fig2, x=None, y=None, w=epw)
            os.remove('Figures/' + fig2)
            pdf.ln(2 * th)
            fig3 = extraPlotTitle3.replace(' ', '_') + '.png'
            pdf.image('Figures/' + fig3, x=None, y=None, w=epw)
            os.remove('Figures/' + fig3)
            for i in range(number_of_vel):
                pdf.ln(2 * th)
                fig4 = vel_dict.get('smoothedVPlotTitle%s' % i).replace(' ', '_') + '.png'
                pdf.image('Figures/' + fig4, x=None, y=None, w=epw)
                os.remove('Figures/' + fig4)
            if glacierCalc.data is not None:
                pdf.ln(2 * th)
                fig_stakes = fig_stakes_title.replace(' ', '_') + '.png'
                pdf.image('Figures/' + fig_stakes, x=None, y=None, w=epw)
                os.remove('Figures/' + fig_stakes)

            pdfName = glacierCalc.name + '_' + str(glacierCalc.res) + 'mRes_' + str(glacierCalc.bin) + 'mBin_' + \
                      time_span + 'dhdt_' + str(raw_data_smooth_factor) + str(divQ_smooth_factor) + 'smoothing_' + \
                      thick_select[0] + '_' + str(vel_select[0:number_of_vel]) + '.pdf'
            pdf.output('PDFs/' + pdfName, 'F')

# to show all the figures from the calculation (not saved .png files) -- make sure livePlots is set to 1 or True
plt.show()

