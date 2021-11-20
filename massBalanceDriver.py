'''
awwells, 12780, final project code
All the necessary modules to import are listed at the top of this document.
The necessary raster files are in this folder
No other files or code is required to run this project
'''
# Import necessary modules
from glacierClass import glacier, fullGlacier
from dataPlots import elevationBinPlot, elevationBinPlot2, elevationBinPlot2Subfigs
from emergenceFunction import emergence_pixels
import numpy as np
import os
import rasterio.plot
from rasterio.plot import show
from smoothingFunctions import sgolay2d, gaussianFilter

# -----------------------------------------------------GENERAL---------------------------------------------------------#
res = (20, 20)              # Resampling resultion (x, y) tuple
avgDensity = 850            # Average glacier mass density (kg/m3)
elevationBinWidth = 50      # Elevation bin width, for elevation binning calculations (m)
vCol = 0.8                  # surface column average velocity scaling

# -----------------------------------------------------GULKANA---------------------------------------------------------#
name = 'Gulkana'            # Glacier name (folder should be the same as glacier name)
type = 'Climatic'           # Mass balance type
demFile = 'RGI_GulkanaRegion_DEM.tif'
hFile = 'GulkanaRegion_thickness.tif'
# hFile = 'Gulkana_Thickness_model1.tif'
dhdtFile = 'dhdt_GulkanaRegion.tif'
vxFile = 'GulkanaRegion_vx.tif'
vyFile = 'GulkanaRegion_vy.tif'
crs = 'EPSG:32606'           # UTM Zone 6 North. Alaska Albers coordinate system: ('EPSG: 3338')

# (glacierName, massBalanceType, demFile, thicknessFile, changeInThicknessFile,
# coordinateSystem, resolution, density, elevationBinWidth, vxFile, vyFile, velocityColumnAvgScaling)
gulkana = fullGlacier(name, type, demFile, hFile, dhdtFile,
                      crs, res, avgDensity, elevationBinWidth, vxFile, vyFile, vCol)
# -----------------------------------------------------WOLVERINE--------------------------------------------------------#
name = 'Wolverine'              # Glacier name (folder should be the same as glacier name)
type = 'Climatic'               # Mass balance type
demFile = 'DEM_Wolverine.tif'
hFile = 'Wolverine_Thickness.tif'
dhdtFile = 'dhdt_Wolverine.tif'
vxFile = 'WolverineGlacier_vx.tif'
vyFile = 'WolverineGlacier_vy.tif'
crs = 'EPSG:32606'           # UTM Zone 6 North.
wolverine = fullGlacier(name, type, demFile, hFile, dhdtFile,
                      crs, res, avgDensity, elevationBinWidth, vxFile, vyFile, vCol)
#-----------------------------------------------------LEMON CREEK------------------------------------------------------#
name = 'LemonCreek'            # Glacier name (folder should be the same as glacier name)
type = 'Climatic'               # Mass balance type
demFile = 'DEM_LemonCreek.tif'
hFile = 'LemonCreek_Thickness.tif'
dhdtFile = 'dhdt_LemonCreek.tif'
vxFile = 'LemonCreek_vx.tif'
vyFile = 'LemonCreek_vy.tif'
crs = 'EPSG:32608'           # UTM Zone 8 North
lemonCreek = fullGlacier(name, type, demFile, hFile, dhdtFile,
                      crs, res, avgDensity, elevationBinWidth, vxFile, vyFile, vCol)
# ------------------------------------------------------COLUMBIA-------------------------------------------------------#
name = 'Columbia'            # Glacier name (folder should be the same as glacier name)
type = 'Climatic'           # Mass balance type
demFile = 'DEM_Columbia.tif'
hFile = 'Columbia_Thickness.tif'
dhdtFile = 'dhdt_Columbia.tif'
vxFile = 'Columbia_vx.tif'
vyFile = 'Columbia_vy.tif'
crs = 'EPSG:32606'           # UTM Zone 6 North
columbia = fullGlacier(name, type, demFile, hFile, dhdtFile,
                      crs, res, avgDensity, elevationBinWidth, vxFile, vyFile, vCol)
# -------------------------------------------------------RAINBOW-------------------------------------------------------#
name = 'Rainbow'            # Glacier name (folder should be the same as glacier name)
type = 'Climatic'           # Mass balance type
demFile = 'DEM_Rainbow.tif'
hFile = 'Rainbow_Thickness.tif'
dhdtFile = 'dhdt_Rainbow.tif'
vxFile = 'Rainbow_vx.tif'
vyFile = 'Rainbow_vy.tif'
crs = 'EPSG:32610'           # UTM Zone 10 North
rainbow = fullGlacier(name, type, demFile, hFile, dhdtFile,
                      crs, res, avgDensity, elevationBinWidth, vxFile, vyFile, vCol)
# ----------------------------------------------------SOUTH CASCADE----------------------------------------------------#
name = 'SouthCascade'        # Glacier name (folder should be the same as glacier name)
type = 'Climatic'           # Mass balance type
demFile = 'DEM_SouthCascade.tif'
hFile = 'SouthCascade_Thickness.tif'
dhdtFile = 'dhdt_SouthCascade.tif'
vxFile = 'SouthCascade_vx.tif'
vyFile = 'SouthCascade_vy.tif'
crs = 'EPSG:32610'           # UTM Zone 10 North.
southCascade = fullGlacier(name, type, demFile, hFile, dhdtFile,
                      crs, res, avgDensity, elevationBinWidth, vxFile, vyFile, vCol)
# -----------------------------------------------------RIKHA SAMBA-----------------------------------------------------#
name = 'RikhaSamba'              # Glacier name (folder should be the same as glacier name)
type = 'Climatic'                # Mass balance type
demFile = 'DEM_RikhaSamba.tif'
hFile = 'RikhaSamba_Thickness.tif'
dhdtFile = 'dhdt_RikhaSamba.tif'
# dhdtFile = 'dhdt_ASTER_n28_e083.tif'
vxFile = 'RikhaSamba_vx.tif'
vyFile = 'RikhaSamba_vy.tif'
crs = 'EPSG:32644'           # UTM Zone 44 North.
rikhaSamba = fullGlacier(name, type, demFile, hFile, dhdtFile,
                      crs, res, avgDensity, elevationBinWidth, vxFile, vyFile, vCol)
# ---------------------------------------------------------------------------------------------------------------------#
glacierCalc = gulkana                              # chose glacier to perform calculations

path = os.getcwd() + '/' + glacierCalc.name         # navigate to working directory
os.chdir(path)

# STEP 1: CONVERT COORDINATE SYSTEMS AND RESAMPLE
glacierCalc.tifReprojectionResample()

# STEP 2: CLIP RASTERS TO SAME EXTENT
glacierCalc.tifClip()

# QUICK DATA FILTER: FILL HOLES IN ANY FILES (DEM, H, or DHDT)
glacierCalc.fillHole()
glacierCalc.glacierOutline()

# CHECK TO SEE IF THINGS WORKED
check = 0
if check == True:
    checkFile = glacierCalc.dem
    src = rasterio.open(checkFile)
    dst = rasterio.open(checkFile[:-4]+'_reproj.tif')
    dst_clip = rasterio.open(checkFile[:-4] + '_clip.tif')

    print('Source coordinate system of', checkFile, 'is: ', src.crs)
    print('Destination coordinate system of', checkFile, 'is: ', dst.crs)

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

maxvalh = np.max(hArray)
maxindexh = np.where(hArray == maxvalh)
indexrowh = maxindexh[0][0]
indexcolh = maxindexh[1][0]

vxmaxh = vxArray[indexrowh, indexcolh]
vymaxh = vyArray[indexrowh, indexcolh]
if abs(vxmaxh) > abs(vymaxh):
    vprimarydir = [vxArray, 'vx']
    vsecondarydir = [vyArray, 'vy']
else:
    vprimarydir = [vyArray, 'vy']
    vsecondarydir = [vxArray, 'vx']

drc = [0, 0, 0]
dcc = [0, 0, 0]
for i in [0, 1, 2]:
    drc[i] = demArray[indexrowh + ((i-1)*1), indexcolh]
    dcc[i] = demArray[indexrowh, indexcolh + ((i-1)*1)]

if abs(drc[0]-drc[2]) > abs(dcc[0]-dcc[2]):
    # if glacier is primarily north-south
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
    # if glacier is primarily east-west
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

# Overwrite velocity files with new velocity files: new directions
kwargs = rasterio.open(glacierCalc.vx[:-4] + '_clip.tif').meta.copy()
dst_file_vx = glacierCalc.vx[:-4] + '_clip.tif'
with rasterio.open(dst_file_vx, 'w', **kwargs) as dst:
    dst.write(x_vel_array_dir, 1)

kwargs = rasterio.open(glacierCalc.vy[:-4] + '_clip.tif').meta.copy()
dst_file_vy = glacierCalc.vy[:-4] + '_clip.tif'
with rasterio.open(dst_file_vy, 'w', **kwargs) as dst:
    dst.write(y_vel_array_dir, 1)

# x_vel_array_dir=vxArray
# y_vel_array_dir=vyArray
# ------------------------------------------PART 3 DISPLAY OPTIONS-----------------------------------------#
showArea = 0                                    # to print glacier area
showTotalMassBalanceValue = 1                   # to print the total mass balance value
showClimMassBalanceValue = 1                    # to print the climatic mass balance value
# ---------------------------------------------------------------------------------------------------------#
# STEP 3: MANIPULATE AND COMBINE DATA TO FULFILL CONTINUITY EQUATION
glacierCalc.totalMassBalance()
glacierCalc.climMassBalance()

area = glacierCalc.glacierArea()
if showArea == True:
    print('Area (sq. km):', area/1000000)

totalMassBalanceValue = glacierCalc.totalMassBalanceValue()
climMassBalanceValue = glacierCalc.climMassBalanceValue()

if showTotalMassBalanceValue == True:
    exp = 'The annual TOTAL mass balance for ' + glacierCalc.name + ' Glacier is: '
    unit = '(mm w.e.)'
    print(exp, totalMassBalanceValue, unit) # units of kg/m2-yr is also just mm w.e.

if showClimMassBalanceValue == True:
    exp = 'The annual CLIMATIC mass balance for ' + glacierCalc.name + ' Glacier is: '
    unit = '(mm w.e.)'
    print(exp, climMassBalanceValue, unit) # units of kg/m2-yr is also just mm w.e.

# ------------------------------------------PART 4 DISPLAY OPTIONS-----------------------------------------#
use_smooth_vel = 0                          # 1 to use smooth velocity, 0 to use raw velocity

showSmoothFilter = 0                        # to show the difference in raw data and smoothed data
showSmoothClimMassBalance = 0               # to show the difference in climatic mass balance using vel vs smoothed vel
# ---------------------------------------------------------------------------------------------------------#
# STEP 4: SMOOTH VELOCITY DATA WITH LOW PASS MOVING WINDOW FILTER, CLIMATIC MASS BALANCE FOR SMOOTH VELOCITY DATA
glacierCalc.velocitySmoothing(windowSize=5, polyOrder=1, std=12, smoothFilter='gauss') # smoothFiler='golay', otherwise 'gauss'
glacierCalc.climMassBalance(smoothVel=use_smooth_vel)

# show(rasterio.open(glacierCalc.vx[:-4]+'_smooth.tif'))  #smooth velocity_x file
if showSmoothFilter == True:
    show(rasterio.open(glacierCalc.h).read(1), title='original thickness file')
    show(sgolay2d(rasterio.open(glacierCalc.h).read(1), 3, 1), title='savitzky-golay filter: thickness')
    show(gaussianFilter(rasterio.open(glacierCalc.h).read(1), 2), title='gaussian filter: thickness')

if showSmoothClimMassBalance == True:
    show(rasterio.open(glacierCalc.name+'TotalMassBalance.tif').read(1), title='Total Mass Balance Result')
    show(rasterio.open(glacierCalc.name+'ClimaticMassBalance.tif').read(1), title='Climatic Mass Balance Result')
    show(rasterio.open(glacierCalc.name+'ClimaticMassBalanceSmooth.tif').read(1),
         title='Smoothed-velocity Climatic Mass Balance Result')

# ------------------------------------PART 5 INPUTS AND DISPLAY OPTIONS------------------------------------#
showAltResMassLoss = 1                  # to print the altitudinally-resolved mass loss
showMassLossElevation = 0               # to plot of the mean mass balance contributions per elevation bin (np.gradient)
# ---------------------------------------------------------------------------------------------------------#
# STEP 5: ALTITUDINALLY RESOLVE MASS BALANCE
# run function to estimate the altitudinally resolved mass balance. this function actually returns a few things:
# calculated statistic at each elevation bin, elevation bin boundaries, and elevation bin counts

if use_smooth_vel == True:
    massBalanceClimFile = glacierCalc.name + 'ClimaticMassBalanceSmooth.tif'
    balanceType = '(smooth velocity)'
else:
    massBalanceClimFile = glacierCalc.name + 'ClimaticMassBalance.tif'
    balanceType = ''
# change mass balance file in altitudeAggregation to change this back to original un-smoothed clim mass balance

massBalanceTotalFile = glacierCalc.name + 'TotalMassBalance.tif'


binTotMassMean, binElevations, binCounts, binNumber = glacierCalc.altitudeAggregation(massBalanceTotalFile, stat='mean')
binClimMassMean, binElevations, binCounts, binNumber = glacierCalc.altitudeAggregation(massBalanceClimFile, stat='mean')

demTotBinTotalMass = binCounts * binTotMassMean * glacierCalc.res[0] * glacierCalc.res[1]  # this is in kg/yr
demClimBinTotalMass = binCounts * binClimMassMean * glacierCalc.res[0] * glacierCalc.res[1]
altResTotMass = np.sum(demTotBinTotalMass)
altResClimMass = np.sum(demClimBinTotalMass)

altResTotMassBalance = altResTotMass/area                       # divide by total glacier area for alt-resolved mass balance
altResClimMassBalance = altResClimMass/area

if showAltResMassLoss == True:
    # to print the values for altitudinally resolved mass balance (kg/m2-yr or mm w.e.)
    print('The altitudinally resolved TOTAL mass balance is:', altResTotMassBalance, '(mm w.e.)')
    print('The altitudinally resolved CLIMATIC', balanceType, 'mass balance is:', altResClimMassBalance, '(mm w.e.)')
    # print('Elevation bin size:', elevationBinWidth, 'm')

if showMassLossElevation == True:
    # to show a scatter plot of elevation vs mean mass loss at each elevation bin
    binElevationsList = binElevations.tolist()
    for i in range(len(binElevationsList)):
        binElevationsList[i] = binElevationsList[i] + (elevationBinWidth*0.5)
    binTotMassMeanList = binTotMassMean.tolist()
    binClimMassMeanList = binClimMassMean.tolist()
    binAreasList = (binCounts*glacierCalc.res[0]*glacierCalc.res[1]).tolist()

    x_values1 = binTotMassMeanList
    x_values1_lab = 'Total Mass Balance'
    x_values1_2 = binClimMassMeanList
    x_values1_2_lab = ' '.join(['Climatic Mass Balance', balanceType])
    y_values1 = binElevationsList[0:len(binElevationsList)-1]
    x_values2 = binElevationsList[0:len(binElevationsList)-1]
    y_values2 = np.divide(binAreasList, 1000000)    # sq.km area
    xLabel1 = 'Bin Mean Mass Balance (mm w.e.)'
    yLabel1 = 'Bin Mean Elevation (m)'
    xLabel2 = 'Bin Area ($km^2$)'
    title = 'Glacier mean mass balance per elevation bin: np.gradient method'
    min = 'default'
    buff = 500
    alpha = 0.3

    elevationBinPlot2(x_values1, x_values1_lab, x_values1_2, x_values1_2_lab, y_values1, x_values2, y_values2, xLabel1,
                      yLabel1, xLabel2, title, elevationBinWidth, min, buff, alpha)

# ------------------------------------------PART 6 DISPLAY OPTIONS-----------------------------------------#
# show(rasterio.open(glacierCalc.vx[:-4]+'_clip.tif').read(1)*outlineArray)
# show(rasterio.open(glacierCalc.vy[:-4]+'_clip.tif').read(1)*outlineArray)
plotMassBalance = 1             # to plot the flux gate climatic mass balance and total mass balance

# STEP 6: EMERGENCE METHOD
'''
#we use total mass balance, altitudinally resolve it, and then account for binDivQ after
#run function to estimate the altitudinally resolved mass balance. this function actually returns a few things:
#calculated statistic at each elevation bin, elevation bin boundaries, and elevation bin counts
totalMassBalanceFile = glacierCalc.name + 'TotalMassBalance.tif'
binMassMean, binElevations, binCounts = glacierCalc.altitudeAggregation(totalMassBalanceFile, stat='mean')
'''

vel_x = x_vel_array_dir
vel_y = y_vel_array_dir
# show(vel_x*outlineArray, title='vel x')
# show(vel_y*outlineArray, title='vel y')
vel_total = np.power((np.square(vel_x) + np.square(vel_y)), 0.5)
icethickness_fn = glacierCalc.h[:-4] + '_clip.tif'
icethickness = rasterio.open(icethickness_fn).read(1)

max_velocity = 200
vmin = 0

emergence_velocity = emergence_pixels(vel_x, vel_y, vel_total, vmin, max_velocity, vCol, icethickness, glacierCalc.res[0])
binEmergenceMean, binElevations, binCounts, binNumber = glacierCalc.altitudeAggregation(emergence_velocity, 'mean')

dhdtFile = glacierCalc.dhdt[:-4] + '_clip.tif'
binMeandhdt, binElevations, binCounts, binNumber = glacierCalc.altitudeAggregation(dhdtFile, stat='mean')

binClimChange = (binMeandhdt - binEmergenceMean)*avgDensity
binTotalChange = binMeandhdt*avgDensity

# apply bin SMB to each value in that bin: for plotting ONLY
binnedTotalMassBalance = np.zeros(binNumber.shape)
binnedClimaticMassBalance = np.zeros(binNumber.shape)
binnedEmergence = np.zeros(binNumber.shape)
for i in range(len(binNumber)):
    binnedTotalMassBalance[i] = binTotalChange[binNumber[i]-1]
    binnedClimaticMassBalance[i] = binClimChange[binNumber[i]-1]
    binnedEmergence[i] = binEmergenceMean[binNumber[i]-1]

outlineArrayShape = outlineArray.shape
binnedTotalMassBalance = np.array(binnedTotalMassBalance).reshape(outlineArrayShape)
binnedClimaticMassBalance = np.array(binnedClimaticMassBalance).reshape(outlineArrayShape)
binnedEmergence = np.array(binnedEmergence).reshape(outlineArrayShape)
binnedElevationNumber = np.array(binNumber).reshape(outlineArrayShape)

# for plotting: each subplot has: [data array, title, colorbar label]
emergence_plot = [binnedEmergence * outlineArray, 'Emergence Map', 'Mean Elevation Bin Emergence (m/a)']
h_plot = [icethickness, 'Thickness Map', 'Glacier Thickness (m)']
v_plot = [vel_total, 'Magnitude of Velocity Map', 'Speed (m/a)']
z_plot = [demArray, 'Elevation Map', 'Elevation (m)']
b_tot_plot = [binnedTotalMassBalance * outlineArray, 'Total Mass Balance Map', 'Mean Elevation Bin SMB (mm w.e.)']
b_clim_plot = [binnedClimaticMassBalance * outlineArray, 'Climatic Mass Balance Map', 'Mean Elevation Bin SMB (mm w.e.)']

# -----------chose subplots to show:------------
subplot2 = b_tot_plot
subplot3 = z_plot
subplot4 = emergence_plot

# to show a scatter plot of elevation vs mean mass loss at each elevation bin, as well as 3 other figures
if plotMassBalance == True:
    binElevationsList = binElevations.tolist()
    for i in range(len(binElevationsList)):
        binElevationsList[i] = binElevationsList[i] + (elevationBinWidth * 0.5)
    binClimChangeList = binClimChange.tolist()
    binTotalChangeList = binTotalChange.tolist()
    binAreasList = (binCounts * glacierCalc.res[0] * glacierCalc.res[1]).tolist()

    x_subplot_nw = binTotalChangeList
    x_subplot_nw_lab = 'Total Mass Balance'
    x_subplot_nw_2 = binClimChangeList
    x_subplot_nw_2_lab = 'Climatic Mass Balance'
    y_values1 = binElevationsList[0:len(binElevationsList) - 1]
    x_values2 = binElevationsList[0:len(binElevationsList) - 1]
    y_values2 = np.divide(binAreasList, 1000000)    # sq.km area
    xLabel1 = 'Bin Mean Mass Balance (mm w.e.)'
    yLabel1 = 'Bin Mean Elevation (m)'
    xLabel2 = 'Bin Area ($km^2$)'
    title1 = 'Binned SMB Plot'
    title = glacierCalc.name + ' Glacier Plots: (from Emergence Method)'
    min = 'default'
    buff = 500
    alpha = 0.3

    subplot_ne = subplot2[0]
    subplot_ne_title = subplot2[1]
    subplot_ne_label = subplot2[2]
    subplot_sw = subplot3[0]
    subplot_sw_title = subplot3[1]
    subplot_sw_label = subplot3[2]
    subplot_sw_cbar_ticks = None
    subplot_se = subplot4[0]
    subplot_se_title = subplot4[1]
    subplot_se_label = subplot4[2]

    elevationBinPlot2Subfigs(x_subplot_nw, x_subplot_nw_lab, x_subplot_nw_2, x_subplot_nw_2_lab, y_values1, x_values2,
                             y_values2, xLabel1, yLabel1, xLabel2, title1, elevationBinWidth, min, buff, alpha, title,
                             subplot_ne, subplot_ne_title, subplot_ne_label,
                             subplot_sw, subplot_sw_title, subplot_sw_label,
                             subplot_se, subplot_se_title, subplot_se_label, subplot_sw_cbar_ticks)

