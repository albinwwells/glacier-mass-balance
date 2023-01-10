'''
Code for comparing velocity products from each glacier
Automatically obtains glacier data using RGI shapefiles, not specification/prep needed
Outputs pdf with plot comparing result from velocity products
'''

# Import necessary modules
from glacierDatabase import glacierInfo, glacierVelUncertainty
from Modules.glacierGUI import pickGlacierData, pickGlacierTime, pickIcepackVelocity, pickCorrectionsScalings

from Modules.geotiffPrep import rasterLike, getCRS, shpReprojection, tifReprojectionResample, tifClip, shpClip, fillHole, missingPixels
from Modules.baseFunctions import glacierOutline, altitudeAggregation, binPercentile, latlonTiffIndex

from Modules.glacierFunctions import glacierArea, totalMassBalance, totalMassBalanceValue, divQ
from Modules.glacierFunctions import glacierAttributes, glacierSlope, demHillshade
from Modules.glacierFunctions import velocityAspect, velocityAspectAngle, velAspectCorrection
from Modules.glacierFunctions import velAspectDirection, velAspectSlopeThreshold, velAspectSlopeAverage
from Modules.glacierFunctions import slope_vel_plot, h_vel_plot, stress_vel_plot
from Modules.smoothingFunctions import sgolay2d, gaussianFilter, dynamicSmoothing

from Modules.dataPlots import show_fig, plotMany, scatterplot
from Modules.dataPlots import plotData, plotData3, plotData6, plotDataPoints, velPlot, plotClassify, plotContinuous
from Modules.dataPlots import elevationBinPlot, elevationBinPlot2, elevationBinPlot3Subfigs, elevationBinPlot3data3Subfigs

import glob, os, sys, shutil, warnings, itertools
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.warp import Resampling
import rasterio.plot
from rasterio.plot import show
from numpy import savetxt
from sklearn.metrics import mean_absolute_error
from fpdf import FPDF

# warnings.filterwarnings(action='ignore', message='Mean of empty slice')
warnings.filterwarnings(action='ignore')  # ignoring all runtime warnings

# gulkana is 570, wolverine is 9162, lemon creek 1104
vel_dict = {}
# -----------------------------------------------------GENERAL---------------------------------------------------------#
main = ''  # This is either empty of 'main'. If it is 'main', mass balance aggregation will only consider the main trunk
res = 20  # Resampling resolution (x, y) tuple
avgDensity = 850  # Average glacier mass density (kg/m3)
elevationBinWidth = 50  # Elevation bin width, for elevation binning calculations (m)
vCol = 0.8  # surface column average velocity scaling

glac, dem_select, thick_select, vel_select, shp_select = pickGlacierData()
glacier_number_from, glacier_number_to, time_select = pickGlacierTime(glac)
icepack_nums_1 = pickIcepackVelocity(vel_select)
icepack_nums_2 = icepack_nums_1.copy()
filtered_vel, vel_scaling_factor = pickCorrectionsScalings()

vel_select = [x + str(icepack_nums_1.pop(0)) if 'Icepack' in x else x for x in vel_select]  # add selected icepack output number to vel_select
nums = [icepack_nums_2.pop(0) if 'Icepack' in x else None for x in vel_select]

# --append values to 'glac' based on input
if glacier_number_to == '':
    glacier_number_to = int(glacier_number_from) + 1
if int(glacier_number_from) >= 27112 or int(glacier_number_from) < 1:
    sys.exit('Value out of range (must be between 1 and 27,112).')
if int(glacier_number_to) > 27112 or int(glacier_number_to) <= 1:
    sys.exit('Value out of range (must be between 1 and 27,112).')
if int(glacier_number_from) >= int(glacier_number_to):
    sys.exit('First entry must be smaller than second entry')
if len(vel_select) >= 4:
    sys.exit('Too many velocity products selected. Maximum of 3 products allowed')

other_glac_list = list(map(str, range(int(glacier_number_from), int(glacier_number_to))))
glac = glac[1:]
glac.extend(other_glac_list)

# ---------------------------------------------------------------------------------------------------------------------#
time_span = time_select[0]
number_of_glac = len(glac)
number_of_thick = len(thick_select)
if number_of_thick > 1:
    sys.exit('Too many ice thickness selections. May only select one ice thickness file when comparing velocity')
number_of_vel = len(vel_select)
number_of_files = 3+(2*number_of_vel)

g = glac[0]  # ONLY USING FIRST GLACIER FOR NOW
print('\n--------------------------------', g, 'Glacier Calculations --------------------------------')

# create an object to define each glacier that specifies input parameters and files
glacierVels = [glacierInfo(g, res, avgDensity, elevationBinWidth, vCol, dem_select, thick_select[0], v, shp_select[0])
               for v in vel_select]
glacierSame = glacierInfo(g, res, avgDensity, elevationBinWidth, vCol, dem_select, thick_select[0], vel_select[0], shp_select[0])
glacierSame.name = g.zfill(5)
glacierSame.data, glacierSame.dataVariability, dataCoords = glacierSame.getReferenceData(time_span)


# navigate to appropriate directory (only once per glacier)
if os.path.exists(os.getcwd() + '/Other') == True:
    os.chdir(os.getcwd() + '/Other')  # navigate to working directory
elif os.path.exists(os.path.dirname(os.getcwd()) + '/Other') == True:
    os.chdir(os.path.dirname(os.getcwd()))
    os.chdir(os.getcwd() + '/Other')
else:
    sys.exit('Error navigating to the proper folder. Check that the folder for this glacier exists.')

# create necessary folders IF they don't already exist: PDF and Figure
if os.path.exists(os.getcwd() + '/Figures') == False:
    os.makedirs(os.getcwd() + '/Figures')
if os.path.exists(os.getcwd() + '/PDFs') == False:
    os.makedirs(os.getcwd() + '/PDFs')
if os.path.exists(os.getcwd() + '/PDFs/' + glacierSame.name) == False:  # create folder for specific glacier
    os.makedirs(os.getcwd() + '/PDFs/' + glacierSame.name)


if shp_select[0] == 'RGI':
    shp_dataframe = gpd.GeoDataFrame.from_file(glacierSame.shape)
    shp_attributes = shp_dataframe.loc[shp_dataframe['RGIId'] == 'RGI60-01.' + glacierSame.name]
    shp_attributes.to_file('RGI60-01.' + glacierSame.name + '.shp')
    glacierSame.shape = 'RGI60-01.' + glacierSame.name + '.shp'

glacierSame.crs = getCRS(glacierSame.shape)                 # OBTAIN CRS
glacierSame.h = glacierSame.getThickness(thick_select[0])   # OBTAIN THICKNESS FILE
glacierSame.dem = glacierSame.getDEM(dem_select[0])         # OBTAIN DEM FILE
glacierSame.dhdt = glacierSame.getDhdtFile(time_span)       # OBTAIN CHANGE IN THICKNESS FILE
for j in range(number_of_vel):                              # OBTAIN VELOCITY FILE
    glacierVels[j].vx, glacierVels[j].vy, glacierVels[j].vCor = glacierSame.getVelocity(vel_select[j], n=nums[j])


# GEOTIFF PROCESSING STEPS:
# Create lists for all of the processing
input_tif = [glacierSame.dem, glacierSame.h, glacierSame.dhdt]
for j in range(number_of_vel):
    input_tif.extend([glacierVels[j].vx, glacierVels[j].vy])

output_tif = [glacierSame.name+'_dem.tif', glacierSame.name+'_h.tif', glacierSame.name+'_dhdt.tif',
                glacierSame.name+'_vx1.tif', glacierSame.name+'_vy1.tif',
                glacierSame.name+'_vx2.tif', glacierSame.name+'_vy2.tif',
                glacierSame.name+'_vx3.tif', glacierSame.name+'_vy3.tif'][:number_of_files]

shape_list = [glacierSame.shape for x in input_tif]
dem_list = [glacierSame.name+'_dem.tif' for x in input_tif]
outline_list = [glacierSame.name + 'Outline.tif' for x in input_tif]
pad_size = list(np.ones(number_of_files)*50)
crs_list = [glacierSame.crs for x in input_tif]
res_list = [glacierSame.res for x in input_tif]
vCol_list = [glacierSame.vCol for x in input_tif]
interp_list = [Resampling.cubic_spline for x in input_tif]

# STEP 0: PRE-CLIP EXTENT SO WE DON'T RESAMPLE HUGE TIFS OF THE ENTIRE REGION
# pre filtering to clip extent of regional files (exclude DEM!)
list(map(shpClip, input_tif[1:], shape_list[1:], output_tif[1:], pad_size[1:]))
shutil.copy2(glacierSame.dem, glacierSame.name+'_dem.tif')   # save DEM new name
# rasterLike(rasterio.open(glacierSame.dem).read(1), glacierSame.name+'_dem.tif', glacierSame.dem)

# rename: these are the files we will actually work with an edit, not the originals
glacierSame.dem = glacierSame.name + '_dem.tif'
glacierSame.h = glacierSame.name + '_h.tif'
glacierSame.dhdt = glacierSame.name + '_dhdt.tif'
for j in range(number_of_vel):
    glacierVels[j].vx = output_tif[3+(2*j)]
    glacierVels[j].vy = output_tif[4+(2*j)]

# STEP 1: CONVERT COORDINATE SYSTEMS AND RESAMPLE
shpReprojection(glacierSame.shape, glacierSame.crs, glacierSame.shape)  # reproject shapefile
list(map(tifReprojectionResample, output_tif, output_tif, crs_list, res_list, interp_list))
# v = [[1100, 1800, 2500], [0, 20, 250], [-5, 0, 5], [-20, 0, 20], [-20, 0, 20], [-20, 0, 20], [-20, 0, 20], [-20, 0, 20], [-20, 0, 20]]
array1 = [rasterio.open(output_tif[i]).read(1) for i in range(len(output_tif))]
# plotMany('Metric (m or m/yr)', 'BrBG', 'Step 1: Reproject Resample', list(zip(array1, output_tif, v)))
# rasterLike(array1[0], glacierSame.name + 'dem_step1.tif', glacierSame.dem)

# STEP 2: CLIP RASTERS BY SHAPEFILE
shpClip(output_tif[0], shape_list[0], output_tif[0], fill=False)  # clip rasters by glacier RGI shape, DEM has fill=0
list(map(shpClip, output_tif[1:], shape_list[1:], output_tif[1:]))

# STEP 3: REPROJECT TO THE SAME EXTENT
list(map(tifReprojectionResample, output_tif, output_tif, crs_list, res_list, interp_list, dem_list))
tifReprojectionResample(output_tif[0], output_tif[0], crs_list[0], res_list[0], interp_list[0], dem_list[0])
# list(map(fillHole, output_tif, output_tif))  # this only works for h and dem

# Get glacier outline and check for missing data / holes
rasterLike(np.ones_like(rasterio.open(glacierSame.h).read(1), dtype=np.int8), 'ones_raster.tif', glacierSame.h)
outlineArray = glacierOutline('ones_raster.tif', glacierSame.shape, glacierSame.name + 'Outline.tif')
# outlineArray = glacierOutline(glacierSame.h)
# rasterLike(outlineArray, glacierSame.name+'Outline.tif', glacierSame.h)
# list(map(missingPixels, output_tif, outline_list, res_list))  # make sure data does not have large gaps or holes

# save velocity files
# arrays = [rasterio.open(output_tif[i]).read(1) for i in range(len(output_tif))]
# rasterLike(arrays[3], 'Saved Velocity Outputs/' + glacierVels[0].vx, glacierVels[0].vx)
# rasterLike(arrays[4], 'Saved Velocity Outputs/' + glacierVels[0].vy, glacierVels[0].vy)
# rasterLike(arrays[5], 'Saved Velocity Outputs/' + glacierVels[1].vx, glacierVels[1].vx)
# rasterLike(arrays[6], 'Saved Velocity Outputs/' + glacierVels[1].vy, glacierVels[1].vy)
# rasterLike(arrays[7], 'Saved Velocity Outputs/' + glacierVels[2].vx, glacierVels[2].vx)
# rasterLike(arrays[8], 'Saved Velocity Outputs/' + glacierVels[2].vy, glacierVels[2].vy)

# ----------------------------------------------FIX VELOCITY DIRECTIONS----------------------------------------------#
# get list of arrays: DEM, h, dhdt, vx, vy, ... vx2, vy2, vx3, vy3
output_array = [rasterio.open(x).read(1)*outlineArray for x in output_tif]
demArray = output_array[0]
hArray = output_array[1]
dhdtArray = output_array[2]
vel_x_list = []
vel_y_list = []
vel_total_list = []

for i in range(len(output_array)):
    if i == 3 or i == 5 or i == 7:
        j = int((i - 3)/2)
        # correct for x and y velocity, if needed (usually needed ITS_LIVE in AK, but not for other velocity products)
        vel_x_list.append(output_array[i] * glacierVels[j].vCor[0])
        vel_y_list.append(output_array[i+1] * glacierVels[j].vCor[1])
        vel_x_list[-1][abs(vel_x_list[-1]) > 1e30] = 0   # I have no clue why, but sometimes this creates very large values that need to be masked out
        vel_y_list[-1][abs(vel_y_list[-1]) > 1e30] = 0
        rasterLike(vel_x_list[-1], glacierVels[j].vx, glacierVels[j].vx)
        rasterLike(vel_y_list[-1], glacierVels[j].vy, glacierVels[j].vy)
        vel_total_list.append(np.power((np.square(vel_x_list[-1]) + np.square(vel_y_list[-1])), 0.5))

# -------------------------------------STEP 3: CALCULATE AREA AND TOTAL MB-------------------------------#
area = glacierArea(outlineArray, glacierSame.res)  # calculate and print glacier area
massBalanceTotal = totalMassBalance(dhdtArray, outlineArray, glacierSame.rho)
rasterLike(massBalanceTotal, glacierSame.name + 'TotalMassBalance.tif', glacierSame.dhdt)
totalMassBalanceVal = totalMassBalanceValue(massBalanceTotal, area, glacierSame.res)
print(glacierSame.name, 'Area (sq. km):', area / 1000000)
print('The TOTAL mass balance is: ', totalMassBalanceVal, '(mm w.e.)')

# filter out glaciers with small areas (1 sq. km), and remove any created/saved files in the process
skip = False
if area < 1000000:
    skip = True
    print('AREA IS TOO SMALL (' + str(area/1000000) + ' sq. km)')
    os.remove(glacierSame.h)
    os.remove(glacierSame.dhdt)
    for j in range(number_of_vel):
        os.remove(glacierVels[j].vx)
        os.remove(glacierVels[j].vy)
    os.remove(glacierSame.shape)
    os.remove(glacierSame.shape[:-4] + '.shx')
    os.remove(glacierSame.shape[:-4] + '.dbf')
    os.remove(glacierSame.shape[:-4] + '.prj')
    os.remove(glacierSame.shape[:-4] + '.cpg')
    os.remove(glacierSame.name + 'Outline.tif')
    os.remove('ones_raster.tif')
    # break  # we only need to break if we're in a loop; e.g. to move on to next glacier
    sys.exit('Glacier area too small! It is less than 1 sq.km')  # remove this if we have 'break' and multiple glaciers

# -----------------------------------------GLACIER SLOPE AND ASPECT----------------------------------------#
# obtain slope and aspect arrays, then smooth them with gaussian filters
slope_array_dem = glacierAttributes(glacierSame.dem, 'slope_degrees')
# slope_array_dem = gaussianFilter(slope_array_dem_i, st_dev=1, truncate=4) * outlineArray  # smooth array
slope_map_title = glacierSame.name + ' Slope Map'
plotData(slope_array_dem, 'Slope (degrees)', 'copper_r', slope_map_title, colorbarMax=50)

aspect_array_dem = glacierAttributes(glacierSame.dem, 'aspect')
# aspect_array_dem = gaussianFilter(aspect_array_dem_i, st_dev=1, truncate=4) * outlineArray  # smooth array
aspect_map_title = glacierSame.name + ' Aspect Map'
plotData(aspect_array_dem, 'Aspect (degrees from N)', 'twilight', aspect_map_title, cluster=12)

hillshade_array_dem = demHillshade(demArray, glacierSame.res)
hillshade_map_title = glacierSame.name + ' Hillshade Map'
plotData(hillshade_array_dem, 'Hillshade', 'Greys', hillshade_map_title)

# velocity and thickness gradients (from raw data, not smoothed)
h_array_gradient = glacierSlope(hArray, glacierSame.res) * outlineArray
h_gradient_title = glacierSame.name + ' Thickness Gradient Map ' + thick_select[0]
plotData(h_array_gradient, 'Slope (rise/run)', 'copper_r', h_gradient_title, colorbarMax=1)

vel_array_gradient_list = list(map(glacierSlope, vel_total_list, res_list[:number_of_vel]))
vel_gradient_title_list = [glacierSame.name + ' Vel Gradient Map ' + vel_select[j] for j in range(number_of_vel)]
cbarTitle_list = ['Slope (rise/run)' for x in range(number_of_vel)]
color_list = ['copper_r' for x in range(number_of_vel)]
none_list = [None for x in range(number_of_vel)]
colorbarMax_list = [0.3 for x in range(number_of_vel)]
list(map(plotData, vel_array_gradient_list, cbarTitle_list, color_list, vel_gradient_title_list, none_list, none_list, colorbarMax_list))


# ---------------------------------------GLACIER VELOCITY UNCERTAINTY--------------------------------------#
vx_err_list = [glacierVelUncertainty(vel_select[j], glacierVels[j])[0] for j in range(number_of_vel)]
vy_err_list = [glacierVelUncertainty(vel_select[j], glacierVels[j])[1] for j in range(number_of_vel)]
vx_err_array_list = np.zeros_like(vel_total_list)
vy_err_array_list = np.zeros_like(vel_total_list)
v_err_array_list = np.zeros_like(vel_total_list)
v_err_rel_array_list = np.zeros_like(vel_total_list)

print('WE NEED TO FIX ERROR (LINE 282) -- REMOVE TIFCLIP FUNCTION')
for j in range(number_of_vel):
    if shp_select[0] == 'USGS':
        vx_err_list[j] = None
    if vx_err_list[j] is not None:
        # clip large files to near the extent of our glacier
        vx_err_name = 'vx' + str(j+1) + '_err.tif'
        vy_err_name = 'vy' + str(j+1) + '_err.tif'
        # initial clip, reproject, resample, and clip by glacier outline shape (we only care about the first return)
        shpClip(vx_err_list[j], glacierSame.shape, vx_err_name, pad_size=50)
        shpClip(vy_err_list[j], glacierSame.shape, vy_err_name, pad_size=50)
        tifReprojectionResample(vx_err_name, vx_err_name, glacierSame.crs, glacierSame.res, interp_list[0])
        tifReprojectionResample(vy_err_name, vy_err_name, glacierSame.crs, glacierSame.res, interp_list[0])
        tifClip(vx_err_name, vx_err_name, glacierSame.dem)
        tifClip(vy_err_name, vy_err_name, glacierSame.dem)
        shpClip(vx_err_name, glacierSame.shape, vx_err_name)
        shpClip(vy_err_name, glacierSame.shape, vy_err_name)

        vx_err_array_list[j] = rasterio.open(vx_err_name).read(1)
        vy_err_array_list[j] = rasterio.open(vy_err_name).read(1)

        # relative/absolute errors using raw velocity data
        v_err_rel_array_list[j] = 0.5 * np.divide((
                np.multiply(2 * np.divide(vx_err_array_list[j], vel_x_list[j], out=np.zeros_like(vx_err_array_list[j]),
                                          where=vel_x_list[j] != 0), vel_x_list[j]) +
                np.multiply(2 * np.divide(vy_err_array_list[j], vel_y_list[j], out=np.zeros_like(vy_err_array_list[j]),
                                          where=vel_x_list[j] != 0), vel_y_list[j])),
                vel_total_list[j], out=np.zeros_like(vel_total_list[j]), where=vel_total_list[j] != 0)

        v_err_array_list[j] = np.multiply(v_err_rel_array_list[j], vel_total_list[j])


# ------------------------------------------PART 4 DISPLAY OPTIONS-----------------------------------------#
printMassLoss = 1  # to print the mass loss values
# STEP 4: SMOOTH VELOCITY DATA WITH LOW PASS MOVING WINDOW FILTER, CLIMATIC MASS BALANCE FOR SMOOTH VELOCITY DATA
raw_data_smooth_factor = 4      # smoothing factor for h and v files (factor*h is window size) -- recommend 4
raw_data_smooth_factor_list = list(np.ones(number_of_vel, dtype=np.int8) * raw_data_smooth_factor)
divQ_smooth_factor = 1          # smoothing factor for divQ file (factor*h is window size) -- recommend 1
divQ_smooth_factor_list = list(np.ones(number_of_vel, dtype=np.int8) * divQ_smooth_factor)

smoothhArray = dynamicSmoothing(hArray, hArray, glacierSame.res, raw_data_smooth_factor)
smoothvxArray_list = list(map(dynamicSmoothing, vel_x_list, [hArray, hArray, hArray], res_list[:number_of_vel],
                              raw_data_smooth_factor_list))
smoothvyArray_list = list(map(dynamicSmoothing, vel_y_list, [hArray, hArray, hArray], res_list[:number_of_vel],
                              raw_data_smooth_factor_list))
smoothvArray_list = [np.power((np.square(x) + np.square(y)), 0.5) for x, y in zip(smoothvxArray_list, smoothvyArray_list)]

# -------------------------------------CORRECT VELOCITY BASED ON ASPECT------------------------------------#
# filter velocity to remove vels that oppose aspect by at least the threshold
area_list = [area, area, area][:number_of_vel]
aspect_array_dem_list = [aspect_array_dem, aspect_array_dem, aspect_array_dem][:number_of_vel]
thresh_list = np.ones(number_of_vel)
vel_cor1 = list(map(velAspectCorrection, aspect_array_dem_list, smoothvxArray_list, smoothvyArray_list, thresh_list*90))

vel_x_cor1_list, vel_y_cor1_list = np.array([x[0] for x in vel_cor1]), np.array([x[1] for x in vel_cor1])
vel_total_cor1_list = np.array([np.power((np.square(x) + np.square(y)), 0.5) for x, y in zip(vel_x_cor1_list, vel_y_cor1_list)])

corrected_vels_quiver1_list = list(map(velPlot, vel_x_cor1_list, vel_y_cor1_list, vel_total_cor1_list, area_list, thresh_list))
aspect_cor_title1_list = [vel_select[j] + ' Aspect-Corrected Map' for j in range(number_of_vel)]
cbarTitle_list = ['Velocity (m/a)' for x in range(number_of_vel)]
c_list = ['BrBG' for x in range(number_of_vel)]
list(map(plotData, vel_total_cor1_list, cbarTitle_list, c_list, aspect_cor_title1_list, none_list, corrected_vels_quiver1_list))
print('WE NEED TO EITHER MASK BASED ON ELEVATION (ABOVE THE TERMINUS) OR PRESCRIBE A SPECIFIC, NON-PHYSICAL VALUE TO MASK\n')



# GOOD/BAD VELOCITY PRODUCTS: create a binary mask: 1 where velocity products are bad, 0 where they are good
vel_good_bad_criteria = '(based on vel aspect)'
vel_good_bad = vel_total_cor1_list.copy()
vel_good_bad[(vel_good_bad > 0) & (vel_good_bad < 0.01)] = -1  # our 'bad' values are very small and positive
vel_good_bad[vel_good_bad != -1] = 1  # assign our good values to 1
vel_good_bad[vel_good_bad == -1] = 0  # assign our bad values to 0

# -------------RANGE OF ASPECT FOR VELOCITY, FROM UNCERTAINTY-------------
# Find the range of aspects based on the uncertainty files
vel_x_plus = np.array(vel_x_list) + np.array(vx_err_array_list)
vel_x_minus = np.array(vel_x_list) - np.array(vx_err_array_list)
vel_y_plus = np.array(vel_y_list) + np.array(vy_err_array_list)
vel_y_minus = np.array(vel_y_list) - np.array(vy_err_array_list)
vel_aspect_o0 = np.array(list(map(velocityAspect, vel_x_list, vel_y_list)))     # actual velocity vector aspect
vel_aspect_o1 = np.array(list(map(velocityAspect, vel_x_plus, vel_y_plus)))     # now, every possible aspect based on uncertainty
vel_aspect_o2 = np.array(list(map(velocityAspect, vel_x_plus, vel_y_minus)))
vel_aspect_o3 = np.array(list(map(velocityAspect, vel_x_minus, vel_y_plus)))
vel_aspect_o4 = np.array(list(map(velocityAspect, vel_x_minus, vel_y_minus)))

# this is the range of aspect values we have
v_aspect_list = [[vel_aspect_o1[x], vel_aspect_o2[x], vel_aspect_o3[x], vel_aspect_o4[x]] for x in range(number_of_vel)]
vel_aspect_range_list = list(map(velocityAspectAngle, vel_aspect_o0, v_aspect_list))
vel_aspect_range_list = [x*2*outlineArray for x in vel_aspect_range_list]
vel_aspect_range_title_list = [glacierSame.name+' '+vel_select[j]+' Range of Velocity Aspects' for x in range(number_of_vel)]
cbarTitle_list = ['Velocity Aspect Range' for x in range(number_of_vel)]
c_list = ['Spectral' for x in range(number_of_vel)]
colorbarMax_list = [90, 90, 1][:number_of_vel]
list(map(plotData, vel_aspect_range_list, cbarTitle_list, c_list, vel_aspect_range_title_list, none_list, none_list, colorbarMax_list))


if filtered_vel[0] == 'Aspect-Corrected-Removed':
    # NOTE: THIS DOES NOT MAINTAIN CONTINUITY. DO NOT USE THIS
    outlineArrayNew = outlineArray.copy()
    # vel_total_cor1 = np.minimum.reduce(np.array(vel_total_cor1_list))   # only keep areas where all products work
    vel_total_cor1 = np.maximum.reduce(np.array(vel_total_cor1_list))   # only remove areas where no products work
    outlineArrayNew[vel_total_cor1 <= 0.001] = 0  # mask out velocity values less than 0.001 (these are stagnant)
    rasterLike(outlineArrayNew, glacierSame.name + 'Outline.tif', glacierSame.name + 'Outline.tif')
    outlineArray = rasterio.open(glacierSame.name + 'Outline.tif').read(1)

if main == 'main':     # IF STATEMENT GIVES US MAIN TRUNK ONLY (00570)
    main_trunk_shp = 'Gulkana-main-trunk.shp'
    shp_attributes = shp_dataframe.loc[shp_dataframe['RGIId'] == glacierSame.name + '-Main']
    shp_attributes.to_file(main_trunk_shp)
    shpReprojection(main_trunk_shp, glacierSame.crs, main_trunk_shp)
    outlineArray = glacierOutline('ones_raster.tif', main_trunk_shp, glacierSame.name + 'Outline.tif')
    area = glacierArea(outlineArray, glacierSame.res)  # calculate and print glacier area
    print('We are only using the main trunk for analysis! Glacier Main Trunk Area (sq. km):', area/1000000)
    massBalanceTotal = totalMassBalance(dhdtArray, outlineArray, glacierSame.rho)
    totalMassBalanceVal = totalMassBalanceValue(massBalanceTotal, area, glacierSame.res)
    print('The TOTAL mass balance for the main trunk is: ', totalMassBalanceVal, '(mm w.e.)')


print('THE FOLLOWING IS THE "GULKANA CORRECTION". I HAVE REMOVED IT FOR  NOW, MUST FIX IT IF WE NEED IT (LINE 355)')
# # 'Gulkana'Correction' -- try to correct the problematic regions of Gulkana (hard-coded)
# # create a mask by point locations
# buff = 35  # the number of pixels to correct near a point (10 is 200m window size for 20m pixels)
# correctRegions = [[63.294753, -145.42061], [63.296, -145.401],
#                   [63.2865, -145.431], [63.2901, -145.4432]]  # point v
# outline_m = np.ma.getmask(np.ma.masked_where(outlineArray == 1, outlineArray))  # glacier outline mask
# vel_m = np.ma.getmask(np.ma.masked_where(vel_total_cor1 < 2, vel_total_cor1))
# m = np.full(demArray.shape, False)
# for coords_row in correctRegions:
#     row, col = glacierSame.latlonTiffIndex(demArray, coords_row)
#     m[row - buff:row + buff + 1, col - buff:col + buff + 1] = True
# m = m*outline_m*vel_m  # only take points that are on the glacier, near our points, and small vel signals
#
# # # create a boolean mask where our velocities are bad (0 < vel_x < 0.01) and elevation is > 1500
# # m = np.ma.getmask(np.ma.masked_where((vel_x_cor1 > 0) & (vel_x_cor1 < 0.01) & (demArray > 1500), vel_x_cor1))
#
# # now, we want to use the mask to apply values to this region. this prescribes a velocity magnitude
# vel_x_aspect, vel_y_aspect = velAspectDirection(aspect_array_dem, np.ones_like(smoothvArray)*10)  # aspect-directed vels
# vel_x_cor2 = vel_x_cor1_list.copy()
# vel_y_cor2 = vel_y_cor1_list.copy()
# vel_x_cor2[m == True] = vel_x_aspect[m == True]
# vel_y_cor2[m == True] = vel_y_aspect[m == True]
# vel_total_cor2 = np.power((np.square(vel_x_cor2) + np.square(vel_y_cor2)), 0.5)
# corrected_vels_quiver2 = velPlot(vel_x_cor2, vel_y_cor2, vel_total_cor2, area, threshold=1.0)
# aspect_cor_title2 = vel_select[j] + ' Gulkana-Corrected Map'
# plotData(vel_total_cor2, 'Velocity (m/a)', 'BrBG', aspect_cor_title2,
#          cluster=None, quiver=corrected_vels_quiver2)
# # ---- ^End of Gulkana-Correction^ ----

print('BELOW ARE VELOCITY FILTERING METHODS THAT HAVE NOT BEEN USED IN A WHILE (LINE 385)')
# # filter velocity to use vel product magnitude but use aspect for the direction
# vel_x_cor2, vel_y_cor2 = velAspectDirection(aspect_array_dem, smoothvArray)
# vel_total_cor2 = np.power((np.square(vel_x_cor2) + np.square(vel_y_cor2)), 0.5)
# corrected_vels_quiver2 = velPlot(vel_x_cor2, vel_y_cor2, vel_total_cor2, area, threshold=1.0)
# aspect_cor_title2 = vel_select[j] + ' Aspect-Directed Map'
# plotData(vel_total_cor2, 'Velocity (m/a)', 'BrBG', aspect_cor_title2,
#          cluster=None, quiver=corrected_vels_quiver2)
#
# # filter velocity to use vel product magnitude but use aspect for the direction when slope exceeds a threshold
# vel_x_cor3, vel_y_cor3 = velAspectSlopeThreshold(aspect_array_dem, vel_x_cor1, vel_y_cor1, slope_array_dem, 10)
# vel_total_cor3 = np.power((np.square(vel_x_cor3) + np.square(vel_y_cor3)), 0.5)
# corrected_vels_quiver3 = velPlot(vel_x_cor3, vel_y_cor3, vel_total_cor3, area, threshold=1.0)
# aspect_cor_title3 = vel_select[j] + ' Aspect-Slope Directed Map'
# plotData(vel_total_cor3, 'Velocity (m/a)', 'BrBG', aspect_cor_title3,
#          cluster=None, quiver=corrected_vels_quiver3)
#
# # filter velocity to use vel product magnitude but use weighted average pf raw data and DEM aspect for direction
# vel_x_cor4, vel_y_cor4 = velAspectSlopeAverage(aspect_array_dem, vel_x_cor1, vel_y_cor1, slope_array_dem, 10)
# vel_total_cor4 = np.power((np.square(vel_x_cor4) + np.square(vel_y_cor4)), 0.5)
# corrected_vels_quiver4 = velPlot(vel_x_cor4, vel_y_cor4, vel_total_cor4, area, threshold=1.0)
# aspect_cor_title4 = vel_select[j] + ' Weighted Mean of Vel and DEM-based Aspect Map'
# plotData(vel_total_cor4, 'Velocity (m/a)', 'BrBG', aspect_cor_title4,
#          cluster=None, quiver=corrected_vels_quiver4)

print('BELOW ARE SLOPE-VELOCITY PLOTS. THESE HAVE BEEN REMOVED FOR A WHILE. THESE ARE THE SCREENSHOTS ON MY DESKTOP (LINE 410)')
# slope_vel_plot(slope_array_dem, vArray, title='Slope vs Velocity (' + vel_select[j] + ')', showPlot=0)
# # slope_vel_plot(slope_array_dem, smoothvArray, showPlot=False)
# h_vel_plot(slope_array_dem, hArray, vArray,
#            title='Thickness vs Velocity (' + thick_select[0] + ', ' + vel_select[j] + ')', showPlot=0)
# # h_vel_plot(slope_array_dem, smoothhArray, smoothvArray, showPlot=False)
# stress_vel_plot(slope_array_dem, hArray, vArray,
#                 title='Driving Stress vs Velocity (' + thick_select[0] + ', ' + vel_select[j] + ')', showPlot=0)
# # stress_vel_plot(slope_array_dem, smoothhArray, smoothvArray, showPlot=False)



# ---------------------------------------------------------------------------------------------------------#
outlineArray_list = [outlineArray, outlineArray, outlineArray][:number_of_vel]
demArray_list = [demArray, demArray, demArray][:number_of_vel]
hArray_list = [hArray, hArray, hArray][:number_of_vel]
smoothhArray_list = [smoothhArray, smoothhArray, smoothhArray][:number_of_vel]
if filtered_vel[0] == 'Aspect-Corrected' or filtered_vel[0] == 'Aspect-Corrected-Removed':
    divQarray_list = list(map(divQ, vel_x_cor1_list, vel_y_cor1_list, smoothhArray_list,
                              res_list[:number_of_vel], vCol_list[:number_of_vel]))
# elif filtered_vel[0] == 'Gulkana-Correction':
#     divQarray_list = divQ(vel_x_cor2, vel_y_cor2, smoothhArray, glacierSame.res, glacierSame.vCol)
# elif filtered_vel[0] == 'Aspect-Directed':
#     divQarray_list = divQ(vel_x_cor2, vel_y_cor2, smoothhArray, glacierSame.res, glacierSame.vCol)
# elif divQarray[0] == 'Aspect-Slope Directed':
#     divQRaster_list = divQ(vel_x_cor3, vel_y_cor3, smoothhArray, glacierSame.res, glacierSame.vCol)
# elif filtered_vel[0] == 'Weighted Mean Aspect':
#     divQarray_list = divQ(vel_x_cor4, vel_y_cor4, smoothhArray, glacierSame.res, glacierSame.vCol)
else:
    divQarray_list = list(map(divQ, smoothvxArray_list, smoothvyArray_list, smoothhArray_list,
                              res_list[:number_of_vel], vCol_list[:number_of_vel]))

divQRasterSmooth_list = list(map(dynamicSmoothing, divQarray_list, hArray_list, res_list[:number_of_vel], divQ_smooth_factor_list))


# total SMB calculations
binMeandhdt, binElevations1, binCounts1, binNumber1, bin_std_dhdt, bin_min_dhdt, bin_max_dhdt = \
    altitudeAggregation(dhdtArray, demArray, outlineArray, stat='mean', bin_z=glacierSame.bin)

binTotalChange = binMeandhdt * glacierSame.rho
demTotBinTotalMass = binCounts1 * binTotalChange * glacierSame.res * glacierSame.res  # this is in kg/yr
altResTotMassBalance = np.sum(demTotBinTotalMass) / area  # divide by total area for alt-resolved mb value

# climatic SMB calculations
clim_smb_pixel_smooth_list = []
climMassBalanceValueSmooth_list = []
stat_list = ['mean', 'mean', 'mean'][:number_of_vel]
bin_list = [glacierSame.bin, glacierSame.bin, glacierSame.bin][:number_of_vel]
for j in range(number_of_vel):
    clim_smb_pixel_smooth_list.append((dhdtArray + divQRasterSmooth_list[j]) * glacierSame.rho * outlineArray)
    climMassBalanceValueSmooth_list.append(np.sum(clim_smb_pixel_smooth_list[-1] * glacierSame.res * glacierSame.res) / area)

r = list(map(altitudeAggregation, divQRasterSmooth_list, demArray_list, outlineArray_list, stat_list, bin_list))
bindivQSmoothMean, binElevations2, binCounts2, binNumber2 = [np.array(x[0]) for x in r], r[0][1], r[0][2], r[0][3]

binClimChangeSmooth_list = [(binMeandhdt + bindivQSmoothMean[j]) * glacierSame.rho for j in range(number_of_vel)]
demClimBinTotalMassSmooth_list = [np.array(binCounts2 * binClimChangeSmooth_list[j] * glacierSame.res * glacierSame.res)
                                  for j in range(number_of_vel)]
altResClimMassBalanceSmooth_list = [(np.sum(demClimBinTotalMassSmooth_list[j]) / area) for j in range(number_of_vel)]

# print('The altitudinally resolved TOTAL mass balance is:', altResTotMassBalance, '(mm w.e.)')
print('The CLIMATIC (SMOOTH) mass balance is: ', climMassBalanceValueSmooth_list, '(mm w.e.) for\n                    ',
      vel_select)
# print('The altitudinally resolved CLIMATIC (SMOOTH) MB is:', altResClimMassBalanceSmooth_list, '(mm w.e.)')

# ---------------------------------------------STEP 6: BIN ERRORS--------------------------------------------#
errorVal = 'percent'  # 'std', 'se', 'minmax', 'percent'
p = 25
p_list = [p, p, p][:number_of_vel]

binTotalSMBMean, binElevations3, binCounts3, binNumber3, bin_std_totSmb, bin_min_totSmb, bin_max_totSmb = \
    altitudeAggregation(massBalanceTotal, demArray, outlineArray, stat='mean', bin_z=glacierSame.bin)
r = list(map(altitudeAggregation, clim_smb_pixel_smooth_list, demArray_list, outlineArray_list, stat_list, bin_list))
binClimSMBSmoothMean, binCounts4 = [np.array(x[0]) for x in r], r[0][2]
bin_std_climSmbSmooth = [np.array(x[4]) for x in r]
bin_min_climSmbSmooth, bin_max_climSmbSmooth = [np.array(x[5]) for x in r], [np.array(x[6]) for x in r]

if errorVal == 'std':
    error_name = 'standard deviation'
    totalSMBErr = bin_std_totSmb
    climSMBSmoothErr_list = [bin_std_climSmbSmooth[j] for j in range(number_of_vel)]
if errorVal == 'se':
    error_name = 'standard error'
    totalSMBErr = bin_std_totSmb / (binCounts3 ** 0.5)
    climSMBSmoothErr_list = [(bin_std_climSmbSmooth[j] / (binCounts4 ** 0.5)) for j in range(number_of_vel)]
if errorVal == 'minmax':
    error_name = 'min and max value'
    totalSMBErr = np.array([binTotalSMBMean - bin_min_totSmb, bin_max_totSmb - binTotalSMBMean])
    climSMBSmoothErr_list = [np.array([binClimSMBSmoothMean[j] - bin_min_climSmbSmooth[j], bin_max_climSmbSmooth[j] -
                                       binClimSMBSmoothMean[j]]) for j in range(number_of_vel)]
if errorVal == 'percent':
    error_name = 'percentile: lower bound at ' + str(p) + '%, upper bound at ' + str(100 - p) + '%'
    binTotalSMB_lower_p, binTotalSMB_upper_p = binPercentile(massBalanceTotal, demArray, outlineArray, p, glacierSame.bin)
    r = list(map(binPercentile, clim_smb_pixel_smooth_list, demArray_list, outlineArray_list, p_list, bin_list))
    binClimSMBSmooth_lower_p, binClimSMBSmooth_upper_p = [x[0] for x in r], [x[1] for x in r]

    totalSMBErr = abs(np.array([binTotalSMB_lower_p - binTotalSMBMean, binTotalSMB_upper_p - binTotalSMBMean]))
    climSMBSmoothErr_list = [abs(np.array([binClimSMBSmooth_lower_p[j] - binClimSMBSmoothMean[j],
                                           binClimSMBSmooth_upper_p[j] - binClimSMBSmoothMean[j]])) for j in
                             range(number_of_vel)]

if errorVal == 'none':
    error_name = 'no error bars'
    totalSMBErr = None
    climSMBSmoothErr_list = [None for j in range(number_of_vel)]


# ------------------------------------------STEP 7: PLOTTING/RESULTS-----------------------------------------#
livePlots = 0  # to show the plots as live plots (not just reopened saved files)
create_pdf = 1  # to create pdf containing glacier mass balance information

# For some reason, we get a pixel in it's own elevation bin, without a recorded elevation bin. This is a problem
final_bin_count = np.unique(binNumber1, return_counts=True)[1][-1]

if final_bin_count <= 5 and len(binTotalChange) != int(binNumber1.max()):
    print('Top bin pixel count was: ' + str(final_bin_count) + '. It has been merged with the second-to-top bin.')
    binNumber1[binNumber1 == binNumber1.max()] = binNumber1.max() - 1
    binNumber2[binNumber2 == binNumber2.max()] = binNumber2.max() - 1
    binNumber3[binNumber3 == binNumber3.max()] = binNumber3.max() - 1

# we'll remove the last value. note these correspond to the lower bound of each bin now
binElevations1 = binElevations1[0:-1]
binElevations2 = binElevations2[0:-1]
binElevations3 = binElevations3[0:-1]

# apply bin SMB to each value in that bin: FOR PLOTTING ONLY
binnedTotalMassBalance = np.zeros(binNumber1.shape)
binnedClimaticMassBalanceSmooth = np.zeros(binNumber2.shape)
binnedEmergenceSmooth = np.zeros(binNumber2.shape)
binnedClimaticMassBalanceSmooth_list = []
binnedEmergenceSmooth_list = []

for j in range(number_of_vel):
    for i in range(len(binNumber1)):
        binnedTotalMassBalance[i] = binTotalChange[binNumber1[i] - 1]
        binnedClimaticMassBalanceSmooth[i] = binClimChangeSmooth_list[j][binNumber2[i] - 1]
        binnedEmergenceSmooth[i] = bindivQSmoothMean[j][binNumber2[i] - 1]
    binnedClimaticMassBalanceSmooth_list.append(binnedClimaticMassBalanceSmooth)
    binnedEmergenceSmooth_list.append(binnedEmergenceSmooth)

outlineArrayShape = outlineArray.shape
binnedTotalMassBalance = np.array(binnedTotalMassBalance).reshape(outlineArrayShape) * outlineArray
binnedClimaticMassBalanceSmooth_list = [np.array(x).reshape(outlineArrayShape) * outlineArray for x in binnedClimaticMassBalanceSmooth_list]
binnedEmergenceSmooth_list = [np.array(x).reshape(outlineArrayShape) * outlineArray for x in binnedEmergenceSmooth_list]
binnedElevationNumber = np.array(binNumber1).reshape(outlineArrayShape) * outlineArray

# for plotting: each subplot has: [data array, title, colorbar label]
h_plot = [hArray, 'Thickness Map', 'Glacier Thickness (m)', 'ocean']
h_plot_smooth = [smoothhArray, 'Smoothed Thickness Map', 'Glacier Thickness (m)', 'ocean']
z_plot = [demArray, 'Elevation Map', 'Elevation (m)', 'gist_earth']  # gist_earth, viridis, jet
b_tot_plot = [binnedTotalMassBalance, 'Total Mass Balance Map', 'Mean Elevation Bin SMB (mm w.e.)', 'RdBu']
b_tot_plot_pixel = [massBalanceTotal * outlineArray, 'Total Mass Balance Map', 'Pixel Mass Balance (mm w.e.)', 'RdBu']
dhdt_plot_pixel = [dhdtArray, 'Change in Thickness Map', 'dh/dt (m/a)', 'RdBu']

v_plot_list = [[vel_total_list[j], 'Velocity Map', ' Velocity (m/a)', 'BrBG'] for j in range(number_of_vel)]
v_plot_smooth_list = [[smoothvArray_list[j], 'Smoothed Velocity Map', 'Velocity (m/a)', 'BrBG'] for j in range(number_of_vel)]
# vx_plot = [[vel_x_list[j], vel_select[j] + ' X-Velocity Map', 'Velocity (m/a)', 'BrBG'] for j in range(number_of_vel)]
# vx_plot_smooth = [[smoothvxArray_list[j], vel_select[j] + ' Smoothed X-Velocity Map', 'Velocity (m/a)', 'BrBG']
#                   for j in range(number_of_vel)]
# vy_plot = [[vel_y_list[j], vel_select[j] + ' Y-Velocity Map', 'Velocity (m/a)', 'BrBG'] for j in range(number_of_vel)]
# vy_plot_smooth = [[smoothvyArray_list[j], vel_select[j] + ' Smoothed Y-Velocity Map', 'Velocity (m/a)', 'BrBG']
#                   for j in range(number_of_vel)]
v_plot_err_list = [[np.nan_to_num(v_err_array_list[j]), 'Velocity Uncertainty Map', 'Velocity Uncertainty (m/a)',
                    'BrBG'] for j in range(number_of_vel)]
v_plot_err_rel_list = [[np.nan_to_num(v_err_rel_array_list[j])*100, 'Velocity Relative Uncertainty Map',
                        'Relative Uncertainty (%)', 'BrBG'] for j in range(number_of_vel)]
b_clim_plot_smooth_list = [[binnedClimaticMassBalanceSmooth_list[j], vel_select[j] + ' Climatic Mass Balance Map (smooth)',
                            'Mean Elevation Bin SMB (mm w.e.)', 'RdBu'] for j in range(number_of_vel)]
b_clim_plot_pixel_smooth_list = [[clim_smb_pixel_smooth_list[j] * outlineArray, vel_select[j] + ' Climatic Mass Balance Map (smooth)',
                                  'Pixel Mass Balance (mm w.e.)', 'RdBu'] for j in range(number_of_vel)]


emergence_plot_smooth_list = [[binnedEmergenceSmooth_list[j], vel_select[j] + ' Emergence Map (smooth)',
                               'Mean Elevation Bin Emergence (m/a)', 'RdBu'] for j in range(number_of_vel)]
emergence_plot_pixel_smooth_list = [[divQRasterSmooth_list[j] * outlineArray, 'Smoothed Emergence Map',
                                     'Pixel Emergence (m/a)', 'RdBu'] for j in range(number_of_vel)]

# -------QUANTIFY DIFFERENCES IN VELOCITY AND GLACIOLOGICAL MEASUREMENTS--------#
# Comparing MB to glaciologic measurements for each velocity product (where WGMS data exist)
if glacierSame.data is not None:
    one_to_one_cmb_scatterplot_data = []
    one_to_one_scatterplot_label_list = []
    one_to_one_divq_scatterplot_data = []
    for j in range(number_of_vel):
        # APPROACH 1: mean errors for elevation bins with glaciologic measurements
        binAreasList = (binCounts2 * glacierSame.res * glacierSame.res).tolist()
        binElevationsList = binElevations2.tolist()

        mb_model_prediction = []
        mb_model_prediction_avg = []
        data_emergence = []
        emergence_model_prediction = []
        for i in range(len(glacierSame.data[1])):
            # find index value where reference data elevation exceeds elevation bin
            indexVal = next(x[0] for x in enumerate(binElevationsList) if x[1] >= glacierSame.data[1][i]) - 1
            # value of MB model for elevation bin where reference data exists
            mb_model_prediction.append(binClimChangeSmooth_list[j][indexVal])
            data_emergence.append((glacierSame.data[0][i] - binTotalChange[indexVal])/glacierSame.rho)
            emergence_model_prediction.append((binClimChangeSmooth_list[j][indexVal] -
                                               binTotalChange[indexVal])/glacierSame.rho)

            # alternative approach: area-weighted mean of 3 adjacent elevation bins
            # find the total area of 3 adjacent elevation bins
            areaAdjacent = binAreasList[indexVal - 1] + binAreasList[indexVal] + binAreasList[indexVal + 1]
            # weighted mean of 3 adjacted elevation bins
            weighted_mean = \
                np.nanmean([binClimChangeSmooth_list[j][indexVal - 1] * binAreasList[indexVal - 1] / areaAdjacent,
                           binClimChangeSmooth_list[j][indexVal] * binAreasList[indexVal] / areaAdjacent,
                           binClimChangeSmooth_list[j][indexVal + 1] * binAreasList[indexVal + 1] / areaAdjacent])
            mb_model_prediction_avg.append(weighted_mean)

        # mean absolute error and mean bias error:
        meanAbsErr = mean_absolute_error(glacierSame.data[0], mb_model_prediction)
        meanBiasErr = np.mean([L1 - L2 for L1, L2 in zip(glacierSame.data[0], mb_model_prediction)])
        # mean absolute error and mean bias error, area-weighted by 3 adjacent bin averages
        meanAbsErr3BinWeighted = mean_absolute_error(glacierSame.data[0], mb_model_prediction_avg)
        meanBiasErr3BinWeighted = np.mean([L1 - L2 for L1, L2 in zip(glacierSame.data[0],
                                                                     mb_model_prediction_avg)])  # (subtract lists)
        # plot 1v1 CMB scatterplot
        one_to_one_cmb_scatterplot_data.append([glacierSame.data[0], mb_model_prediction])
        one_to_one_cmb_scatterplot_title = 'Climatic Mass Balance from Calculated and Reference Data'
        one_to_one_scatterplot_label_list.append(vel_select[j] + '(Data Points)')

        # APPROACH 2: comparison of values to line of best fit from glaciological measurements
        # let's only consider bins from the min to max total mass balance (removes initial, low-area bins)
        min_index = binTotalChange.tolist().index(min(binTotalChange))
        # max bin index must be after min bin index and must meet a criteria
                    # (e.g. have an area at least as large as min index, or at least be 2% of total glacier area)
        min_bin_area = binAreasList[min_index]
        max_index = binTotalChange.tolist().index(max(binTotalChange[min_index:]))
        max_bin_area = binAreasList[max_index]
        # make sure max index has a large enough area:
        # if not, find the max index before max_index with a large enough area
        criteria = 0.02*area
        # criteria = min_bin_area
        if max_bin_area < criteria:
            max_index = max(i for i, a in enumerate(binAreasList[:max_index]) if a >= criteria)

        validIndexes = list(range(min_index, max_index+1))
        valid_mb_vals = [binClimChangeSmooth_list[j][x] for x in validIndexes]
        # obtain accompanying values from line of best fit (lobf)
        # get elevations associated with the MB points
        for i in range(len(binElevationsList)):
            binElevationsList[i] = binElevationsList[i] + (glacierSame.bin * 0.5)
        valid_elevation_vals = [binElevationsList[x] for x in validIndexes]
        # get the lobf and the lobf values at these elevations
        m, b = np.polyfit(glacierSame.data[0], glacierSame.data[1], 1)
        accompanying_lobf_vals = [yb / m for yb in [y - b for y in valid_elevation_vals]]  # (binElevationsList-b)/m
        # mean absolute error and mean bias error:
        meanAbsErrLine = mean_absolute_error(accompanying_lobf_vals, valid_mb_vals)
        meanBiasErrLine = np.mean([L1 - L2 for L1, L2 in zip(accompanying_lobf_vals, valid_mb_vals)])
        # 1 to 1 scatterplot with line of best fit
        one_to_one_cmb_scatterplot_data.append([accompanying_lobf_vals, valid_mb_vals])
        one_to_one_scatterplot_label_list.append(vel_select[j] + '(LOBF)')

        # plot 1v1 EMERGENCE scatterplot
        one_to_one_divq_scatterplot_data.append([data_emergence, emergence_model_prediction])
        one_to_one_divq_scatterplot_title = 'Emergence from Calculated and Reference Data'
        valid_emergence_vals = [(binClimChangeSmooth_list[j][x] - binTotalChange[x])/glacierSame.rho for x in validIndexes]
        accompanying_lobf_divq = [(lobf - tmb)/glacierSame.rho for lobf, tmb in zip(accompanying_lobf_vals, binTotalChange)]
        one_to_one_divq_scatterplot_data.append([accompanying_lobf_divq, valid_emergence_vals])

        # APPROACH 3: clim MB gradient from model to slope of line of best fit (lobf)
        m_mb_model, b_mb_model = np.polyfit(valid_mb_vals, valid_elevation_vals, 1)
        # positive error means model has more mass loss per elevation than reference data
        clim_mb_grad_err = 100 * ((1/m_mb_model) - (1/m)) / (1/m)

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
            rasterLike(clim_smb_pixel_smooth_list[j], glacierSame.name + '_array_temp.tif', glacierSame.name + 'TotalMassBalance.tif')
            row, col = latlonTiffIndex(glacierSame.name + '_array_temp.tif', coords_row, glacierSame.crs)
            dataCoords_pixel_col.append([col])
            dataCoords_pixel_row.append([row])
            dataCoords_labels.append(coords_row[4])
            local_avg_mb = np.nanmean(clim_smb_pixel_smooth_list[j][row-app4_pixel_window_avg:row+app4_pixel_window_avg+1,
                                      col-app4_pixel_window_avg:col+app4_pixel_window_avg+1])  # window avg at point
            pixel_location_clim_mb_abs.append(abs(local_avg_mb - coords_row[2]))
            pixel_location_clim_mb_diff.append(local_avg_mb - coords_row[2])
        meanAbsErrPixelCrs = np.nanmean(pixel_location_clim_mb_abs)
        meanBiasErrPixelCrs = np.nanmean(pixel_location_clim_mb_diff)

        vel_dict['MeanAbsoluteError%s' % j] = meanAbsErr
        vel_dict['WeightedMeanAbsoluteError%s' % j] = meanAbsErr3BinWeighted
        vel_dict['MeanBiasError%s' % j] = meanBiasErr
        vel_dict['WeightedMeanBiasError%s' % j] = meanBiasErr3BinWeighted
        vel_dict['MeanAbsoluteErrorLine%s' % j] = meanAbsErrLine
        vel_dict['MeanBiasErrorLine%s' % j] = meanBiasErrLine
        vel_dict['ClimMBGradError%s' % j] = clim_mb_grad_err
        vel_dict['MeanAbsoluteErrorPixel%s' % j] = meanAbsErrPixelCrs
        vel_dict['MeanBiasErrorPixel%s' % j] = meanBiasErrPixelCrs

    # compare emergence to Lucas Zeller emergence data: do we need more/less smoothing?
        # quantitative and qualitative analysis: similar to elevation vs MB bias/mean absolute error quantification above
        # map differencing

scatterplot(one_to_one_cmb_scatterplot_data, 'Reference Stake CMB (mm w.e.)', 'Calculated Remote Sensing CMB (mm w.e.)',
            one_to_one_cmb_scatterplot_title, ['#01665e', '#01665e', '#8c510a', '#8c510a', '#67000d', '#67000d'],
            one_to_one_scatterplot_label_list, ['+', 'o', '+', 'o', '+', 'o'], xyLine=True)
scatterplot(one_to_one_divq_scatterplot_data, 'Reference Stake Flux Divergence (m)', 'Calculated Remote Sensing Flux Divergence (m)',
            one_to_one_divq_scatterplot_title, ['#01665e', '#01665e', '#8c510a', '#8c510a', '#67000d', '#67000d'],
            one_to_one_scatterplot_label_list, ['+', 'o', '+', 'o', '+', 'o'], xyLine=True, buff=1)

# -------PLOTTING FOR EACH VELOCITY PRODUCT--------#
fig4_1 = emergence_plot_pixel_smooth_list  # fig4: raw vs smooth velocity v
fig4_2 = v_plot_list
fig4_3 = v_plot_smooth_list

# 'fig4' with 3 subplots
fig4_1Vals = [fig4_1[x][0] for x in range(number_of_vel)]
fig4_1cbar = [fig4_1[x][2] for x in range(number_of_vel)]
fig4_1Title = [fig4_1[x][1] for x in range(number_of_vel)]
fig4_1Color = [fig4_1[x][3] for x in range(number_of_vel)]
fig4_2Vals = [fig4_2[x][0] for x in range(number_of_vel)]
fig4_2cbar = [fig4_2[x][2] for x in range(number_of_vel)]
fig4_2Title = [fig4_2[x][1] for x in range(number_of_vel)]
fig4_2Color = [fig4_2[x][3] for x in range(number_of_vel)]
fig4_2Quiver = list(map(velPlot, vel_x_list, vel_y_list, vel_total_list, area_list, thresh_list))
fig4_3Vals = [fig4_3[x][0] for x in range(number_of_vel)]
fig4_3cbar = [fig4_3[x][2] for x in range(number_of_vel)]
fig4_3Title = [fig4_3[x][1] for x in range(number_of_vel)]
fig4_3Color = [fig4_3[x][3] for x in range(number_of_vel)]
fig4_3Quiver = list(map(velPlot, smoothvxArray_list, smoothvyArray_list, smoothvArray_list, area_list, thresh_list))
smoothedVPlotTitle_list = [glacierSame.name+' '+vel_select[j]+' Velocity Products' for j in range(number_of_vel)]
fig4_div_select_list = [[1, 1, 1] for x in range(number_of_vel)]
fig4_min_list, fig4_max_list = [[-30,0,0] for x in range(number_of_vel)], [[30,50,50] for x in range(number_of_vel)]
fig4_mean_list = [[0,2.5,2.5] for x in range(number_of_vel)]
list(map(plotData3, fig4_1Vals, fig4_1cbar, fig4_1Color, fig4_1Title,
         fig4_2Vals, fig4_2cbar, fig4_2Color, fig4_2Title,
         fig4_3Vals, fig4_3cbar, fig4_3Color, fig4_3Title, smoothedVPlotTitle_list, res_list[:number_of_vel],
         fig4_2Quiver, fig4_3Quiver, fig4_div_select_list, fig4_min_list, fig4_mean_list, fig4_max_list))

# 'fig5' velocity uncertainty plots
fig5_1 = v_plot_err_list
fig5_2 = v_plot_err_rel_list
fig5_3 = v_plot_list

# 'fig5' with 3 subplots
fig5_1Vals = [fig5_1[x][0] for x in range(number_of_vel)]
fig5_1cbar = [fig5_1[x][2] for x in range(number_of_vel)]
fig5_1Title = [fig5_1[x][1] for x in range(number_of_vel)]
fig5_1Color = [fig5_1[x][3] for x in range(number_of_vel)]
fig5_2Vals = [fig5_2[x][0] for x in range(number_of_vel)]
fig5_2cbar = [fig5_2[x][2] for x in range(number_of_vel)]
fig5_2Title = [fig5_2[x][1] for x in range(number_of_vel)]
fig5_2Color = [fig5_2[x][3] for x in range(number_of_vel)]
fig5_2Quiver = [None for x in range(number_of_vel)]
fig5_3Vals = [fig5_3[x][0] for x in range(number_of_vel)]
fig5_3cbar = [fig5_3[x][2] for x in range(number_of_vel)]
fig5_3Title = [fig5_3[x][1] for x in range(number_of_vel)]
fig5_3Color = [fig5_3[x][3] for x in range(number_of_vel)]
fig5_3Quiver = list(map(velPlot, vel_x_list, vel_y_list, vel_total_list, area_list, thresh_list))
uncertaintyVPlotTitle_list = [glacierSame.name+' '+vel_select[j]+' Velocity Uncertainties' for j in range(number_of_vel)]
fig5_div_select_list = [[0, 1, 0] for x in range(number_of_vel)]
fig5_min_list, fig5_max_list = [[0,0,0] for x in range(number_of_vel)], [[100,100,100] for x in range(number_of_vel)]
fig5_mean_list = [[10,10,10] for x in range(number_of_vel)]
list(map(plotData3, fig5_1Vals, fig5_1cbar, fig5_1Color, fig5_1Title,
         fig5_2Vals, fig5_2cbar, fig5_2Color, fig5_2Title,
         fig5_3Vals, fig5_3cbar, fig5_3Color, fig5_3Title, uncertaintyVPlotTitle_list, res_list[:number_of_vel],
         fig5_2Quiver, fig5_3Quiver, fig5_div_select_list, fig5_min_list, fig5_mean_list, fig5_max_list))

if livePlots == False:
    plt.close('all')

# -------------------------------QUANTITATIVE VELOCITY SCALING FACTOR-------------------------#
# # small bin (sbin) mean slope, thickness, and velocity (bins are 20% the size of normal)
# sbin_z = int(glacierSame.bin / 5)
# sbinArea = altitudeAggregation(slope_array_dem, demArray, outlineArray, stat='mean', bin_z=sbin_z)[2] * glacierSame.res * glacierSame.res
# sbinMeanSlope = altitudeAggregation(slope_array_dem, demArray, outlineArray, stat='mean', bin_z=sbin_z)[0]
# sbinMeanThick = altitudeAggregation(smoothhArray, demArray, outlineArray, stat='mean', bin_z=sbin_z)[0]
# sbinMeanVel = altitudeAggregation(vel_total_cor1, demArray, outlineArray, stat='mean', bin_z=sbin_z)[0] # This needs to be iterable

# svalidIndexes = validIndexes * int(glacierCalc.bin / sbin_z)
# a_cs_out = sbinMeanThick[svalidIndexes[0]] * sbinArea[svalidIndexes[0]] / \
#            (sbin_z / np.tan(sbinMeanSlope[svalidIndexes[0]]*np.pi/180))  # area coming out of bin
# a_cs_in = sbinMeanThick[svalidIndexes[1]] * sbinArea[svalidIndexes[1]] / \
#           (sbin_z / np.tan(sbinMeanSlope[svalidIndexes[1]]*np.pi/180))  # area coming into bin
#
# flux_out = a_cs_out * sbinMeanVel[svalidIndexes[0]]
# flux_in = a_cs_in * sbinMeanVel[svalidIndexes[1]]
# em_out = flux_out / binAreasList[validIndexes[0]]
# em_in = flux_out / binAreasList[validIndexes[1]]
# em_diff = em_in - em_out
#
# print('Emergence in:', em_in)
# print('Emergence out:', em_out)

# required_emergence = np.divide(accompanying_lobf_vals, glacierCalc.rho) - binMeandhdt[validIndexes[0]:validIndexes[-1]+1]


print('BELOW IS THE CODE FOR HOW MUCH LARGER VELOCITY NEEDS TO BE TO ACHIEVE EMERGENCE (LINE 770). THIS IS TEMPORARILY DISABLED')
# # let's iteratively see how scaling velocity will match observation/difference, and return that value
#
# # divQRaster = get_divQRaster(smoothvxArray, smoothvyArray, smoothhArray, glacierCalc.res, glacierCalc.vCol)
# if vel_scaling_factor[0] == 'Yes':
#     # Emergence is basically scaling 1:1 with velocity!! Two terms dominate
#     required_emergence = (accompanying_lobf_vals[0] / glacierSame.rho) - binMeandhdt[validIndexes[0]]
#     current_best = required_emergence
#     best_scaling_factor = 'n/a'
#     print('Required emergece of', required_emergence, 'm/yr')
#
#     for i in range(20):
#         divQRaster_test = divQ(i * 0.5 * vel_x_list, i * 0.5 * vel_y_list, smoothhArray, glacierSame.res, glacierSame.vCol)
#         # divQRaster_test = get_divQRaster(i * 0.5 * smoothvxArray, i * 0.5 * smoothvyArray, smoothhArray, glacierCalc.res, glacierCalc.vCol)
#         # divQRaster_test = get_divQRaster(i*0.5*vel_x_cor1, i*0.5*vel_y_cor1, smoothhArray, glacierCalc.res, glacierCalc.vCol)
#         # divQRasterSmooth_test = dynamicSmoothing(divQRaster_test, hArray, glacierCalc.res, divQ_smooth_factor)
#         divQRasterSmooth_test = gaussianFilter(divQRaster_test, 1)
#         bindivQSmoothMean_test = glacierCalc.altitudeAggregation(divQRasterSmooth_test, stat='mean')[0][validIndexes[0]]
#         diff_with_i = required_emergence - bindivQSmoothMean_test
#         if abs(diff_with_i) < abs(current_best):
#             print('For scaling factor', i*0.5, 'we have:', bindivQSmoothMean_test, 'm/yr emergence')
#             current_best = diff_with_i
#             best_scaling_factor = i*0.5
#
#     print(best_scaling_factor)
#     # 1.5 (itslive), 2.0 (millan), 9.5 (retreat) with vel_cor1
#     # 1.5 (itslive), 2.0 (millan), n/a (retreat) with smoothvArrays
#     # 1.5 (itslive), 1.5 (millan), n/a (retreat) with raw vel

# --------------------------------------------- PLOTTING ----------------------------------------------
'''
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
    subplot2 = b_clim_plot_smooth_list[0]
    subplot3 = b_tot_plot
    subplot4 = b_clim_plot_smooth_list[1]
elif number_of_vel == 3:
    subplot2 = b_clim_plot_smooth_list[0]
    subplot3 = b_clim_plot_smooth_list[1]
    subplot4 = b_clim_plot_smooth_list[2]
    # subplot2 = b_clim_plot_pixel_smooth_list[0]
    # subplot3 = b_clim_plot_pixel_smooth_list[1]
    # subplot4 = b_clim_plot_pixel_smooth_list[2]
else:
    subplot2 = b_clim_plot_smooth_list[0]
    subplot3 = b_tot_plot
    subplot4 = b_clim_plot_pixel_smooth_list[0]


# to show a scatter plot of elevation vs mean mass loss at each elevation bin, as well as 3 other figures
binElevationsList = binElevations1.tolist()
for i in range(len(binElevationsList)):
    binElevationsList[i] = binElevationsList[i] + (glacierSame.bin * 0.5)
binTotalChangeList = binTotalChange.tolist()
binAreasList = (binCounts1 * glacierSame.res * glacierSame.res).tolist()

x_subplot_nw = binTotalChangeList
x_subplot_nw_lab = r'$\dot{b}_{tot}$'
x_subplot_nw_2 = binClimChangeSmooth_list[0]
x_subplot_nw_2_err = climSMBSmoothErr_list[0]
x_subplot_nw_2_lab = r'$\dot{b}_{clim}$ (' + vel_select[0] + ')'
try:
    x_subplot_nw_3 = binClimChangeSmooth_list[1]
    x_subplot_nw_3_err = climSMBSmoothErr_list[1]
    x_subplot_nw_3_lab = r'$\dot{b}_{clim}$ (' + vel_select[1] + ')'
except:
    x_subplot_nw_3 = binClimChangeSmooth_list[0]
    x_subplot_nw_3_err = climSMBSmoothErr_list[0]
    x_subplot_nw_3_lab = r'$\dot{b}_{clim}$ (' + vel_select[0] + ')'

if glacierSame.data is not None:  # add reference data points
    x_subplot_nw_ref = glacierSame.data[0]
    y_values1_ref = glacierSame.data[1]
    x_subplot_nw_ref_lab = 'Reference Data'
else:
    x_subplot_nw_ref = None
    y_values1_ref = None
    x_subplot_nw_ref_lab = None
if glacierSame.dataVariability is not None:
    var_values1_ref = glacierSame.dataVariability
else:
    var_values1_ref = None
y_values1 = binElevationsList
x_values2 = binElevationsList
y_values2 = np.divide(binAreasList, 1000000)  # sq.km area
xLabel1 = 'Bin Mean Mass Balance (mm w.e.)'
yLabel1 = 'Bin Mean Elevation (m)'
xLabel2 = 'Bin Area ($km^2$)'
title1 = 'Binned SMB Plot'
mbPlotsTitle = glacierSame.name + ' Velocity Product Comparison'
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
                             elevationBinWidth, buff, alpha, mbPlotsTitle, glacierSame.res,
                             subplot_ne, subplot_ne_title, subplot_ne_label, color_ne,
                             subplot_sw, subplot_sw_title, subplot_sw_label, color_sw,
                             subplot_se, subplot_se_title, subplot_se_label, color_se,
                             x_subplot_nw_2_err, x_subplot_nw_3_err,
                             x_subplot_nw_ref, x_subplot_nw_ref_lab, y_values1_ref, var_values1_ref, subplot_sw_cbar_ticks)
else:
    x_subplot_nw_4 = binClimChangeSmooth_list[2]
    x_subplot_nw_4_err = climSMBSmoothErr_list[2]
    x_subplot_nw_4_lab = r'$\dot{b}_{clim}$ (' + vel_select[2] + ')'
    x_subplot_nw_2_lab = x_subplot_nw_2_lab[:16] + ' (ITS_LIVE)'
    x_subplot_nw_3_lab = x_subplot_nw_3_lab[:16] + ' (Millan)'
    x_subplot_nw_4_lab = x_subplot_nw_4_lab[:16] + ' (RETREAT)'
    x_subplot_nw_ref_lab = r'$\dot{b}_{clim}$ (ablation stakes)'
    xLabel1 = ''
    yLabel1 = ''
    xLabel2 = ''
    title1 = ''
    elevationBinPlot3data3Subfigs(x_subplot_nw, x_subplot_nw_lab, x_subplot_nw_2, x_subplot_nw_2_lab,
                                  x_subplot_nw_3, x_subplot_nw_3_lab, x_subplot_nw_4, x_subplot_nw_4_lab,
                                  y_values1, x_values2, y_values2, xLabel1, yLabel1, xLabel2, title1,
                                  elevationBinWidth, buff, alpha, mbPlotsTitle, glacierSame.res,
                                  subplot_ne, subplot_ne_title, subplot_ne_label, color_ne,
                                  subplot_sw, subplot_sw_title, subplot_sw_label, color_sw,
                                  subplot_se, subplot_se_title, subplot_se_label, color_se,
                                  x_subplot_nw_2_err, x_subplot_nw_3_err, x_subplot_nw_4_err,
                                  x_subplot_nw_ref, x_subplot_nw_ref_lab, y_values1_ref, var_values1_ref,
                                  subplot_sw_cbar_ticks)

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
extraPlotTitle2 = glacierSame.name + ' Thickness, Elevation, Total MB Plots'
plotData3(fig2_1Vals, fig2_1cbar, fig2_1Color, fig2_1Title,
          fig2_2Vals, fig2_2cbar, fig2_2Color, fig2_2Title,
          fig2_3Vals, fig2_3cbar, fig2_3Color, fig2_3Title, extraPlotTitle2, glacierSame.res,
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
extraPlotTitle3 = glacierSame.name + ' Raw Thickness, Elevation, Total MB Pixel Plots'
plotData3(fig3_1Vals, fig3_1cbar, fig3_1Color, fig3_1Title,
          fig3_2Vals, fig3_2cbar, fig3_2Color, fig3_2Title,
          fig3_3Vals, fig3_3cbar, fig3_3Color, fig3_3Title, extraPlotTitle3, glacierSame.res,
          fig3_2Quiver, fig3_3Quiver)

# plot of stake reference data locations
if glacierSame.data is not None:
    reference_plot_background = z_plot
    reference_plot_background2 = b_tot_plot_pixel
    fig_stakes_title = 'Stake Point Plots'
    plotDataPoints(reference_plot_background[0], reference_plot_background[2], reference_plot_background[3],
                   b_tot_plot_pixel[0], b_tot_plot_pixel[2], b_tot_plot_pixel[3],
                   fig_stakes_title, dataCoords_pixel_col, dataCoords_pixel_row, dataCoords_labels)


# ---------------------------------- GOOD-BAD VELOCITIES AND MEAN, MIN, MAX ------------------------------------
vel_gb_sum = sum(vel_good_bad)      # sum up our good/bad velocity region arrays to get the total estimate and plot
vel_gb_sum[outlineArray == 0] = np.nan  # mask out off-glacier values and assign them to np.nan
class_bins = [0, 1, 2, 3]
class_labels = ['0 good product', '1 good products', '2 good products', '3 good products']
colors = ['#ca0020', '#f4a582', '#bababa', '#404040']
title_gb = glacierSame.name + ' Number of good velocity products' + vel_good_bad_criteria + '\n(out of ' + str(number_of_vel) + ')\n'
plotClassify(vel_gb_sum, class_bins, class_labels, colors, title_gb)
# rasterLike(vel_gb_sum, 'good velocity products' + vel_good_bad_criteria + '.tif', glacierSame.dem)


# find the mean and range of velocities, and plot them
np.seterr(divide='ignore')  # ignore divide by zero error warning
vel_list = []
vel_list_vx = []
vel_list_vy = []
for j in range(number_of_vel):
    smoothvArray_list[j][smoothvArray_list[j] == 0] = np.nan    # remove 'bad' values
    smoothvxArray_list[j][smoothvxArray_list[j] == 0] = np.nan
    smoothvyArray_list[j][smoothvyArray_list[j] == 0] = np.nan
    vel_list.append(smoothvArray_list[j])
    vel_list_vx.append(smoothvxArray_list[j])
    vel_list_vy.append(smoothvyArray_list[j])

vel_mean = np.divide(np.nansum(np.dstack(vel_list), 2), vel_gb_sum)
vel_vx_mean = np.divide(np.nansum(np.dstack(vel_list_vx), 2), vel_gb_sum)
vel_vy_mean = np.divide(np.nansum(np.dstack(vel_list_vy), 2), vel_gb_sum)
vel_min = np.nanmin(np.dstack(vel_list), 2)
vel_max = np.nanmax(np.dstack(vel_list), 2)
vel_range = vel_max - vel_min

# mask out 'bad' values
vel_mean[vel_gb_sum == 0] = -2  # these are the sites with NO good velocity products
vel_mean[vel_gb_sum == 1] = -1  # these are the sites with ONE good velocity product
vel_range[vel_gb_sum == 0] = -2  # these are the sites with NO good velocity products
vel_range[vel_gb_sum == 1] = -1  # these are the sites with ONE good velocity product
vel_mean[outlineArray == 0] = np.nan  # mask out off-glacier values and assign them to np.nan
vel_range[outlineArray == 0] = np.nan  # mask out off-glacier values and assign them to np.nan

meanVel_Quiver = velPlot(vel_vx_mean, vel_vy_mean, vel_mean, area, threshold=1.0)  # plot velocity arrows
mean_vel_range = [0, 30]
title_vel_mean = glacierSame.name + ' Mean Velocity (smooth)'
plotContinuous(vel_mean, mean_vel_range, 'YlGnBu', title_vel_mean, 'Velocity (m/yr)', quiver=meanVel_Quiver)
range_vel_range = [0, 50]
title_vel_range = glacierSame.name + ' Range of Velocities (smooth)'
plotContinuous(vel_range, range_vel_range, 'Reds', title_vel_range, 'Velocity (m/yr)')

# let's create a new good-bad velocity product list based on velocity magnitudes
mag_thresh = 0.5  # threshold for vel magnitude agreement: 0.5 indicates vel products must be within 50% of the mean to be considered 'good'
vel_mean_min = vel_mean * (1 - mag_thresh)
vel_mean_max = vel_mean * (1 + mag_thresh)

vel_good_bad_criteria_mag = '(based on vel magnitude within a ' + str(mag_thresh*100) + '% threshold'
vel_good_bad_mag = smoothvArray_list.copy()
for i in range(len(vel_good_bad_mag)):
    check_max = np.greater(vel_good_bad_mag[i], vel_mean_max)
    check_min = np.greater(vel_mean_min, vel_good_bad_mag[i])
    vel_good_bad_mag[i][(check_max == 1) | (check_min == 1)] = -1
    vel_good_bad_mag[i][vel_good_bad_mag[i] != -1] = 1  # assign our good values to 1
    vel_good_bad_mag[i][vel_good_bad_mag[i] == -1] = 0  # assign our bad values to 0
vel_gb_sum_mag = sum(vel_good_bad_mag)
vel_gb_sum_mag[outlineArray == 0] = np.nan
title_gb_mag = glacierSame.name + ' Number of good velocity products' + vel_good_bad_criteria_mag + '\n(out of ' + str(number_of_vel) + ')\n'
plotClassify(vel_gb_sum_mag, class_bins, class_labels, colors, title_gb_mag)
# show_fig(vel_gb_sum_mag, 'good vels', 'Greys', 'number of goods')
rasterLike(vel_gb_sum_mag, 'good velocity products' + vel_good_bad_criteria_mag + main + '.tif', glacierSame.dem)


# --------------------------------- SAVE DATA AS XLSX -----------------------------------
if number_of_vel == 1:
    xlsx_data = list(itertools.zip_longest(binElevationsList, binAreasList, binTotalChangeList,
                                           binClimChangeSmooth_list[0], [], glacierSame.data[1], glacierSame.data[0]))
    df = pd.DataFrame(xlsx_data, columns=['Elevation Bin Center (m)', 'Area (m2)', 'Total MB (mm w.e.)',
                                          'Climatic MB 1 (mm w.e.)', '', 'Stake elevation', 'Stake MB'])
    df.to_excel('PDFs/' + glacierSame.name + '/' + glacierSame.name + '_mb_data.xlsx')
elif number_of_vel == 2:
    xlsx_data = list(itertools.zip_longest(binElevationsList, binAreasList, binTotalChangeList,
                                           binClimChangeSmooth_list[0],  binClimChangeSmooth_list[1], [],
                                           glacierSame.data[1], glacierSame.data[0]))
    df = pd.DataFrame(xlsx_data, columns=['Elevation Bin Center (m)', 'Area (m2)', 'Total MB (mm w.e.)',
                                          'Climatic MB 1 (mm w.e.)', 'Climatic MB 2 (mm w.e.)', '',
                                          'Stake elevation', 'Stake MB'])
    df.to_excel('PDFs/' + glacierSame.name + '/' + glacierSame.name + '_mb_data.xlsx')
else:
    xlsx_data = list(itertools.zip_longest(binElevationsList, binAreasList, binTotalChangeList,
                                           binClimChangeSmooth_list[0], binClimChangeSmooth_list[1],
                                           binClimChangeSmooth_list[2], [], glacierSame.data[1], glacierSame.data[0]))
    df = pd.DataFrame(xlsx_data, columns=['Elevation Bin Center (m)', 'Area (m2)', 'Total MB (mm w.e.)',
                                          'Climatic MB 1 (mm w.e.)', 'Climatic MB 2 (mm w.e.)',
                                          'Climatic MB 3 (mm w.e.)', '', 'Stake elevation', 'Stake MB'])
    df.to_excel('PDFs/' + glacierSame.name + '/' + glacierSame.name + '_mb_data.xlsx')

# ---------------------------------- CONSTRUCT THE PDF -----------------------------------
if livePlots == False:
    plt.close('all')

# to make and save information as pdf file
if create_pdf == True:
    pdf = FPDF('P', 'mm', 'A4')  # initialize pdf with A4 portrait mode size in mm units
    pdf.add_page()
    pdf.set_font('Times', 'BU', 18)  # set Times font (or 'Arial'), regular text (or 'B', 'I', 'U'), size 12
    epw = pdf.w - 2 * pdf.l_margin  # effective page width
    # pdf.cell(w, h=0, txt='', border=0, ln=0, align='', fill=False, link='')
    pdf.cell(epw, 10, glacierSame.name + ' Glacier Info', border=0, ln=1, align='C')

    col_width = epw / 5  # Set column width to 1/2 of effective page width to distribute content
    data = [['DEM file', glacierSame.dem + '(' + dem_select[0] + ')'],
            ['Change in thickness file', glacierSame.dhdt + ' (Hugonnet, ' + time_span + ')'],
            ['Thickness file', glacierSame.h + ' (' + thick_select[0] + ')'],
            ['Velocity file(s)', vel_select[:number_of_vel]]]
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
            [glacierSame.res, glacierSame.rho, glacierSame.vCol, glacierSame.bin, glacierSame.crs]]
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

    pdf.ln(3)
    pdf.set_font('Times', '', 10)
    pdf.cell(epw, 5, 'The following calculations were conducted with velocity products that are based on "' +
             filtered_vel[0] + '" data:')

    pdf.set_font('Times', '', 8)
    pdf.ln(3.5)
    pdf.cell(epw, 5, '    "Aspect-Corrected" removes velocities that oppose the aspect beyond a threshold')
    pdf.ln(3.5)
    pdf.cell(epw, 5, '    "Aspect-Corrected-Removed" also removes Aspect-Corrected values from binned analysis')
    pdf.ln(3.5)
    pdf.cell(epw, 5, '    "Gulkana-Correction" is an attempt to replace regions with bad vels on Gulkana')

    pdf.ln(2 * th)
    pdf.set_font('Times', 'B', 12)
    pdf.cell(epw, 5, 'Calculation Results', ln=1, align='C')
    pdf.ln(0.5)
    col_width = epw / 2
    MB_values_to_report = []
    for j in range(number_of_vel):
        MB_values_to_report.append(['Climatic MB: ' + vel_select[j],
                                    str(round(climMassBalanceValueSmooth_list[j], 2)) + ' (mm w.e.)'])

    data = [['Area', str(round(area / 1000000, 2)) + ' (sq. km)'],
            ['Total MB', str(round(totalMassBalanceVal, 2)) + ' (mm w.e.)'],
            ['Total MB, alt-resolved', str(round(altResTotMassBalance, 2)) + ' (mm w.e.)'],
            [MB_values_to_report[0][0], MB_values_to_report[0][1]]]
    if number_of_vel >= 2:
        data.append([MB_values_to_report[1][0], MB_values_to_report[1][1]])
        if number_of_vel == 3:
            data.append([MB_values_to_report[2][0], MB_values_to_report[2][1]])

    data.append(['Scatter plot error bar value', error_name])
    # climMassBalanceValueSmooth_list[0]

    pdf.set_font('Times', '', 9)
    for row in data:
        for datum in row:
            pdf.cell(col_width, th, str(datum), border=1)
        pdf.ln(th)  # 2 * th is for double padding
    pdf.set_font('Times', '', 8)
    pdf.cell(epw, 5, '**Climatic MB from emergence method does not use smoothed data**')

    if glacierSame.data is not None:
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
                                       str(glacierSame.res * app4_pixel_window_avg) + ' (m)'])

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
    for j in range(number_of_vel):
        pdf.ln(2 * th)
        fig4 = smoothedVPlotTitle_list[j].replace(' ', '_') + '.png'
        pdf.image('Figures/' + fig4, x=None, y=None, w=epw)
        # os.remove('Figures/' + fig4) #1234
    for j in range(number_of_vel):
        pdf.ln(2 * th)
        fig5 = uncertaintyVPlotTitle_list[j].replace(' ', '_') + '.png'
        pdf.image('Figures/' + fig5, x=None, y=None, w=epw)
        os.remove('Figures/' + fig5)
    if glacierSame.data is not None:
        pdf.ln(2 * th)
        fig_stakes = fig_stakes_title.replace(' ', '_') + '.png'
        pdf.image('Figures/' + fig_stakes, x=None, y=None, w=epw)
        os.remove('Figures/' + fig_stakes)

    pdf.ln(2 * th)      # add slope map to pdf
    fig_slope = slope_map_title.replace(' ', '_') + '.png'
    pdf.image('Figures/' + fig_slope, x=None, y=None, w=epw)
    os.remove('Figures/' + fig_slope)
    pdf.ln(2 * th)      # add aspect map to pdf
    fig_aspect = aspect_map_title.replace(' ', '_') + '.png'
    pdf.image('Figures/' + fig_aspect, x=None, y=None, w=epw)
    os.remove('Figures/' + fig_aspect)
    pdf.ln(2 * th)  # add hillshade map to pdf
    fig_hillshade = hillshade_map_title.replace(' ', '_') + '.png'
    pdf.image('Figures/' + fig_hillshade, x=None, y=None, w=epw)
    os.remove('Figures/' + fig_hillshade)
    pdf.ln(2 * th)  # add goob/bad velocity map to pdf
    fig_gb_vel = title_gb.replace(' ', '_') + '.png'
    pdf.image('Figures/' + fig_gb_vel, x=None, y=None, w=epw/2)
    # os.remove('Figures/' + fig_gb_vel)
    fig_gb_vel_mag = title_gb_mag.replace(' ', '_') + '.png'
    pdf.image('Figures/' + fig_gb_vel_mag, x=None, y=None, w=epw/2)
    # os.remove('Figures/' + fig_gb_vel_mag)

    # add 1 to 1 scatterplots to pdf
    pdf.add_page()  # add new pdf page
    pdf.set_y(10)
    y_ordinate = pdf.get_y()
    fig_1to1_cmb = one_to_one_cmb_scatterplot_title.replace(' ', '_') + '.png'
    pdf.image('Figures/' + fig_1to1_cmb, x=10, y=y_ordinate, w=epw)
    os.remove('Figures/' + fig_1to1_cmb)

    pdf.add_page()
    pdf.set_y(10)
    y_ordinate = pdf.get_y()
    fig_1to1_divq = one_to_one_divq_scatterplot_title.replace(' ', '_') + '.png'
    pdf.image('Figures/' + fig_1to1_divq, x=10, y=y_ordinate, w=epw)
    os.remove('Figures/' + fig_1to1_divq)

    pdf.add_page()          # add new pdf page
    pdf.set_y(10)
    y_ordinate = pdf.get_y()
    for j in range(number_of_vel):  # add aspect-corrected velocity plots to pdf
        fig6 = aspect_cor_title1_list[j].replace(' ', '_') + '.png'
        pdf.image('Figures/' + fig6, x=j*epw/3, y=y_ordinate, w=epw/3)
        os.remove('Figures/' + fig6)

    # pdf.ln(epw/3)
    # y_ordinate = pdf.get_y()        # ordinate value of the top left corner of the image
    # for j in range(number_of_vel):  # add aspect-directed velocity plots to pdf
    #     fig7 = aspect_cor_title2_list[j].replace(' ', '_') + '.png'
    #     pdf.image('Figures/' + fig7, x=j*epw/3, y=y_ordinate, w=epw/3)
    #     os.remove('Figures/' + fig7)

    # pdf.ln(epw/3)
    # y_ordinate = pdf.get_y()        # ordinate value of the top left corner of the image
    # for i in range(number_of_vel):  # add aspect-directed velocity plots to pdf
    #     fig7 = vel_dict.get('aspect_cor_title2%s' % i).replace(' ', '_') + '.png'
    #     pdf.image('Figures/' + fig7, x=i*epw/3, y=y_ordinate, w=epw/3)
    #     os.remove('Figures/' + fig7)
    #
    # pdf.ln(epw/3)
    # y_ordinate = pdf.get_y()
    # for i in range(number_of_vel):  # add aspect-directed velocity plots only for slope threshold to pdf
    #     fig8 = vel_dict.get('aspect_cor_title3%s' % i).replace(' ', '_') + '.png'
    #     pdf.image('Figures/' + fig8, x=i*epw/3, y=y_ordinate, w=epw/3)
    #     os.remove('Figures/' + fig8)
    #
    # pdf.ln(epw / 3)
    # y_ordinate = pdf.get_y()
    # for i in range(number_of_vel):  # add weighted avg directed velocity plots to pdf
    #     fig9 = vel_dict.get('aspect_cor_title4%s' % i).replace(' ', '_') + '.png'
    #     pdf.image('Figures/' + fig9, x=i * epw / 3, y=y_ordinate, w=epw / 3)
    #     os.remove('Figures/' + fig9)

    if filtered_vel[0] == 'Original (smoothed)':
        pdfName = glacierSame.name + main + '_' + time_span + 'dhdt_' + shp_select[0] + 'shp_' + dem_select[0] + '_' + \
                  thick_select[0] + '_' + str(raw_data_smooth_factor) + str(divQ_smooth_factor) + 'smoothing_' + \
                  str(vel_select[:number_of_vel]) + '.pdf'
    else:
        pdfName = glacierSame.name + main + '_' + time_span + 'dhdt_' + shp_select[0] + 'shp_' + dem_select[0] + '_' + \
                  thick_select[0] + '_' + str(raw_data_smooth_factor) + str(divQ_smooth_factor) + 'smoothing_' + \
                  str(vel_select[:number_of_vel]) + filtered_vel[0] + '.pdf'
    pdf.output('PDFs/' + glacierSame.name + '/' + pdfName, 'F')

# ------------------------- SAVE VELOCITY OUTPUTS ----------
saveVel = 0
if saveVel == True:
    for j in range(number_of_vel):
        dest_vx1 = 'Saved Velocity Outputs/' + glacierSame.name + '_' + vel_select[j] + '_vx.tif'
        dest_vy1 = 'Saved Velocity Outputs/' + glacierSame.name + '_' + vel_select[j] + '_vy.tif'
        rasterLike(vel_x_list[j], dest_vx1, glacierVels[j].vx)
        rasterLike(vel_y_list[j], dest_vy1, glacierVels[j].vy)
        dest_vx2 = 'Saved Velocity Outputs/' + glacierSame.name + '_' + vel_select[j] + '_vx-aspect-corrected.tif'
        dest_vy2 = 'Saved Velocity Outputs/' + glacierSame.name + '_' + vel_select[j] + '_vy-aspect-corrected.tif'
        rasterLike(vel_x_cor1_list[j], dest_vx2, glacierVels[j].vx)
        rasterLike(vel_y_cor1_list[j], dest_vy2, glacierVels[j].vy)


# ---------------------------------REMOVE SAVED FILES---------------------------#
os.remove(glacierSame.dem)
os.remove(glacierSame.h)
os.remove(glacierSame.dhdt)
[os.remove(glacierVels[j].vx) for j in range(number_of_vel)]
[os.remove(glacierVels[j].vy) for j in range(number_of_vel)]

try:
    os.remove('RGI60-01.' + glacierSame.name + '.shp')
    os.remove('RGI60-01.' + glacierSame.name + '.shx')
    os.remove('RGI60-01.' + glacierSame.name + '.dbf')
    os.remove('RGI60-01.' + glacierSame.name + '.prj')
    os.remove('RGI60-01.' + glacierSame.name + '.cpg')
except:
    pass
try:
    os.remove('RGI60-01.' + glacierSame.name + '-main.shp')
    os.remove('RGI60-01.' + glacierSame.name + '-main.shx')
    os.remove('RGI60-01.' + glacierSame.name + '-main.dbf')
    os.remove('RGI60-01.' + glacierSame.name + '-main.prj')
    os.remove('RGI60-01.' + glacierSame.name + '-main.cpg')
except:
    pass

for tempfile in glob.glob('temps.*'):   # remove the 'temp.*' shapefiles
    os.remove(tempfile)
os.remove(glacierSame.name + 'TotalMassBalance.tif')
os.remove(glacierSame.name + 'Outline.tif')
os.remove('ones_raster.tif')
if os.path.exists(glacierSame.name + '_array_temp.tif'):
    os.remove(glacierSame.name + '_array_temp.tif')
for j in range(number_of_vel):
    if os.path.exists('vx' + str(j+1) + '_err.tif'):
        os.remove('vx' + str(j+1) + '_err.tif')
        os.remove('vy' + str(j+1) + '_err.tif')

# to show all the figures from the calculation (not saved .png files) -- make sure livePlots is set to 1 or True
plt.show()

