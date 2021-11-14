'''
awwells, 12780, final project code
All the necessary modules to import are listed at the top of this document.
The necessary raster files are in this folder
No other files or code is required to run this project
'''
#Import necessary modules
from glacierClass import glacier, fullGlacier
from dataPlots import elevationBinPlot
import numpy as np
import os
import rasterio.plot
from rasterio.plot import show
from functions import sgolay2d, gaussianFilter

#------------------------------------------------------GENERAL---------------------------------------------------------#
res = (20, 20)            #Rsampling resultion (x, y) tuple
avgDensity = 850            #Average glacier mass density (kg/m3)
elevationBinWidth = 50      #Elevation bin width, for elevation binning calculations (m)
vCol = 0.8                  #surface column average velocity scaling

#------------------------------------------------------GULKANA---------------------------------------------------------#
name = 'Gulkana'            #Glacier name (folder should be the same as glacier name)
type = 'Climatic'           #Mass balance type
demFile = 'RGI_GulkanaRegion_DEM.tif'
hFile = 'GulkanaRegion_thickness.tif'
#hFile = 'Gulkana_Thickness_model1.tif'
dhdtFile = 'dhdt_GulkanaRegion.tif'
vxFile = 'GulkanaRegion_vx.tif'
vyFile = 'GulkanaRegion_vy.tif'
crs = 'EPSG:32606'           #UTM Zone 6 North. Alaska Albers coordinate system: ('EPSG: 3338')

#(glacierName, massBalanceType, demFile, thicknessFile, changeInThicknessFile,
    #coordinateSystem, resolution, density, elevationBinWidth, vxFile, vyFile, velocityColumnAvgScaling)
gulkana = fullGlacier(name, type, demFile, hFile, dhdtFile,
                      crs, res, avgDensity, elevationBinWidth, vxFile, vyFile, vCol)
#-----------------------------------------------------WOLVERINE--------------------------------------------------------#
name = 'Wolverine'              #Glacier name (folder should be the same as glacier name)
type = 'Climatic'               #Mass balance type
demFile = 'DEM_Wolverine.tif'
hFile = 'Wolverine_Thickness.tif'
dhdtFile = 'dhdt_Wolverine.tif'
vxFile = 'WolverineGlacier_vx.tif'
vyFile = 'WolverineGlacier_vy.tif'
crs = 'EPSG:32606'           #UTM Zone 6 North.
wolverine = fullGlacier(name, type, demFile, hFile, dhdtFile,
                      crs, res, avgDensity, elevationBinWidth, vxFile, vyFile, vCol)
#-----------------------------------------------------LEMON CREEK------------------------------------------------------#
name = 'LemonCreek'            #Glacier name (folder should be the same as glacier name)
type = 'Climatic'           #Mass balance type
demFile = 'DEM_LemonCreek.tif'
hFile = 'LemonCreek_Thickness.tif'
dhdtFile = 'dhdt_LemonCreek.tif'
vxFile = 'LemonCreek_vx.tif'
vyFile = 'LemonCreek_vy.tif'
crs = 'EPSG:32608'           #UTM Zone 8 North
lemonCreek = fullGlacier(name, type, demFile, hFile, dhdtFile,
                      crs, res, avgDensity, elevationBinWidth, vxFile, vyFile, vCol)
#-------------------------------------------------------COLUMBIA-------------------------------------------------------#
name = 'Columbia'            #Glacier name (folder should be the same as glacier name)
type = 'Climatic'           #Mass balance type
demFile = 'DEM_Columbia.tif'
hFile = 'Columbia_Thickness.tif'
dhdtFile = 'dhdt_Columbia.tif'
vxFile = 'Columbia_vx.tif'
vyFile = 'Columbia_vy.tif'
crs = 'EPSG:32606'           #UTM Zone 6 North
columbia = fullGlacier(name, type, demFile, hFile, dhdtFile,
                      crs, res, avgDensity, elevationBinWidth, vxFile, vyFile, vCol)
#--------------------------------------------------------RAINBOW-------------------------------------------------------#
name = 'Rainbow'            #Glacier name (folder should be the same as glacier name)
type = 'Climatic'           #Mass balance type
demFile = 'DEM_Rainbow.tif'
hFile = 'Rainbow_Thickness.tif'
dhdtFile = 'dhdt_Rainbow.tif'
vxFile = 'Rainbow_vx.tif'
vyFile = 'Rainbow_vy.tif'
crs = 'EPSG:32610'           #UTM Zone 10 North
rainbow = fullGlacier(name, type, demFile, hFile, dhdtFile,
                      crs, res, avgDensity, elevationBinWidth, vxFile, vyFile, vCol)
#-----------------------------------------------------SOUTH CASCADE----------------------------------------------------#
name = 'SouthCascade'            #Glacier name (folder should be the same as glacier name)
type = 'Climatic'           #Mass balance type
demFile = 'DEM_SouthCascade.tif'
hFile = 'SouthCascade_Thickness.tif'
dhdtFile = 'dhdt_SouthCascade.tif'
vxFile = 'SouthCascade_vx.tif'
vyFile = 'SouthCascade_vy.tif'
crs = 'EPSG:32610'           #UTM Zone 10 North.
southCascade = fullGlacier(name, type, demFile, hFile, dhdtFile,
                      crs, res, avgDensity, elevationBinWidth, vxFile, vyFile, vCol)
#------------------------------------------------------RIKHA SAMBA-----------------------------------------------------#
name = 'RikhaSamba'              #Glacier name (folder should be the same as glacier name)
type = 'Climatic'                #Mass balance type
demFile = 'DEM_RikhaSamba.tif'
hFile = 'RikhaSamba_Thickness.tif'
dhdtFile = 'dhdt_RikhaSamba.tif'
vxFile = 'RikhaSamba_vx.tif'
vyFile = 'RikhaSamba_vy.tif'
crs = 'EPSG:32644'           #UTM Zone 44 North.
rikhaSamba = fullGlacier(name, type, demFile, hFile, dhdtFile,
                      crs, res, avgDensity, elevationBinWidth, vxFile, vyFile, vCol)
#----------------------------------------------------------------------------------------------------------------------#
glacierCalc = rikhaSamba                              #chose glacier to perform calculations

path = os.getcwd() + '/' + glacierCalc.name         #navigate to working directory
os.chdir(path)

#STEP 1: CONVERT COORDINATE SYSTEMS AND RESAMPLE
glacierCalc.tifReprojectionResample()

#STEP 2: CLIP RASTERS TO SAME EXTENT
glacierCalc.tifClip()

#QUICK DATA FILTER: FILL HOLES IN ANY FILES (DEM, H, or DHDT)
glacierCalc.fillHole()

#CHECK TO SEE IF THINGS WORKED
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
#-------------------------------------------PART 3 DISPLAY OPTIONS-----------------------------------------#
showTotalMassBalanceValue = 1                   #to print the total mass balance value
showClimMassBalanceValue = 1                    #to print the climatic mass balance value
#----------------------------------------------------------------------------------------------------------#
#STEP 3: MANIPULATE AND COMBINE DATA TO FULFILL CONTINUITY EQUATION
glacierCalc.glacierOutline()

glacierCalc.totalMassBalance()
glacierCalc.climMassBalance()

area = glacierCalc.glacierArea()
print('Area (sq. km):', area/1000000)

totalMassBalanceValue = glacierCalc.totalMassBalanceValue()
climMassBalanceValue = glacierCalc.climMassBalanceValue()

if showTotalMassBalanceValue == True:
    exp = 'The annual TOTAL mass balance for ' + glacierCalc.name + ' Glacier is: '
    unit = '(mm w.e.)'
    print(exp, totalMassBalanceValue, unit) #units of kg/m2-yr is also just mm w.e.

if showClimMassBalanceValue == True:
    exp = 'The annual CLIMATIC mass balance for ' + glacierCalc.name + ' Glacier is: '
    unit = '(mm w.e.)'
    print(exp, climMassBalanceValue, unit) #units of kg/m2-yr is also just mm w.e.

#-------------------------------------------PART 4 DISPLAY OPTIONS-----------------------------------------#
use_smooth_vel = 1                          #1 to use smooth velocity, 0 to use raw velocity

showSmoothFilter = 0                        #to show the difference in raw data and smoothed data
showSmoothClimMassBalance = 0               #to show the difference in climatic mass balance using vel vs smoothed vel
#----------------------------------------------------------------------------------------------------------#
#STEP 4: SMOOTH VELOCITY DATA WITH LOW PASS MOVING WINDOW FILTER, CLIMATIC MASS BALANCE FOR SMOOTH VELOCITY DATA
glacierCalc.velocitySmoothing(windowSize=5, polyOrder=1, std=2, smoothFilter='golay') #smoothFiler='golay', otherwise 'gauss'
glacierCalc.climMassBalance(smoothVel=use_smooth_vel)

#show(rasterio.open(glacierCalc.vx[:-4]+'_smooth.tif'))  #smooth velocity_x file
if showSmoothFilter == True:
    show(rasterio.open(glacierCalc.h).read(1), title='original thickness file')
    show(sgolay2d(rasterio.open(glacierCalc.h).read(1), 3, 1), title='savitzky-golay filter: thickness')
    show(gaussianFilter(rasterio.open(glacierCalc.h).read(1), 2), title='gaussian filter: thickness')

if showSmoothClimMassBalance == True:
    show(rasterio.open(glacierCalc.name+'ClimaticMassBalance.tif').read(1), title='Climatic Mass Balance Result')
    show(rasterio.open(glacierCalc.name+'ClimaticMassBalanceSmooth.tif').read(1),
         title='Smoothed-velocity Climatic Mass Balance Result')

#-------------------------------------PART 5 INPUTS AND DISPLAY OPTIONS------------------------------------#
showMassLossElevationClimatic = 1       #1 to show CLIMATIC mass balance at each elevation, 0 for TOTAL
showAltResMassLoss = 1                  #to print the altitudinally-resolved mass loss
showMassLossElevation = 1               #to show plot of the mean mass balance contributions per elevation bin
#----------------------------------------------------------------------------------------------------------#
#STEP 5: ALTITUDINALLY RESOLVE MASS BALANCE
#run function to estimate the altitudinally resolved mass balance. this function actually returns a few things:
#total altitudinally resolved mass, mean mass at each elevation bin, elevation bin boundaries, and elevation bin counts
if showMassLossElevationClimatic == True:
    altResMass, binMassMean, binElevations, binCounts = glacierCalc.altitudeAggregation(smoothVel=use_smooth_vel)
    if use_smooth_vel == True:
        balanceType = 'CLIMATIC (SMOOTH)'
    else:
        balanceType = 'CLIMATIC'
    # change mass balance file in altitudeAggregration to change this back to original un-smoothed clim mass balance
else:
    altResMass, binMassMean, binElevations, binCounts = glacierCalc.altitudeAggregation(balType='total')
    balanceType = 'TOTAL'

altResMassBalance = altResMass/area                       #divide by total glacier area for alt-resolved mass balance

if showAltResMassLoss == True:
    #to print the values for altitudinally resolved mass balance (kg/m2-yr or mm w.e.)
    print('The altitudinally resolved', balanceType, 'mass balance is:', altResMassBalance, '(mm w.e.)')
    print('Elevation bin size:', elevationBinWidth, 'm')

if showMassLossElevation == True:
    #to show a scatter plot of elevation vs mean mass loss at each elevation bin
    binElevationsList = binElevations.tolist()
    for i in range(len(binElevationsList)):
        binElevationsList[i] = binElevationsList[i] + (elevationBinWidth*0.5)
    binMassMeanList = binMassMean.tolist()
    binAreasList = (binCounts*glacierCalc.res[0]*glacierCalc.res[1]).tolist()

    x_values1 = binMassMeanList
    y_values1 = binElevationsList[0:len(binElevationsList)-1]
    x_values2 = binElevationsList[0:len(binElevationsList)-1]
    y_values2 = binAreasList
    xLabel1 = 'Bin Mean Mass Balance (mm w.e.)'
    yLabel1 = 'Bin Mean Elevation (m)'
    xLabel2 = 'Bin Area ($m^2$)'
    title = 'Glacier mean mass balance (' + balanceType + ') per elevation bin'
    min = 'default'
    buff = 100
    alpha = 0.3

    elevationBinPlot(x_values1, y_values1, x_values2, y_values2, xLabel1, yLabel1, xLabel2, title,
                     elevationBinWidth, min, buff, alpha)

#-------------------------------------------PART 6 DISPLAY OPTIONS-----------------------------------------#
#show(rasterio.open(glacierCalc.vx[:-4]+'_clip.tif').read(1)*rasterio.open(glacierCalc.name+'Outline.tif').read(1)) #show v raster
#show(rasterio.open(glacierCalc.vy[:-4]+'_clip.tif').read(1)*rasterio.open(glacierCalc.name+'Outline.tif').read(1))

showVelocityElevation = 0                   #to show plot of mean bin velocity magnitude per elevation bin
showThicknessElevation = 0                  #to show plot of mean bin ice thickness per elevation bin
showQElevation = 0                          #to show plot of mean bin Q per elevation bin
showMassLossElevationBinQ = 0               #to show plot of the mean mass balance contributions per elevation bin
#----------------------------------------------------------------------------------------------------------#
#STEP 6: MEAN VELOCITY AND ICE THICKNESS PER ELEVATION BIN
#THEN RECALCULATE FLUX DIVERGENCE AT EACH ELEVATION BIN
#ALTITUDINALLY RESOLVE MASS BALANCE USING BIN MEAN DIV Q

#run function to estimate the altitudinally resolved velocity and ice thickness. this function returns a few things:
#mean velocity at each elevation bin, elevation bin boundaries for velocity, and elevation bin counts for velocity
#mean ice thickness at each elevation bin, elevation bin boundaries for thickness, and elevation bin counts for thickness
binMeanVel, binVelBoundaries, binVelCount, binMeanh, binhBoundaries, binhCount = glacierCalc.binMeanVelThickness()

binMeanq = binMeanVel*binMeanh
binMeanqList = binMeanq.tolist()
binDivQ = np.gradient(binMeanqList)

#we use total mass balance, altitudinally resolve it, and then account for binDivQ after
#run function to estimate the altitudinally resolved mass balance. this function actually returns a few things:
#total altitudinally resolved mass, mean mass at each elevation bin, elevation bin boundaries, and elevation bin counts
altResMass, binMassMean, binElevations, binCounts = glacierCalc.altitudeAggregation(balType='total')

if showVelocityElevation == True:
    #to show a scatter plot of elevation vs mean mass loss at each elevation bin
    binVelBoundariesList = binVelBoundaries.tolist()
    for i in range(len(binVelBoundariesList)):
        binVelBoundariesList[i] = binVelBoundariesList[i] + (elevationBinWidth*0.5)
    binMeanVelList = binMeanVel.tolist()
    binVelAreasList = (binVelCount*glacierCalc.res[0]*glacierCalc.res[1]).tolist()

    x_values1 = binMeanVelList
    y_values1 = binVelBoundariesList[0:len(binVelBoundariesList)-1]
    x_values2 = binVelBoundariesList[0:len(binVelBoundariesList)-1]
    y_values2 = binVelAreasList
    xLabel1 = 'Bin Mean Velocity Magnitude (m/yr)'
    yLabel1 = 'Bin Mean Elevation (m)'
    xLabel2 = 'Bin Area ($m^2$)'
    title = 'Glacier mean velocity magnitude per elevation bin'
    min = 0
    buff = 0.5
    alpha = 0.3

    elevationBinPlot(x_values1, y_values1, x_values2, y_values2, xLabel1, yLabel1, xLabel2, title,
                     elevationBinWidth, min, buff, alpha)

if showThicknessElevation == True:
    #to show a scatter plot of elevation vs mean mass loss at each elevation bin
    binhBoundariesList = binhBoundaries.tolist()
    for i in range(len(binhBoundariesList)):
        binhBoundariesList[i] = binhBoundariesList[i] + (elevationBinWidth*0.5)
    binMeanhList = binMeanh.tolist()
    binhAreasList = (binhCount*glacierCalc.res[0]*glacierCalc.res[1]).tolist()

    x_values1 = binMeanhList
    y_values1 = binhBoundariesList[0:len(binhBoundariesList)-1]
    x_values2 = binhBoundariesList[0:len(binhBoundariesList)-1]
    y_values2 = binhAreasList
    xLabel1 = 'Bin Mean Ice Thickness (m)'
    yLabel1 = 'Bin Mean Elevation (m)'
    xLabel2 = 'Bin Area ($m^2$)'
    title = 'Glacier mean ice thickness per elevation bin'
    min = 0
    buff = 10
    alpha = 0.3

    elevationBinPlot(x_values1, y_values1, x_values2, y_values2, xLabel1, yLabel1, xLabel2, title,
                     elevationBinWidth, min, buff, alpha)

if showQElevation == True:
    #to show a scatter plot of elevation vs mean mass loss at each elevation bin
    binVelBoundariesList = binVelBoundaries.tolist()
    for i in range(len(binVelBoundariesList)):
        binVelBoundariesList[i] = binVelBoundariesList[i] + (elevationBinWidth*0.5)
    binMeanQList = binMeanq.tolist()
    binQAreasList = (binVelCount*glacierCalc.res[0]*glacierCalc.res[1]).tolist()

    x_values1 = binMeanQList
    y_values1 = binVelBoundariesList[0:len(binVelBoundariesList)-1]
    x_values2 = binVelBoundariesList[0:len(binVelBoundariesList)-1]
    y_values2 = binQAreasList
    xLabel1 = 'Bin Mean Q ($m^2/yr$)'
    yLabel1 = 'Bin Mean Elevation (m)'
    xLabel2 = 'Bin Area ($m^2$)'
    title = 'Glacier mean Q per elevation bin'
    min = 0
    buff = 0.5
    alpha = 0.3

    elevationBinPlot(x_values1, y_values1, x_values2, y_values2, xLabel1, yLabel1, xLabel2, title,
                     elevationBinWidth, min, buff, alpha)

if showMassLossElevationBinQ == True:
    #to show a scatter plot of elevation vs mean mass loss at each elevation bin
    binElevationsList = binElevations.tolist()
    for i in range(len(binElevationsList)):
        binElevationsList[i] = binElevationsList[i] + (elevationBinWidth*0.5)
    binMassMean = (binMassMean + binDivQ)
    binMassMeanList = binMassMean.tolist()
    binAreas = (binCounts*glacierCalc.res[0]*glacierCalc.res[1])
    binAreasList = binAreas.tolist()

    altResMassBalanceBinQ = np.sum(binAreas*binMassMean)/area
    print('The altitudinally resolved climatic mass balance (using altitudinally-resolved Q and 1D '
          'flux divergence) is:', altResMassBalanceBinQ, '(mm w.e.)')

    x_values1 = binMassMeanList
    y_values1 = binElevationsList[0:len(binElevationsList)-1]
    x_values2 = binElevationsList[0:len(binElevationsList)-1]
    y_values2 = binAreasList
    xLabel1 = 'Bin Mean Mass Balance (mm w.e.)'
    yLabel1 = 'Bin Mean Elevation (m)'
    xLabel2 = 'Bin Area ($m^2$)'
    title = 'Glacier mean climatic mass balance per elevation bin (from altitudinally-resolved q)'
    min = 'default'
    buff = 100
    alpha = 0.3

    elevationBinPlot(x_values1, y_values1, x_values2, y_values2, xLabel1, yLabel1, xLabel2, title,
                     elevationBinWidth, min, buff, alpha)

