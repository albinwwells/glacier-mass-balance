# glacier-mass-balance

# Overview
The folling .py files are used to conduct a total mass balance and climate mass balance for any given glacier. Additionally, an altitudinally-resolved method for estimating mass balance is show. Together, these files are used to apply appropriate raster file geoprocessing and calculations. The last two files in this list are used to aid in raster file smoothing/data filtering and in effectively plotting solutions.

## 1. massBalanceOOP.py
This is the primary file and driver

* Initialized glaciers: selecting calculation parameters and defining appropriate raster files
* Calling methods to manipulate and calculate based on selected glacier
* Visualization and calculation options are offered throughout to show or check certain files and processes. For example:
  - Decisions to show or skip various raster files and calculated values
  - Decisions on using smoothed raster files vs raw data (and decision on smoothing filter type and parameters)
  - Decision on altitudinally resolving using total mass balance or climatic mass balance (it does not do both at once)


## 2. glacierClass.py
This code defines the glacier class
* Two types of glaciers are defined, a parent class 'glacier' and a child class named 'fullGlacier'
* The only difference between these classes are that 'fullGlacier' classes contain velocity data, and can therefore be used for a climatic mass balance calculation in addition to total mass balance calculations
* Methods include:
  - Coordinate system *reprojections*
  - Raster pixel size resolution *resampling*
  - Raster file extent *clipping*
  - Raster file *hole-filling*
  - Raster file *smoothing*
  - Pixel-by-pixel *total mass balance calculation*
  - Pixel-by-pixel *climatic mass balance calculation*
  - Altitude aggregation into elevation bins for *altitudinally-resolved total or climatic mass balance*


## 3. smoothingFunctions.py
This code contains two smoothing functions for raster files / arrays:
  - A 2-D Savitzky-Golay moving window smoothing filter with specified pixel / array window size and fitted function order
  - A Gaussian filter with specified standard deviation


## 4. dataPlots.py
This code is simply for producing data plots
* The code plots mass balance per elevation bin, with elevation bin area plotted on the left
NOTE: Plots are saved into a local '/Figures' folder. Add this folder or remove this section of the code to run
