# glacier-mass-balance
A framework to calculate and plot various metrics regarding glacier mass balance

# Overview
The following .py files are used to conduct a total mass balance and climate mass balance for any given glacier. Additionally, an altitudinally-resolved method for estimating mass balance is show. Together, these files are used to apply appropriate raster file geoprocessing and calculations. The last two files in this list are used to aid in raster file smoothing/data filtering and in effectively plotting solutions.

## 1. main_vel_comparison.py
This is the primary file and driver of the MB calculation

* Initialized glaciers: selecting calculation parameters and defining appropriate raster files
  - Choose any glacier RGI Region 1 or 2 -- contact me and I can share dataset files / folders
  - Use glacierDatabase.py to define files for each glacier
  - GUI on main_vel_comparison.py will draw from regional files or glacierDatabase.py for specific glacier files and features
* Calling methods to manipulate and calculate based on selected glacier
* Compare results using different velocity products (optional)
* Visualization and calculation options are offered throughout to show or check certain files and processes. For example:
* Images are saved into a '/Figures' folder and PDFs containing all relevant information are saved into a '/PDFs' folder
  - These folders are created to store saved files if they do not exist


## 2. glacierClass.py
This code defines the glacier class
* Two types of glaciers are defined, a parent class 'glacier' and a child class named 'fullGlacier'
* The only difference between these classes are that 'fullGlacier' classes contain velocity data, and can therefore be used for a climatic mass balance calculation in addition to total mass balance calculations
* Methods include:
  - Coordinate system *reprojections*
  - Raster pixel size resolution *resampling*
  - Raster file extent *clipping*
  - Raster file *clipping* with a shapefile
  - Raster file *hole-filling*
  - Raster file *smoothing*
  - Pixel-by-pixel *total mass balance calculation*
  - Pixel-by-pixel *climatic mass balance calculation*
  - Altitude aggregation into elevation bins for *altitudinally-resolved total or climatic mass balance*


## 3. smoothingFunctions.py
This code contains two smoothing functions for raster files / arrays:
  - A 2-D Savitzky-Golay moving window smoothing filter with specified pixel / array window size and fitted function order
  - A Gaussian filter with specified standard deviation
  - A Gaussian filter with a moving window size for each pixel, based on a value from another file (e.g. ice thickness)


## 4. dataPlots.py
This code is simply for producing data plots
* The code plots mass balance per elevation bin, with elevation bin area plotted on the left
NOTE: Plots are saved into a local '/Figures' folder. Add this folder or remove this section of the code to run

## 5. emergenceFunction.py
This code is for the emergence calculation at each pixel
This is taken and adapted from the code written by David Rounce found on Github at drounce/Rounce2018JGR

## 6. velocityPlots.py
Basic code for plotting the velocity of a glacier without running other calculations. This file works independently from the others
