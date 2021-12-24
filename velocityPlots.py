# For plotting velocity files

from smoothingFunctions import sgolay2d, gaussianFilter, dynamicSmoothing
import numpy as np
import os
import rasterio
import rasterio.plot
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm

# -----------------------------------------------------GENERAL---------------------------------------------------------#
res = 20                    # Resampling resolution (x, y) tuple
glacier = 'eklutna'
# -------------------------------------------------------EKLUTNA-------------------------------------------------------#
if glacier == 'eklutna':
    name = 'Eklutna'                # Glacier name (folder should be the same as glacier name)
    type = 'vxvy'                   # Velocity 'type', either 'vxvy' or 'v'
    vxFile = 'Eklutna_vy.tif'
    cor_vx = -1
    # vxFile = 'Eklutna_vx_Millan.tif'
    vyFile = 'Eklutna_vx.tif'
    cor_vy = 1
    # vyFile = 'Eklutna_vy_Millan.tif'
    crs = 'EPSG:32606'           # UTM Zone 6 North

# ---------------------------------------------------------------------------------------------------------------------#
path = os.getcwd() + '/' + name         # navigate to working directory
os.chdir(path)

def velReprojectionResample(vel, crs, res):
    with rasterio.open(vel) as src:
        transform, width, height = calculate_default_transform(
            src.crs, crs, src.width, src.height, *src.bounds, resolution=res)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        v = np.ma.zeros((height, width), dtype=float)
        reproject(
            source=rasterio.band(src, 1),
            destination=v,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.cubic_spline)
    return v


if type == 'vxvy':
    # STEP 1: CONVERT COORDINATE SYSTEMS AND RESAMPLE
    vxArray = velReprojectionResample(vxFile, crs, res)
    vyArray = velReprojectionResample(vyFile, crs, res)
    vArray = np.power((np.square(vxArray) + np.square(vyArray)), 0.5)
if type == 'v':
    vArray = velReprojectionResample(vFile, crs, res)

use_filter = 'none'      # 'sg' for savitzky-golay filter, 'gauss' for gaussian filter
if use_filter == 'sg':
    vArray = sgolay2d(vArray, window_size=17, order=1)
if use_filter == 'gauss':
    vArray = gaussianFilter(vArray, st_dev=4, truncate=3)

def velQuiver(vx, vy, v_tot, freq, threshold):
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
    return quiverInput

title_map = name + ' Glacier Velocity Map (m/a)'
fig = plt.figure()
ax = fig.add_subplot(111, label="1")
divnorm = TwoSlopeNorm(vmin=vArray.min(), vcenter=min(vArray.mean(), 10), vmax=min(vArray.max(), 100))
vArray[vArray > 1000] = np.nan  # to remove huge values for white background in plot
vArray[vArray == 0] = np.nan  # to remove 0 values for white background in plot
if type == 'vxvy':
    quiverData = velQuiver(vxArray * cor_vx, vyArray * cor_vy, vArray, freq=20, threshold=2.0)    # plot velocity arrows
    ax.quiver(quiverData[0], quiverData[1], quiverData[2], quiverData[3], quiverData[4],
               cmap=quiverData[5], scale=quiverData[6], width=.003)  # velocity arrows
vArray = vArray.astype(float)
im = ax.imshow(vArray, cmap=plt.cm.get_cmap('YlOrBr'), norm=divnorm)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, label='')
ax.set_title(title_map)

plt.show(block=False)
# figName = title_map.replace(' ', '_') + '.png'
# print(os.getcwd())
# fig.savefig('Figures/' + figName)
