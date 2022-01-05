# For plotting velocity files

from smoothingFunctions import sgolay2d, gaussianFilter, dynamicSmoothing
import numpy as np
import os
import fiona
import geopandas as gpd
import rasterio
import rasterio.plot
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm

# -----------------------------------------------------GENERAL---------------------------------------------------------#
res = 20                    # Resampling resolution (x, y) tuple
glacier = 'wolverine'
# -------------------------------------------------GULKANA---------------------------------------------------------#
if glacier == 'gulkana':
    name = 'Gulkana'
    shapeFile = 'gulkana_reproj.shp'
    # vxFile = 'GulkanaRegion_vy.tif'
    # vyFile = 'GulkanaRegion_vx.tif'
    # vxFile = 'Gulkana_vx_Millan.tif'
    # vyFile = 'Gulkana_vy_Millan.tif'
    # vxFile = 'Gulkana_vx_retreat.tif'                   # 2020 annual velocity data
    # vyFile = 'Gulkana_vy_retreat.tif'
    vxFile = 'Gulkana_vx_retreat_composite.tif'  # 2015-2020 velocity composite
    vyFile = 'Gulkana_vy_retreat_composite.tif'
    cor_vx = 1
    cor_vy = 1
    crs = 'EPSG:32606'  # UTM Zone 6 North. Alaska Albers coordinate system: ('EPSG: 3338')
# -------------------------------------------------WOLVERINE-------------------------------------------------------#
if glacier == 'wolverine':
    name = 'Wolverine'              # Glacier name (folder should be the same as glacier name)
    shapeFile = 'wolverine.shp'
    # vxFile = 'WolverineGlacier_vy.tif'
    # vyFile = 'WolverineGlacier_vx.tif'
    # vxFile = 'WolverineGlacier_vx_Millan.tif'
    # vyFile = 'WolverineGlacier_vy_Millan.tif'
    vxFile = 'Wolverine_vx_retreat.tif'                     # 2020 annual velocity data
    vyFile = 'Wolverine_vy_retreat.tif'
    # vxFile = 'Wolverine_vx_retreat_composite.tif'           # 2015-2020 velocity composite
    # vyFile = 'Wolverine_vy_retreat_composite.tif'
    cor_vx = 1
    cor_vy = 1
    crs = 'EPSG:32606'              # UTM Zone 6 North
# -------------------------------------------------------EKLUTNA-------------------------------------------------------#
if glacier == 'eklutna':
    name = 'Eklutna'                # Glacier name (folder should be the same as glacier name)
    shapeFile = 'eklutna_reproj.shp'
    vxFile = 'Eklutna_vy.tif'
    vyFile = 'Eklutna_vx.tif'
    # vxFile = 'Eklutna_vx_Millan.tif'
    # vyFile = 'Eklutna_vy_Millan.tif'
    # vxFile = 'Eklutna_vx_retreat.tif'                   # 2020 annual velocity data
    # vyFile = 'Eklutna_vy_retreat.tif'
    # vxFile = 'Eklutna_vx_retreat_composite.tif'  # 2015-2020 velocity composite
    # vyFile = 'Eklutna_vy_retreat_composite.tif'
    cor_vx = -1
    cor_vy = 1
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

        with rasterio.open(vel, 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.cubic_spline)


def shpReprojection(shape, crs):
        src = gpd.read_file(shape)
        src = src.to_crs(crs)
        src.to_file(shape)

def shpClip(shape, file):
    with fiona.open(shape) as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
    with rasterio.open(file) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True, nodata=0)
        kwargs = src.meta
        kwargs.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
    return out_image


# STEP 1: CONVERT COORDINATE SYSTEMS AND RESAMPLE
velReprojectionResample(vxFile, crs, res)
velReprojectionResample(vyFile, crs, res)

shpReprojection(shapeFile, crs)
vxArray3 = shpClip(shapeFile, vxFile)
vyArray3 = shpClip(shapeFile, vyFile)
vxArray = vxArray3.reshape(-1, vxArray3.shape[-1])
vyArray = vyArray3.reshape(-1, vyArray3.shape[-1])
vArray = np.power((np.square(vxArray) + np.square(vyArray)), 0.5)

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

title_map = name + ' Glacier Velocity Map'
fig = plt.figure()
ax = fig.add_subplot(111, label="1")
divnorm = TwoSlopeNorm(vmin=0, vcenter=5, vmax=75)
# divnorm = TwoSlopeNorm(vmin=0, vcenter=min(vArray.mean(), 10), vmax=min(vArray.max(), 100))
vArray[vArray > 1000] = np.nan  # to remove huge values for white background in plot
vArray[vArray == 0] = np.nan  # to remove 0 values for white background in plot
quiverData = velQuiver(vxArray * cor_vx, vyArray * cor_vy, vArray, freq=10, threshold=1.0)    # plot velocity arrows
ax.quiver(quiverData[0], quiverData[1], quiverData[2], quiverData[3], quiverData[4],
           cmap=quiverData[5], scale=quiverData[6]*2, width=.002)  # velocity arrows
vArray = vArray.astype(float)
im = ax.imshow(vArray, cmap=plt.cm.get_cmap('BrBG'), norm=divnorm)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, label='Velocity (m/a)')
ax.set_title(title_map)

plt.show()
# figName = title_map.replace(' ', '_') + '.png'
# print(os.getcwd())
# fig.savefig('Figures/' + figName)
