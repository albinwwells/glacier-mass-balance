# Functions for handling raster files
# Clipping, saving, reprojecting, filling gaps, mosaicking
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import rasterio
import rasterio.plot
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform
from rasterio.merge import merge
from rasterio.windows import from_bounds
from rasterio.fill import fillnodata
import fiona
import pandas as pd
import geopandas as gpd
import numpy as np
import math
import warnings
import glob, os


def rasterLike(array, destination, geotiff):
    # Create a raster (with name destination) from an array, with the same metadata as an existing geotiff
    with rasterio.open(geotiff) as src:
        kwargs = src.meta.copy()
    with rasterio.open(destination, 'w', **kwargs) as dst:
        dst.write(array, 1)

def show_fig(image, title, color, ctitle, bounds=None, res=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(12,6))
    c = ax.imshow(image, cmap=color, extent=bounds, vmin=vmin, vmax=vmax)
    if res != None:
        ax.add_artist(ScaleBar(dx=res, units='m')) # add scalebar
    fig.colorbar(c, label=ctitle)
    fig.suptitle(title)
    plt.show()

# def tifReprojectionResample(file, reprojected_tif, crs, res, interp):
#     # open and obtain file transformation information, then perform reprojection
#     with rasterio.open(file) as src:
#         transform, width, height = calculate_default_transform(
#             src.crs,
#             crs,
#             src.width,
#             src.height,
#             *src.bounds,
#             resolution=res
#         )
#
#         kwargs = src.meta.copy()
#         kwargs.update({
#             'crs': crs,
#             'transform': transform,
#             'width': width,
#             'height': height
#         })
#
#         with rasterio.open(reprojected_tif, 'w', **kwargs) as dst:
#             reproject(
#                 source=rasterio.band(src, 1),
#                 destination=rasterio.band(dst, 1),
#                 src_transform=src.transform,
#                 src_crs=src.crs,
#                 dst_transform=transform,
#                 dst_crs=crs,
#                 resampling=interp   # Resampling.cubic_spline
#             )

def tifReprojectionResample(file, reprojected_tif, crs, res, interp, extent_file=None):
    # open and obtain file transformation information, then perform reprojection
    with rasterio.open(file) as src:
        if extent_file is None:
            transform, width, height = calculate_default_transform(
                src.crs,
                crs,
                src.width,
                src.height,
                *src.bounds,
                resolution=res
            )
        else:
            dst = rasterio.open(extent_file)
            transform, width, height = calculate_default_transform(
                src.crs,
                crs,
                dst.width,
                dst.height,
                *dst.bounds,
                resolution=res
            )

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(reprojected_tif, 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=interp   # Resampling.cubic_spline
            )


def shpReprojection(shapefile, crs, dst):
    src = gpd.read_file(shapefile)
    src = src.to_crs(crs)
    src.to_file(dst)

def tifClip(geotiff, destination, geotiff_clip):
    # clip a geotiff to the bounds of another geotiff_clip
    with rasterio.open(geotiff) as src:
        dst = rasterio.open(geotiff_clip)
        dstBounds = dst.bounds  # bounds of destination tif (left, bottom, right, top)
        kwargs = src.meta.copy()
        kwargs.update({
            'height': dst.height,
            'width': dst.width,
            'transform': dst.transform
        })

        with rasterio.open(destination, 'w', **kwargs) as dst:
            w = from_bounds(dstBounds[0], dstBounds[1], dstBounds[2], dstBounds[3], src.transform)
            dst.write(src.read(window=w))

def shpClip(geotiff, shapefile, destination, pad_size=0, nan_val=0, fill=True, crop=True):
    # clip a geotiff with a shapefile
    with rasterio.open(geotiff) as src:
        # we need to make a temporary shapefile with the same crs as the geotiff
        shpReprojection(shapefile, src.crs, dst='temps.shp')
        with fiona.open('temps.shp', 'r') as shapefile:
            shapes = [feature['geometry'] for feature in shapefile]

        out_image, out_transform = rasterio.mask.mask(
            src,
            shapes,
            crop=crop,
            filled=fill,
            nodata=nan_val,
            pad=True,
            pad_width=pad_size
        )

        out_image[np.isnan(out_image)] = 0
        kwargs = src.meta
        kwargs.update({'driver': 'GTiff',
                       'height': out_image.shape[1],
                       'width': out_image.shape[2],
                       'transform': out_transform})
        with rasterio.open(destination, 'w', **kwargs) as dst:
            dst.write(out_image)

    for f in glob.glob('temps.*'):  # remove the 'temp.*' shapefiles
        os.remove(f)

def fillHole(file, destination, dist=10, iters=1):
    with rasterio.open(file) as src:
        profile = src.profile
        inputs = src.read(1)

        fillmask = inputs.copy()  # fillnodata is applied where the mask=0
        fillmask[inputs >= 0] = 1
        fillmask[fillmask != 1] = 0

        inputFilled = fillnodata(inputs, mask=fillmask, max_search_distance=dist, smoothing_iterations=iters)
        inputFilled[pd.isnull(inputFilled) == True] = 0

    with rasterio.open(destination, 'w', **profile) as dst:
        dst.write_band(1, inputFilled)

def mosaic_files(files, mosaic_output):
    # create a mosiac of the obtained raster files. reproject files to the same crs if needed
    src_files_to_mosaic = []
    dst_crs = rasterio.open(files[0]).crs
    for file in files:
        src = rasterio.open(file)
        if src.crs != dst_crs:  # NOTE: files must have the same crs
            raise 'Both datasets do not share a common coordinate system. One must be reprojected.'
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src.meta.copy()
    out_meta.update({'driver': 'GTiff',
                     'height': mosaic.shape[1],
                     'width': mosaic.shape[2],
                     'transform': out_trans})
    with rasterio.open(mosaic_output, 'w', **out_meta) as dest:
        dest.write(mosaic)


def missingPixels(input, outline, res):
    with rasterio.open(input) as src:
        file = src.read(1)
        totalPixels = file.shape[0] * file.shape[1]
        outlineFile = rasterio.open(outline).read(1)
        missingPixels = (file == 0).sum() + (outlineFile != 0).sum() - totalPixels
        # threshold of 0.5 sq km
        threshold = 500000
        if missingPixels * res * res > threshold:
            warnings.warn('\nToo many missing pixels in data' +
                          '\nTotal missing area (sq. km): ' +
                          str(missingPixels * res * res / 1000000) +
                          '\nTotal missing pixels: ' + str(missingPixels))

def getCRS(shapefile):
    # obtain northing and easting values from shapefile
    zips = gpd.read_file(shapefile)
    zips = zips.to_crs('epsg:4326')
    list_of_coords = list(zips['geometry'][0].exterior.coords)
    easting = []
    for north_east_tuple in list_of_coords:
        easting.append(int(abs(north_east_tuple[0])+1))     # converting to easting and accounting for notation
    utm_num = str(math.ceil((- np.median(easting) + 180) / 6)).zfill(2)
    crs = 'EPSG:326' + utm_num
    return crs

