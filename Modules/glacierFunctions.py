# Same as glacier class but files aren't saved, simply returned as array
import rasterio
import rasterio.plot
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform
from rasterio.windows import from_bounds
from rasterio.fill import fillnodata
import fiona
import pandas
import geopandas as gpd
import numpy as np
from scipy import stats
from scipy.interpolate import NearestNDInterpolator
from .smoothingFunctions import sgolay2d, gaussianFilter

def tifPrep(file, shape, crs, res):
    with rasterio.open(file) as src:
        transform, width, height = calculate_default_transform(src.crs, crs, src.width, src.height,
                                                               *src.bounds, resolution=res)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        dest = 'temp_file.tif'
        with rasterio.open(dest, 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.cubic_spline)

        with rasterio.open(dest) as src:
            profile = src.profile
            input = src.read(1)
            # call fillnodata function to fill this hole. we can later check that it filled (and with a reasonable value)!
            inputFilled = fillnodata(input, mask=src.read_masks(1), max_search_distance=10, smoothing_iterations=0)
            inputFilled[pandas.isnull(inputFilled) == True] = 0
        with rasterio.open(dest, 'w', **profile) as dst:
            dst.write_band(1, inputFilled)

        with fiona.open(shape) as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
        with rasterio.open(dest) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True, nodata=0)
            kwargs = src.meta
            kwargs.update({"driver": "GTiff",
                           "height": out_image.shape[1],
                           "width": out_image.shape[2],
                           "transform": out_transform})
        with rasterio.open(dest, 'w', **kwargs) as dst:
            dst.write(out_image.reshape(-1, out_image.shape[-1]), 1)

        bounds = rasterio.open(dest).bounds
        transform = rasterio.open(dest).transform
        meta = rasterio.open(dest).profile
        return out_image, bounds, transform, meta

def shpReprojection(shape, crs):
        src = gpd.read_file(shape)
        src = src.to_crs(crs)
        src.to_file(shape)

def clip(input_file, h, w, src_meta, src_transform, dst_transform, bounds):
    src = 'temp_file1.tif'
    dest = 'temp_file.tif'
    with rasterio.Env():
        with rasterio.open(src, 'w', **src_meta) as dst:
            dst.write(input_file, 1)

    kwargs = src_meta
    kwargs.update({
        'height': h,
        'width': w,
        'transform': dst_transform
    })

    with rasterio.open(dest, 'w', **kwargs) as dst:
        w = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], src_transform)
        dst.write(rasterio.open(src).read(window=w))

    windowArray = rasterio.open(dest).read(1)
    return windowArray

def glacierOutline(outlineFile):
    currentGlacierOutline = pandas.DataFrame(outlineFile)  # ice thickness file only has values on the glacier
    currentGlacierOutline[currentGlacierOutline != 0] = 1  # make mask binary (0 and 1)
    return currentGlacierOutline.to_numpy()

def glacierArea(file, res):
    totalArea = np.sum(file) * res * res  # area in m2
    return totalArea

def totalMassBalance(dhdt, outline, density):
    # total (dhdt) mass balance (kg/m2-yr) = dhdht (m/yr) * density (kg/m3)
    massBalanceTotal = dhdt * density * outline
    return massBalanceTotal

def totalMassBalanceValue(totalMB, outline, res):
    area = np.sum(outline) * res * res  # area in m2

    # totalMB[totalMB == 0] = np.nan
    totalMassBalanceVal = np.sum(totalMB * res * res) / area  # total mass balance in kg/m2-yr
    return totalMassBalanceVal

def altitudeAggregation(dem, outline, calcFile, bins, stat):
    '''
    Returns elevation-binned mass balance means
    :param calcFile:
    :param stat: statistic to be calculated ('mean', 'count', 'sum'
    :return: array of the altitudinally-binned statistic and elevation bin boundaries
    '''
    demGlacier = dem * outline
    demGlacierArray = demGlacier[~np.isnan(demGlacier)]

    calcFileArray = calcFile * outline
    calcFileArray = calcFileArray[~np.isnan(calcFileArray)]

    demGlacier_findMin = demGlacier
    demGlacier_findMin[demGlacier_findMin == 0] = np.nan
    demGlacier_findMin = demGlacier_findMin[~np.isnan(demGlacier_findMin)]

    # bin from dem raster, but operate on massBalance raster
    # returns: stats array for each bin, the bin edges, and the indices of each value in the bin edges
    demBinCount = stats.binned_statistic(demGlacierArray, calcFileArray, statistic='count',
                        bins=range(int(demGlacier_findMin.min()), int(demGlacier_findMin.max()+bins), bins))
    demBinStat = stats.binned_statistic(demGlacierArray, calcFileArray, statistic=stat,
                                        bins=range(int(demGlacier_findMin.min()),
                                                   int(demGlacier_findMin.max() + bins), bins))
    binStat = demBinStat[0]
    binBoundaries = demBinStat[1]
    binCount = demBinCount[0]
    binNumber = demBinStat[2]
    return binStat, binBoundaries, binCount, binNumber

