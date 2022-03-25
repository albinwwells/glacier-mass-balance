# Final project classes and methods
import rasterio
import rasterio.plot
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform
from rasterio.merge import merge
from rasterio.windows import from_bounds
from rasterio.fill import fillnodata
import pyproj
from pyproj import transform
import fiona
import pandas
import geopandas as gpd
import numpy as np
from scipy import stats
import math
from .smoothingFunctions import sgolay2d, gaussianFilter
import warnings
import os

class glacier:
    #glacier represents a glacier with at least a DEM, ice thickness, and change in ice thickness raster file
    def __init__(self, glacierName, massBalanceType, shapeFile, demFile, thicknessFile, changeInThicknessFile,
                 coordinateSystem, resolution, density, elevationBinWidth, data, data_var):
        self.name = glacierName
        self.type = massBalanceType     # this is either 'Total' or 'Climatic'
        self.shape = shapeFile
        self.dem = demFile
        self.h = thicknessFile
        self.dhdt = changeInThicknessFile
        self.crs = coordinateSystem
        self.res = resolution
        self.rho = density
        self.bin = elevationBinWidth
        self.data = data
        self.dataVariability = data_var

    def tifReprojectionResample(self):
        fileList = [self.dem, self.h, self.dhdt]
        # index through each file, open and obtain transformation information, then perform reprojection
        for file in fileList:
            with rasterio.open(file) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, self.crs, src.width, src.height, *src.bounds, resolution=self.res)
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': self.crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })

                destination = file[:-4] + '_reproj.tif'
                with rasterio.open(destination, 'w', **kwargs) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=self.crs,
                        resampling=Resampling.cubic_spline)

    def shpReprojection(self):
        src = gpd.read_file(self.shape)
        src = src.to_crs(self.crs)
        dst = self.shape[:-4] + '_reproj.shp'
        src.to_file(dst)

    def tifClip(self):
        reprojDEM = self.dem[:-4] + '_reproj.tif'
        reprojH = self.h[:-4] + '_reproj.tif'
        reprojDHDT = self.dhdt[:-4] + '_reproj.tif'
        clipfile = reprojH              # file for clipping bounds
        fileList = [reprojDEM, reprojH, reprojDHDT]
        for file in fileList:
            with rasterio.open(file) as src:
                dst = rasterio.open(clipfile)
                dstBounds = dst.bounds  # bounds of destination tif (left, bottom, right, top)
                kwargs = src.meta.copy()
                kwargs.update({
                    'height': dst.height,
                    'width': dst.width,
                    'transform': dst.transform
                })

                destination = file[:-10] + 'clip.tif'
                with rasterio.open(destination, 'w', **kwargs) as dst:
                    w = from_bounds(dstBounds[0], dstBounds[1], dstBounds[2], dstBounds[3], src.transform)
                    dst.write(src.read(window=w))

    def fillHole(self):
        fillDEM = self.dem[:-4] + '_clip.tif'
        fillH = self.h[:-4] + '_clip.tif'
        fillDHDT = self.dhdt[:-4] + '_clip.tif'
        fillFile = [fillDEM, fillH, fillDHDT]

        for file in fillFile:
            with rasterio.open(file) as src:
                profile = src.profile
                input = src.read(1)
                # call fillnodata function to fill this hole. we can later check that it filled (and with a reasonable value)!
                inputFilled = fillnodata(input, mask=src.read_masks(1), max_search_distance=10, smoothing_iterations=0)
                inputFilled[pandas.isnull(inputFilled) == True] = 0

            with rasterio.open(file, 'w', **profile) as dst:
                dst.write_band(1, inputFilled)

    def glacierOutline(self):
        outlineFile = self.dem[:-4] + '_clip.tif'
        src = rasterio.open(outlineFile)
        kwargs = src.meta.copy()
        dst_file = self.name + 'Outline.tif'

        outlineFile = src.read(1)
        currentGlacierOutline = pandas.DataFrame(outlineFile)  # ice thickness file only has values on the glacier
        currentGlacierOutline[currentGlacierOutline != 0] = 1  # make mask binary (0 and 1)

        with rasterio.open(dst_file, 'w', **kwargs) as dst:
            dst.write(currentGlacierOutline, 1)

    def glacierArea(self):
        outlineFile = self.name + 'Outline.tif'
        totalArea = np.sum(rasterio.open(outlineFile).read(1)) * self.res * self.res  # area in m2
        return totalArea

    def totalMassBalance(self):
        dhdtClip = self.dhdt[:-4] + '_clip.tif'
        src = rasterio.open(dhdtClip)
        kwargs = src.meta.copy()

        glacierOutlineFile = self.name + 'Outline.tif'
        glacierOutlineFileRead = rasterio.open(glacierOutlineFile).read(1)

        # total (dhdt) mass balance (kg/m2-yr) = dhdht (m/yr) * density (kg/m3)
        massBalanceTotal = src.read(1) * self.rho * glacierOutlineFileRead

        destinationTotal = self.name + 'TotalMassBalance.tif'
        with rasterio.open(destinationTotal, 'w', **kwargs) as dst:
            dst.write(massBalanceTotal, 1)

    def totalMassBalanceValue(self):
        totalMassBalanceFile = self.name + 'TotalMassBalance.tif'
        outlineFile = self.name + 'Outline.tif'
        area = np.sum(rasterio.open(outlineFile).read(1)) * self.res * self.res  # area in m2
        dst_total = rasterio.open(totalMassBalanceFile)
        rdst_total = dst_total.read(1)
        rdst_total[rdst_total == 0] = np.nan  # rdst is dst but 0 values are eliminated
        totalMassBalanceVal = np.sum(dst_total.read(1) * self.res * self.res) / area  # total mass balance in kg/m2-yr
        return totalMassBalanceVal

    def altitudeAggregation(self, calcFile, stat):
        '''
        Returns elevation-binned mass balance means
        :param calcFile:
        :param stat: statistic to be calculated ('mean', 'count', 'sum'
        :return: array of the altitudinally-binned statistic and elevation bin boundaries
        '''
        dem = self.dem[:-4] + '_clip.tif'
        if isinstance(dem, np.ndarray) == False:
            src = rasterio.open(dem)
            dem = src.read(1)

        glacierOutlineFile = self.name + 'Outline.tif'
        glacierOutlineFileRead = rasterio.open(glacierOutlineFile).read(1)
        demGlacier = dem * glacierOutlineFileRead
        demGlacierArray = demGlacier[~np.isnan(demGlacier)]

        if isinstance(calcFile, np.ndarray) == False:
            src = rasterio.open(calcFile)
            calcFile = src.read(1)

        calcFileArray = calcFile * glacierOutlineFileRead
        calcFileArray = calcFileArray[~np.isnan(calcFileArray)]

        demGlacier_findMin = demGlacier
        demGlacier_findMin = demGlacier_findMin[demGlacier_findMin != 0]

        # bin from dem raster, but operate on massBalance raster
        # returns: stats array for each bin, the bin edges, and the indices of each value in the bin edges
        demBinCount = stats.binned_statistic(demGlacierArray, calcFileArray, statistic='count',
                            bins=range(int(demGlacier_findMin.min()), int(demGlacier_findMin.max()+self.bin), self.bin))
        demBin_std = stats.binned_statistic(demGlacierArray, calcFileArray, statistic='std',
                            bins=range(int(demGlacier_findMin.min()), int(demGlacier_findMin.max()+self.bin), self.bin))
        demBin_min = stats.binned_statistic(demGlacierArray, calcFileArray, statistic='min',
                            bins=range(int(demGlacier_findMin.min()), int(demGlacier_findMin.max()+self.bin), self.bin))
        demBin_max = stats.binned_statistic(demGlacierArray, calcFileArray, statistic='max',
                            bins=range(int(demGlacier_findMin.min()), int(demGlacier_findMin.max()+self.bin), self.bin))
        demBinStat = stats.binned_statistic(demGlacierArray, calcFileArray, statistic=stat,
                                            bins=range(int(demGlacier_findMin.min()),
                                                       int(demGlacier_findMin.max() + self.bin), self.bin))
        binStat = demBinStat[0]
        binBoundaries = demBinStat[1]
        binCount = demBinCount[0]
        binNumber = demBinStat[2]
        return binStat, binBoundaries, binCount, binNumber, demBin_std[0], demBin_min[0], demBin_max[0]

    def binPercentile(self, calcFile, lower_percentile):
        '''
        Returns elevation-binned mass balance percentiles
        :param calcFile: mass balance file
        :param lower_percentile: lower percentile to be calculated
        :return:
        '''
        dem = self.dem[:-4] + '_clip.tif'
        if isinstance(dem, np.ndarray) == False:
            src = rasterio.open(dem)
            dem = src.read(1)

        glacierOutlineFile = self.name + 'Outline.tif'
        glacierOutlineFileRead = rasterio.open(glacierOutlineFile).read(1)
        demGlacier = dem * glacierOutlineFileRead
        demGlacierArray = demGlacier[~np.isnan(demGlacier)]

        if isinstance(calcFile, np.ndarray) == False:
            src = rasterio.open(calcFile)
            calcFile = src.read(1)

        calcFileArray = calcFile * glacierOutlineFileRead
        calcFileArray = calcFileArray[~np.isnan(calcFileArray)]

        demGlacier_findMin = demGlacier
        demGlacier_findMin = demGlacier_findMin[demGlacier_findMin != 0]

        # bin from dem raster, but operate on massBalance raster
        # returns: stats array for each bin, the bin edges, and the indices of each value in the bin edges
        demBinPercentile_low = stats.binned_statistic(demGlacierArray, calcFileArray,
                                                      statistic=lambda calcFileArray: np.percentile(calcFileArray,
                                                                                                    lower_percentile),
                                                      bins=range(int(demGlacier_findMin.min()),
                                                                 int(demGlacier_findMin.max() + self.bin), self.bin))
        demBinPercentile_high = stats.binned_statistic(demGlacierArray, calcFileArray,
                                                       statistic=lambda calcFileArray: np.percentile(calcFileArray,
                                                                                                     100 - lower_percentile),
                                                       bins=range(int(demGlacier_findMin.min()),
                                                                  int(demGlacier_findMin.max() + self.bin), self.bin))
        binStat_low = demBinPercentile_low[0]
        binStat_high = demBinPercentile_high[0]
        # binBoundaries = demBinPercentile_low[1]
        # binNumber = demBinPercentile_low[2]
        return binStat_low, binStat_high


class fullGlacier(glacier):
    # fullGlacier represents a glacier that also has velocity raster files
    def __init__(self, glacierName, massBalanceType, shapeFile, demFile, thicknessFile, changeInThicknessFile,
                 coordinateSystem, resolution, density, elevationBinWidth, data, data_var, vxFile, vyFile,
                 velocityCorrection, velocityColumnAvgScaling):
        super().__init__(glacierName, massBalanceType, shapeFile, demFile, thicknessFile, changeInThicknessFile,
                         coordinateSystem, resolution, density, elevationBinWidth, data, data_var)
        self.vx = vxFile
        self.vy = vyFile
        self.vCor = velocityCorrection
        self.vCol = velocityColumnAvgScaling

    def velReprojectionResample(self):
        velocities = [self.vx, self.vy]
        for vel in velocities:
            with rasterio.open(vel) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, self.crs, src.width, src.height, *src.bounds, resolution=self.res)
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': self.crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })

                destination = vel[:-4] + '_reproj.tif'
                with rasterio.open(destination, 'w', **kwargs) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=self.crs,
                        resampling=Resampling.cubic_spline)

    def tifClip(self):
        reprojDEM = self.dem[:-4] + '_reproj.tif'
        reprojH = self.h[:-4] + '_reproj.tif'
        reprojDHDT = self.dhdt[:-4] + '_reproj.tif'
        reprojVX = self.vx[:-4] + '_reproj.tif'
        reprojVY = self.vy[:-4] + '_reproj.tif'
        clipfile = reprojDEM      # or reprojDEM
        fileList = [reprojDEM, reprojH, reprojDHDT, reprojVX, reprojVY]
        for file in fileList:
            with rasterio.open(file) as src:
                dst = rasterio.open(clipfile)
                dstBounds = dst.bounds  # bounds of destination tif (left, bottom, right, top)
                kwargs = src.meta.copy()
                kwargs.update({
                    'height': dst.height,
                    'width': dst.width,
                    'transform': dst.transform
                })

                destination = file[:-10] + 'clip.tif'
                with rasterio.open(destination, 'w', **kwargs) as dst:
                    w = from_bounds(dstBounds[0], dstBounds[1], dstBounds[2], dstBounds[3], src.transform)
                    dst.write(src.read(window=w))

    def shpClip(self):
        reprojDEM = self.dem[:-4] + '_clip.tif'
        reprojH = self.h[:-4] + '_clip.tif'
        reprojDHDT = self.dhdt[:-4] + '_clip.tif'
        reprojVX = self.vx[:-4] + '_clip.tif'
        reprojVY = self.vy[:-4] + '_clip.tif'
        fileList = [reprojDEM, reprojH, reprojDHDT, reprojVX, reprojVY]
        for file in fileList:
            with fiona.open(self.shape[:-4] + '_reproj.shp', 'r') as shapefile:
                shapes = [feature["geometry"] for feature in shapefile]
            with rasterio.open(file) as src:
                out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True, nodata=0)
                kwargs = src.meta
                kwargs.update({"driver": "GTiff",
                                 "height": out_image.shape[1],
                                 "width": out_image.shape[2],
                                 "transform": out_transform})
            destination = file[:-8] + 'clip.tif'
            with rasterio.open(destination, 'w', **kwargs) as dst:
                dst.write(out_image)

    def missingPixels(self):
        demArray = rasterio.open(self.dem[:-4] + '_clip.tif').read(1)
        hArray = rasterio.open(self.h[:-4] + '_clip.tif').read(1)
        dhdtArray = rasterio.open(self.dhdt[:-4] + '_clip.tif').read(1)
        vxArray = rasterio.open(self.vx[:-4] + '_clip.tif').read(1)
        vyArray = rasterio.open(self.vy[:-4] + '_clip.tif').read(1)
        outlineFile = rasterio.open(self.name + 'Outline.tif').read(1)
        fileList = [demArray, hArray, dhdtArray, vxArray, vyArray]
        i = 0
        for file in fileList:
            i = i + 1
            totalPixels = file.shape[0] * file.shape[1]
            missingPixels = (file == 0).sum() + (outlineFile != 0).sum() - totalPixels
            # threshold of 0.5 sq km
            threshold = 500000
            if missingPixels * self.res * self.res > threshold:
                warnings.warn('\nToo many missing pixels in data #' + str(i) + '\nTotal missing area (sq. km): ' +
                              str(missingPixels * self.res * self.res / 1000000) + \
                              '\nTotal missing pixels: ' + str(missingPixels))

    def fillVel(self):
        fillVx = self.vx[:-4] + '_clip.tif'
        fillVy = self.vy[:-4] + '_clip.tif'
        fillFile = [fillVx, fillVy]

        for file in fillFile:
            with rasterio.open(file) as src:
                profile = src.profile
                input = src.read(1)
                # call fillnodata function to fill this hole. we can later check that it filled (and with a reasonable value)!
                inputFilled = fillnodata(input, mask=src.read_masks(1), max_search_distance=10, smoothing_iterations=0)
                inputFilled[pandas.isnull(inputFilled) == True] = 0

            with rasterio.open(file, 'w', **profile) as dst:
                dst.write_band(1, inputFilled)

    def initialClip(self):
        H = self.h
        DHDT = self.dhdt
        VX = self.vx
        VY = self.vy
        clipfile = self.shape
        fileList = [H, DHDT, VX, VY]
        fileName = ['_h.tif', '_dhdt.tif', '_vx.tif', '_vy.tif']
        for file in range(len(fileList)):
            with rasterio.open(fileList[file]) as src:
                # crop_extent = gpd.read_file(clipfile)
                # # Reproject the crop extent data to match the roads layer.
                # crop_extent_wgs84 = crop_extent.to_crs(src.crs)
                #
                # # Clip the buildings and roads to the extent of the study area using geopandas
                # roads_clip = gpd.clip(fileList[file], crop_extent_wgs84)
                # destination = self.name + fileName[file]
                # with rasterio.open(destination, 'w', **kwargs) as dst:
                #     dst.write(roads_clip, 1)
                clip = gpd.read_file(clipfile)
                clip_crs = clip.to_crs(src.crs)
                clip_crs.to_file('temp.shp')

                with fiona.open('temp.shp', 'r') as shapefile:
                    shapes = [feature["geometry"] for feature in shapefile]

                out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True, nodata=0, pad=True, pad_width=10)
                out_image[np.isnan(out_image)] = 0
                kwargs = src.meta
                kwargs.update({"driver": "GTiff",
                               "height": out_image.shape[1],
                               "width": out_image.shape[2],
                               "transform": out_transform})
                destination = self.name + fileName[file]
                with rasterio.open(destination, 'w', **kwargs) as dst:
                    dst.write(out_image)

    def getCRS(self):
        # obtain northing and easting values from shapefile
        zips = gpd.read_file(self.shape)
        zips = zips.to_crs('epsg:4326')
        list_of_coords = list(zips['geometry'][0].exterior.coords)
        easting = []
        for north_east_tuple in list_of_coords:
            easting.append(int(abs(north_east_tuple[0])+1))     # converting to easting and accounting for notation
        utm_num = str(math.ceil((- np.median(easting) + 180) / 6)).zfill(2)
        crs = 'EPSG:326' + utm_num
        return crs

    def latlonTiffIndex(self, array, coordinates):
        # obtain raster array index values at a lat / lon coordinate location
        # takes in an array and tuple of lat lon coordinates
        src = rasterio.open(self.name + 'TotalMassBalance.tif')
        kwargs = src.meta.copy()
        dest = self.name + '_array_temp.tif'
        with rasterio.open(dest, 'w', **kwargs) as dst:
            dst.write(array, 1)             # save file as raster with geo coordinates

        geo_array = rasterio.open(dest)
        transformer = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), pyproj.CRS(self.crs))
        east, north = transformer.transform(coordinates[0], coordinates[1])         # find utm coordinates from lat/lon
        row, col = geo_array.index(east, north)                 # obtain raster row/col closest to utm coordinates
        return row, col

    def getDhdtFile(self, time_span):
        # obtain northing and easting values from shapefile
        zips = gpd.read_file(self.shape)
        zips = zips.to_crs('epsg:4326')
        list_of_coords = list(zips['geometry'][0].exterior.coords)
        easting = []
        northing = []
        for north_east_tuple in list_of_coords:
            easting.append(int(abs(north_east_tuple[0])+1))     # converting to easting and accounting for notation
            northing.append(int(north_east_tuple[1]))

        # find and open the dhdt file given the northing and easting values
        # if more than one northing and/or easting exists, we need multiple files
        if min(northing) == max(northing):
            northing_str = str(min(northing))
            if min(easting) == max(easting):
                easting_str = str(min(easting))
                dhdt_file = 'RGI1 Elevation Change ' + time_span + '/N' + northing_str + 'W' + \
                            easting_str + '_' + time_span[0:4] + '-01-01_' + time_span[-4:] + '-01-01_dhdt.tif'
                return dhdt_file
            else:
                easting_str1 = str(min(easting))
                easting_str2 = str(max(easting))
                dhdt_file = [('RGI1 Elevation Change ' + time_span + '/N' + northing_str + 'W' + easting_str1 +
                              '_' + time_span[0:4] + '-01-01_' + time_span[-4:] + '-01-01_dhdt.tif'),
                             ('RGI1 Elevation Change ' + time_span + '/N' + northing_str + 'W' + easting_str2 +
                              '_' + time_span[0:4] + '-01-01_' + time_span[-4:] + '-01-01_dhdt.tif')]
        else:
            if min(easting) == max(easting):
                easting_str = str(min(easting))
                northing_str1 = str(min(northing))
                northing_str2 = str(max(northing))
                dhdt_file = [('RGI1 Elevation Change ' + time_span + '/N' + northing_str1 + 'W' + easting_str +
                              '_' + time_span[0:4] + '-01-01_' + time_span[-4:] + '-01-01_dhdt.tif'),
                             ('RGI1 Elevation Change ' + time_span + '/N' + northing_str2 + 'W' + easting_str +
                              '_' + time_span[0:4] + '-01-01_' + time_span[-4:] + '-01-01_dhdt.tif')]
            else:
                northing_str1 = str(min(northing))
                northing_str2 = str(max(northing))
                easting_str1 = str(min(easting))
                easting_str2 = str(max(easting))
                dhdt_file = [('RGI1 Elevation Change ' + time_span + '/N' + northing_str1 + 'W' + easting_str1 +
                              '_' + time_span[0:4] + '-01-01_' + time_span[-4:] + '-01-01_dhdt.tif'),
                             ('RGI1 Elevation Change ' + time_span + '/N' + northing_str1 + 'W' + easting_str2 +
                              '_' + time_span[0:4] + '-01-01_' + time_span[-4:] + '-01-01_dhdt.tif'),
                             ('RGI1 Elevation Change ' + time_span + '/N' + northing_str2 + 'W' + easting_str1 +
                              '_' + time_span[0:4] + '-01-01_' + time_span[-4:] + '-01-01_dhdt.tif'),
                             ('RGI1 Elevation Change ' + time_span + '/N' + northing_str2 + 'W' + easting_str2 +
                              '_' + time_span[0:4] + '-01-01_' + time_span[-4:] + '-01-01_dhdt.tif')]

        src_files_to_mosaic = []
        x = []
        # create a mosiac of the obtained raster files. reproject files to the same crs if needed
        for i in range(len(dhdt_file)):
            src = rasterio.open(dhdt_file[i])
            dst_crs = rasterio.open(dhdt_file[1]).crs
            if src.crs == dst_crs:
                src_files_to_mosaic.append(src)
            else:
                x.append(1)
                transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': dst_crs,
                    'transform': transform,
                    'width': width,
                    'height': height})
                with rasterio.open('temp_mosaic' + str(i) + '.tif', 'w', **kwargs) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.cubic_spline)
                src_new = rasterio.open('temp_mosaic' + str(i) + '.tif')
                src_files_to_mosaic.append(src_new)

        mosaic, out_trans = merge(src_files_to_mosaic)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                                 "height": mosaic.shape[1],
                                 "width": mosaic.shape[2],
                                 "transform": out_trans})
        destination = 'temp_dhdt.tif'
        with rasterio.open(destination, "w", **out_meta) as dest:
            dest.write(mosaic)

        if os.path.exists('temp_mosaic1.tif') == True:
            for i in range(len(x)):
                os.remove('temp_mosaic' + str(i) + '.tif')
        return destination

    def MillanThicknessRGI1_2(self):
        # obtain northing and easting values from shapefile
        zips = gpd.read_file(self.shape)
        zips = zips.to_crs('epsg:4326')
        list_of_coords = list(zips['geometry'][0].exterior.coords)
        easting = []
        for north_east_tuple in list_of_coords:
            easting.append(north_east_tuple[0])

        if min(easting) and max(easting) < -136:
            return 'Millan_Thickness_RGI1.tif'
        elif min(easting) and max(easting) > -136.3:
            return 'Millan_Thickness_RGI2.tif'
        else:
            h_files = ['Millan_Thickness_RGI1.tif', 'Millan_Thickness_RGI2.tif']

        src_files_to_mosaic = []
        # create a mosiac of the obtained raster files. reproject files to the same crs if needed
        for i in range(2):
            src = rasterio.open(h_files[i])
            dst_crs = rasterio.open(h_files[1]).crs
            if src.crs == dst_crs:
                src_files_to_mosaic.append(src)
            else:
                transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': dst_crs,
                    'transform': transform,
                    'width': width,
                    'height': height})
                with rasterio.open('temp_h_mosaic' + str(i) + '.tif', 'w', **kwargs) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.cubic_spline)
                src_new = rasterio.open('temp_h_mosaic' + str(i) + '.tif')
                src_files_to_mosaic.append(src_new)

        mosaic, out_trans = merge(src_files_to_mosaic)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                                 "height": mosaic.shape[1],
                                 "width": mosaic.shape[2],
                                 "transform": out_trans})
        destination = 'temp_h.tif'
        with rasterio.open(destination, "w", **out_meta) as dest:
            dest.write(mosaic)

        if os.path.exists('temp_h_mosaic.tif') == True:
            for i in range(2):
                os.remove('temp_h_mosaic' + str(i) + '.tif')
        return destination

    def MillanVelocityRGI1_2(self):
        # obtain northing and easting values from shapefile
        zips = gpd.read_file(self.shape)
        zips = zips.to_crs('epsg:4326')
        list_of_coords = list(zips['geometry'][0].exterior.coords)
        easting = []
        for north_east_tuple in list_of_coords:
            easting.append(north_east_tuple[0])

        if min(easting) and max(easting) < -136:
            return 'vx_Millan_RGI1.tif', 'vy_Millan_RGI1.tif'
        elif min(easting) and max(easting) > -136.3:
            return 'vx_Millan_RGI2.tif', 'vy_Millan_RGI2.tif'
        else:
            vx_files = ['vx_Millan_RGI1.tif', 'vx_Millan_RGI2.tif']
            vy_files = ['vy_Millan_RGI1.tif', 'vy_Millan_RGI2.tif']

        src_files_to_mosaic = []
        # create a mosiac of the obtained raster files. reproject files to the same crs if needed
        for i in range(2):
            src = rasterio.open(vx_files[i])
            dst_crs = rasterio.open(vx_files[1]).crs
            if src.crs == dst_crs:
                src_files_to_mosaic.append(src)
            else:
                transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height,
                                                                       *src.bounds)
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': dst_crs,
                    'transform': transform,
                    'width': width,
                    'height': height})
                with rasterio.open('temp_vx_mosaic' + str(i) + '.tif', 'w', **kwargs) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.cubic_spline)
                src_new = rasterio.open('temp_vx_mosaic' + str(i) + '.tif')
                src_files_to_mosaic.append(src_new)

        mosaic, out_trans = merge(src_files_to_mosaic)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans})
        destination_vx = 'temp_h.tif'
        with rasterio.open(destination_vx, "w", **out_meta) as dest:
            dest.write(mosaic)

        src_files_to_mosaic = []
        # create a mosiac of the obtained raster files. reproject files to the same crs if needed
        for i in range(2):
            src = rasterio.open(vy_files[i])
            dst_crs = rasterio.open(vy_files[1]).crs
            if src.crs == dst_crs:
                src_files_to_mosaic.append(src)
            else:
                transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height,
                                                                       *src.bounds)
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': dst_crs,
                    'transform': transform,
                    'width': width,
                    'height': height})
                with rasterio.open('temp_vy_mosaic' + str(i) + '.tif', 'w', **kwargs) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.cubic_spline)
                src_new = rasterio.open('temp_vy_mosaic' + str(i) + '.tif')
                src_files_to_mosaic.append(src_new)

        mosaic, out_trans = merge(src_files_to_mosaic)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans})
        destination_vy = 'temp_vy.tif'
        with rasterio.open(destination_vy, "w", **out_meta) as dest:
            dest.write(mosaic)

        if os.path.exists('temp_vx_mosaic.tif') == True:
            for i in range(2):
                os.remove('temp_vx_mosaic' + str(i) + '.tif')
        if os.path.exists('temp_vy_mosaic.tif') == True:
            for i in range(2):
                os.remove('temp_vy_mosaic' + str(i) + '.tif')
        return destination_vx, destination_vy


    def getReferenceData(self, time_span):
        # get reference data for certain glaciers. By RGI ID and time span
        if self.name == '00570':
            if time_span == '2000-2020':
                data = [[-1254, 327], [1690, 1850]]     # [[MB],[Elevation]]
                data_min = [-3880, -720]
                data_max = [330, 2030]
                data_var = abs(np.array([np.array(data_min) - np.array(data[0]), np.array(data_max) - np.array(data[0])]))
                data_coords = [[63.285514, -145.41034, -1254, '1690m', 'B'],
                               [63.284888, -145.38505, -327, '1850m', 'D']]    # [Lat, Lon, Avg MB, Elevation, Point ID]
                return data, data_var, data_coords
            elif time_span == '2015-2020':
                data = [[-3295, -2757, -1508, -250, -604, 670], [1451, 1544, 1690, 1850, 1879, 2030]]
                data_min = [-4290, -3780, -2460, -720, -1720, -440]
                data_max = [-2450, -1810, -700, 780, 400, 1500]
                data_var = abs(np.array([np.array(data_min) - np.array(data[0]), np.array(data_max) - np.array(data[0])]))
                data_coords = [[63.272055, -145.41668, -2757, '1544m', 'AB'],
                      [63.264893, -145.41675, -3295, '1451m', 'AU'],
                      [63.285514, -145.41034, -1508, '1690m', 'B'],
                      [63.284888, -145.38505, -250, '1850m', 'D'],
                      [63.294753, -145.42061, -604, '1879m', 'V'],
                      [63.285958, -145.48020, 670, '2030m', 'X']]
                return data, data_var, data_coords
        elif self.name == '09162':
            if time_span == '2000-2020':
                data = [[-1616, 919], [1067, 1298]]
                data_min = [-3060, -510]
                data_max = [-40, 2740]
                data_var = abs(np.array([np.array(data_min) - np.array(data[0]), np.array(data_max) - np.array(data[0])]))
                data_coords = [[60.404161, -148.90667, -1616, '1067m', 'B'],
                               [60.419739, -148.92072, 919, '1298m', 'C']]
                return data, data_var, data_coords
            elif time_span == '2015-2020':
                data = [[-6797, -2462, -1797, -452, 542, 758, 1285], [624, 1004, 1067, 1235, 1298, 1370, 1371]]
                data_min = [-8810, -3200, -2650, -1110, 30, -10, 450]
                data_max = [-5860, -1750, -1170, 310, 1660, 2100, 2780]
                data_var = abs(np.array([np.array(data_min) - np.array(data[0]), np.array(data_max) - np.array(data[0])]))
                data_coords = [[60.380534, -148.91834, -6797, '624m', 'AU'],
                               [60.404161, -148.90667, -1797, '1067m', 'B'],
                               [60.419739, -148.92072, 542, '1298m', 'C'],
                               [60.396430, -148.90871, -2462, '1004m', 'N'],
                               [60.405785, -148.87642, -452, '1235m', 'S'],
                               [60.418867, -148.89160, 758, '1370m', 'T'],
                               [60.424946, -148.93705, 1285, '1371m', 'Y']]
                return data, data_var, data_coords
        elif self.name == '01104':
            if time_span == '2000-2020':
                data = [[894, 1887, 1073, 1751], [1074, 1143, 1184, 1198]]
                data_min = [-2600, 560, -1880, 680]
                data_max = [3900, 3562, 3570, 3070]
                data_var = abs(np.array([np.array(data_min) - np.array(data[0]), np.array(data_max) - np.array(data[0])]))
                data_coords = [[58.380890, -134.34600, 894, '1074m', 'C'],
                               [58.365092, -134.35500, 1073, '1184m', 'D'],
                               [58.361808, -134.36100, 1751, '1198m', 'G'],
                               [58.372287, -134.34626, 1887, '1143m', 'H']]
                return data, data_var, data_coords
            elif time_span == '2015-2020':
                data = [[-3743, -2553, -845, 1452, -435, 6], [823, 943, 1074, 1143, 1184, 1234]]
                data_min = [-5590, -4550, -2600, 560, -1880, -1740]
                data_max = [-1760, -810, 1260, 1910, 1400, 1240]
                data_var = abs(np.array([np.array(data_min) - np.array(data[0]), np.array(data_max) - np.array(data[0])]))
                data_coords = [[58.400605, -134.36192, -3743, '823m', 'A'],
                               [58.393298, -134.35204, -2553, '943m', 'B'],
                               [58.380890, -134.34600, 845, '1074m', 'C'],
                               [58.365092, -134.35500, -435, '1184m', 'D'],
                               [58.361634, -134.33745, 6, '1234m', 'E'],
                               [58.372287, -134.34626, 1452, '1143m', 'H']]
                return data, data_var, data_coords
        else:
            data = None
            data_var = None
            data_coords = None
            return data, data_var, data_coords


    ## -------- THESE ARE A WORK IN PROGRESS / ABANDONED ---------
    def velAdjusted(self, *args):
        # adjust velocity products so they have the same mean values
        # input of 2 or 3 velocity arrays
        num_zeros = []
        for ar in args:
            # get the number of zeros in each array
            num_zeros = np.append(num_zeros, ar.size - np.count_nonzero(ar))

        if len(args) == 2:
            # get velocity arrays
            vel1 = args[0]
            vel2 = args[1]
            # find average values, excluding zero values (off-glacier terrain)
            avg_vel1 = np.sum(vel1) / (vel1.size - min(num_zeros))
            avg_vel2 = np.sum(vel2) / (vel2.size - min(num_zeros))
            # adjust velocity by applying a factor based on other products
            vel1_adj = vel1 * max([avg_vel1, avg_vel2]) / avg_vel1
            vel2_adj = vel2 * max([avg_vel1, avg_vel2]) / avg_vel2
            adjusted_vels = [vel1_adj, vel2_adj]
            return adjusted_vels

        elif len(args) == 3:
            # get velocity arrays
            vel1 = args[0]
            vel2 = args[1]
            vel3 = args[2]
            # find average values, excluding zero values (off-glacier terrain)
            avg_vel1 = np.sum(vel1) / (vel1.size - min(num_zeros))
            avg_vel2 = np.sum(vel2) / (vel2.size - min(num_zeros))
            avg_vel3 = np.sum(vel3) / (vel3.size - min(num_zeros))
            # adjust velocity by applying a factor based on other products
            vel1_adj = vel1 * max([avg_vel1, avg_vel2, avg_vel3]) / avg_vel1
            vel2_adj = vel2 * max([avg_vel1, avg_vel2, avg_vel3]) / avg_vel2
            vel3_adj = vel3 * max([avg_vel1, avg_vel2, avg_vel3]) / avg_vel3
            adjusted_vels = [vel1_adj, vel2_adj, vel3_adj]            # factor by which to multiply velocity
            return adjusted_vels

    def velDiscrepancyRegions(self, adjusted_vels, discrepancy_threshold):
        # takes the adjusted velocities and returns a binary mask identifying regions where velocity products differ
        # 'differ' depends on the prescribed discrepancy threshold
        if len(adjusted_vels) == 2:
            # returns boolean array where two arrays are within a tolerance: true are locations of product deviance
            masked_array = np.invert(np.isclose(adjusted_vels[0], adjusted_vels[1], rtol=discrepancy_threshold))
            return masked_array

        elif len(adjusted_vels) == 3:
            # returns locations where arrays are within a tolerance: true are locations of product deviance
            masked_array1 = np.invert(np.isclose(adjusted_vels[0], adjusted_vels[1], rtol=discrepancy_threshold))
            masked_array2 = np.invert(np.isclose(adjusted_vels[0], adjusted_vels[2], rtol=discrepancy_threshold))
            masked_array3 = np.invert(np.isclose(adjusted_vels[1], adjusted_vels[2], rtol=discrepancy_threshold))
            # returns array with locations where ANY two velocity products differ
            masked_array = ((masked_array1 == masked_array2) & (masked_array1 == masked_array3) &
                            (masked_array2 == masked_array3))
            return masked_array



