# Final project classes and methods
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
from smoothingFunctions import sgolay2d, gaussianFilter

class glacier:
    #glacier represents a glacier with at least a DEM, ice thickness, and change in ice thickness raster file
    def __init__(self, glacierName, massBalanceType, shapeFile, demFile, thicknessFile, changeInThicknessFile,
                 coordinateSystem, resolution, density, elevationBinWidth, data):
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
        outlineFile = self.h[:-4] + '_clip.tif'
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
        demGlacier_findMin[demGlacier_findMin == 0] = np.nan
        demGlacier_findMin = demGlacier_findMin[~np.isnan(demGlacier_findMin)]

        # bin from dem raster, but operate on massBalance raster
        # returns: stats array for each bin, the bin edges, and the indices of each value in the bin edges
        demBinCount = stats.binned_statistic(demGlacierArray, calcFileArray, statistic='count',
                            bins=range(int(demGlacier_findMin.min()), int(demGlacier_findMin.max()+self.bin), self.bin))
        demBinStat = stats.binned_statistic(demGlacierArray, calcFileArray, statistic=stat,
                                            bins=range(int(demGlacier_findMin.min()),
                                                       int(demGlacier_findMin.max() + self.bin), self.bin))
        binStat = demBinStat[0]
        binBoundaries = demBinStat[1]
        binCount = demBinCount[0]
        binNumber = demBinStat[2]
        return binStat, binBoundaries, binCount, binNumber


class fullGlacier(glacier):
    # fullGlacier represents a glacier that also has velocity raster files
    def __init__(self, glacierName, massBalanceType, shapeFile, demFile, thicknessFile, changeInThicknessFile,
                 coordinateSystem, resolution, density, elevationBinWidth, data, vxFile, vyFile,
                 velocityCorrection, velocityColumnAvgScaling):
        super().__init__(glacierName, massBalanceType, shapeFile, demFile, thicknessFile, changeInThicknessFile,
                         coordinateSystem, resolution, density, elevationBinWidth, data)
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
