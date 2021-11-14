#Final project classes and methods
import rasterio
import rasterio.plot
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.fill import fillnodata
import pandas
import numpy as np
from scipy import stats
from functions import sgolay2d, gaussianFilter

class glacier:
    '''glacier represents a glacier with at least a DEM, ice thickness, and change in ice thickness raster file'''
    def __init__(self, glacierName, massBalanceType, demFile, thicknessFile, changeInThicknessFile,
                 coordinateSystem, resolution, density, elevationBinWidth):
        self.name = glacierName
        self.type = massBalanceType #this is either 'Total' or 'Climatic'
        self.dem = demFile
        self.h = thicknessFile
        self.dhdt = changeInThicknessFile
        self.crs = coordinateSystem
        self.res = resolution
        self.rho = density
        self.bin = elevationBinWidth

    def tifReprojectionResample(self):
        fileList = [self.dem, self.h, self.dhdt]
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

    def tifClip(self):
        reprojDEM = self.dem[:-4] + '_reproj.tif'
        reprojH = self.h[:-4] + '_reproj.tif'
        reprojDHDT = self.dhdt[:-4] + '_reproj.tif'
        clipfile = reprojH
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
                inputFilled = fillnodata(input, mask=src.read_masks(1), max_search_distance=100, smoothing_iterations=0)

            with rasterio.open(file, 'w', **profile) as dst:
                dst.write_band(1, inputFilled)

    def glacierOutline(self):
        outlineFile = self.h[:-4] + '_clip.tif'
        min_h = 0                               #minimum thickness threshold
        src = rasterio.open(outlineFile)
        kwargs = src.meta.copy()
        dst_file = self.name + 'Outline.tif'

        outlineFile = src.read(1)
        df = pandas.DataFrame(outlineFile)  # ice thickness file only has values on the glacier
        currentGlacierOutline = df.mask(df <= min_h, 0)  # mask to fit shape of glacier
        currentGlacierOutline[currentGlacierOutline != 0] = 1  # make mask binary (0 and 1)

        with rasterio.open(dst_file, 'w', **kwargs) as dst:
            dst.write(currentGlacierOutline, 1)

    def glacierArea(self):
        outlineFile = self.name + 'Outline.tif'
        totalArea = np.sum(rasterio.open(outlineFile).read(1)) * self.res[0] * self.res[1]  #area in m2
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
        area = np.sum(rasterio.open(outlineFile).read(1)) * self.res[0] * self.res[1]  #area in m2
        dst_total = rasterio.open(totalMassBalanceFile)
        rdst_total = dst_total.read(1)
        rdst_total[rdst_total == 0] = np.nan  # rdst is dst but 0 values are eliminated
        totalMassBalanceVal = np.sum(dst_total.read(1) * self.res[0] * self.res[1]) / area  # total mass balance in kg/m2-yr
        return totalMassBalanceVal

    def altitudeAggregation(self, smoothVel=0, balType='climatic'):
        dem = self.dem[:-4] + '_clip.tif'
        if isinstance(dem, np.ndarray) == False:
            src = rasterio.open(dem)
            dem = src.read(1)

        glacierOutlineFile = self.name + 'Outline.tif'
        glacierOutlineFileRead = rasterio.open(glacierOutlineFile).read(1)
        demGlacier = dem * glacierOutlineFileRead
        demGlacierArray = demGlacier[~np.isnan(demGlacier)]

        if balType == 'total':
            massBalanceFile = self.name + 'TotalMassBalance.tif'
        elif smoothVel == True:
            massBalanceFile = self.name + 'ClimaticMassBalanceSmooth.tif'
        else:
            massBalanceFile = self.name + 'ClimaticMassBalance.tif'
        massBalanceArray = rasterio.open(massBalanceFile).read(1)
        massBalanceArray = massBalanceArray[~np.isnan(massBalanceArray)]

        demGlacier_findMin = demGlacier
        demGlacier_findMin[demGlacier_findMin == 0] = np.nan
        demGlacier_findMin = demGlacier_findMin[~np.isnan(demGlacier_findMin)]

        # bin from dem raster, but operate on massBalance raster
        # returns: stats array for each bin, the bin edges, and the indices of each value in the bin edges
        demBinCount = stats.binned_statistic(demGlacierArray, massBalanceArray, statistic='count',
                                             bins=range(int(demGlacier_findMin.min()),
                                                        int(demGlacier_findMin.max() + self.bin), self.bin))
        demBinMean = stats.binned_statistic(demGlacierArray, massBalanceArray, statistic='mean',
                                            bins=range(int(demGlacier_findMin.min()),
                                                       int(demGlacier_findMin.max() + self.bin), self.bin))
        demBinTotalMass = demBinCount[0] * demBinMean[0] * self.res[0] * self.res[1]  # this is in kg/yr
        binMeanMassBalance = demBinMean[0]
        binBoundaries = demBinMean[1]
        binCount = demBinCount[0]
        altitudeResolvedMassLoss = np.sum(demBinTotalMass)
        return altitudeResolvedMassLoss, binMeanMassBalance, binBoundaries, binCount


class fullGlacier(glacier):
    '''fullGlacier represents a glacier that also has velocity raster files'''
    def __init__(self, glacierName, massBalanceType, demFile, thicknessFile, changeInThicknessFile,
                 coordinateSystem, resolution, density, elevationBinWidth, vxFile, vyFile, velocityColumnAvgScaling):
        super().__init__(glacierName, massBalanceType, demFile, thicknessFile, changeInThicknessFile,
                         coordinateSystem, resolution, density, elevationBinWidth)
        self.vx = vxFile
        self.vy = vyFile
        self.vCol = velocityColumnAvgScaling

    def tifReprojectionResample(self):
        fileList = [self.dem, self.h, self.dhdt, self.vx, self.vy]
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

    def tifClip(self):
        reprojDEM = self.dem[:-4] + '_reproj.tif'
        reprojH = self.h[:-4] + '_reproj.tif'
        reprojDHDT = self.dhdt[:-4] + '_reproj.tif'
        reprojVX = self.vx[:-4] + '_reproj.tif'
        reprojVY = self.vy[:-4] + '_reproj.tif'
        clipfile = reprojDEM
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

    def velocitySmoothing(self, windowSize, polyOrder, std, smoothFilter='golay', der=None):
        '''

        :param windowSize: odd integer. pixel size for smoothing window. ex: 5 would be 5x5 pixel smoothing window
        :param polyOrder: polynomial order for smoothing curve fit. cannot be greater than windowSize
        :param std: standard deviation for gaussian filter
        :param smoothFilter: select 'golay' or 'gauss' filter. golay is chosen by default
        :param der: compute derivative instead of function. entries are 'row', 'col', or 'both'
        :return: raster with smoothed values as determined by this filter
        '''
        vxArray = rasterio.open(self.vx[:-4] + '_clip.tif').read(1)*rasterio.open(self.name+'Outline.tif').read(1)
        vyArray = rasterio.open(self.vy[:-4] + '_clip.tif').read(1)*rasterio.open(self.name+'Outline.tif').read(1)
        hArray = rasterio.open(self.h[:-4] + '_clip.tif').read(1) * rasterio.open(self.name + 'Outline.tif').read(1)
        if smoothFilter == 'golay':
            vxSmooth = sgolay2d(vxArray, window_size=windowSize, order=polyOrder, derivative=der)
            vySmooth = sgolay2d(vyArray, window_size=windowSize, order=polyOrder, derivative=der)
            hSmooth = sgolay2d(hArray, window_size=windowSize, order=polyOrder, derivative=der)
        elif smoothFilter == 'gauss':
            vxSmooth = gaussianFilter(vxArray, st_dev=std)
            vySmooth = gaussianFilter(vyArray, st_dev=std)
            hSmooth = gaussianFilter(hArray, st_dev=std)

        kwargs = rasterio.open(self.vx[:-4] + '_clip.tif').meta.copy()
        dst_file_vx = self.vx[:-4] + '_smooth.tif'
        with rasterio.open(dst_file_vx, 'w', **kwargs) as dst:
            dst.write(vxSmooth, 1)

        kwargs = rasterio.open(self.vy[:-4] + '_clip.tif').meta.copy()
        dst_file_vy = self.vy[:-4] + '_smooth.tif'
        with rasterio.open(dst_file_vy, 'w', **kwargs) as dst:
            dst.write(vySmooth, 1)

        kwargs = rasterio.open(self.h[:-4] + '_clip.tif').meta.copy()
        dst_file_h = self.h[:-4] + '_smooth.tif'
        with rasterio.open(dst_file_h, 'w', **kwargs) as dst:
            dst.write(hSmooth, 1)

    def climMassBalance(self, smoothVel=0):
        dhdtClip = self.dhdt[:-4] + '_clip.tif'
        hClip = self.h[:-4] + '_clip.tif'
        if smoothVel == True:
            vxClip = self.vx[:-4] + '_smooth.tif'
            vyClip = self.vy[:-4] + '_smooth.tif'
        else:
            vxClip = self.vx[:-4] + '_clip.tif'
            vyClip = self.vy[:-4] + '_clip.tif'
        sourceList = [dhdtClip, hClip, vxClip, vyClip]

        src = rasterio.open(dhdtClip)
        kwargs = src.meta.copy()
        for i in range(len(sourceList)):
            with rasterio.open(sourceList[i]) as src:
                sourceList[i] = src.read(1)

        glacierOutlineFile = self.name + 'Outline.tif'
        glacierOutlineFileRead = rasterio.open(glacierOutlineFile).read(1)

        '''
        div(Q) = dqx/dx + dqy/dy = gradient(h*vx, x) + gradient(h*vy, y)

        np.gradient returns derivatives relative to axis number, so (y, x) in this case: 
            I double checked this: [1] is the x (horizontal rows) and [0] is the y (vertical columns)

        Velocity files are not intuitive: x is the vertical axis in relation to the glacier. 
            This is because our reprojection actually rotated the image so much
            Checking velocities with known direction of motion for Gulkana glacier:
                show(sourceList[3]*gulkanaGlacierOutline)
                show(sourceList[4]*gulkanaGlacierOutline)

        vx is positive in the 'north' direction. vy is positive in the 'west' direction
        Thus, our 'x-direction' is actually the negative y, and 'y-direction' is x

        We take qx and derive it with regards to np.gradient x
        We take qy and derive it with regards to np.gradient y
        '''
        df_vx = pandas.DataFrame(sourceList[2] * glacierOutlineFileRead) #Remove very negative velocity regions (we assume they are erroneous)
        df_vy = pandas.DataFrame(sourceList[3] * glacierOutlineFileRead)
        vx_mask = df_vx.mask(df_vx <= df_vx.stack().mean() - df_vx.stack().std(), 0)
        vy_mask = df_vy.mask(df_vy <= df_vy.stack().mean() - df_vy.stack().std(), 0)

        qxRaster = sourceList[1] * self.vCol * np.array(-sourceList[3])
        qyRaster = sourceList[1] * self.vCol * np.array(sourceList[2])
        #qxRaster = sourceList[1] * self.vCol * np.array(-sourceList[3]) * glacierOutlineFileRead
        #qyRaster = sourceList[1] * self.vCol * np.array(sourceList[2]) * glacierOutlineFileRead
        divQRaster = np.gradient(qxRaster)[1] + np.gradient(qyRaster)[0]

        massBalanceClim = (sourceList[0] + divQRaster) * self.rho * glacierOutlineFileRead

        if smoothVel == True:
            destinationClimSmooth = self.name + 'ClimaticMassBalanceSmooth.tif'
            with rasterio.open(destinationClimSmooth, 'w', **kwargs) as dst:
                dst.write(massBalanceClim, 1)
        else:
            destinationClim = self.name + 'ClimaticMassBalance.tif'
            with rasterio.open(destinationClim, 'w', **kwargs) as dst:
                dst.write(massBalanceClim, 1)

    def climMassBalanceValue(self):
        climMassBalanceFile = self.name + 'ClimaticMassBalance.tif'
        outlineFile = self.name + 'Outline.tif'
        area = np.sum(rasterio.open(outlineFile).read(1)) * self.res[0] * self.res[1]  #area in m2
        dst_clim = rasterio.open(climMassBalanceFile)
        rdst_clim = dst_clim.read(1)
        rdst_clim[rdst_clim == 0] = np.nan  # rdst_clim is dst_clim but 0 values are eliminated

        climMassBalanceVal = np.sum(dst_clim.read(1) * self.res[0] * self.res[1]) / area  #climatic mass balance in kg/m2-yr
        return climMassBalanceVal

    def binMeanVelThickness(self):
        dem = self.dem[:-4] + '_clip.tif'
        if isinstance(dem, np.ndarray) == False:
            src = rasterio.open(dem)
            dem = src.read(1)

        glacierOutlineFile = self.name + 'Outline.tif'
        glacierOutlineFileRead = rasterio.open(glacierOutlineFile).read(1)
        demGlacier = dem*glacierOutlineFileRead
        demGlacierArray = demGlacier[~np.isnan(demGlacier)]

        hFile = self.h[:-4] + '_clip.tif'
        vxFile = self.vx[:-4] + '_clip.tif'
        vyFile = self.vy[:-4] + '_clip.tif'
        vxArray = rasterio.open(vxFile).read(1)
        vxArray = vxArray[~np.isnan(vxArray)]
        vyArray = rasterio.open(vyFile).read(1)
        vyArray = vyArray[~np.isnan(vyArray)]
        hArray = rasterio.open(hFile).read(1)
        hArray = hArray[~np.isnan(hArray)]
        vArray = np.power((np.square(vxArray) + np.square(vyArray)), 0.5)

        demGlacier_findMin = demGlacier
        demGlacier_findMin[demGlacier_findMin == 0] = np.nan
        demGlacier_findMin = demGlacier_findMin[~np.isnan(demGlacier_findMin)]

        #bin from dem raster, but operate on massBalance raster
        #returns: stats array for each bin, the bin edges, and the indices of each value in the bin edges
        demBinVelCount = stats.binned_statistic(demGlacierArray, vArray, statistic='count',
                            bins=range(int(demGlacier_findMin.min()), int(demGlacier_findMin.max()+self.bin), self.bin))
        demBinMeanVel = stats.binned_statistic(demGlacierArray, vArray, statistic='mean',
                            bins=range(int(demGlacier_findMin.min()), int(demGlacier_findMin.max()+self.bin), self.bin))
        demBinhCount = stats.binned_statistic(demGlacierArray, hArray, statistic='count',
                            bins=range(int(demGlacier_findMin.min()), int(demGlacier_findMin.max()+self.bin), self.bin))
        demBinMeanh = stats.binned_statistic(demGlacierArray, hArray, statistic='mean',
                            bins=range(int(demGlacier_findMin.min()), int(demGlacier_findMin.max()+self.bin), self.bin))

        binMeanVel = demBinMeanVel[0]
        binVelBoundaries = demBinMeanVel[1]
        binVelCount = demBinVelCount[0]
        binMeanh = demBinMeanh[0]
        binhBoundaries = demBinMeanh[1]
        binhCount = demBinhCount[0]

        return binMeanVel, binVelBoundaries, binVelCount, binMeanh, binhBoundaries, binhCount
