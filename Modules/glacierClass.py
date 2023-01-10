# Final project classes and methods
import rasterio
import rasterio.plot
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform
from rasterio.merge import merge
import geopandas as gpd
import numpy as np
import os

class glacier:
    def __init__(self, glacierName, massBalanceType, shapeFile, demFile, thicknessFile, changeInThicknessFile,
                 coordinateSystem, resolution, density, elevationBinWidth, data, data_var, vxFile, vyFile,
                 velocityCorrection, velocityColumnAvgScaling):
        self.name = glacierName
        self.type = massBalanceType  # this is either 'Total' or 'Climatic'
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
        self.vx = vxFile
        self.vy = vyFile
        self.vCor = velocityCorrection
        self.vCol = velocityColumnAvgScaling

    def getThickness(self, h_select):
        # OBTAIN DEM FILE
        if h_select == 'MillanThickness':
            h = self.MillanThicknessRGI1_2()
        elif h_select == 'FarinottiThickness':
            h = 'RGI1_Thickness_Farinotti_Composite/RGI60-01.' + self.name + '_thickness.tif'
        elif h_select == 'OGGM_Thickness':
            oggm_filepath = os.path.abspath(os.path.join(os.getcwd(), '../../../../../Volumes/LaCie' \
                                                                      '/Glacier Data Sets 2-18/Full Data Sets' \
                                                                      '/RGI Region 1 and 2/Farinotti Thickness and DEMs' \
                                                                      '/results_model_3_thickness/RGI60-01'))
            h = oggm_filepath + '/thickness_RGI60-01.' + self.name + '.tif'
        elif h_select == 'FarinottiThicknessFlowlineCorr':
            h = 'Thickness Corrections/00570_h_farinotti_flowline_corrected.tif'
        elif h_select == 'MillanThicknessFlowlineCorr':
            h = 'Thickness Corrections/00570_h_millan_flowline_corrected.tif'
        elif h_select == 'OGGMThicknessFlowlineCorr':
            h = 'Thickness Corrections/00570_h_oggm_flowline_corrected.tif'
        return h

    def getDEM(self, dem_select):
        # OBTAIN DEM FILE
        if dem_select == 'Farinotti DEM':
            dem = 'RGI1_DEM/RGI60-01.' + self.name + '_dem.tif'
        elif dem_select == 'Copernicus DEM':
            dem = 'Copernicus and OGGM DEM/Copernicus DEM/RGI60.01.' + self.name + '_output_COP30.tif'
        elif dem_select == 'OGGM DEM':
            dem = 'Copernicus and OGGM DEM/OGGM DEM/RGI60.01.' + self.name + '_dem_oggm.tif'
        elif dem_select == 'USGS DEM':
            dem = 'Copernicus and OGGM DEM/USGS DEM/RGI60.01.' + self.name + '_USGS_2m_DEM_filled.tif'
        return dem

    def getVelocity(self, v_data, n=None):
        if v_data == 'ITS_LIVE_20yrComposite':
            vxFile = 'ITS_LIVE_vy_20yr_composite.tif'
            vyFile = 'ITS_LIVE_vx_20yr_composite.tif'
            vel_cor = [-1, 1]
        elif v_data == 'ITS_LIVE_2017-2018':
            vxFile = 'ITS_LIVE_vy_2017_2018_composite.tif'
            vyFile = 'ITS_LIVE_vx_2017_2018_composite.tif'
            vel_cor = [-1, 1]
        elif v_data == 'ITS_LIVE_2018':
            vxFile = 'ITS_LIVE_vy_2018.tif'
            vyFile = 'ITS_LIVE_vx_2018.tif'
            vel_cor = [-1, 1]
        elif v_data == 'MillanVelocity_2017-2018':
            vxFile, vyFile = self.MillanVelocityRGI1_2()
            vel_cor = [1, 1]
        elif v_data == 'RETREAT_2017-2018':
            vxFile = 'RETREAT_vx_2017-2018_composite.tif'
            vyFile = 'RETREAT_vy_2017-2018_composite.tif'
            vel_cor = [1, 1]
        elif v_data == 'RETREAT_2018':
            vxFile = 'RETREAT_vx_2018.tif'
            vyFile = 'RETREAT_vy_2018.tif'
            vel_cor = [1, 1]
        elif v_data == 'RETREAT_2020':
            sys.exit('RETREAT (2020) file does not exist for ' + self.name)
        elif v_data == 'RETREAT_2015-2020':
            sys.exit('RETREAT (2015-2020) file does not exist for ' + self.name)
        # elif v_data == 'ISSM_Model_Farinotti':
        #     sys.exit('ISSM Model velocity (from Farinotti h) file does not exist for ' + self.name)
        #     vel_cor = [1, -1]
        # elif v_data == 'ISSM_Model_Millan':
        #     vxFile = 'Modeled Velocities/00570_ISSM_Millan_vx.tif'
        #     vyFile = 'Modeled Velocities/00570_ISSM_Millan_vy.tif'
        #     vel_cor = [1, -1]
        # elif n is not None:
        elif 'Icepack_Model' in v_data:
            vxFile = 'Modeled Velocities/00570_Icepack_output' + str(n) + '_vx.tif'
            vyFile = 'Modeled Velocities/00570_Icepack_output' + str(n) + '_vy.tif'
            vel_cor = [1, 1]
        elif v_data == 'Fabricated Velocity (570)':
            vxFile = 'Saved Velocity Outputs/00570_fabricated_vx.tif'
            vyFile = 'Saved Velocity Outputs/00570_fabricated_vy.tif'
            vel_cor = [1, 1]
        elif v_data == 'Fabricated Velocity Corrected (570)':
            vxFile = 'Saved Velocity Outputs/00570_fabricated_vx_corrected.tif'
            vyFile = 'Saved Velocity Outputs/00570_fabricated_vy_corrected.tif'
            vel_cor = [1, 1]
        return vxFile, vyFile, vel_cor

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
        # create a mosaic if glacier falls between/overlaps RGI1 and 2 raster files
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
        destination_vx = 'temp_vx.tif'
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


    def MillanVelocityErrRGI1_2(self):
        # create a mosaic if glacier falls between/overlaps RGI1 and 2 raster files
        # obtain northing and easting values from shapefile
        # nearly identical to MillanVelocityRGI1_2 but this filepaths are to error files
        zips = gpd.read_file(self.shape)
        zips = zips.to_crs('epsg:4326')
        list_of_coords = list(zips['geometry'][0].exterior.coords)
        easting = []
        subfolder = 'Velocity Error Files/'
        for north_east_tuple in list_of_coords:
            easting.append(north_east_tuple[0])

        if min(easting) and max(easting) < -136:
            return subfolder + 'std_vx_Millan_RGI1.tif', subfolder + 'std_vy_Millan_RGI1.tif'
        elif min(easting) and max(easting) > -136.3:
            return subfolder + 'std_vx_Millan_RGI2.tif', subfolder + 'std_vy_Millan_RGI2.tif'
        else:
            vx_files = [subfolder + 'std_vx_Millan_RGI1.tif', subfolder + 'std_vx_Millan_RGI2.tif']
            vy_files = [subfolder + 'std_vy_Millan_RGI1.tif', subfolder + 'std_vy_Millan_RGI2.tif']

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
        destination_vx = 'temp_vx_err.tif'
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
        destination_vy = 'temp_vy_err.tif'
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



