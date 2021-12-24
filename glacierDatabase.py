from glacierClass import glacier, fullGlacier

def glacierInfo(glac_name, res, avgDensity, elevationBinWidth, vCol):
    # -------------------------------------------------GULKANA---------------------------------------------------------#
    if glac_name == 'Gulkana':
        name = 'Gulkana'            # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'           # Mass balance type
        shape = 'gulkana.shp'
        demFile = 'RGI_GulkanaRegion_DEM.tif'
        hFile = 'Gulkana_Thickness_Millan.tif'
        # hFile = 'GulkanaRegion_thickness.tif'
        # hFile = 'Gulkana_Thickness_model1.tif'
        dhdtFile = 'dhdt_GulkanaRegion.tif'
        # vxFile = 'GulkanaRegion_vy.tif'
        # vyFile = 'GulkanaRegion_vx.tif'
        vxFile = 'Gulkana_vx_Millan.tif'
        vyFile = 'Gulkana_vy_Millan.tif'
        vel_cor = [1, 1]               # velocity direction correction [vxFile_cor, vyFile_cor]
        crs = 'EPSG:32606'              # UTM Zone 6 North. Alaska Albers coordinate system: ('EPSG: 3338')
        # data = [[-3970, -1240, -20], [1355, 1678, 1834]]   # USGS stake data 2001 (index sites A, B, D) (table 29/30)
        # data = [[-2590, -10, 1080], [1357, 1678, 1833]]    # USGS stake data 2000 (index sites A, B, D)
        # data = [[-4370, -1620, -250], [1360, 1679, 1833]]  # USGS stake data 1999 (index sites A, B, D)
        data = [[-3846, -1154, 192], [1368, 1678, 1833]]    # USGS index site stake data annual balance 5yr average (1997-2001)

        # (glacierName, massBalanceType, demFile, thicknessFile, changeInThicknessFile,
        # coordinateSystem, resolution, density, elevationBinWidth, vxFile, vyFile, velocityColumnAvgScaling, mb_data)
        gulkana = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, vxFile, vyFile, vel_cor, vCol)
        return gulkana
    # -------------------------------------------------WOLVERINE-------------------------------------------------------#
    elif glac_name == 'Wolverine':
        name = 'Wolverine'              # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'               # Mass balance type
        shape = 'wolverine.shp'
        demFile = 'DEM_Wolverine.tif'
        hFile = 'Wolverine_Thickness_Millan.tif'
        # hFile = 'Wolverine_Thickness_model3.tif'      # much better data. composite thickness had some major errors
        dhdtFile = 'dhdt_Wolverine.tif'
        # vxFile = 'WolverineGlacier_vy.tif'
        # vyFile = 'WolverineGlacier_vx.tif'
        vxFile = 'WolverineGlacier_vx_Millan.tif'
        vyFile = 'WolverineGlacier_vy_Millan.tif'
        vel_cor = [1, 1]               # velocity direction correction [vxFile_cor, vyFile_cor]
        crs = 'EPSG:32606'              # UTM Zone 6 North
        # data = [[-10000, -5000, -500, 2000], [600, 900, 1200, 1500]]    # USGS stake derived data, annual 2019 (Zeller fig 22)
        data = [[-8000, -4000, 0, 4000], [600, 900, 1200, 1500]]  # USGS stake derived data, annual 2016 (Zeller fig 22)
        wolverine = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                                crs, res, avgDensity, elevationBinWidth, data, vxFile, vyFile, vel_cor, vCol)
        return wolverine
    # -------------------------------------------------LEMON CREEK-----------------------------------------------------#
    elif glac_name == 'LemonCreek':
        name = 'LemonCreek'            # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'               # Mass balance type
        shape = 'lemonCreek.shp'
        demFile = 'DEM_LemonCreek.tif'
        hFile = 'LemonCreek_Thickness_model3.tif'
        dhdtFile = 'dhdt_LemonCreek.tif'
        vxFile = 'LemonCreek_vy.tif'
        vyFile = 'LemonCreek_vx.tif'
        vel_cor = [-1, 1]               # velocity direction correction [vxFile_cor, vyFile_cor]
        crs = 'EPSG:32608'              # UTM Zone 8 North
        data = None
        lemonCreek = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, vxFile, vyFile, vel_cor, vCol)
        return lemonCreek
    # ---------------------------------------------------EKLUTNA-------------------------------------------------------#
    elif glac_name == 'Eklutna':
        name = 'Eklutna'              # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'                # Mass balance type
        shape = 'eklutna.shp'
        demFile = 'Eklutna_dem.tif'
        # hFile = 'Eklutna_thickness.tif'
        hFile = 'Eklutna_thickness_Millan.tif'
        dhdtFile = 'dhdt_Eklutna.tif'
        # vxFile = 'Eklutna_vy.tif'
        # vyFile = 'Eklutna_vx.tif'
        vxFile = 'Eklutna_vx_Millan.tif'
        vyFile = 'Eklutna_vy_Millan.tif'
        vel_cor = [1, 1]               # velocity direction correction [vxFile_cor, vyFile_cor]
        crs = 'EPSG:32606'              # UTM Zone 6 North
        # WATCH OUT FOR VELOCITIES! MAKE SURE ITS CORRECT (SLOPE FROM DEM IS MESSING IT UP)
        data = [[-4000, -1500, 0],[1100, 1400, 1500]]   # estimated from Sass et al., 2017
        eklutna = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, vxFile, vyFile, vel_cor, vCol)
        return eklutna
    # ---------------------------------------------------RAINBOW-------------------------------------------------------#
    elif glac_name == 'Rainbow':
        name = 'Rainbow'            # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'           # Mass balance type
        shape = 'rainbow.shp'
        demFile = 'DEM_Rainbow.tif'
        hFile = 'Rainbow_Thickness.tif'
        dhdtFile = 'dhdt_Rainbow.tif'
        vxFile = 'Rainbow_vy.tif'
        vyFile = 'Rainbow_vx.tif'
        vel_cor = [-1, 1]               # velocity direction correction [vxFile_cor, vyFile_cor]
        crs = 'EPSG:32610'              # UTM Zone 10 North
        data = None
        rainbow = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, vxFile, vyFile, vel_cor, vCol)
        return rainbow
    # ------------------------------------------------SOUTH CASCADE----------------------------------------------------#
    elif glac_name == 'SouthCascade':
        name = 'SouthCascade'        # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'           # Mass balance type
        shape = 'southCascade.shp'
        demFile = 'DEM_SouthCascade.tif'
        hFile = 'SouthCascade_Thickness.tif'
        dhdtFile = 'dhdt_SouthCascade.tif'
        vxFile = 'SouthCascade_vx.tif'
        vyFile = 'SouthCascade_vy.tif'
        vel_cor = [1, 1]                # velocity direction correction [vxFile_cor, vyFile_cor]
        crs = 'EPSG:32610'              # UTM Zone 10 North
        data = None
        southCascade = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, vxFile, vyFile, vel_cor, vCol)
        return  southCascade
    # -------------------------------------------------RIKHA SAMBA-----------------------------------------------------#
    elif glac_name == 'RikhaSamba':
        name = 'RikhaSamba'              # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'                # Mass balance type
        shape = 'rikhaSamba.shp'
        demFile = 'DEM_RikhaSamba.tif'
        hFile = 'RikhaSamba_Thickness.tif'
        dhdtFile = 'dhdt_RikhaSamba.tif'
        # dhdtFile = 'dhdt_ASTER_n28_e083.tif'
        vxFile = 'RikhaSamba_vx.tif'
        vyFile = 'RikhaSamba_vy.tif'
        vel_cor = [1, 1]                # velocity direction correction [vxFile_cor, vyFile_cor]
        crs = 'EPSG:32644'              # UTM Zone 44 North
        # reference data from Miles sup. fig 49, SMB ('published data' plot shows -4 m w.e. at 5400m)
        data = [[-2000, -600, -200, -100, -100, -100], [5400, 5600, 5800, 6000, 6200, 6400]]
        rikhaSamba = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, vxFile, vyFile, vel_cor, vCol)
        return rikhaSamba
    # ---------------------------------------------------CONRAD--------------------------------------------------------#
    elif glac_name == 'Conrad':
        name = 'Conrad'              # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'                # Mass balance type
        shape = 'conrad.shp'
        demFile = 'Conrad_dem.tif'
        hFile = 'Conrad_thickness.tif'
        dhdtFile = 'dhdt_Conrad.tif'
        vxFile = 'Conrad_vy.tif'
        vyFile = 'Conrad_vx.tif'
        vel_cor = [-1, 1]               # velocity direction correction [vxFile_cor, vyFile_cor]
        crs = 'EPSG:32611'              # UTM Zone 11 North
        # estimated from Pelto & Menounos 2021 (estimated from 2016, 2017, 2018 averages)
        data = [[-6000, -3700, -1700, 0, 600, 1000],[2000, 2200, 2400, 2600, 2800, 3000]]
        # WATCH OUT FOR VELOCITIES! MAKE SURE ITS CORRECT (SLOPE FROM DEM IS MESSING IT UP)
        conrad = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, vxFile, vyFile, vel_cor, vCol)
        return conrad
    # -----------------------------------------------ILLECILLEWAET-----------------------------------------------------#
    elif glac_name == 'Illecillewaet':
        name = 'Illecillewaet'              # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'                # Mass balance type
        shape = 'illecillewaet.shp'
        demFile = 'Illecillewaet_dem.tif'
        hFile = 'Illecillewaet_thickness.tif'
        dhdtFile = 'dhdt_Illecillewaet.tif'
        vxFile = 'Illecillewaet_vy.tif'
        vyFile = 'Illecillewaet_vx.tif'
        vel_cor = [-1, 1]               # velocity direction correction [vxFile_cor, vyFile_cor]
        crs = 'EPSG:32611'              # UTM Zone 11 North
        # estimated from Pelto & Menounos 2021 (estimated from 2016, 2017, 2018 averages)
        data = [[-5000, -3800, -2600, -1600, 0, 1200],[2100, 2200, 2300, 2400, 2500, 2600]]
        illecillewaet = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, vxFile, vyFile, vel_cor, vCol)
        return illecillewaet
    # ---------------------------------------------------NORDIC--------------------------------------------------------#
    elif glac_name == 'Nordic':
        name = 'Nordic'              # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'                # Mass balance type
        shape = 'nordic.shp'
        demFile = 'Nordic_dem.tif'
        hFile = 'Nordic_thickness.tif'
        dhdtFile = 'dhdt_Nordic.tif'
        vxFile = 'Nordic_vy.tif'
        vyFile = 'Nordic_vx.tif'
        vel_cor = [-1, 1]               # velocity direction correction [vxFile_cor, vyFile_cor]
        crs = 'EPSG:32611'              # UTM Zone 11 North
        # estimated from Pelto & Menounos 2021 (estimated from 2016, 2017, 2018 averages)
        data = [[-2200, -200, 200, 800],[2200, 2400, 2600, 2800]]
        nordic = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, vxFile, vyFile, vel_cor, vCol)
        return nordic
    # --------------------------------------------------ZILLMER--------------------------------------------------------#
    elif glac_name == 'Zillmer':
        name = 'Zillmer'              # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'                # Mass balance type
        shape = 'zillmer.shp'
        demFile = 'Zillmer_dem.tif'
        hFile = 'Zillmer_thickness.tif'
        dhdtFile = 'dhdt_Zillmer.tif'
        vxFile = 'Zillmer_vy.tif'
        vyFile = 'Zillmer_vx.tif'
        vel_cor = [-1, 1]               # velocity direction correction [vxFile_cor, vyFile_cor]
        crs = 'EPSG:32611'              # UTM Zone 11 North
        # estimated from Pelto & Menounos 2021 (estimated from 2016, 2017, 2018 averages)
        data = [[-1800, -400, 1000],[2200, 2400, 2600]]
        # WATCH OUT FOR VELOCITIES! MAKE SURE ITS CORRECT (SLOPE FROM DEM IS MESSING IT UP)
        zillmer = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, vxFile, vyFile, vel_cor, vCol)
        return zillmer
    # -----------------------------------------------------------------------------------------------------------------#


