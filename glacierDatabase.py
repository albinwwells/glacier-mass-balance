from Modules.glacierClass import glacier, fullGlacier
import numpy as np
import sys

def glacierInfo(glac_name, res, avgDensity, elevationBinWidth, vCol, h_data, v_data):
    # -------------------------------------------------GULKANA---------------------------------------------------------#
    if glac_name == 'Gulkana':
        name = 'Gulkana'            # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'           # Mass balance type
        shape = 'gulkana.shp'
        demFile = 'RGI_GulkanaRegion_DEM.tif'
        dhdtFile = 'dhdt_GulkanaRegion.tif'
        if h_data == 'FarinottiThickness':
            hFile = 'GulkanaRegion_thickness.tif'               # Farinotti composite
            # hFile = 'Gulkana_Thickness_model1.tif'
        elif h_data == 'MillanThickness':
            hFile = 'Gulkana_Thickness_Millan.tif'
        if v_data == 'ITS_LIVE_20yrComposite':
            vxFile = 'GulkanaRegion_vy.tif'                     # ITS_LIVE 120m composite, 20yr avg
            vyFile = 'GulkanaRegion_vx.tif'
            vel_cor = [-1, 1]                       # velocity direction correction [vxFile_cor, vyFile_cor]
        elif v_data == 'ITS_LIVE_2017-2018':
            vxFile = 'Gulkana_vy_ITSLIVE_2017_2018_composite.tif'
            vyFile = 'Gulkana_vx_ITSLIVE_2017_2018_composite.tif'
            vel_cor = [-1, 1]
        elif v_data == 'MillanVelocity_2017-2018':
            vxFile = 'Gulkana_vx_Millan.tif'                    # 2017-2018 composite
            vyFile = 'Gulkana_vy_Millan.tif'
            vel_cor = [1, 1]
        elif v_data == 'RETREAT_2020':
            vxFile = 'Gulkana_vx_retreat2020.tif'                   # 2020 annual velocity data
            vyFile = 'Gulkana_vy_retreat2020.tif'
            vel_cor = [1, 1]
        elif v_data == 'RETREAT_2015-2020':
            vxFile = 'Gulkana_vx_retreat_composite2015_2020.tif'         # 2015-2020 velocity composite
            vyFile = 'Gulkana_vy_retreat_composite2015_2020.tif'
            vel_cor = [1, 1]
        elif v_data == 'RETREAT_2017-2018':
            vxFile = 'Gulkana_vx_retreat_2017_2018.tif'        # 2017-2018 velocity composite
            vyFile = 'Gulkana_vy_retreat_2017_2018.tif'
            vel_cor = [1, 1]
        crs = 'EPSG:32606'              # UTM Zone 6 North. Alaska Albers coordinate system: ('EPSG: 3338')
        # WGMS 2015-2020 stake data
        # data = [[-3295, -2757, -1508, -250, -604, 670], [1451, 1544, 1690, 1850, 1879, 2030]]
        # data_min = [-4290, -3780, -2460, -720, -1720, -440]
        # data_max = [-2450, -1810, -700, 780, 400, 1500]

        # WGMS 2000-2020 stake data
        data = [[-1254, 327], [1690, 1850]]    # USGS index site stake data annual balance 5yr average (1997-2001)
        data_min = [-3880, -720]
        data_max = [330, 2030]

        data_var = abs(np.array([np.array(data_min) - np.array(data[0]), np.array(data_max) - np.array(data[0])]))
        # (glacierName, massBalanceType, demFile, thicknessFile, changeInThicknessFile,
        # coordinateSystem, resolution, density, elevationBinWidth, vxFile, vyFile, velocityColumnAvgScaling, mb_data)
        gulkana = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, data_var, vxFile, vyFile, vel_cor, vCol)
        return gulkana
    # -------------------------------------------------WOLVERINE-------------------------------------------------------#
    elif glac_name == 'Wolverine':
        name = 'Wolverine'              # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'               # Mass balance type
        shape = 'wolverine.shp'
        demFile = 'DEM_Wolverine.tif'
        dhdtFile = 'dhdt_Wolverine.tif'
        if h_data == 'FarinottiThickness':
            #hFile = 'Wolverine_Thickness.tif'
            hFile = 'Wolverine_Thickness_model3.tif'      # much better data. composite thickness had some major errors
        elif h_data == 'MillanThickness':
            hFile = 'Wolverine_Thickness_Millan.tif'
        if v_data == 'ITS_LIVE_20yrComposite':
            vxFile = 'WolverineGlacier_vy.tif'                      # ITS_LIVE 120m composite, 20yr avg
            vyFile = 'WolverineGlacier_vx.tif'
            vel_cor = [-1, 1]
        elif v_data == 'ITS_LIVE_2017-2018':
            vxFile = 'Wolverine_vy_ITSLIVE_2017_2018_composite.tif'
            vyFile = 'Wolverine_vx_ITSLIVE_2017_2018_composite.tif'
            vel_cor = [-1, 1]
        elif v_data == 'MillanVelocity_2017-2018':
            vxFile = 'WolverineGlacier_vx_Millan.tif'  # 2017-2018 composite
            vyFile = 'WolverineGlacier_vy_Millan.tif'
            vel_cor = [1, 1]
        elif v_data == 'RETREAT_2020':
            vxFile = 'Wolverine_vx_retreat2020.tif'  # 2020 annual velocity data
            vyFile = 'Wolverine_vy_retreat2020.tif'
            vel_cor = [1, 1]
        elif v_data == 'RETREAT_2015-2020':
            vxFile = 'Wolverine_vx_retreat_composite2015_2020.tif'  # 2015-2020 velocity composite
            vyFile = 'Wolverine_vy_retreat_composite2015_2020.tif'
            vel_cor = [1, 1]
        elif v_data == 'RETREAT_2017-2018':
            vxFile = 'Wolverine_vx_retreat_2017_2018.tif'  # 2017-2018 velocity composite
            vyFile = 'Wolverine_vy_retreat_2017_2018.tif'
            vel_cor = [1, 1]
        crs = 'EPSG:32606'              # UTM Zone 6 North
        # WGMS 2015-2020 stake data
        # data = [[-6797, -2462, -1797, -452, 542, 758, 1285], [624, 1004, 1067, 1235, 1298, 1370, 1371]]
        # data_min = [-8810, -3200, -2650, -1110, 30, -10, 450]
        # data_max = [-5860, -1750, -1170, 310, 1660, 2100, 2780]

        # WGMS 2000-2020 stake data
        data = [[-1616, 919], [1067, 1298]]
        data_min = [-3060, -510]
        data_max = [-40, 2740]

        data_var = abs(np.array([np.array(data_min) - np.array(data[0]), np.array(data_max) - np.array(data[0])]))
        wolverine = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                                crs, res, avgDensity, elevationBinWidth, data, data_var, vxFile, vyFile, vel_cor, vCol)
        return wolverine
    # ---------------------------------------------------EKLUTNA-------------------------------------------------------#
    elif glac_name == 'Eklutna':
        name = 'Eklutna'              # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'                # Mass balance type
        shape = 'eklutna.shp'
        demFile = 'Eklutna_dem.tif'
        dhdtFile = 'dhdt_Eklutna.tif'
        if h_data == 'FarinottiThickness':
            hFile = 'Eklutna_thickness.tif'
        elif h_data == 'MillanThickness':
            hFile = 'Eklutna_thickness_Millan.tif'
        if v_data == 'ITS_LIVE_20yrComposite':
            vxFile = 'Eklutna_vy.tif'  # ITS_LIVE 120m composite, 20yr avg
            vyFile = 'Eklutna_vx.tif'
            vel_cor = [-1, 1]
        elif v_data == 'ITS_LIVE_2017-2018':
            vxFile = 'Eklutna_vy_ITSLIVE_2017_2018_composite.tif'
            vyFile = 'Eklutna_vx_ITSLIVE_2017_2018_composite.tif'
            vel_cor = [-1, 1]
        elif v_data == 'MillanVelocity_2017-2018':
            vxFile = 'Eklutna_vx_Millan.tif'  # 2017-2018 composite
            vyFile = 'Eklutna_vy_Millan.tif'
            vel_cor = [1, 1]
        elif v_data == 'RETREAT_2020':
            vxFile = 'Eklutna_vx_retreat.tif'  # 2020 annual velocity data
            vyFile = 'Eklutna_vy_retreat.tif'
            vel_cor = [1, 1]
        elif v_data == 'RETREAT_2015-2020':
            vxFile = 'Eklutna_vx_retreat_composite.tif'  # 2015-2020 velocity composite
            vyFile = 'Eklutna_vy_retreat_composite.tif'
            vel_cor = [1, 1]
        elif v_data == 'RETREAT_2017-2018':
            vxFile = 'Eklutna_vx_retreat_2017_2018.tif'  # 2017-2018 velocity composite
            vyFile = 'Eklutna_vy_retreat_2017_2018.tif'
            vel_cor = [1, 1]
        crs = 'EPSG:32606'              # UTM Zone 6 North
        # WATCH OUT FOR VELOCITIES! MAKE SURE ITS CORRECT (SLOPE FROM DEM IS MESSING IT UP)
        data = [[-4000, -1500, 0],[1100, 1400, 1500]]   # estimated from Sass et al., 2017
        data_var = None
        eklutna = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, data_var, vxFile, vyFile, vel_cor, vCol)
        return eklutna
    # -------------------------------------------------LEMON CREEK-----------------------------------------------------#
    elif glac_name == 'LemonCreek':
        name = 'LemonCreek'            # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'               # Mass balance type
        shape = 'lemonCreek.shp'
        demFile = 'DEM_LemonCreek.tif'
        dhdtFile = 'dhdt_LemonCreek.tif'
        if h_data == 'FarinottiThickness':
            hFile = 'LemonCreek_Thickness_model3.tif'  # Farinotti model 3: OGGM/PyGEM
        elif h_data == 'MillanThickness':
            hFile = 'LemonCreek_Thickness_Millan.tif'
        if v_data == 'ITS_LIVE_20yrComposite':
            vxFile = 'LemonCreek_vy.tif'  # ITS_LIVE 120m composite, 20yr avg
            vyFile = 'LemonCreek_vx.tif'
            vel_cor = [-1, 1]
        elif v_data == 'ITS_LIVE_2017-2018':
            sys.exit('ITS_LIVE (2017-2018) file does not exist for ' + name)
        elif v_data == 'MillanVelocity_2017-2018':
            vxFile = 'LemonCreek_vx_Millan.tif'  # 2017-2018 composite
            vyFile = 'LemonCreek_vy_Millan.tif'
            vel_cor = [1, 1]
        elif v_data == 'RETREAT_2020':
            sys.exit('RETREAT (2020) file does not exist for ' + name)
        elif v_data == 'RETREAT_2015-2020':
            sys.exit('RETREAT (2015-2020) file does not exist for ' + name)
        elif v_data == 'RETREAT_2017-2018':
            vxFile = 'LemonCreek_vx_retreat_2017_2018.tif'  # 2017-2018 velocity composite
            vyFile = 'LemonCreek_vy_retreat_2017_2018.tif'
            vel_cor = [1, 1]
        crs = 'EPSG:32608'              # UTM Zone 8 North
        # WGMS 2015-2020 stake data
        data = [[-3743, -2553, -845, 1452, -435, 6], [823, 943, 1074, 1143, 1184, 1234]]
        data_min = [-5590, -4550, -2600, 560, -1880, -1740]
        data_max = [-1760, -810, 1260, 1910, 1400, 1240]

        # WGMS 2000-2020 stake data
        # data = [[894, 1887, 1073, 1751], [1074, 1143, 1184, 1198]]
        # data_min = [-2600, 560, -1880, 680]
        # data_max = [3900, 3562, 3570, 3070]

        data_var = abs(np.array([np.array(data_min) - np.array(data[0]), np.array(data_max) - np.array(data[0])]))
        lemonCreek = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, data_var, vxFile, vyFile, vel_cor, vCol)
        return lemonCreek
    # ---------------------------------------------------RAINBOW-------------------------------------------------------#
    elif glac_name == 'Rainbow':
        name = 'Rainbow'            # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'           # Mass balance type
        shape = 'rainbow.shp'
        demFile = 'DEM_Rainbow.tif'
        dhdtFile = 'dhdt_Rainbow.tif'
        if h_data == 'FarinottiThickness':
            hFile = 'Rainbow_Thickness.tif'  # Farinotti composite
        elif h_data == 'MillanThickness':
            sys.exit('Millan Thickness file does not exist for ' + name)
        if v_data == 'ITS_LIVE_20yrComposite':
            vxFile = 'Rainbow_vy.tif'                 # ITS_LIVE 120m composite, 20yr avg
            vyFile = 'Rainbow_vx.tif'
            vel_cor = [-1, 1]
        elif v_data == 'ITS_LIVE_2017-2018':
            sys.exit('ITS_LIVE (2017-2018) file does not exist for ' + name)
        elif v_data == 'MillanVelocity_2017-2018':
            sys.exit('Millan Velocity (2017-2018) file does not exist for ' + name)
        elif v_data == 'RETREAT_2020':
            sys.exit('RETREAT (2020) file does not exist for ' + name)
        elif v_data == 'RETREAT_2015-2020':
            vxFile = 'Rainbow_vx_retreat_composite.tif'
            vyFile = 'Rainbow_vy_retreat_composite.tif'
            vel_cor = [1, 1]
        crs = 'EPSG:32610'              # UTM Zone 10 North
        data = None
        data_var = None
        rainbow = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, data_var, vxFile, vyFile, vel_cor, vCol)
        return rainbow
    # ------------------------------------------------SOUTH CASCADE----------------------------------------------------#
    elif glac_name == 'SouthCascade':
        name = 'SouthCascade'        # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'           # Mass balance type
        shape = 'southCascade.shp'
        demFile = 'DEM_SouthCascade.tif'
        dhdtFile = 'dhdt_SouthCascade.tif'
        if h_data == 'FarinottiThickness':
            hFile = 'SouthCascade_Thickness.tif'  # Farinotti composite
        elif h_data == 'MillanThickness':
            sys.exit('Millan Thickness file does not exist for ' + name)
        if v_data == 'ITS_LIVE_20yrComposite':
            vxFile = 'SouthCascade_vx.tif'  # ITS_LIVE 120m composite, 20yr avg
            vyFile = 'SouthCascade_vy.tif'
            vel_cor = [-1, 1]
        elif v_data == 'ITS_LIVE_2017-2018':
            sys.exit('ITS_LIVE (2017-2018) file does not exist for ' + name)
        elif v_data == 'MillanVelocity_2017-2018':
            sys.exit('Millan Velocity (2017-2018) file does not exist for ' + name)
        elif v_data == 'RETREAT_2020':
            sys.exit('RETREAT (2020) file does not exist for ' + name)
        elif v_data == 'RETREAT_2015-2020':
            vxFile = 'SouthCascade_vx_retreat_composite.tif'
            vyFile = 'SouthCascade_vy_retreat_composite.tif'
            vel_cor = [1, 1]
        crs = 'EPSG:32610'              # UTM Zone 10 North
        data = None
        data_var = None
        southCascade = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, data_var, vxFile, vyFile, vel_cor, vCol)
        return  southCascade
    # ---------------------------------------------------CONRAD--------------------------------------------------------#
    elif glac_name == 'Conrad':
        name = 'Conrad'              # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'                # Mass balance type
        shape = 'conrad.shp'
        demFile = 'Conrad_dem.tif'
        dhdtFile = 'dhdt_Conrad.tif'
        if h_data == 'FarinottiThickness':
            hFile = 'Conrad_thickness.tif'  # Farinotti composite
        elif h_data == 'MillanThickness':
            hFile = 'Conrad_Thickness_Millan.tif'
        if v_data == 'ITS_LIVE_20yrComposite':
            vxFile = 'Conrad_vy.tif'  # ITS_LIVE 120m composite, 20yr avg
            vyFile = 'Conrad_vx.tif'
            vel_cor = [-1, 1]
        elif v_data == 'ITS_LIVE_2017-2018':
            vxFile = 'Conrad_vy_ITSLIVE_2017_2018_composite.tif'
            vyFile = 'Conrad_vx_ITSLIVE_2017_2018_composite.tif'
            vel_cor = [-1, 1]
        elif v_data == 'MillanVelocity_2017-2018':
            vxFile = 'Conrad_vx_Millan.tif'  # 2017-2018 composite
            vyFile = 'Conrad_vy_Millan.tif'
            vel_cor = [1, 1]
        elif v_data == 'RETREAT_2020':
            sys.exit('RETREAT (2020) file does not exist for ' + name)
        elif v_data == 'RETREAT_2015-2020':
            sys.exit('RETREAT (2015-2020) file does not exist for ' + name)
        crs = 'EPSG:32611'              # UTM Zone 11 North
        # estimated from Pelto & Menounos 2021 (estimated from 2016, 2017, 2018 averages)
        data = [[-6000, -3700, -1700, 0, 600, 1000],[2000, 2200, 2400, 2600, 2800, 3000]]
        data_var = None
        # WATCH OUT FOR VELOCITIES! MAKE SURE ITS CORRECT (SLOPE FROM DEM IS MESSING IT UP)
        conrad = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, data_var, vxFile, vyFile, vel_cor, vCol)
        return conrad
    # -----------------------------------------------ILLECILLEWAET-----------------------------------------------------#
    elif glac_name == 'Illecillewaet':
        name = 'Illecillewaet'              # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'                # Mass balance type
        shape = 'illecillewaet.shp'
        demFile = 'Illecillewaet_dem.tif'
        dhdtFile = 'dhdt_Illecillewaet.tif'
        if h_data == 'FarinottiThickness':
            hFile = 'Illecillewaet_thickness.tif'  # Farinotti composite
        elif h_data == 'MillanThickness':
            hFile = 'Illecillewaet_Thickness_Millan.tif'
        if v_data == 'ITS_LIVE_20yrComposite':
            vxFile = 'Illecillewaet_vy.tif'  # ITS_LIVE 120m composite, 20yr avg
            vyFile = 'Illecillewaet_vx.tif'
            vel_cor = [-1, 1]
        elif v_data == 'ITS_LIVE_2017-2018':
            vxFile = 'Illecillewaet_vy_ITSLIVE_2017_2018_composite.tif'
            vyFile = 'Illecillewaet_vx_ITSLIVE_2017_2018_composite.tif'
            vel_cor = [-1, 1]
        elif v_data == 'MillanVelocity_2017-2018':
            vxFile = 'Illecillewaet_vx_Millan.tif'  # 2017-2018 composite
            vyFile = 'Illecillewaet_vy_Millan.tif'
            vel_cor = [1, 1]
        elif v_data == 'RETREAT_2020':
            sys.exit('RETREAT (2020) file does not exist for ' + name)
        elif v_data == 'RETREAT_2015-2020':
            sys.exit('RETREAT (2015-2020) file does not exist for ' + name)
        crs = 'EPSG:32611'              # UTM Zone 11 North
        # estimated from Pelto & Menounos 2021 (estimated from 2016, 2017, 2018 averages)
        data = [[-5000, -3800, -2600, -1600, 0, 1200],[2100, 2200, 2300, 2400, 2500, 2600]]
        data_var = None
        illecillewaet = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, data_var, vxFile, vyFile, vel_cor, vCol)
        return illecillewaet
    # ---------------------------------------------------NORDIC--------------------------------------------------------#
    elif glac_name == 'Nordic':
        name = 'Nordic'              # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'                # Mass balance type
        shape = 'nordic.shp'
        demFile = 'Nordic_dem.tif'
        dhdtFile = 'dhdt_Nordic.tif'
        if h_data == 'FarinottiThickness':
            hFile = 'Nordic_thickness.tif'  # Farinotti composite
        elif h_data == 'MillanThickness':
            hFile = 'Nordic_Thickness_Millan.tif'
        if v_data == 'ITS_LIVE_20yrComposite':
            vxFile = 'Nordic_vy.tif'  # ITS_LIVE 120m composite, 20yr avg
            vyFile = 'Nordic_vx.tif'
            vel_cor = [-1, 1]
        elif v_data == 'ITS_LIVE_2017-2018':
            vxFile = 'Nordic_vy_ITSLIVE_2017_2018_composite.tif'
            vyFile = 'Nordic_vx_ITSLIVE_2017_2018_composite.tif'
            vel_cor = [-1, 1]
        elif v_data == 'MillanVelocity_2017-2018':
            vxFile = 'Nordic_vx_Millan.tif'  # 2017-2018 composite
            vyFile = 'Nordic_vy_Millan.tif'
            vel_cor = [1, 1]
        elif v_data == 'RETREAT_2020':
            sys.exit('RETREAT (2020) file does not exist for ' + name)
        elif v_data == 'RETREAT_2015-2020':
            sys.exit('RETREAT (2015-2020) file does not exist for ' + name)
        crs = 'EPSG:32611'              # UTM Zone 11 North
        # estimated from Pelto & Menounos 2021 (estimated from 2016, 2017, 2018 averages)
        data = [[-2200, -200, 200, 800],[2200, 2400, 2600, 2800]]
        data_var = None
        nordic = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, data_var, vxFile, vyFile, vel_cor, vCol)
        return nordic
    # --------------------------------------------------ZILLMER--------------------------------------------------------#
    elif glac_name == 'Zillmer':
        name = 'Zillmer'              # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'                # Mass balance type
        shape = 'zillmer.shp'
        demFile = 'Zillmer_dem.tif'
        dhdtFile = 'dhdt_Zillmer.tif'
        if h_data == 'FarinottiThickness':
            hFile = 'Zillmer_thickness.tif'  # Farinotti composite
        elif h_data == 'MillanThickness':
            hFile = 'Zillmer_Thickness_Millan.tif'
        if v_data == 'ITS_LIVE_20yrComposite':
            vxFile = 'Zillmer_vy.tif'  # ITS_LIVE 120m composite, 20yr avg
            vyFile = 'Zillmer_vx.tif'
            vel_cor = [-1, 1]
        elif v_data == 'ITS_LIVE_2017-2018':
            vxFile = 'Zillmer_vy_ITSLIVE_2017_2018_composite.tif'
            vyFile = 'Zillmer_vx_ITSLIVE_2017_2018_composite.tif'
            vel_cor = [-1, 1]
        elif v_data == 'MillanVelocity_2017-2018':
            vxFile = 'Zillmer_vx_Millan.tif'  # 2017-2018 composite
            vyFile = 'Zillmer_vy_Millan.tif'
            vel_cor = [1, 1]
        elif v_data == 'RETREAT_2020':
            sys.exit('RETREAT (2020) file does not exist for ' + name)
        elif v_data == 'RETREAT_2015-2020':
            sys.exit('RETREAT (2015-2020) file does not exist for ' + name)
        crs = 'EPSG:32611'              # UTM Zone 11 North
        # estimated from Pelto & Menounos 2021 (estimated from 2016, 2017, 2018 averages)
        data = [[-1800, -400, 1000],[2200, 2400, 2600]]
        data_var = None
        # WATCH OUT FOR VELOCITIES! MAKE SURE ITS CORRECT (SLOPE FROM DEM IS MESSING IT UP)
        zillmer = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, data_var, vxFile, vyFile, vel_cor, vCol)
        return zillmer
    # -------------------------------------------------RIKHA SAMBA-----------------------------------------------------#
    elif glac_name == 'RikhaSamba':
        name = 'RikhaSamba'              # Glacier name (folder should be the same as glacier name)
        type = 'Climatic'                # Mass balance type
        shape = 'rikhaSamba.shp'
        demFile = 'DEM_RikhaSamba.tif'
        dhdtFile = 'dhdt_RikhaSamba.tif'
        # dhdtFile = 'dhdt_ASTER_n28_e083.tif'
        if h_data == 'FarinottiThickness':
            hFile = 'RikhaSamba_Thickness.tif'  # Farinotti composite
        elif h_data == 'MillanThickness':
            sys.exit('Millan Thickness file does not exist for ' + name)
        if v_data == 'ITS_LIVE_20yrComposite':
            vxFile = 'RikhaSamba_vx.tif'  # ITS_LIVE 120m composite, 20yr avg
            vyFile = 'RikhaSamba_vy.tif'
            vel_cor = [-1, 1]
        elif v_data == 'ITS_LIVE_2017-2018':
            sys.exit('ITS_LIVE (2017-2018) file does not exist for ' + name)
        elif v_data == 'MillanVelocity_2017-2018':
            sys.exit('Millan Velocity (2017-2018) file does not exist for ' + name)
        elif v_data == 'RETREAT_2020':
            sys.exit('RETREAT (2020) file does not exist for ' + name)
        elif v_data == 'RETREAT_2015-2020':
            sys.exit('RETREAT (2015-2020) file does not exist for ' + name)
        crs = 'EPSG:32644'              # UTM Zone 44 North
        # reference data from Miles sup. fig 49, SMB ('published data' plot shows -4 m w.e. at 5400m)
        data = [[-2000, -600, -200, -100, -100, -100], [5400, 5600, 5800, 6000, 6200, 6400]]
        data_var = None
        rikhaSamba = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, data_var, vxFile, vyFile, vel_cor, vCol)
        return rikhaSamba
    # ---------------------------------------------ALL OTHER GLACIERS--------------------------------------------------#
    else:
        name = 'Other'                                  # Glacier name!!!
        type = 'Climatic'                               # Mass balance type
        shape = '01_rgi60_Alaska.shp'                   # shapefile from RGI
        demFile = ''                                    # placeholder: get RGI ID, then update this in main_vel_comp
        dhdtFile = ''                                   # placeholder: get N and W values, then update in main_vel_comp
        if h_data == 'FarinottiThickness':
            hFile = ''                                  # placeholder: get RGI ID, then update this in main_vel_comp
        elif h_data == 'MillanThickness':
            hFile = ''
        if v_data == 'ITS_LIVE_20yrComposite':
            vxFile = 'ITS_LIVE_vy_20yr_composite.tif'
            vyFile = 'ITS_LIVE_vx_20yr_composite.tif'
            vel_cor = [-1, 1]
        elif v_data == 'ITS_LIVE_2017-2018':
            vxFile = 'ITS_LIVE_vy_2017_2018_composite.tif'
            vyFile = 'ITS_LIVE_vx_2017_2018_composite.tif'
            vel_cor = [-1, 1]
        elif v_data == 'MillanVelocity_2017-2018':
            vxFile = 'vx_Millan_RGI1.tif'
            vyFile = 'vy_Millan_RGI1.tif'
            vel_cor = [1, 1]
        elif v_data == 'RETREAT_2017-2018':
            vxFile = 'RETREAT_vx_2017-2018_composite.tif'
            vyFile = 'RETREAT_vy_2017-2018_composite.tif'
            vel_cor = [1, 1]
        elif v_data == 'RETREAT_2020':
            sys.exit('RETREAT (2020) file does not exist for ' + name)
        elif v_data == 'RETREAT_2015-2020':
            sys.exit('RETREAT (2015-2020) file does not exist for ' + name)
        crs = 'EPSG:32606'              # UTM Zone 6 North
        data = None
        data_var = None
        otherGlacier = fullGlacier(name, type, shape, demFile, hFile, dhdtFile,
                              crs, res, avgDensity, elevationBinWidth, data, data_var, vxFile, vyFile, vel_cor, vCol)
        return otherGlacier
    # -----------------------------------------------------------------------------------------------------------------#


