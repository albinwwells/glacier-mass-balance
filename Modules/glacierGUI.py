from tkinter import *

def pickGlacierData():
    # MUST DOWNLOAD FOLDERS FROM EXTERNAL HARD DRIVE FOR ANY OF THESE EXCEPT G12 (ALL OTHER GLACIERS)
    g1 = 'Gulkana'
    g2 = 'Wolverine'
    g3 = 'Eklutna'
    g4 = 'LemonCreek'
    g5 = 'Rainbow'
    g6 = 'SouthCascade'
    g7 = 'Conrad'
    g8 = 'Illecillewaet'
    g9 = 'Nordic'
    g10 = 'Zillmer'
    g11 = 'RikhaSamba'
    g12 = 'All Other Glaciers'

    d1 = 'Farinotti DEM'
    d2 = 'Copernicus DEM'  # https://portal.opentopography.org/raster?jobId=rt1653410714595
    d3 = 'OGGM DEM'
    d4 = 'USGS DEM'

    h1 = 'FarinottiThickness'
    h2 = 'MillanThickness'
    h3 = 'OGGM_Thickness'
    h4 = 'FarinottiThicknessFlowlineCorr'
    h5 = 'MillanThicknessFlowlineCorr'
    h6 = 'OGGMThicknessFlowlineCorr'

    v1 = 'ITS_LIVE_20yrComposite'
    v2 = 'ITS_LIVE_2017-2018'
    v2_2 = 'ITS_LIVE_2018'
    v3 = 'MillanVelocity_2017-2018'
    v4 = 'RETREAT_2017-2018'
    v4_2 = 'RETREAT_2018'
    v5 = 'RETREAT_2020'
    v6 = 'RETREAT_2015-2020'
    v7 = 'ISSM_Model_Farinotti'
    v8 = 'ISSM_Model_Millan'
    v9 = 'Icepack_Model'
    v10 = 'Fabricated Velocity (570)'
    v11 = 'Fabricated Velocity Corrected (570)'

    s1 = 'RGI'
    s2 = 'USGS'

    glaciers = ([g12])    #, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11)
    dem_data = (d1, d2, d3, d4)
    thick_data = (h1, h2, h3, h4, h5, h6)
    vel_data = (v1, v2, v2_2, v3, v4, v4_2, v5, v6, v7, v8, v9, v9, v9, v10, v11)
    shp_data = (s1, s2)

    # chose glacier to perform calculations with GUI
    # select one of multiple glaciers in the set
    # double click, press the return key, or click the 'SELECT' button to run calculations
    def handler(event=None):
        global glac, dem_select, thick_select, vel_select, shp_select
        glac = lb1.curselection()
        glac = [lb1.get(int(x)) for x in glac]
        dem_select = lb2.curselection()
        dem_select = [lb2.get(int(x)) for x in dem_select]
        thick_select = lb3.curselection()
        thick_select = [lb3.get(int(x)) for x in thick_select]
        vel_select = lb4.curselection()
        vel_select = [lb4.get(int(x)) for x in vel_select]
        shp_select = lb5.curselection()
        shp_select = [lb5.get(int(x))[:-10] for x in shp_select]
        t.destroy()


    t = Tk()
    t.title('Glacier and Data Selection')
    lb1 = Listbox(t, height=15, exportselection=FALSE)
    lb2 = Listbox(t, height=15, exportselection=FALSE)
    lb3 = Listbox(t, height=15, exportselection=FALSE)
    lb4 = Listbox(t, height=15, exportselection=FALSE)
    lb5 = Listbox(t, height=15, exportselection=FALSE)
    lb1.config(selectmode=EXTENDED)  # select multiple items with CTRL and ranges of items with SHIFT
    lb2.config(selectmode=SINGLE)  # select only one item
    lb3.config(selectmode=EXTENDED)
    lb4.config(selectmode=EXTENDED)
    lb5.config(selectmode=SINGLE)
    for i in range(len(glaciers)):
        lb1.insert(i + 1, glaciers[i])
    for i in range(len(dem_data)):
        lb2.insert(i + 1, dem_data[i])
    for i in range(len(thick_data)):
        lb3.insert(i + 1, thick_data[i])
    for i in range(len(vel_data)):
        lb4.insert(i + 1, vel_data[i])
    for i in range(len(shp_data)):
        lb5.insert(i + 1, shp_data[i] + ' Shapefile')
    lb1.select_set(0)  # default beginning selection in GUI
    lb2.select_set(3)
    lb3.select_set(1)
    lb4.select_set(1, 3)
    lb5.select_set(0)
    lb1.pack(side=LEFT)
    lb2.pack(side=LEFT)
    lb3.pack(side=LEFT)
    lb4.pack(side=LEFT)
    lb5.pack(side=LEFT)
    t.bind('<Return>', handler)
    t.bind('<Double-1>', handler)
    b = Button(command=handler, text='SELECT', fg='black')
    b.pack(side=LEFT)

    w = 1000  # tk window width
    h = 220  # tk window height
    ws = t.winfo_screenwidth()  # width of the screen
    hs = t.winfo_screenheight()  # height of the screen
    x = (ws / 2) - (w / 2)
    y = (hs / 2) - (h / 2)
    t.geometry('%dx%d+%d+%d' % (w, h, x, y))
    t.mainloop()
    return glac, dem_select, thick_select, vel_select, shp_select

def pickGlacierTime(glac):
    time1 = '2000-2020'
    time2 = '2015-2020'
    time_data = (time1, time2)

    # create a GUI to handle regional analysis for all other glaciers
    if glac[0] == 'All Other Glaciers':
        def handler2(event=None):
            global glacier_number_from, glacier_number_to, time_select
            glacier_number_from = ent.get()
            glacier_number_to = ent2.get()
            time_select = lb4.curselection()
            time_select = [lb4.get(int(x)) for x in time_select]
            t2.destroy()

        t2 = Tk()
        t2.title('Glacier RGI Number')
        ent = Entry(t2, exportselection=FALSE, justify='center')
        ent.insert(END, '570')          # gulkana is 570, wolverine is 9162, lemon creek 1104,
        ent.pack(side=TOP)
        ent2 = Entry(t2, exportselection=FALSE, justify='center')
        # ent2.insert(END, '571')
        ent2.pack(side=TOP)

        lb4 = Listbox(t2, height=3, exportselection=FALSE)
        lb4.config(selectmode=SINGLE)
        for i in range(len(time_data)):
            lb4.insert(i + 1, time_data[i])
        lb4.select_set(1)
        lb4.configure(justify=CENTER)
        lb4.pack()

        t2.bind('<Return>', handler2)
        t2.bind('<Double-1>', handler2)
        b2 = Button(command=handler2, text='ENTER', fg='black')
        b2.pack(side=BOTTOM)
        w = 300  # tk window width
        h = 130  # tk window height
        ws = t2.winfo_screenwidth()  # width of the screen
        hs = t2.winfo_screenheight()  # height of the screen
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        t2.geometry('%dx%d+%d+%d' % (w, h, x, y))
        t2.mainloop()
        return glacier_number_from, glacier_number_to, time_select

def pickIcepackVelocity(vel_select):
    # create a GUI to handle selection of icepack velocity
    if 'Icepack_Model' in vel_select:
        def handler5(event=None):
            global icepack_number1, icepack_number2, icepack_number3
            icepack_number1 = ent1.get()
            icepack_number2 = ent2.get()
            icepack_number3 = ent3.get()
            t5.destroy()

        t5 = Tk()
        t5.title('Icepack Velocity Number')
        ent1 = Entry(t5, exportselection=FALSE, justify='center')
        ent1.insert(END, '1')
        ent1.pack(side=TOP)
        ent2 = Entry(t5, exportselection=FALSE, justify='center')
        ent2.insert(END, '3')
        ent2.pack(side=TOP)
        ent3 = Entry(t5, exportselection=FALSE, justify='center')
        ent3.insert(END, '5')
        ent3.pack(side=TOP)

        t5.bind('<Return>', handler5)
        t5.bind('<Double-1>', handler5)
        b2 = Button(command=handler5, text='ENTER', fg='black')
        b2.pack(side=BOTTOM)
        w = 300  # tk window width
        h = 130  # tk window height
        ws = t5.winfo_screenwidth()  # width of the screen
        hs = t5.winfo_screenheight()  # height of the screen
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        t5.geometry('%dx%d+%d+%d' % (w, h, x, y))
        t5.mainloop()
        return [icepack_number1, icepack_number2, icepack_number3]
    else:
        return [None, None, None]

def pickCorrectionsScalings():
        c1 = 'Original (smoothed)'
        c2 = 'Aspect-Corrected'  # this removes velocities that are in the opposite direction as the aspect
        c3 = 'Aspect-Corrected-Removed'   # removes velocities opposing the aspect and removes them from bins
        c4 = 'Gulkana-Correction'
        # c4 = 'Aspect-Slope Directed'  # uses vel magnitude but direction is based on aspect WHEN A SLOPE THRESHOLD IS EXCEEDED
        # c5 = 'Weighted Mean Aspect'   # direction is based on slope-based weighted mean of raw vel and DEM aspect
        s1 = 'Yes'
        s2 = 'No'
        correct_vels = (c1, c2, c3, c4)
        uncertainty = (s1, s2)

        # Add GUI for Aspect-Corrected Velocities - do we want to calculate CMB with them or not
        def handler3(event=None):
            global filtered_vel
            filtered_vel = lb5.curselection()
            filtered_vel = [lb5.get(int(x)) for x in filtered_vel]
            t3.destroy()

        t3 = Tk()
        t3.title('Use corrected or original velocities for SMB?')
        lb5 = Listbox(t3, height=len(correct_vels), exportselection=FALSE, justify='center')
        lb5.config(selectmode=SINGLE)  # select multiple items with CTRL and ranges of items with SHIFT
        for i in range(len(correct_vels)):
            lb5.insert(i + 1, correct_vels[i])
        lb5.select_set(1)  # default beginning selection in GUI
        lb5.configure(justify=CENTER)
        lb5.pack()
        t3.bind('<Return>', handler3)
        t3.bind('<Double-1>', handler3)
        b3 = Button(command=handler3, text='SELECT', fg='black')
        b3.pack(side=BOTTOM)
        w = 350  # tk window width
        h = 120  # tk window height
        ws = t3.winfo_screenwidth()  # width of the screen
        hs = t3.winfo_screenheight()  # height of the screen
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        t3.geometry('%dx%d+%d+%d' % (w, h, x, y))
        t3.mainloop()

        # Add GUI for Velocity scaling factor - do we want to calculate it or not
        def handler4(event=None):
            global vel_scaling_factor
            vel_scaling_factor = lb6.curselection()
            vel_scaling_factor = [lb6.get(int(x)) for x in vel_scaling_factor]
            t4.destroy()

        t4 = Tk()
        t4.title('Calculate velocity scaling factor?')
        lb6 = Listbox(t4, height=2, exportselection=FALSE, justify='center')
        lb6.config(selectmode=SINGLE)  # select multiple items with CTRL and ranges of items with SHIFT
        for i in range(len(uncertainty)):
            lb6.insert(i + 1, uncertainty[i])
        lb6.select_set(0)  # default beginning selection in GUI
        lb6.configure(justify=CENTER)
        lb6.pack()
        t4.bind('<Return>', handler4)
        t4.bind('<Double-1>', handler4)
        b4 = Button(command=handler4, text='SELECT', fg='black')
        b4.pack(side=BOTTOM)
        w = 250  # tk window width
        h = 60  # tk window height
        ws = t4.winfo_screenwidth()  # width of the screen
        hs = t4.winfo_screenheight()  # height of the screen
        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        t4.geometry('%dx%d+%d+%d' % (w, h, x, y))
        t4.mainloop()
        return filtered_vel, vel_scaling_factor
