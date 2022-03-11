# For plotting velocity files

from Modules.smoothingFunctions import sgolay2d, gaussianFilter, dynamicSmoothing
from glacierDatabase import glacierInfo
import numpy as np
import scipy
import os
import geopandas as gpd
import fiona
import geopandas as gpd
import rasterio
import rasterio.plot
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from tkinter import *

arrowFreq = 10
res = 20        # resampling resolution (x, y) tuple

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

v1 = 'ITS_LIVE_20yrComposite'
v2 = 'ITS_LIVE_2017-2018'
v3 = 'MillanVelocity_2017-2018'
v4 = 'RETREAT_2017-2018'
v5 = 'RETREAT_2020'
v6 = 'RETREAT_2015-2020'

# chose glacier to perform calculations with GUI
# select one of multiple glaciers in the set
# double click, press the return key, or click the 'SELECT' button to run calculations
def handler(event=None):
    global name, thick_select, vel_select
    name = lb1.curselection()
    name = [lb1.get(int(x)) for x in name]
    vel_select = lb3.curselection()
    vel_select = [lb3.get(int(x)) for x in vel_select]
    t.destroy()


t = Tk()
t.title('Glacier and Data Selection')
glaciers = (g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11)
vel_data = (v1, v2, v3, v4, v5, v6)
lb1 = Listbox(t, height=15, exportselection=FALSE)
lb3 = Listbox(t, height=15, exportselection=FALSE)
lb1.config(selectmode=SINGLE)   # select only one item
lb3.config(selectmode=SINGLE)
for i in range(len(glaciers)):
    lb1.insert(i + 1, glaciers[i])
for i in range(len(vel_data)):
    lb3.insert(i + 1, vel_data[i])
lb1.select_set(0)  # default beginning selection in GUI
lb3.select_set(2)
lb1.pack(side=LEFT)
lb3.pack(side=LEFT)
t.bind('<Return>', handler)
t.bind('<Double-1>', handler)
b = Button(command=handler, text='SELECT', fg='black')
b.pack(side=LEFT)

w = 450  # tk window width
h = 200  # tk window height
ws = t.winfo_screenwidth()  # width of the screen
hs = t.winfo_screenheight()  # height of the screen
x = (ws / 2) - (w / 2)
y = (hs / 2) - (h / 2)
t.geometry('%dx%d+%d+%d' % (w, h, x, y))
t.mainloop()

# ---------------------------------------------------------------------------------------------------------------------#
# thickness file is irrelevant for this code, so we just always use Farinotti since it exists for all glaciers
# most inputs are irrelevant for this code, so I don't have them as variables
glacierCalc = glacierInfo(name[0], res, 850, 50, 0.8, 'FarinottiThickness', vel_select[0])
os.chdir(os.getcwd() + '/' + glacierCalc.name)      # navigate to working directory

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

# CONVERT COORDINATE SYSTEMS AND RESAMPLE
velReprojectionResample(glacierCalc.vx, glacierCalc.crs, glacierCalc.res)
velReprojectionResample(glacierCalc.vy, glacierCalc.crs, glacierCalc.res)

shpReprojection(glacierCalc.shape, glacierCalc.crs)

vxArray3 = shpClip(glacierCalc.shape, glacierCalc.vx)
vyArray3 = shpClip(glacierCalc.shape, glacierCalc.vy)
vxArray = vxArray3.reshape(-1, vxArray3.shape[-1])
vyArray = vyArray3.reshape(-1, vyArray3.shape[-1])
vArray = np.power((np.square(vxArray) + np.square(vyArray)), 0.5)
# fill in data with griddata

use_filter = 'none'      # 'sg' for savitzky-golay filter, 'gauss' for gaussian filter
if use_filter == 'sg':
    vArray = sgolay2d(vArray, window_size=17, order=1)
    vxArray = sgolay2d(vxArray, window_size=17, order=1)
    vyArray = sgolay2d(vyArray, window_size=17, order=1)
if use_filter == 'gauss':
    vArray = gaussianFilter(vArray, st_dev=4, truncate=3)
    vxArray = gaussianFilter(vxArray, st_dev=4, truncate=3)
    vyArray = gaussianFilter(vyArray, st_dev=4, truncate=3)

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

title_map = glacierCalc.name + ' Glacier Velocity Map'
fig = plt.figure()
ax = fig.add_subplot(111, label="1")
divnorm = TwoSlopeNorm(vmin=0, vcenter=5, vmax=50)
# divnorm = TwoSlopeNorm(vmin=0, vcenter=min(vArray.mean(), 10), vmax=min(vArray.max(), 100))
vArray[vArray > 1000] = np.nan  # to remove huge values for white background in plot
vArray[vArray == 0] = np.nan  # to remove 0 values for white background in plot
quiverData = velQuiver(vxArray * glacierCalc.vCor[0], vyArray * glacierCalc.vCor[1], vArray, freq=arrowFreq, threshold=1.0)
ax.quiver(quiverData[0], quiverData[1], quiverData[2], quiverData[3], quiverData[4],
           cmap=quiverData[5], scale=quiverData[6]*2, width=.002)  # velocity arrows
vArray = vArray.astype(float)
im = ax.imshow(vArray, cmap=plt.cm.get_cmap('BrBG'), norm=divnorm)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, label='Velocity (m/a)')
ax.set_title(title_map)

scalebar = AnchoredSizeBar(ax.transData,
                           1000/res, '1 km', 'lower left',
                           pad=0.1,
                           color='k',
                           frameon=False,
                           size_vertical=1,
                           fontproperties=fm.FontProperties(size=12))
ax.add_artist(scalebar)

plt.show()
# figName = title_map.replace(' ', '_') + '.png'
# print(os.getcwd())
# fig.savefig('Figures/' + figName)
