# This code is for opening the saved pdf file from the main.py code
import os
import glob
from tkinter import *
import webbrowser

g1 = 'Gulkana'
g2 = 'Wolverine'
g3 = 'Eklutna'
g4 = 'LemonCreek'
g5 = 'SouthCascade'
g6 = 'Rainbow'
g7 = 'Conrad'
g8 = 'Illecillewaet'
g9 = 'Nordic'
g10 = 'Zillmer'
g11 = 'RikhaSamba'

# chose glacier to view images with GUI
# select one of multiple glaciers in the set
# double click, press the return key, or click the 'SELECT' button to open saved images
def handler(event=None):
    global glac
    glac = lb.curselection()
    glac = [lb.get(int(x)) for x in glac]
    t.destroy()

t = Tk()
t.title('Glacier Figure Selection')
glaciers = (g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11)
lb = Listbox(t, height=15)
lb.config(selectmode=EXTENDED)      # select multiple items with CTRL and ranges of items with SHIFT
for i in range(len(glaciers)):
       lb.insert(i+1, glaciers[i])
lb.select_set(0)                    # default beginning selection in GUI
lb.pack(side=LEFT)
t.bind('<Return>', handler)
t.bind('<Double-1>', handler)
b = Button(command=handler, text='SELECT', fg='black')
b.pack(side=LEFT)

w = 250                             # tk window width
h = 200                             # tk window height
ws = t.winfo_screenwidth()          # width of the screen
hs = t.winfo_screenheight()         # height of the screen
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)
t.geometry('%dx%d+%d+%d' % (w, h, x, y))
t.mainloop()

# ---------------------------------------------------------------------------------------------------------------------#
for glacier in glac:
    if os.path.exists(os.getcwd() + '/' + glacier + '/PDFs') == True:
        os.chdir(os.getcwd() + '/' + glacier + '/PDFs')      # navigate to working directory
    else:
        prev_directory = os.path.dirname(os.getcwd())
        os.chdir(prev_directory)
        main_directory = os.path.dirname(os.getcwd())
        os.chdir(main_directory)
        path = os.getcwd() + '/' + glacier + '/PDFs'
        os.chdir(path)

    '/Users/albinwells/Desktop/MassBalanceCode/Gulkana/PDFs'
    filenames = glob.glob(os.getcwd() + '/*.pdf')
    for file in filenames:
        chrome = webbrowser.get('chrome')
        chrome.open_new('file://' + file)
