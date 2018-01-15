import os
# os.environ["https_proxy"] = 'https://proxy.lehi.micron.com:8080'
# os.environ["http_proxy"] = 'http://proxy.lehi.micron.com:8080'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table
import subprocess
import imgkit

# Custom method adapted from Stackoverflow
# This is used as a comparison to compare Matplotlib to alternatives
def checkerboard_table(data, fmt='{:.2f}', bkg_colors=['yellow', 'white']):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0,0,1,1])

    nrows, ncols = data.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i,j), val in np.ndenumerate(data):
        # Index either the first or second item of bkg_colors based on
        # a checker board pattern
        idx = [j % 2, (j + 1) % 2][i % 2]
        color = bkg_colors[idx]

        tb.add_cell(i, j, width, height, text=fmt.format(val),
                    loc='center', facecolor=color)

    # Row Labels...
    for i, label in enumerate(data.index):
        tb.add_cell(i, -1, width, height, text=label, loc='right',
                    edgecolor='none', facecolor='none')
    # Column Labels...
    for j, label in enumerate(data.columns):
        tb.add_cell(-1, j, width, height/2, text=label, loc='center',
                           edgecolor='none', facecolor='none')
    ax.add_table(tb)
    return fig

# Pull in data and create
def main():
    # Let's use the iris dataset for our test runs
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    # Function only does numerical columns (this could be changed)
    # Notice in figure table is too dense to read for big tables
    # If there are long column/index names Matplotlib tables will not display correctly even for small tables
    checkerboard_table(df.loc[:,0:3])
    plt.savefig('matplotlibTable.png')

    # Alternative is to create an html file and convert that an images
    # This will keep it looking good/legible but may be a mute point if you just keep your tables small
    # This method is preferred/needed if column titles are long/descriptive
    df.to_html('table.html')

    # Wrapper is needed for imgemagick
    # html = pt.to_html()
    # htmlfile = open('table.html', 'w')
    # htmlfile.write('<html>'+html+'</html>')
    # htmlfile.close()

    # wkhtmltoimage/wkhtmltopdf simple program to convert html to an alternative filters
    # from command line, gives crisp image
    subprocess.call('wkhtmltoimage -f png --width 0 table.html wkhtmltoimage.png', shell=True)
    # image has extra white space (can fix this with four point transform)
    imgkit.from_file('table.html', 'imgkit.png')
    # Imagemagick is a powerful alternative as well
    # It will also create extra white space
    # subprocess.call('convert -density 72 table.html imagemacick.png', shell=True)

    

# Only run example code if specifically running this code, other wise makes checkerboard_table available
if __name__ == '__main__':
    main()
