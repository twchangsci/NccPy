# This program Regpick, written by Dwee, plots the resulting locations on map
# Requires GMT >= 6.0 (for the pygmt and the other modules) (tested on v6.2.0)

import numpy as np
import os
import shutil
import subprocess
# import pygmt


def nccpy_plot_map(para_file_pass):

    print(f'\n>>>> This is NccPy_plot_map.py  (Preliminary plotting script) <<<<')

    # #########################################################################
    # ## I.  Initialize the program: input, generate, and log parameters ######
    # #########################################################################

    # Load the parameter file
    filee = open(para_file_pass)
    para_master = filee.readlines()

    # The Final Catalog to Plot
    catt = np.genfromtxt('reloc.joint.full')
    catthy = np.genfromtxt('reloc.solo.hypo.full')
    cattce = np.genfromtxt('reloc.solo.cent.full')

    # The number of effective links
    equsg = np.genfromtxt('reloc.joint.equsage')
    equsghy = np.genfromtxt('reloc.solo.hypo.equsage')
    equsgce = np.genfromtxt('reloc.solo.cent.equsage')

    # Set stress drop (for sizing events; MPa)
    strdrop = float(para_master[22].split()[0])

    # Set lower limit of effective links
    equsgthres = 0.5 * (len(catt) - 1)

    # Set coloring [[>= Mag1, >= Mag2, ...], [Color1, Color2, ...]]
    trivial = 1e-15  # Trivial value used to judge if float is really zero!
    color_mag = [float(ii) for ii in para_master[23].split()[0:3]]
    color_def = [str(ii) for ii in para_master[24].split()[0:3]]

    colorrset = [[color_mag[0] - trivial, color_mag[1] - trivial, color_mag[2] - trivial], color_def]
    # e.g. colorrset = [[4.5, 3.2, 2.0], ['red', 'green', 'blue']]

    # The name tag of folder to save figure
    figdirfn = f'Figure'
    figtag = f'Preliminary'
    if os.path.exists(figdirfn) == 0:      # Make master figure folder
        os.mkdir(figdirfn)
    else:
        if os.path.exists(f'{figdirfn}/{figtag}'):
            shutil.rmtree(f'{figdirfn}/{figtag}')    # Remove sub figure folder
    os.mkdir(f'{figdirfn}/{figtag}')

    # Determine the default region to plot
    # Latitude
    latt = np.array([[catt[:, 6]]])
    latt = np.r_[latt, [[catt[:, 9]]]]
    latt = np.r_[latt, [[catt[:, 12]]]]

    # Longitude
    lonn = np.array([[catt[:, 7]]])
    lonn = np.r_[lonn, [[catt[:, 10]]]]
    lonn = np.r_[lonn, [[catt[:, 13]]]]

    # Depth
    depp = np.array([[catt[:, 10]]])
    depp = np.r_[depp, [[catt[:, 13]]]]

    # Region to plot (default to: round up / down to 2nd digit)
    regionn = [np.floor((np.min(lonn) - 0.01) * 100) / 100, np.ceil((np.max(lonn) + 0.01) * 100) / 100,
               np.floor((np.min(latt) - 0.01) * 100) / 100, np.ceil((np.max(latt) + 0.01) * 100) / 100]
    # regionn = []   # Manually set region to plot

    # Locations of scale
    scaleeloc = [regionn[0] + 0.8 * (regionn[1] - regionn[0]),
                 regionn[2] + 0.05 * (regionn[3] - regionn[2])]

    # #################################################
    # ## 1. Plot the original distribution  ###########
    # #################################################
    # Figure file name (no file extension: default to pdf)
    fignmori = f'{figdirfn}/{figtag}/Ori'

    # Initiate plotting session on GMT
    subprocess.run([f'gmt', f'begin', f'{fignmori}'])

    # Coastline
    # The GMT-calling version
    subprocess.run([f'gmt', f'coast',
                    f'-R{regionn[0]}/{regionn[1]}/{regionn[2]}/{regionn[3]}',
                    f'-JT{(regionn[0] + regionn[1]) / 2}/5i', f'-Df', f'-W1',
                    f'-Ba0.01f0.005', f'-Ggray90',
                    f'-Lg{scaleeloc[0]}/{scaleeloc[1]}+c+w0.5'])

    # figori = pygmt.Figure()
    #
    # figori.coast(
    #         region=regionn,
    #         projection=f'T{(regionn[0] + regionn[1]) / 2}/5i',
    #         land='gray90',
    #         water='white',
    #         shorelines='1/0.5p',
    #         resolution='f',
    #         frame=['a0.01f0.005', 'WSen'],
    # )

    # figori.show()

    # The Circles for events
    for ievt in range(len(catt)):

        # Set size based on magnitude (diameter)
        diamm = (2 / 1000) * \
            ((7 * 10 ** (1.5 * catt[ievt, 15] + 9.1))/(16 * 1000000 * strdrop)) \
            ** (1 / 3)

        # Set colors based on magnitude
        colorr = ''
        if catt[ievt, 15] >= colorrset[0][0]:
            colorr = colorrset[1][0]
        elif catt[ievt, 15] >= colorrset[0][1]:
            colorr = colorrset[1][1]
        elif catt[ievt, 15] >= colorrset[0][2]:
            colorr = colorrset[1][2]

        # Plot them!
        # Send to pipe
        echoo = subprocess.Popen(
                [f'echo', f'{catt[ievt, 7]}', f'{catt[ievt, 6]}', f'{diamm}', '0'],
                stdout=subprocess.PIPE)

        # Grab from pipe
        subprocess.run([f'gmt', f'plot', f'-SE-', f'-W1.2,{colorr}'],
                       stdin=echoo.stdout)

    # Saves the figure and end plotting session
    subprocess.run([f'gmt', f'end'])
    # subprocess.run([f'gmt', f'end', f'show'])

    # ########################################################
    # ## 2. Plot the joint-relocated distribution  ###########
    # ########################################################
    # Figure file name (no file extension: default to pdf)
    fignmjoint = f'{figdirfn}/{figtag}/Reloc_joint'

    # Initiate plotting session on GMT
    subprocess.run([f'gmt', f'begin', f'{fignmjoint}'])

    # Coast line
    subprocess.run([f'gmt', f'coast',
                    f'-R{regionn[0]}/{regionn[1]}/{regionn[2]}/{regionn[3]}',
                    f'-JT{(regionn[0] + regionn[1]) / 2}/5i', f'-Df', f'-W1',
                    f'-Ba0.01f0.005', f'-Ggray90',
                    f'-Lg{scaleeloc[0]}/{scaleeloc[1]}+c+w0.5'])

    # The Circles for events
    for ievt in range(len(catt)):

        # Plot hypocenters first

        # Set colors based on magnitude and number of effective connections
        colorr = ''

        # Smallest events
        if colorrset[0][1] > catt[ievt, 15] >= colorrset[0][2]:
            if (equsg[ievt, 1] < equsgthres and
                    equsg[ievt, 2] < equsgthres):
                colorr = 'grey60'
            else:
                colorr = colorrset[1][2]

        # Larger events
        else:
            if equsg[ievt, 1] < equsgthres:
                colorr = 'grey60'
            else:
                if catt[ievt, 15] >= colorrset[0][0]:
                    colorr = colorrset[1][0]
                elif catt[ievt, 15] >= colorrset[0][1]:
                    colorr = colorrset[1][1]

        # Plot
        # Send to pipe
        echoo = subprocess.Popen([f'echo', f'{catt[ievt, 10]}', f'{catt[ievt, 9]}',
                                  f'0.05', '0'], stdout=subprocess.PIPE)

        # Grab from pipe
        subprocess.run([f'gmt', f'plot', f'-SE-', f'-W1.5,{colorr}'],
                       stdin=echoo.stdout)
        # echoo.communicate()

        # Plot centroids
        # Set size based on magnitude (diameter)
        diamm = (2 / 1000) * \
            ((7 * 10 ** (1.5 * catt[ievt, 15] + 9.1))/(16 * 1000000 * strdrop)) \
            ** (1 / 3)

        # Set colors based on magnitude and number of effective connections
        colorr = ''

        # Smallest events
        if colorrset[0][1] > catt[ievt, 15] >= colorrset[0][2]:
            if (equsg[ievt, 1] < equsgthres and
                    equsg[ievt, 2] < equsgthres):
                colorr = 'grey60'
            else:
                colorr = colorrset[1][2]

        # Larger events
        else:
            if equsg[ievt, 2] < equsgthres:
                colorr = 'grey60'
            else:
                if catt[ievt, 15] >= colorrset[0][0]:
                    colorr = colorrset[1][0]
                elif catt[ievt, 15] >= colorrset[0][1]:
                    colorr = colorrset[1][1]

        # Plot
        # Send to pipe
        echoo = subprocess.Popen([f'echo', f'{catt[ievt, 13]}', f'{catt[ievt, 12]}',
                                  f'{diamm}', '0'], stdout=subprocess.PIPE)

        # Grab from pipe
        subprocess.run([f'gmt', f'plot', f'-SE-', f'-W0.5,{colorr},-'],
                       stdin=echoo.stdout)

    # Saves the figure and end plotting session
    subprocess.run([f'gmt', f'end'])
    # subprocess.run([f'gmt', f'end', f'show'])

    # ##########################3###################################
    # ## 3. Plot the independently-relocated hypocenters  ##########
    # ##############################################################
    # Figure file name (no file extension: default to pdf)
    fignmsolohy = f'{figdirfn}/{figtag}/Reloc_solo_hypo'

    # Initiate plotting session on GMT
    subprocess.run([f'gmt', f'begin', f'{fignmsolohy}'])

    # Coast line
    subprocess.run([f'gmt', f'coast',
                    f'-R{regionn[0]}/{regionn[1]}/{regionn[2]}/{regionn[3]}',
                    f'-JT{(regionn[0] + regionn[1]) / 2}/5i', f'-Df', f'-W1',
                    f'-Ba0.01f0.005', f'-Ggray90',
                    f'-Lg{scaleeloc[0]}/{scaleeloc[1]}+c+w0.5'])

    # The Circles for events
    for ievt in range(len(catthy)):

        # Set colors based on magnitude and number of effective connections
        colorr = ''

        # Smallest events
        if colorrset[0][1] > catthy[ievt, 12] >= colorrset[0][2]:
            if equsghy[ievt, 1] < equsgthres:
                colorr = 'grey60'
            else:
                colorr = colorrset[1][2]

        # Larger events
        else:
            if equsghy[ievt, 1] < equsgthres:
                colorr = 'grey60'
            else:
                if catthy[ievt, 12] >= colorrset[0][0]:
                    colorr = colorrset[1][0]
                elif catthy[ievt, 12] >= colorrset[0][1]:
                    colorr = colorrset[1][1]

        # Plot
        # Send to pipe
        echoo = subprocess.Popen([f'echo', f'{catthy[ievt, 10]}',
                                  f'{catthy[ievt, 9]}', f'0.05', '0'], stdout=subprocess.PIPE)

        # Grab from pipe
        subprocess.run([f'gmt', f'plot', f'-SE-', f'-W1.5,{colorr}'],
                       stdin=echoo.stdout)

    # Saves the figure and end plotting session
    subprocess.run([f'gmt', f'end'])
    # subprocess.run([f'gmt', f'end', f'show'])

    # ##########################3#################################
    # ## 4. Plot the independently-relocated centroids  ##########
    # ############################################################
    # Figure file name (no file extension: default to pdf)
    fignmce = f'{figdirfn}/{figtag}/Reloc_solo_cent'

    # Initiate plotting session on GMT
    subprocess.run([f'gmt', f'begin', f'{fignmce}'])

    # Coast line
    subprocess.run([f'gmt', f'coast',
                    f'-R{regionn[0]}/{regionn[1]}/{regionn[2]}/{regionn[3]}',
                    f'-JT{(regionn[0] + regionn[1]) / 2}/5i', f'-Df', f'-W1',
                    f'-Ba0.01f0.005', f'-Ggray90',
                    f'-Lg{scaleeloc[0]}/{scaleeloc[1]}+c+w0.5'])

    # The Circles for events
    for ievt in range(len(cattce)):

        # Set size based on magnitude (diameter)
        diamm = (2 / 1000) * \
            ((7 * 10 ** (1.5 * cattce[ievt, 12] + 9.1))/(16 * 1000000 * strdrop)) \
            ** (1 / 3)

        # Set colors based on magnitude and number of effective connections
        colorr = ''

        # Smallest events
        if colorrset[0][1] > cattce[ievt, 12] >= colorrset[0][2]:
            if equsgce[ievt, 1] < equsgthres:
                colorr = 'grey60'
            else:
                colorr = colorrset[1][2]

        # Larger events
        else:
            if equsgce[ievt, 1] < equsgthres:
                colorr = 'grey60'
            else:
                if cattce[ievt, 12] >= colorrset[0][0]:
                    colorr = colorrset[1][0]
                elif cattce[ievt, 12] >= colorrset[0][1]:
                    colorr = colorrset[1][1]

        # Plot
        # Send to pipe
        echoo = subprocess.Popen([f'echo', f'{cattce[ievt, 10]}',
                                  f'{cattce[ievt, 9]}', f'{diamm}', '0'], stdout=subprocess.PIPE)

        # Grab from pipe
        subprocess.run([f'gmt', f'plot', f'-SE-', f'-W0.5,{colorr},-'],
                       stdin=echoo.stdout)

    # Saves the figure and end plotting session
    subprocess.run([f'gmt', f'end'])
    # subprocess.run([f'gmt', f'end', f'show'])

    return None
