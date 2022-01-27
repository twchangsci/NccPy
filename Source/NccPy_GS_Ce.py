# This program determines the relative centroid location between event pairs
#   through grid searches that maximize the sum of NCC values
# It is rewritten and modified from nccgrd.f90 in the hyponcc package (by Dr. Satoshi Ide & Kazuaki Ohta)

# Major requirements:
#   [Parameter file]
#   [Original catalog]: specified in parameter file
#   [Waveform data]: specified in parameter file
#   [Appropriate travel-time table]: specified in parameter file
# Major output:
#   nccgs.cent.tbl: the relative location file (to be passed into inversion)
# Minor outputs:
#   Logs/GS_cent/nccgs.cent.out: additional results
#   Logs/GS_cent/nccgs.cent.log: detailed logs
#   Logs/GS_cent/stainfo.~: information of station for the number of waveform data
#   Logs/GS_cent/wvindex.~: information of waveform alignment prior to/ after the grid search
#   Logs/GS_cent/nccmap.~ : the complete NCC value records

# V0.6.0 (2022.01)
# Author: Ta-Wei Chang  (twchangsci@gmail.com)


import os
import shutil
import numpy as np
import math
import obspy.taup
import scipy.signal
import time
import datetime
import pandas

from NccPy_GS_Ce_aux import cc_cent_custom, vincenty_custom, ncc_gridsearch


def nccpy_gs_ce(para_file_pass):

    # Time the calculation
    start_time = time.time()

    print(f'\n>>>> This is NccPy_GS_Ce.py  (Grid search: centroid) <<<<')

    # ## I. Initialize the program: input, generate, and log parameters #####
    # Load the parameter file
    filee = open(para_file_pass)
    para_master = filee.readlines()

    # Assign the parameters
    # Catalog, data, travel-time table
    oricatfn = f'{para_master[0].rstrip()}'         # Original catalog
    datadir = f'{para_master[1].rstrip()}/Vel'      # Waveform data
    vtbroot = f'{para_master[3].rstrip()}'          # Directory and specification of travel-time table

    # Grid spacings
    [ala1, alo1, adp1, adt1, dla1, dlo1, ddp1, ddt1] = [float(ii) for ii in para_master[5].split()[0:8]]
    [ala2, alo2, adp2, adt2, dla2, dlo2, ddp2, ddt2] = [float(ii) for ii in para_master[6].split()[0:8]]

    # Some settings
    [sta_wv_out_tag, nccmap_out_tag, namefix_tag] = [int(ii) for ii in para_master[8].split()[0:3]]
    [iconv, strdrop, vrup] = \
        [int(para_master[8].split()[3]), float(para_master[8].split()[4]), float(para_master[8].split()[5])]

    # For the data:
    [twin, thead] = [float(ii) for ii in para_master[9].split()[0:2]]
    [minwv, maxwv] = [int(ii) for ii in para_master[9].split()[2:4]]
    maxstadist = float(para_master[9].split()[4])
    maxdist = [float(ii) for ii in para_master[9].split()[5:7]]

    # For S/N:
    [noscritsn, pre_window_sn, post_window_sn, noscritmean] = [float(ii) for ii in para_master[10].split()[0:4]]

    # Load the travel-time table:
    vtbpara = pandas.read_csv(f'{vtbroot}.para.txt', delim_whitespace=True, header=None).values  # Parameters
    vtbarrp = pandas.read_csv(f'{vtbroot}.P.txt', delim_whitespace=True, header=None).values  # P arrivals
    vtbarrs = pandas.read_csv(f'{vtbroot}.S.txt', delim_whitespace=True, header=None).values  # S arrivals

    vtbdistini = vtbpara[0][0]  # Distance: start    (degree)
    vtbdistend = vtbpara[0][1]  # Distance: end      (degree)
    vtbdiststep = vtbpara[0][2]  # Distance: interval (degree)

    vtbdepini = vtbpara[0][3]  # Depth: start    (km)
    vtbdepend = vtbpara[0][4]  # Depth: end      (km)
    vtbdepstep = vtbpara[0][5]  # Depth: interval (km)

    # Load original catalog:
    master_cat = np.genfromtxt(oricatfn)

    # Round depth in catalog using spacing in travel-time table (sampling interval issues)
    master_cat[:, 8] = np.round(master_cat[:, 8], decimals=int(np.log10(round(1 / vtbdepstep, 4))))

    # Initialization of some parameters
    nla1 = int(round(ala1 / dla1, 0))  # Number of grids: coarse, lat
    nlo1 = int(round(alo1 / dlo1, 0))  # Number of grids: coarse, lon
    ndp1 = int(round(adp1 / ddp1, 0))  # Number of grids: coarse, dep
    ndt1 = int(round(adt1 / ddt1, 0))  # Number of grids: coarse, time

    nla2 = int(round(ala2 / dla2, 0))  # Number of grids: fine, lat
    nlo2 = int(round(alo2 / dlo2, 0))  # Number of grids: fine, lon
    ndp2 = int(round(adp2 / ddp2, 0))  # Number of grids: fine, dep
    ndt2 = int(round(adt2 / ddt2, 0))  # Number of grids: fine, time

    ncar1 = (2 * nla1 + 1) * (2 * nlo1 + 1) * (2 * ndp1 + 1) * (2 * ndt1 + 1)  # Number of grid points: coarse
    ncar2 = (2 * nla2 + 1) * (2 * nlo2 + 1) * (2 * ndp2 + 1) * (2 * ndt2 + 1)  # Number of grid points: fine

    vs_ak135_top = 3.2     # S velocity of the layer [3.3, 10] km for the AK135 velocity model
    # vsjma2001_1 = 2.844  # S velocity of the first layer for the JMA2001 velocity model
    # vsiasp1 = 3.36       # S velocity of the first layer for the IASP91 velocity model
    # The max time length on each side to be extended for the grid search
    tadd = int(np.ceil((np.sqrt((ala1 + ala2) ** 2 + (alo1 + alo2) ** 2 + (adp1 + adp2) ** 2) /
                        vs_ak135_top + adt1 + adt2) * 1.2))

    earth_rad = (2 * 6378137 + 6356752.314245) / 3 / 1000    # In km (same as geopy.distance.EARTH_RADIUS)
    trivial = 1e-15  # Trivial value used to judge if float is really zero

    # Additionally recognize these as P or S waves just in case
    p_list = ['U', 'Z', 'BHZ', 'HHZ']
    s_list = ['E', 'N', 'X', 'Y', 'BHE', 'BHN', 'BH1', 'BH2']

    # Make output directories
    if os.path.exists(f'Logs') == 0:  # Make log folder if not existing
        os.mkdir(f'Logs')
    else:
        if os.path.exists(f'Logs/GS_cent'):  # Remove result directory if existing
            shutil.rmtree(f'Logs/GS_cent')
    os.mkdir(f'Logs/GS_cent')

    # Initiate output files
    nccgstbl = open(f'nccgs.cent.tbl', 'w')  # log file: relative locations from grid search
    nccgsout = open(f'Logs/GS_cent/nccgs.cent.out', 'w')    # log file: detailed outputs
    nccgslog = open(f'Logs/GS_cent/nccgs.cent.log', 'w')    # log file: detailed logs

    # Output the parameters: screen
    print(f'Waveform duration: {twin:5.1f}  pre-signal: {thead:4.1f}')
    print(f'The coarse grid: lat: {ala1:5.1f}/{dla1:4.1f} lon: {alo1:5.1f}/{dlo1:4.1f} '
          f'dep: {adp1:5.1f}/{ddp1:4.1f} time: {adt1:5.2f}/{ddt1:5.2f}')
    print(f'The fine grid:   lat: {ala2:5.1f}/{dla2:4.1f} lon: {alo2:5.1f}/{dlo2:4.1f} '
          f'dep: {adp2:5.1f}/{ddp2:4.1f} time: {adt2:5.2f}/{ddt2:5.2f}')
    print(f'Skip events with separation over: hor {maxdist[0]:4.1f} , dep {maxdist[1]:4.1f}\n')
    if iconv == 1:
        print(f'The convolve is on; stress drop: {strdrop} MPa')

    print(f'Num of coarse grid: lat:{nla1:.0f} lon:{nlo1:.0f} dep: {ndp1:.0f} time: {ndt1:.0f}')
    print(f'Num of fine grid:   lat:{nla2:.0f} lon:{nlo2:.0f} dep: {ndp2:.0f} time: {ndt2:.0f}')
    print(f'Duration of waveform for one side of grid (s): {tadd:.0f}\n')

    print(f'Travel-time table used: {vtbroot}\n'
          f'Distance (deg): {vtbdistini:.4f} to {vtbdistend:.4f} in resolution {vtbdiststep:.4f}\n'
          f'Depth (km):     {vtbdepini:.4f} to {vtbdepend:.4f} in resolution {vtbdepstep:.4f}\n')

    # Log them as well
    print(f'Waveform duration: {twin:5.1f}  pre-signal: {thead:4.1f}', file=nccgslog)
    print(f'The coarse grid: lat: {ala1:5.1f}/{dla1:4.1f} lon: {alo1:5.1f}/{dlo1:4.1f} '
          f'dep: {adp1:5.1f}/{ddp1:4.1f} time: {adt1:5.2f}/{ddt1:5.2f}', file=nccgslog)
    print(f'The fine grid:   lat: {ala2:5.1f}/{dla2:4.1f} lon: {alo2:5.1f}/{dlo2:4.1f} '
          f'dep: {adp2:5.1f}/{ddp2:4.1f} time: {adt2:5.2f}/{ddt2:5.2f}', file=nccgslog)
    print(f'Skip events with separation over: hor {maxdist[0]:4.1f} , dep {maxdist[1]:4.1f}\n', file=nccgslog)
    if iconv == 1:
        print(f'The convolve is on; stress drop: {strdrop} MPa', file=nccgslog)

    print(f'Num of coarse grid: lat:{nla1:.0f} lon:{nlo1:.0f} dep: {ndp1:.0f} time: {ndt1:.0f}', file=nccgslog)
    print(f'Num of fine grid:   lat:{nla2:.0f} lon:{nlo2:.0f} dep: {ndp2:.0f} time: {ndt2:.0f}', file=nccgslog)
    print(f'Duration of waveform for crossing radius of grid (s): {tadd:.0f}\n', file=nccgslog)

    print(f'Travel-time table used: {vtbroot}\n'
          f'Distance (deg): {vtbdistini:.4f} to {vtbdistend:.4f} in resolution {vtbdiststep:.4f}\n'
          f'Depth (km):     {vtbdepini:.4f} to {vtbdepend:.4f} in resolution {vtbdepstep:.4f}\n', file=nccgslog)

    # ## II. Major loop of calculation #####
    # List of seismogram data folders
    master_name = []

    for i1 in range(len(master_cat)):
        master_name.append(f'{datadir:s}/'
                           f'{master_cat[i1, 0]:04.0f}{master_cat[i1, 1]:02.0f}{master_cat[i1, 2]:02.0f}'
                           f'{master_cat[i1, 3]:02.0f}{master_cat[i1, 4]:02.0f}{math.floor(master_cat[i1, 5]):02.0f}')

    # ## II. 1. Load and process data for each event pair #####
    for i1 in range(len(master_cat) - 1):

        # Load the waveforms of the i1 events (moved here to save time: no need to load the first one again and again)
        wave1_ori = obspy.read(f'{master_name[i1]}/*')

        for i2 in range(i1 + 1, len(master_cat)):

            # Display/ log name of the two events
            print(f'Evt0 ({i1})  {master_cat[i1, 0]:4.0f} {master_cat[i1, 1]:2.0f} {master_cat[i1, 2]:2.0f} '
                  f'{master_cat[i1, 3]:2.0f} {master_cat[i1, 4]:2.0f} {master_cat[i1, 5]:5.2f} '
                  f'{master_cat[i1, 6]:8.4f} {master_cat[i1, 7]:8.4f} {master_cat[i1, 8]:6.2f} '
                  f'{master_cat[i1, 9]:4.1f}')
            print(f'Evt1 ({i2})  {master_cat[i2, 0]:4.0f} {master_cat[i2, 1]:2.0f} {master_cat[i2, 2]:2.0f} '
                  f'{master_cat[i2, 3]:2.0f} {master_cat[i2, 4]:2.0f} {master_cat[i2, 5]:5.2f} '
                  f'{master_cat[i2, 6]:8.4f} {master_cat[i2, 7]:8.4f} {master_cat[i2, 8]:6.2f} '
                  f'{master_cat[i2, 9]:4.1f}')

            print(f'\nEvt0 ({i1})  {master_cat[i1, 0]:4.0f} {master_cat[i1, 1]:2.0f} {master_cat[i1, 2]:2.0f} '
                  f'{master_cat[i1, 3]:2.0f} {master_cat[i1, 4]:2.0f} {master_cat[i1, 5]:5.2f} '
                  f'{master_cat[i1, 6]:8.4f} {master_cat[i1, 7]:8.4f} {master_cat[i1, 8]:6.2f} '
                  f'{master_cat[i1, 9]:4.1f}\n'
                  f'Evt1 ({i2})  {master_cat[i2, 0]:4.0f} {master_cat[i2, 1]:2.0f} {master_cat[i2, 2]:2.0f} '
                  f'{master_cat[i2, 3]:2.0f} {master_cat[i2, 4]:2.0f} {master_cat[i2, 5]:5.2f} '
                  f'{master_cat[i2, 6]:8.4f} {master_cat[i2, 7]:8.4f} {master_cat[i2, 8]:6.2f} '
                  f'{master_cat[i2, 9]:4.1f}\n', file=nccgslog)

            # ## II. 2. Preliminary removal of distant pairs/ waveforms with different sampling frequency #####
            # Some criterions to reject event pairs
            # Reject if catalog distance is too great between them
            evloc1 = (master_cat[i1, 6], master_cat[i1, 7])
            evloc2 = (master_cat[i2, 6], master_cat[i2, 7])
            evtdist_hor = vincenty_custom(evloc1, evloc2)
            # evtdist_hor = geopy.distance.geodesic(evloc1, evloc2).km
            evtdist_ver = np.abs(master_cat[i1, 8] - master_cat[i2, 8])

            if evtdist_hor > maxdist[0] or evtdist_ver > maxdist[1]:
                print(f'Inter-event distance (hor / ver):({evtdist_hor:.3f} / {evtdist_ver:.3f}) '
                      f'> limit ({maxdist[0]}/{maxdist[1]})')
                print(f'Inter-event distance (hor / ver):({evtdist_hor:.3f} / {evtdist_ver:.3f}) '
                      f'> limit ({maxdist[0]}/{maxdist[1]})', file=nccgslog)
                continue

            # Load the waveforms of the i2 events
            wave2_ori = obspy.read(f'{master_name[i2]}/*')

            # Copy them to further process them
            wave1_0 = wave1_ori.copy()
            wave2_0 = wave2_ori.copy()

            # Finalizing parameter assignment
            sample_mas = int(round(wave1_0.traces[0].meta.sampling_rate, 3))  # Regulate all to the same sampling rate
            nptwave = int(round(twin * sample_mas))  # Length of waveform (templete) to cross-correlate, in samples
            npthead = int(round(thead * sample_mas))  # Header length, in samples
            nptgrid = int(round(tadd * sample_mas))  # Time required to travel through grid radius, in samples
            nptccln = nptwave + 2 * nptgrid  # Length of waveform (longer one) to cross-correlate, in sample
            minsigl = int(np.ceil((twin + 2 * tadd) * 1.2))    # We now set the min wave length through necessity!!

            # Check if matching station/ component exist and have the same sampling rate in both
            # In wave1/2_1: common stations & components
            wave1_1 = wave1_0.copy()
            wave1_1.clear()
            wave2_1 = wave2_0.copy()
            wave2_1.clear()

            for ii3 in range(len(wave1_0)):
                for ii4 in range(len(wave2_0)):

                    name1 = ((wave1_0.traces[ii3].meta.station, wave1_0.traces[ii3].meta.channel,
                              wave1_0.traces[ii3].meta.network, wave1_0.traces[ii3].meta.location))

                    name2 = ((wave2_0.traces[ii4].meta.station, wave2_0.traces[ii4].meta.channel,
                              wave2_0.traces[ii4].meta.network, wave2_0.traces[ii4].meta.location))

                    # To determine that NGUH & N.NGUH are actually the same station
                    if namefix_tag == 1:
                        # Evt 1
                        if name1[0].rfind('.') == -1:  # No period found
                            nnname1 = name1[0]
                        else:  # period found: skip to one digit after period
                            nnname1 = name1[0][name1[0].rfind('.') + 1:]

                        # Evt 2
                        if name2[0].rfind('.') == -1:  # No period found
                            nnname2 = name2[0]
                        else:  # period found: skip to one digit after period
                            nnname2 = name2[0][name2[0].rfind('.') + 1:]

                    else:
                        nnname1 = name1[0]
                        nnname2 = name2[0]

                    # if nnname1 == nnname2 and name1[1] == name2[1] and name1[2] == name2[2] and name1[3] == name2[3]:
                    if nnname1 == nnname2 and name1[1] == name2[1]:
                        if int(round(wave1_0.traces[ii3].meta.sampling_rate, 3)) == sample_mas and \
                                int(round(wave2_0.traces[ii4].meta.sampling_rate, 3)) == sample_mas:
                            if round(wave1_0.traces[ii3].meta.sac.stla, 5) == round(wave2_0.traces[ii4].meta.sac.stla, 5) and \
                                    round(wave1_0.traces[ii3].meta.sac.stlo, 5) == round(wave2_0.traces[ii4].meta.sac.stlo, 5):
                                if np.isfinite(wave1_0.traces[ii3].data).all() == 1:
                                    if np.isfinite(wave2_0.traces[ii4].data).all() == 1:
                                        if wave1_0.traces[ii3].data.size / sample_mas >= minsigl:
                                            if wave2_0.traces[ii4].data.size / sample_mas >= minsigl:

                                                # Re-gather these traces (now the same order iin the two events)
                                                wave1_1.append(wave1_0[ii3])
                                                wave2_1.append(wave2_0[ii4])

                                            else:
                                                print(
                                                    f'Evt2: {wave2_0.traces[ii4].meta.network}.{wave2_0.traces[ii4].meta.station}.'
                                                    f'{wave2_0.traces[ii4].meta.location}.{wave2_0.traces[ii4].meta.channel} : '
                                                    f'waveform too short', file=nccgslog)
                                        else:
                                            print(f'Evt1: {wave1_0.traces[ii3].meta.network}.{wave1_0.traces[ii3].meta.station}.'
                                                  f'{wave1_0.traces[ii3].meta.location}.{wave1_0.traces[ii3].meta.channel} : '
                                                  f'waveform too short', file=nccgslog)
                                    else:
                                        print(
                                            f'Evt2: {wave2_0.traces[ii4].meta.network}.{wave2_0.traces[ii4].meta.station}.'
                                            f'{wave2_0.traces[ii4].meta.location}.{wave2_0.traces[ii4].meta.channel} : '
                                            f'NaN/ Inf in waveform', file=nccgslog)

                                else:
                                    print(
                                        f'Evt1: {wave1_0.traces[ii3].meta.network}.{wave1_0.traces[ii3].meta.station}.'
                                        f'{wave1_0.traces[ii3].meta.location}.{wave1_0.traces[ii3].meta.channel} : '
                                        f'NaN/ Inf in waveform', file=nccgslog)

            # If there are insufficient common data:
            if len(wave1_1.traces) < minwv:
                print(f'Too few data after co-exist station & signal length test: {len(wave1_1.traces)}\n')
                print(f'Too few data after co-exist station & signal length test: {len(wave1_1.traces)}\n',
                      file=nccgslog)
                continue

            # Detrend the data!
            for ii34 in range(len(wave1_1)):
                wave1_1.traces[ii34].detrend(type='linear')
                wave2_1.traces[ii34].detrend(type='linear')

            # Now, wave*_1 series is the common & detrended waveform set
            # In wave1/2_2: sort by distance
            wave1_2 = wave1_1.copy()
            wave1_2.clear()
            wave2_2 = wave2_1.copy()
            wave2_2.clear()

            # Sort them from the clostest! <In rows!>
            distsort = np.zeros((2, len(wave1_1)))
            for ii34 in range(len(wave1_1)):
                staloc1 = (wave1_1.traces[ii34].meta.sac.stla, wave1_1.traces[ii34].meta.sac.stlo)
                distsort[0][ii34] = vincenty_custom(staloc1, evloc1)   # 0th row: the distance

            distsort[1] = np.argsort(distsort[0])  # 1st row: the sequence of ranking

            for ii34 in range(len(wave1_1)):
                wave1_2.append(wave1_1[int(round(distsort[1][ii34]))])
                wave2_2.append(wave2_1[int(round(distsort[1][ii34]))])

            # Now, wave*_2 series is the distance-sorted, common waveform set

            # ## II. 3. Apply Convolution #####
            # Copy these to wave1/2_3: for convolution & S/N screening
            wave1_3 = wave1_2.copy()
            wave2_3 = wave2_2.copy()

            if iconv == 1:  # Convolution applied

                for ii5 in range(len(wave1_3)):
                    # Wave 1
                    # Calculate duration of rupture
                    dur1 = (2 / vrup) * \
                           ((7 * 10 ** (1.5 * master_cat[i2, 9] + 9.1)) / (16 * strdrop * 1000000)) ** (1 / 3)

                    # Only convolve if the triangle is long enough (>= 3)!
                    if int(round(dur1 * sample_mas)) - 2 > 0:
                        # Generate the triangle, normalize it to area = 1, add zeros so that the triangle start at 0
                        triang1 = np.r_[0, scipy.signal.triang(int(round(dur1 * sample_mas)) - 2) / dur1 * 2, 0]

                        # Apply the convolution of the triangle
                        wave1_3.traces[ii5].data = np.convolve(wave1_3.traces[ii5].data, triang1)

                    # Wave 2
                    # Calculate duration of rupture
                    dur2 = (2 / vrup) * \
                           ((7 * 10 ** (1.5 * master_cat[i1, 9] + 9.1)) / (16 * strdrop * 1000000)) ** (1 / 3)

                    # Only convolve if the triangle is long enough (>= 3)!
                    if int(round(dur2 * sample_mas)) - 2 > 0:
                        # Generate the triangle, normalize it to area = 1, add zeros so that the triangle start at 0
                        triang2 = np.r_[0, scipy.signal.triang(int(round(dur2 * sample_mas)) - 2) / dur2 * 2, 0]

                        # Apply the convolution of the triangle
                        wave2_3.traces[ii5].data = np.convolve(wave2_3.traces[ii5].data, triang2)

            # ## II. 4. Apply S/N and date quality test #####
            # In wave1/2_4: for S/N & the final one if passes all S/N, quality tests, etc
            wave1_4 = wave1_3.copy()
            wave2_4 = wave2_3.copy()

            wave1_4.clear()
            wave2_4.clear()

            arrival = np.array([])  # Memorize arrival time for those passing S/N test

            for ii6 in range(len(wave1_3)):

                # Determine which wave types they are
                ips = -1    # The judgement of wave type:   -1: unset   0: P   1: S
                try:
                    if -1 <= wave1_3.traces[ii6].meta.sac.cmpinc <= 1:
                        ips = 0
                    elif 89 <= wave1_3.traces[ii6].meta.sac.cmpinc <= 91:
                        ips = 1
                # in case cmpinc is not available
                except:
                    if wave1_3.traces[ii6].stats.channel in p_list:
                        ips = 0
                    elif wave1_3.traces[ii6].stats.channel in s_list:
                        ips = 1

                # if ips == -1:
                #     print(f'No cmpinc & cannot recognize channel for wave1_3[{ii6}]: '
                #           f'{wave1_3.traces[ii6].meta.station}.{wave1_3.traces[ii6].meta.channel}')

                # Retrieve event(catalog) - station(SAC header) distance
                staloc1 = (wave1_3.traces[ii6].meta.sac.stla, wave1_3.traces[ii6].meta.sac.stlo)
                dist1 = vincenty_custom(staloc1, evloc1)
                distdeg1 = dist1 / earth_rad / np.pi * 180

                staloc2 = (wave2_3.traces[ii6].meta.sac.stla, wave2_3.traces[ii6].meta.sac.stlo)
                dist2 = vincenty_custom(staloc2, evloc2)
                distdeg2 = dist2 / earth_rad / np.pi * 180

                # Move the station distance screening here, or the travel1 or travel2 will give bug
                if dist1 <= maxstadist and dist2 <= maxstadist and ips != -1:

                    # Retrieve station- event travel time
                    # Event 1
                    disthere = int(np.floor((distdeg1 - vtbdistini) / vtbdiststep))
                    x0 = (distdeg1 - (vtbdistini + disthere * vtbdiststep)) / vtbdiststep
                    x1 = (-distdeg1 + (vtbdistini + (disthere + 1) * vtbdiststep)) / vtbdiststep
                    dephere = int(round((master_cat[i1, 8] - vtbdepini) / vtbdepstep))

                    if ips == 0:
                        travel1 = vtbarrp[disthere, dephere] * x1 + vtbarrp[disthere + 1, dephere] * x0
                    elif ips == 1:
                        travel1 = vtbarrs[disthere, dephere] * x1 + vtbarrs[disthere + 1, dephere] * x0

                    # Event 2
                    disthere = int(np.floor((distdeg2 - vtbdistini) / vtbdiststep))
                    x0 = (distdeg2 - (vtbdistini + disthere * vtbdiststep)) / vtbdiststep
                    x1 = (-distdeg2 + (vtbdistini + (disthere + 1) * vtbdiststep)) / vtbdiststep
                    dephere = int(round((master_cat[i2, 8] - vtbdepini) / vtbdepstep))

                    if ips == 0:
                        travel2 = vtbarrp[disthere, dephere] * x1 + vtbarrp[disthere + 1, dephere] * x0
                    elif ips == 1:
                        travel2 = vtbarrs[disthere, dephere] * x1 + vtbarrs[disthere + 1, dephere] * x0

                    # Evt1
                    # Catalog-determined arrival time, IN #SAMPLES
                    locarr1 = (wave1_3.traces[ii6].meta.sac.o + travel1 - wave1_3.traces[ii6].meta.sac.b) * sample_mas
                    o_b1 = (wave1_3.traces[ii6].meta.sac.o - wave1_3.traces[ii6].meta.sac.b) * sample_mas

                    # For pre-signal times
                    locpre1_1 = int(round(locarr1 + (-pre_window_sn - post_window_sn) * sample_mas))
                    locpre1_2 = int(round(locarr1 + (-pre_window_sn) * sample_mas))

                    # For post-signal times
                    locpost1_1 = int(round(locarr1 + (-pre_window_sn) * sample_mas))
                    locpost1_2 = int(round(locarr1 + (-pre_window_sn + post_window_sn) * sample_mas))

                    # So, the waveforms:
                    wavepre1_2 = np.array(wave1_2.traces[ii6].data[locpre1_1:locpre1_2], dtype=float)  # Un-convolved (Wave2)
                    wavepost1_2 = np.array(wave1_2.traces[ii6].data[locpost1_1:locpost1_2], dtype=float)
                    wavepre1_3 = np.array(wave1_3.traces[ii6].data[locpre1_1:locpre1_2], dtype=float)  # Convolved (Wave3)
                    wavepost1_3 = np.array(wave1_3.traces[ii6].data[locpost1_1:locpost1_2], dtype=float)

                    # Evt2
                    # Catalog-determined arrival time, IN #SAMPLES
                    locarr2 = (wave2_3.traces[ii6].meta.sac.o + travel2 - wave2_3.traces[ii6].meta.sac.b) * sample_mas
                    o_b2 = (wave2_3.traces[ii6].meta.sac.o - wave2_3.traces[ii6].meta.sac.b) * sample_mas

                    # For pre-signal times
                    locpre2_1 = int(round(locarr2 + (-pre_window_sn - post_window_sn) * sample_mas))
                    locpre2_2 = int(round(locarr2 + (-pre_window_sn) * sample_mas))

                    # For post-signal times
                    locpost2_1 = int(round(locarr2 + (-pre_window_sn) * sample_mas))
                    locpost2_2 = int(round(locarr2 + (-pre_window_sn + post_window_sn) * sample_mas))

                    # So, the waveforms:
                    wavepre2_2 = np.array(wave2_2.traces[ii6].data[locpre2_1:locpre2_2], dtype=float)  # Un-convolved (Wave2)
                    wavepost2_2 = np.array(wave2_2.traces[ii6].data[locpost2_1:locpost2_2], dtype=float)
                    wavepre2_3 = np.array(wave2_3.traces[ii6].data[locpre2_1:locpre2_2], dtype=float)  # Convolved (Wave3)
                    wavepost2_3 = np.array(wave2_3.traces[ii6].data[locpost2_1:locpost2_2], dtype=float)

                    # Screen waveform length based on ACTUAL USAGE: length AT THE SEGMENT WE WANT
                    locreal1_1 = int(round(locarr1 + (travel2 - travel1) * sample_mas - npthead - nptgrid))
                    locreal1_2 = int(round(locarr1 + (travel2 - travel1) * sample_mas - npthead - nptgrid + nptccln))
                    locreal2_1 = int(round(locarr2 + (travel1 - travel2) * sample_mas - npthead - nptgrid))
                    locreal2_2 = int(round(locarr2 + (travel1 - travel2) * sample_mas - npthead - nptgrid + nptccln))

                    wavereal1 = np.array(wave1_3.traces[ii6].data[locreal1_1:locreal1_2], dtype=float)
                    wavereal2 = np.array(wave2_3.traces[ii6].data[locreal2_1:locreal2_2], dtype=float)

                    # Only proceed on with the calculations when length exceeds min requirements
                    if len(wavereal1) == nptccln and len(wavereal2) == nptccln:
                        if np.std(wavepre1_3, ddof=1) * np.std(wavepost1_3, ddof=1) * \
                           np.std(wavepre2_3, ddof=1) * np.std(wavepost2_3, ddof=1) > trivial:

                            # Calculation of S/N -(1): standard deviation of convoluted waveforms
                            sn1_1 = np.std(wavepost1_3, ddof=1) / np.std(wavepre1_3, ddof=1)
                            sn2_1 = np.std(wavepost2_3, ddof=1) / np.std(wavepre2_3, ddof=1)

                            # Calculation of S/N -(2): mean value of un-convoluted waveforms
                            sn1_2 = np.abs(np.mean(wavepost1_2) - np.mean(wavepre1_2)) / \
                                np.max(np.abs(wavepost1_2 - np.mean(wavepre1_2)))
                            sn2_2 = np.abs(np.mean(wavepost2_2) - np.mean(wavepre2_2)) / \
                                np.max(np.abs(wavepost2_2 - np.mean(wavepre2_2)))

                            # S/N and signal quality judgements!
                            if np.isfinite(sn1_1) == 1 and np.std(wavepre1_3, ddof=1) != 0 and \
                                    np.std(wavepost1_3, ddof=1) != 0:
                                if np.isfinite(sn2_1) == 1 and np.std(wavepre2_3, ddof=1) != 0 and \
                                        np.std(wavepost2_3, ddof=1) != 0:
                                    if sn1_1 >= noscritsn and sn1_2 <= noscritmean:
                                        if sn2_1 >= noscritsn and sn2_2 <= noscritmean:

                                            # Limit number of waveform
                                            if len(arrival) < maxwv:

                                                wave1_4.append(wave1_3[ii6])
                                                wave2_4.append(wave2_3[ii6])

                                                if len(arrival) == 0:
                                                    arrival = np.array([[locarr1, locarr2, travel1 * sample_mas,
                                                                         travel2 * sample_mas, ips, o_b1, o_b2]])
                                                else:
                                                    arrival = np.r_[arrival, [[locarr1, locarr2, travel1 * sample_mas,
                                                                               travel2 * sample_mas, ips, o_b1, o_b2]]]

                                            # Log down reasons of rejection!
                                            else:
                                                print(f'# Waveform count ({len(arrival)}) exceeds limit', file=nccgslog)
                                                break

                                        else:
                                            print(f'Evt2: {wave2_3.traces[ii6].meta.network}.'
                                                  f'{wave2_3.traces[ii6].meta.station}.'
                                                  f'{wave2_3.traces[ii6].meta.location}.'
                                                  f'{wave2_3.traces[ii6].meta.channel} : '
                                                  f'waveform fails S/N test', file=nccgslog)

                                    else:
                                        print(f'Evt1: {wave1_3.traces[ii6].meta.network}.'
                                              f'{wave1_3.traces[ii6].meta.station}.'
                                              f'{wave1_3.traces[ii6].meta.location}.'
                                              f'{wave1_3.traces[ii6].meta.channel} : '
                                              f'waveform fails S/N test', file=nccgslog)

                                else:
                                    print(f'Evt2: {wave2_3.traces[ii6].meta.network}.'
                                          f'{wave2_3.traces[ii6].meta.station}.'
                                          f'{wave2_3.traces[ii6].meta.location}.{wave2_3.traces[ii6].meta.channel} : '
                                          f'waveform fails finite (Inf/ NaN) test', file=nccgslog)
                            else:
                                print(f'Evt1: {wave1_3.traces[ii6].meta.network}.{wave1_3.traces[ii6].meta.station}.'
                                      f'{wave1_3.traces[ii6].meta.location}.{wave1_3.traces[ii6].meta.channel} : '
                                      f'waveform fails finite (Inf/ NaN) test', file=nccgslog)

                        elif np.std(wavepre1_3, ddof=1) < trivial:
                            print(f'Evt1: {wave1_3.traces[ii6].meta.network}.{wave1_3.traces[ii6].meta.station}.'
                                  f'{wave1_3.traces[ii6].meta.location}.{wave1_3.traces[ii6].meta.channel} : '
                                  f'wavepre = 0', file=nccgslog)
                        elif np.std(wavepost1_3, ddof=1) < trivial:
                            print(f'Evt1: {wave1_3.traces[ii6].meta.network}.{wave1_3.traces[ii6].meta.station}.'
                                  f'{wave1_3.traces[ii6].meta.location}.{wave1_3.traces[ii6].meta.channel} : '
                                  f'wavepost = 0', file=nccgslog)
                        elif np.std(wavepre2_3, ddof=1) < trivial:
                            print(f'Evt2: {wave2_3.traces[ii6].meta.network}.{wave2_3.traces[ii6].meta.station}.'
                                  f'{wave2_3.traces[ii6].meta.location}.{wave2_3.traces[ii6].meta.channel} : '
                                  f'wavepre = 0', file=nccgslog)
                        elif np.std(wavepost2_3, ddof=1) < trivial:
                            print(f'Evt2: {wave2_3.traces[ii6].meta.network}.{wave2_3.traces[ii6].meta.station}.'
                                  f'{wave2_3.traces[ii6].meta.location}.{wave2_3.traces[ii6].meta.channel} : '
                                  f'wavepost = 0', file=nccgslog)

                    if len(wavereal1) < nptccln:
                        if locreal1_1 >= 0:
                            print(f'Evt1: {wave1_3.traces[ii6].meta.network}.{wave1_3.traces[ii6].meta.station}.'
                                  f'{wave1_3.traces[ii6].meta.location}.{wave1_3.traces[ii6].meta.channel} : '
                                  f'waveform too short: {len(wavereal1)} < nptccln: {nptccln}', file=nccgslog)
                        else:
                            print(f'Evt1: {wave1_3.traces[ii6].meta.network}.{wave1_3.traces[ii6].meta.station}.'
                                  f'{wave1_3.traces[ii6].meta.location}.{wave1_3.traces[ii6].meta.channel} : '
                                  f'waveform too short (presignal): {locreal1_1}', file=nccgslog)

                    if len(wavereal2) < nptccln:
                        if locreal2_1 >= 0:
                            print(f'Evt2: {wave2_3.traces[ii6].meta.network}.{wave2_3.traces[ii6].meta.station}.'
                                  f'{wave2_3.traces[ii6].meta.location}.{wave2_3.traces[ii6].meta.channel} : '
                                  f'waveform too short: {len(wavereal2)} < nptccln: {nptccln}', file=nccgslog)
                        else:
                            print(f'Evt2: {wave2_3.traces[ii6].meta.network}.{wave2_3.traces[ii6].meta.station}.'
                                  f'{wave2_3.traces[ii6].meta.location}.{wave2_3.traces[ii6].meta.channel} : '
                                  f'waveform too short (presignal): {locreal2_1}', file=nccgslog)

                if ips == -1:
                    print(f'No cmpinc & cannot recognize channel for wave1_3[{ii6}]: '
                          f'{wave1_3.traces[ii6].meta.station}.{wave1_3.traces[ii6].meta.channel}', file=nccgslog)

                if dist1 > maxstadist or dist2 > maxstadist:
                    print(f'Station-event distance ({dist1} and {dist2}) exceeds limit: wave1_3[{ii6}]', file=nccgslog)

            # If there are insufficient common data after S/N test for the pair:
            if len(wave1_4.traces) < minwv:
                print(f'Too few data after S/N and data finity test: {len(wave1_4.traces)}\n')
                print(f'Too few data after S/N and data finity test: {len(wave1_4.traces)}\n', file=nccgslog)
                continue

            # ## II. 5. The cross-correlations for different time shiftings #####
            for ii7 in range(len(wave1_4)):

                # Case1 (UP): Event i1 as reference (templete)
                # Time setting: one(template) + two(longer one for CC)
                onestart_u = int(round(arrival[ii7, 0] - npthead))
                oneend_u = int(round(arrival[ii7, 0] - npthead + nptwave))

                twostart_u = int(round(arrival[ii7, 1] + (arrival[ii7, 2] - arrival[ii7, 3]) - npthead - nptgrid))
                twoend_u = int(round(arrival[ii7, 1] + (arrival[ii7, 2] - arrival[ii7, 3]) - npthead - nptgrid + nptccln))

                waveone_u = np.float64(np.array(wave1_4.traces[ii7].data[onestart_u:oneend_u]))
                wavetwo_u = np.float64(np.array(wave2_4.traces[ii7].data[twostart_u:twoend_u]))

                cc1_calc = cc_cent_custom(waveone_u, wavetwo_u, nptccln, nptwave)

                # Save them into a bigger array
                if ii7 == 0:
                    cc1_save = np.array(cc1_calc)

                else:
                    cc1_save = np.c_[cc1_save, cc1_calc]

                # Case2 (DOWN): Event i2 as reference (templete)
                # Time setting: one(template) + two(longer one for CC)
                onestart_d = int(round(arrival[ii7, 1] - npthead))
                oneend_d = int(round(arrival[ii7, 1] - npthead + nptwave))

                twostart_d = int(round(arrival[ii7, 0] + (arrival[ii7, 3] - arrival[ii7, 2]) - npthead - nptgrid))
                twoend_d = int(round(arrival[ii7, 0] + (arrival[ii7, 3] - arrival[ii7, 2]) - npthead - nptgrid + nptccln))

                waveone_d = np.float64(np.array(wave2_4.traces[ii7].data[onestart_d:oneend_d]))
                wavetwo_d = np.float64(np.array(wave1_4.traces[ii7].data[twostart_d:twoend_d]))

                cc2_calc = cc_cent_custom(waveone_d, wavetwo_d, nptccln, nptwave)

                # Save them into a bigger array
                if ii7 == 0:
                    cc2_save = np.array(cc2_calc)
                else:
                    cc2_save = np.c_[cc2_save, cc2_calc]

            # ## II. 6. The grid search for the maximum NCC #####
            # ## II. 6. 1.  Parameter assignment for the grid search #####

            for iupdown in range(2):  # 0: up (1 --> 2);   1: down (2 --> 1)

                rla = 0
                rlo = 0
                rdp = 0
                rdt = 0
                ex = 0
                ex2 = 0

                for iitr in range(2):  # Whether it's coarser (0) or finer (1) grid (always run both)

                    if iitr == 0:  # Coarser grid search

                        # default to catalog location
                        if iupdown == 0:
                            evla = master_cat[i1, 6]
                            evlo = master_cat[i1, 7]
                            evdp = master_cat[i1, 8]
                            evdt = 0

                            tp = arrival[:, 2]   # Note that this tp is arrival of phase * sampling!
                            cc = cc1_save        # The cross-correlations
                            o_b = arrival[:, 6]  # The O - B time in SAC for evt1

                        elif iupdown == 1:
                            evla = master_cat[i2, 6]
                            evlo = master_cat[i2, 7]
                            evdp = master_cat[i2, 8]
                            evdt = 0

                            tp = arrival[:, 3]   # Note that this tp is arrival of phase * sampling!
                            cc = cc2_save        # The cross-correlations
                            o_b = arrival[:, 5]  # The O - B time in SAC for evt0

                        nla = nla1
                        nlo = nlo1
                        ndp = ndp1
                        ndt = ndt1

                        dla = dla1
                        dlo = dlo1
                        ddp = ddp1
                        ddt = ddt1

                        adp = adp1
                        ncar = ncar1

                    elif iitr == 1:  # The finer grid search
                        nla = nla2
                        nlo = nlo2
                        ndp = ndp2
                        ndt = ndt2

                        dla = dla2
                        dlo = dlo2
                        ddp = ddp2
                        ddt = ddt2

                        adp = adp2
                        ncar = ncar2

                    gdla = dla / earth_rad / np.pi * 180
                    gdlo = dlo / earth_rad / np.pi * 180 / np.cos(np.mean(master_cat[:, 6]) / 180 * np.pi)

                    # ## II. 6. 2  The grid searching #####
                    # Prepare component-dependent log
                    wvvec = np.array(np.zeros((int(len(wave1_4)), 5)))
                    for iiii in range(len(wave1_4)):
                        wvvec[iiii] = [wave1_4.traces[iiii].meta.sac.stla, wave1_4.traces[iiii].meta.sac.stlo,
                                       arrival[iiii][4], tp[iiii], o_b[iiii]]  # That arrival[iiii,4] is ips!

                    # Passing info to temp vectors to utilize externally-compiled function
                    passvec1 = (evla, evlo, evdp, evdt, nla, nlo, ndp, ndt, dla, dlo, ddp, ddt)
                    passvec2 = (iitr, adp, ncar, sample_mas, nptccln, nptwave, nptgrid, gdla, gdlo, ex, ex2)
                    passvec3 = (vtbdistini, vtbdiststep, vtbdepini, vtbdepstep)

                    # Use external grid search
                    (ovec, nccrecord, itdifrec, ccmaxrec) = \
                        ncc_gridsearch(passvec1, passvec2, passvec3, wvvec, cc, vtbarrp, vtbarrs)

                    # Recover parameters
                    (nccmax, mla, mlo, mdp, mdt, ex, ex2) = ovec[0:7]

                    if iitr == 0:
                        ex2 = ex2 / ncar
                        ex = ex / ncar
                        sig2 = ex2 - ex * ex

                    # Main outputs: normalized NCC, relative locations
                    nccmax_std = (nccmax - ex) / math.sqrt(sig2)  # Standard-deviation-normalized max NCC
                    rla = rla + mla * dla  # Relative lat (accurate)
                    evla = evla + mla * gdla  # Presumed lat

                    rlo = rlo + mlo * dlo  # Relative lon (accurate)
                    evlo = evlo + mlo * gdlo  # Presumed lon

                    if evdp < adp:  # Grid goes above sea level!
                        rdp = mdp * ddp - evdp  # Subtract original for relative
                        evdp = mdp * ddp  # Already the absolute dep
                    elif evdp >= adp:  # Grid does not go above sea level
                        rdp = rdp + mdp * ddp  # Relative dep
                        evdp = mdp * ddp + evdp  # Presumed dep

                    rdt = rdt + mdt * ddt  # Time shift
                    evdt = evdt + mdt * ddt  # Time shift (including the previous iitr)

                    # Output the NCC matrix
                    if nccmap_out_tag == 1:
                        nccmapfn = f'Logs/GS_cent/nccmap.{iupdown:.0f}.{iitr:.0f}.{i1:.0f}.{i2:.0f}.dat'
                        nccmap = open(nccmapfn, 'w')
                        for lattmp in range(int(2 * nla + 1)):
                            for lontmp in range(int(2 * nlo + 1)):
                                for deptmp in range(int(2 * ndp + 1)):
                                    print(f'{lattmp - nla:.0f}  {lontmp - nlo:.0f}  {deptmp - ndp:.0f}  '
                                          f'{(nccrecord[lattmp, lontmp, deptmp] - ex)/ math.sqrt(sig2):.6f}  '
                                          f'{nccrecord[lattmp, lontmp, deptmp] / len(wave1_4):.6f}  '
                                          f'{nccrecord[lattmp, lontmp, deptmp]:.6f}', file=nccmap)
                        nccmap.close()

                    # Output indices to plot waveforms
                    if sta_wv_out_tag == 1:
                        wvindfn = f'Logs/GS_cent/wvindex.{iupdown:.0f}.{iitr:.0f}.{i1:.0f}.{i2:.0f}.dat'
                        wvind = open(wvindfn, 'w')
                        for iwv in range(len(wave1_4)):
                            if iupdown == 0:
                                print(f'{iwv:.0f}  {npthead:.0f}  {nptwave:.0f}  {arrival[iwv, 0] - npthead:.0f}  '
                                      f'{int(round(itdifrec[iwv] + arrival[iwv, 1] + arrival[iwv, 2] - arrival[iwv, 3] - npthead))}'
                                      , file=wvind)
                            elif iupdown == 1:
                                print(f'{iwv:.0f}  {npthead:.0f}  {nptwave:.0f}  {arrival[iwv, 1] - npthead:.0f}  '
                                      f'{int(round(itdifrec[iwv] + arrival[iwv, 0] + arrival[iwv, 3] - arrival[iwv, 2] - npthead))}'
                                      , file=wvind)
                        wvind.close()

                    # Output more detailed results
                    if iupdown == 0:
                        if iitr == 0:
                            print(f'{i1:.0f}  {i2:.0f}', file=nccgsout)
                            print(f'------------------------------------------', file=nccgsout)
                            print(f'== UP ====  # Waveform: {len(wave1_4):.0f}', file=nccgsout)
                        elif iitr == 1:
                            print(f'== UP ====', file=nccgsout)
                    elif iupdown == 1:
                        print(f'== DOWN ==', file=nccgsout)

                    print(f'L1: {ex:.4f}  L2: {ex2:.4f}  var.: {sig2:.4f}  ', file=nccgsout)
                    print(f'{mla:.0f}  {mlo:.0f}  {mdp:.0f}  {mdt:.0f}  '
                          f'{rla:.5f}  {rlo:.5f}  {rdp:.5f}  {rdt:.3f}  '
                          f'{evla:.5f}  {evlo:.5f}  {evdp:.5f}  {evdt:.3f}  '
                          f'{nccmax:.5f}  {nccmax_std:.5f}  {nccmax / len(wave1_4):.5f}\n', file=nccgsout)

                # Save things for up vs down
                if iupdown == 0:
                    rla_u = rla
                    rlo_u = rlo
                    rdp_u = rdp
                    rdt_u = rdt
                    nccmax_u = nccmax
                    nccmax_std_u = nccmax_std
                    iitr_u = iitr

            # Output to nccgs.tbl
            ddif = math.sqrt((rla_u + rla) ** 2 + (rlo_u + rlo) ** 2 + (rdp_u + rdp) ** 2)
            print(f'{i1:4.0f}  {i2:4.0f}  {rla_u:9.4f}  {rlo_u:9.4f}  {rdp_u:9.4f}  {rdt_u:7.3f}  '
                  f'{nccmax_std_u:8.4f}  {ddif:8.4f}  {iitr_u:2.0f}  '
                  f'{nccmax_u / len(wave1_4):7.4f}  {len(wave1_4):3.0f}', file=nccgstbl)
            print(f'{i2:4.0f}  {i1:4.0f}  {rla:9.4f}  {rlo:9.4f}  {rdp:9.4f}  {rdt:7.3f}  '
                  f'{nccmax_std:8.4f}  {ddif:8.4f}  {iitr:2.0f}  '
                  f'{nccmax / len(wave1_4):7.4f}  {len(wave1_4):3.0f}', file=nccgstbl)

            # Save station info
            if sta_wv_out_tag == 1:
                stalogfn = f'Logs/GS_cent/stainfo.{i1:.0f}.{i2:.0f}.dat'
                stalog = open(stalogfn, 'w')
                for iwv in range(len(wave1_4)):
                    print(f'{iwv:5.0f}   {wave1_4.traces[iwv].meta.station:10s}  {wave1_4.traces[iwv].meta.channel:5s}  '
                          f'{wave1_4.traces[iwv].meta.sac.stla:8.4f}  {wave1_4.traces[iwv].meta.sac.stlo:9.4f}  '
                          f'{master_cat[i1, 6]:8.4f}  {master_cat[i1, 7]:8.4f}  {master_cat[i1, 8]:8.4f}', file=stalog)
                stalog.close()

            # Flush out real-time outputs
            nccgsout.flush()
            nccgstbl.flush()
            nccgslog.flush()

            # Time the execution
            end_time = time.time()
            total_time = end_time - start_time

            print(f'Time after pair i1:{i1}  i2:{i2}    ',
                  str(datetime.timedelta(seconds=total_time)), f'\n')

    # Time the execution
    end_time = time.time()
    total_time = end_time - start_time

    print(f'Time of execution', str(datetime.timedelta(seconds=total_time)))
    print(f'\nTime of execution', str(datetime.timedelta(seconds=total_time)), file=nccgslog)

    # Put these at the very end
    nccgsout.close()
    nccgstbl.close()
    nccgslog.close()

    print(f'\n>>>> Finishes NccPy_GS_Ce.py <<<<')
    return None
