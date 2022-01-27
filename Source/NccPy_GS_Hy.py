# This program determines the relative hypocenter location between event pairs
#   through grid searches that maximize the sum of NCC values

# Major requirements:
#   [Parameter file]
#   [Original catalog]: specified in parameter file
#   [Waveform data]: specified in parameter file
#   [Appropriate travel-time table]: specified in parameter file
# Major output:
#   nccgs.hypo.tbl: the relative location file (to be passed into inversion)
# Minor outputs:
#   Logs/GS_hypo/nccgs.hypo.out: additional results
#   Logs/GS_hypo/nccgs.hypo.log: detailed logs
#   Logs/GS_hypo/stainfo.~: information of station for the number of waveform data
#   Logs/GS_hypo/wvindex.~: information of waveform alignment prior to/ after the grid search & auto-picking
#   Logs/GS_hypo/nccmap.~ : the complete NCC value records

# V0.6.0 (2022.01)
# Author: Ta-Wei Chang  (twchangsci@gmail.com)


import os
import shutil
import numpy as np
import math
import obspy.taup
import time
import datetime
import numba.typed
import pandas

from NccPy_GS_Hy_aux import cc_hypo_full, cc_hypo_max_declare, vincenty_custom, udcheck
from NccPy_GS_Hy_aux import ncc_gridsearch_hypo_pre, ncc_gridsearch_hypo_core, ncc_gridsearch_hypo_post


def nccpy_gs_hy(para_file_pass):

    # Time the calculation
    start_time = time.time()

    print(f'\n>>>> This is NccPy_GS_Hy.py  (Grid search: hypocenter) <<<<')

    # ## I. Initialize the program: input, generate, and log parameters #####
    # Load the parameter file
    filee = open(para_file_pass)
    para_master = filee.readlines()

    # Assign the parameters
    # Catalog, data, travel-time table
    oricatfn = f'{para_master[0].rstrip()}'        # Original catalog
    datadir = f'{para_master[1].rstrip()}/Acc'     # Waveform data
    vtbroot = f'{para_master[3].rstrip()}'         # Directory and specification of travel-time table

    # Grid spacings
    [ala1, alo1, adp1, adt1, dla1, dlo1, ddp1, ddt1] = [float(ii) for ii in para_master[5].split()[0:8]]
    [ala2, alo2, adp2, adt2, dla2, dlo2, ddp2, ddt2] = [float(ii) for ii in para_master[6].split()[0:8]]

    # Some settings
    [sta_wv_out_tag, nccmap_out_tag, namefix_tag] = [int(ii) for ii in para_master[12].split()[0:3]]
    [extdwin, db_sep] = [float(ii) for ii in para_master[12].split()[3:5]]

    # For the data:
    [twin, thead] = [float(ii) for ii in para_master[13].split()[0:2]]
    [minwv, maxwv] = [int(ii) for ii in para_master[13].split()[2:4]]
    maxstadist = float(para_master[13].split()[4])
    maxdist = [float(ii) for ii in para_master[13].split()[5:7]]

    # For S/N:
    [noscritsn, pre_window_sn, post_window_sn, noscritmean, preratio] = \
        [float(ii) for ii in para_master[14].split()[0:5]]

    # For 1st motion picking
    [uddur, iud1, ud1bg, ud1crit, iud2, ud2seg, ud2crit, iud3, ud3seg, ud3crit] = \
        [float(ii) for ii in para_master[15].split()[0:10]]

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
    nla1 = int(round(ala1 / dla1, 0))  # Number of grids: entire, lat
    nlo1 = int(round(alo1 / dlo1, 0))  # Number of grids: entire, lon
    ndp1 = int(round(adp1 / ddp1, 0))  # Number of grids: entire, dep
    ndt1 = int(round(adt1 / ddt1, 0))  # Number of grids: entire, time

    nla2 = int(round(ala2 / dla2, 0))  # Number of grids: fine, lat
    nlo2 = int(round(alo2 / dlo2, 0))  # Number of grids: fine, lon
    ndp2 = int(round(adp2 / ddp2, 0))  # Number of grids: fine, dep
    ndt2 = int(round(adt2 / ddt2, 0))  # Number of grids: fine, time

    ncar1 = (2 * nla1 + 1) * (2 * nlo1 + 1) * (2 * ndp1 + 1) * (2 * ndt1 + 1)  # Number of grid points: entire
    ncar2 = (2 * nla2 + 1) * (2 * nlo2 + 1) * (2 * ndp2 + 1) * (2 * ndt2 + 1)  # Number of grid points: fine

    vp_ak135_top = 5.8    # P velocity of the layer [3.3, 10] km for the AK135 velocity model
    # vpjma2001_1 = 4.8   # P velocity of the first layer for the JMA2001 velocity model
    # vsiasp1 = 3.36      # S velocity of the first layer for the IASP91 velocity model
    # The max time length on each side to be extended for the grid search
    tadd = int(np.ceil((np.sqrt((ala1 + ala2) ** 2 + (alo1 + alo2) ** 2 + (adp1 + adp2) ** 2) /
                        vp_ak135_top + adt1 + adt2) * 1.2))

    earth_rad = (2 * 6378137 + 6356752.314245) / 3 / 1000    # In km (same as geopy.distance.EARTH_RADIUS)
    trivial = 1e-15  # Trivial value used to judge if float is really zero!

    # Additionally recognize these as P or S waves just in case
    p_list = ['U', 'Z', 'BHZ', 'HHZ']
    s_list = ['E', 'N', 'X', 'Y', 'BHE', 'BHN', 'BH1', 'BH2']

    # Make output directories
    if os.path.exists(f'Logs') == 0:  # Make log folder if not existing
        os.mkdir(f'Logs')
    else:
        if os.path.exists(f'Logs/GS_hypo'):  # Remove result directory if existing
            shutil.rmtree(f'Logs/GS_hypo')
    os.mkdir(f'Logs/GS_hypo')

    # Initiate output files
    nccgstbl = open(f'nccgs.hypo.tbl', 'w')  # log file: relative locations from grid search
    nccgsout = open(f'Logs/GS_hypo/nccgs.hypo.out', 'w')    # log file: detailed outputs
    nccgslog = open(f'Logs/GS_hypo/nccgs.hypo.log', 'w')     # log file: detailed logs

    # Output the parameters: screen
    print(f'Waveform duration: {twin:5.1f}  pre-signal: {thead:4.1f}')
    print(f'The coarse grid: lat: {ala1:5.1f}/{dla1:4.1f} lon: {alo1:5.1f}/{dlo1:4.1f} '
          f'dep: {adp1:5.1f}/{ddp1:4.1f} time: {adt1:5.2f}/{ddt1:5.2f}')
    print(f'The fine grid:   lat: {ala2:5.1f}/{dla2:4.1f} lon: {alo2:5.1f}/{dlo2:4.1f} '
          f'dep: {adp2:5.1f}/{ddp2:4.1f} time: {adt2:5.2f}/{ddt2:5.2f}')
    print(f'The extra window to look for best CC: {extdwin:4.1f}')
    print(f'Sample background with presignal waveform: {db_sep:5.1f}')
    print(f'Number of stations: min {minwv:3.0f}  max {maxwv:3.0f}')
    print(f'Skip events with separation over: hor {maxdist[0]:4.1f} , dep {maxdist[1]:4.1f}\n')

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
    print(f'The extra window to look for best CC: {extdwin:4.1f}', file=nccgslog)
    print(f'Sample background with presignal waveform: {db_sep:5.1f}', file=nccgslog)
    print(f'Number of stations: min {minwv:3.0f} ,  max {maxwv:3.0f}', file=nccgslog)
    print(f'Skip events with separation over: hor {maxdist[0]:4.1f} , dep {maxdist[1]:4.1f}\n', file=nccgslog)

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
            nptextd = int(round(extdwin * sample_mas))  # Time of additional window for best CC <hyocenter>
            nptdb = int(round(db_sep * sample_mas))    # The duration to the double-grid-search for background std. dev.
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

                    # To include P waves only! (hypocenter version)
                    # Evt 1
                    ips1 = -1  # The judgement of wave type:   -1: unset   0: P   1: S
                    try:
                        if -1 <= wave1_0.traces[ii3].meta.sac.cmpinc <= 1:
                            ips1 = 0
                        elif 89 <= wave1_0.traces[ii3].meta.sac.cmpinc <= 91:
                            ips1 = 1
                    except:
                        if wave1_0.traces[ii3].stats.channel in p_list:
                            ips1 = 0
                        elif wave1_0.traces[ii3].stats.channel in s_list:
                            ips1 = 1

                    # Evt 2
                    ips2 = -1  # The judgement of wave type:   -1: unset   0: P   1: S
                    try:
                        if -1 <= wave2_0.traces[ii4].meta.sac.cmpinc <= 1:
                            ips2 = 0
                        elif 89 <= wave2_0.traces[ii4].meta.sac.cmpinc <= 91:
                            ips2 = 1
                    except:
                        if wave2_0.traces[ii4].stats.channel in p_list:
                            ips2 = 0
                        elif wave2_0.traces[ii4].stats.channel in s_list:
                            ips2 = 1

                    if ips1 == 0 and ips2 == 0:     # Hypocenter version only:  take P wave only!
                        # if nnname1==nnname2 and name1[1]==name2[1] and name1[2]==name2[2] and name1[3]==name2[3]:
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
                distsort[0][ii34] = vincenty_custom(staloc1, evloc1)  # 0th row: the distance

            distsort[1] = np.argsort(distsort[0])  # 1st row: the sequence of ranking

            for ii34 in range(len(wave1_1)):
                wave1_2.append(wave1_1[int(round(distsort[1][ii34]))])
                wave2_2.append(wave2_1[int(round(distsort[1][ii34]))])

            # Now, wave*_2 series is the sorted, common waveform set

            # Matching NccPy_GS_Ce syntex; deprecated here
            wave1_3 = wave1_2.copy()
            wave2_3 = wave2_2.copy()

            # ## II. 3. Apply S/N and date quality test #####
            # In wave1/2_4: for S/N & the final one if passes all S/N, quality tests, etc
            wave1_4 = wave1_2.copy()
            wave2_4 = wave2_2.copy()

            wave1_4.clear()
            wave2_4.clear()

            arrival = np.array([])  # Memorize arrival time for those passing S/N test

            for ii6 in range(len(wave1_3)):

                # Determine wave types (the S wave part kept for potential future usages of S wave data)
                ips = -1    # The judgement of wave type:   -1: unset   0: P   1: S
                try:
                    if -1 <= wave1_3.traces[ii6].meta.sac.cmpinc <= 1:
                        ips = 0
                    elif 89 <= wave1_3.traces[ii6].meta.sac.cmpinc <= 91:
                        ips = 1
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
                    # disthere = int(round((distdeg2 - vtbdistini) / vtbdiststep))
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
                    wavepre2_3 = np.array(wave2_3.traces[ii6].data[locpre2_1:locpre2_2], dtype=float)
                    wavepost2_3 = np.array(wave2_3.traces[ii6].data[locpost2_1:locpost2_2], dtype=float)

                    # Screen waveform length based on ACTUAL USAGE: length AT THE SEGMENT WE WANT
                    # In hypocenter version, expects continuous from the double grid search using pre-signal time
                    locreal1_1 = int(round(locarr1 + (travel2 - travel1) * sample_mas - npthead - nptgrid - nptdb))
                    locreal1_2 = int(round(locarr1 + (travel2 - travel1) * sample_mas - npthead - nptgrid + nptccln))

                    locreal2_1 = int(round(locarr2 + (travel1 - travel2) * sample_mas - npthead - nptgrid - nptdb))
                    locreal2_2 = int(round(locarr2 + (travel1 - travel2) * sample_mas - npthead - nptgrid + nptccln))

                    wavereal1 = np.array(wave1_3.traces[ii6].data[locreal1_1:locreal1_2], dtype=float)
                    wavereal2 = np.array(wave2_3.traces[ii6].data[locreal2_1:locreal2_2], dtype=float)

                    # Only proceed on with the calculations when length exceeds min requirements
                    if len(wavereal1) == len(wavereal2) == (nptccln + nptdb):   # P selection is done previously

                        # Only those with non-zero signals! (or the division later complains)
                        if np.std(wavepre1_3, ddof=1) * np.std(wavepost1_3, ddof=1) * \
                           np.std(wavepre2_3, ddof=1) * np.std(wavepost2_3, ddof=1) > trivial:

                            # Calculation of S/N -(1): standard deviation of waveforms
                            sn1_1 = np.std(wavepost1_3, ddof=1) / np.std(wavepre1_3, ddof=1)
                            sn2_1 = np.std(wavepost2_3, ddof=1) / np.std(wavepre2_3, ddof=1)

                            # Calculation of S/N -(2): mean value of waveforms
                            sn1_2 = np.abs(np.mean(wavepost1_3) - np.mean(wavepre1_3)) / \
                                np.max(np.abs(wavepost1_3 - np.mean(wavepre1_3)))
                            sn2_2 = np.abs(np.mean(wavepost2_3) - np.mean(wavepre2_3)) / \
                                np.max(np.abs(wavepost2_3 - np.mean(wavepre2_3)))

                            # Calculation of S/N -(3): standard deviation of waveforms
                            sn_3 = np.std(wavepre1_3, ddof=1) / np.std(wavepre2_3, ddof=1)

                            # S/N and signal quality judgements!
                            if np.isfinite(sn1_1) == 1:
                                if np.isfinite(sn2_1) == 1:
                                    if sn1_1 >= noscritsn and sn1_2 <= noscritmean:
                                        if sn2_1 >= noscritsn and sn2_2 <= noscritmean:

                                            # For the hypocenter version: check pre-signal noise level difference
                                            if preratio >= sn_3 >= 1 / preratio:

                                                # For the hypocenter version: check polarity
                                                # Segment of waveform to be passed in
                                                udini_1 = int(round(locarr1 + (-ud1bg - uddur) * sample_mas))
                                                udend_1 = int(round(locarr1 + (uddur + ud3seg) * sample_mas)) + 1
                                                udini_2 = int(round(locarr2 + (-ud1bg - uddur) * sample_mas))
                                                udend_2 = int(round(locarr2 + (uddur + ud3seg) * sample_mas)) + 1

                                                # The waveform themselves
                                                udwave1 = np.array(wave1_3.traces[ii6].data[udini_1:udend_1],
                                                                   dtype=float)
                                                udwave2 = np.array(wave2_3.traces[ii6].data[udini_2:udend_2],
                                                                   dtype=float)

                                                udprm = (uddur, ud1bg, ud1crit, ud2seg, ud2crit, ud3seg, ud3crit,
                                                         sample_mas, iud1, iud2, iud3)

                                                (ud1, ud2, pick1, pick2) = udcheck(udprm, udwave1, udwave2)

                                                # If they are both up (1) or down(0) going, and not "not found"(-1)
                                                if ud1 == ud2 and ud1 != -1:

                                                    # Limit number of waveform & sta-evt distance!
                                                    if len(arrival) < maxwv:

                                                        wave1_4.append(wave1_3[ii6])
                                                        wave2_4.append(wave2_3[ii6])

                                                        if len(arrival) == 0:
                                                            arrival = np.array([[locarr1, locarr2, travel1 * sample_mas,
                                                                                 travel2 * sample_mas, ips,
                                                                                 pick1, pick2, o_b1, o_b2]])

                                                        else:
                                                            arrival = np.r_[arrival, [[locarr1, locarr2,
                                                                                       travel1 * sample_mas,
                                                                                       travel2 * sample_mas,
                                                                                       ips, pick1, pick2, o_b1, o_b2]]]

                                                    # Log down reasons of rejection!
                                                    else:
                                                        print(f'# Waveform count ({len(arrival)}) exceeds limit',
                                                              file=nccgslog)
                                                        break

                                                if ud1 == -1:
                                                    print(f'UD check: {wave1_3.traces[ii6].meta.network}.'
                                                          f'{wave1_3.traces[ii6].meta.station}.'
                                                          f'{wave1_3.traces[ii6].meta.location}.'
                                                          f'{wave1_3.traces[ii6].meta.channel} : '
                                                          f'pick not found in event 1', file=nccgslog)

                                                if ud2 == -1:
                                                    print(f'UD check: {wave1_3.traces[ii6].meta.network}.'
                                                          f'{wave1_3.traces[ii6].meta.station}.'
                                                          f'{wave1_3.traces[ii6].meta.location}.'
                                                          f'{wave1_3.traces[ii6].meta.channel} : '
                                                          f'pick not found in event 2', file=nccgslog)

                                                if ud1 == 1 and ud2 == 0:
                                                    print(f'UD check: {wave1_3.traces[ii6].meta.network}.'
                                                          f'{wave1_3.traces[ii6].meta.station}.'
                                                          f'{wave1_3.traces[ii6].meta.location}.'
                                                          f'{wave1_3.traces[ii6].meta.channel} : '
                                                          f'opposite polarity: wave1 (+); wave2 (-)  '
                                                          f'picking (0: catalog) 1st: {pick1}  2nd: {pick2}',
                                                          file=nccgslog)

                                                elif ud1 == 0 and ud2 == 1:
                                                    print(f'UD check: {wave1_3.traces[ii6].meta.network}.'
                                                          f'{wave1_3.traces[ii6].meta.station}.'
                                                          f'{wave1_3.traces[ii6].meta.location}.'
                                                          f'{wave1_3.traces[ii6].meta.channel} : '
                                                          f'opposite polarity: wave1 (-); wave2 (+)  '
                                                          f'picking (0: catalog) 1st: {pick1}  2nd: {pick2}',
                                                          file=nccgslog)

                                            else:
                                                print(f'Pre-signal amplitude check: {wave1_3.traces[ii6].meta.network}.'
                                                      f'{wave1_3.traces[ii6].meta.station}.'
                                                      f'{wave1_3.traces[ii6].meta.location}.'
                                                      f'{wave1_3.traces[ii6].meta.channel} : '
                                                      f'Large pre-signal mean-level ratio: {sn_3}', file=nccgslog)

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

            # ## II. 4. The cross-correlations for different time shiftings #####
            # Move everything into subroutine: speed + the np_c may be messy for 3-D

            # Pass waveform in! In case of failure when the data length is not equal, pass to list first
            # Initialize lists (Typed list in numba)
            wavelist1 = numba.typed.List()
            wavelist2 = numba.typed.List()

            # Note that these will not work due to numba's deprecation of support of reflected list
            # wavelist1 = []    # All the waveform in wave1_4
            # wavelist2 = []    # All the waveform in wave2_4

            for ii7 in range(len(wave1_4.traces)):
                wavelist1.append(np.float64(wave1_4.traces[ii7].data))
                wavelist2.append(np.float64(wave2_4.traces[ii7].data))

            # Two outputs: 1. CC_save in [idt][iextd][iwv];  2.  CC_save_db in [idt][iwv]
            passvec_cc = (npthead, nptccln, nptwave, nptgrid, nptdb, nptextd)
            (cc1_save, cc2_save, cc1_save_db, cc2_save_db) = cc_hypo_full(passvec_cc, arrival, wavelist1, wavelist2)

            # Get the max values and indices of CC for the addiional grid search for best window to CC (onset)
            (cc1_max, cc1_argmax, cc2_max, cc2_argmax) = cc_hypo_max_declare(cc1_save, cc2_save, nptextd)

            # ## II. 5. The grid search for the maximum NCC #####
            # ## II. 5. 1.  Parameter assignment for the grid search #####

            for iupdown in range(2):  # 0: up (1 --> 2);   1: down (2 --> 1)

                rla = 0
                rlo = 0
                rdp = 0
                rdt = 0
                ex = 0
                ex2 = 0
                ex_db = 0
                ex2_db = 0

                for iitr in range(2):  # Whether it's coarser (0) or finer (1) grid (always run both)

                    if iitr == 0:  # Coarser grid search

                        # default to catalog location
                        if iupdown == 0:
                            evla = master_cat[i1, 6]
                            evlo = master_cat[i1, 7]
                            evdp = master_cat[i1, 8]
                            evdt = 0

                            tp = arrival[:, 2]  # Note that this tp is arrival of phase * sampling!
                            cc = cc1_save  # The cross-correlations
                            cc_db = cc1_save_db  # The cross-correlations for extra grid search
                            cc_max = cc1_max     # The smaller-sized vector of max of the index
                            cc_argmax = cc1_argmax  # The smaller-sized vector of indice of max of the index
                            o_b = arrival[:, 8]   # The O - B time in SAC for evt1

                        elif iupdown == 1:
                            evla = master_cat[i2, 6]
                            evlo = master_cat[i2, 7]
                            evdp = master_cat[i2, 8]
                            evdt = 0

                            tp = arrival[:, 3]  # Note that this tp is arrival of phase * sampling!
                            cc = cc2_save  # The cross-correlations
                            cc_db = cc2_save_db  # The cross-correlations for extra grid search
                            cc_max = cc2_max  # The smaller-sized vector of max of the index
                            cc_argmax = cc2_argmax  # The smaller-sized vector of indice of max of the index
                            o_b = arrival[:, 7]  # The O - B time in SAC for evt0

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

                    # ## II. 8. 2  The grid searching  ######################

                    # Prepare component-dependent log
                    wvvec = np.array(np.zeros((int(len(wave1_4)), 5)))
                    for iiii in range(len(wave1_4)):
                        wvvec[iiii] = [wave1_4.traces[iiii].meta.sac.stla, wave1_4.traces[iiii].meta.sac.stlo,
                                       arrival[iiii][4], tp[iiii], o_b[iiii]]    # That arrival[iiii,4] is ips!

                    # Passing info to temp vectors to utilize externally-compiled function
                    passvec1 = (evla, evlo, evdp, evdt, nla, nlo, ndp, ndt, dla, dlo, ddp, ddt)
                    passvec2 = (iitr, adp, ncar, sample_mas, nptccln, nptwave, nptgrid, nptdb, nptextd, gdla, gdlo,
                                ex, ex2, ex_db, ex2_db)
                    passvec3 = (vtbdistini, vtbdiststep, vtbdepini, vtbdepstep)

                    # Use external grid search!
                    (nccrecord, nccrecord_db, extdrecord, ex2record, ex2record_db, gridstatrvmat, indices_mat) =\
                        ncc_gridsearch_hypo_pre(passvec1, passvec2, passvec3, wvvec, vtbarrp, vtbarrs)

                    (nccrecord, nccrecord_db, extdrecord) = \
                        ncc_gridsearch_hypo_core(passvec1, passvec2, wvvec, cc_max, cc_db,
                                                 nccrecord, nccrecord_db, extdrecord, gridstatrvmat, indices_mat)

                    (ovec, nccrecord_out, itdifrec, ccmaxrec, extdrec) = \
                        ncc_gridsearch_hypo_post(passvec1, passvec2, passvec3, wvvec, cc_max, cc_argmax, vtbarrp, vtbarrs,
                                                 nccrecord, nccrecord_db, ex2record, ex2record_db)

                    # Recover parameters
                    (nccmax, mla, mlo, mdp, mdt, ex, ex2, ex_db, ex2_db) = ovec[0:9]

                    if iitr == 0:
                        ex2 = ex2 / ncar
                        ex = ex / ncar
                        sig2 = ex2 - ex * ex

                        # The double version
                        ex2_db = ex2_db / ncar
                        ex_db = ex_db / ncar
                        sig2_db = ex2_db - ex_db * ex_db

                    # Main outputs: normalized NCC, relative locs!
                    nccmax_std = (nccmax - ex) / math.sqrt(sig2)  # Standard-deviation-normalized max NCC
                    nccmax_std_db = (nccmax - ex) / math.sqrt(sig2_db)  # <Double> Standard-deviation-normalized max NCC
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
                        nccmapfn = f'Logs/GS_hypo/nccmap.{iupdown:.0f}.{iitr:.0f}.{i1:.0f}.{i2:.0f}.dat'
                        nccmap = open(nccmapfn, 'w')
                        for lattmp in range(int(2 * nla + 1)):
                            for lontmp in range(int(2 * nlo + 1)):
                                for deptmp in range(int(2 * ndp + 1)):
                                    print(f'{lattmp - nla:.0f}  {lontmp - nlo:.0f}  {deptmp - ndp:.0f}  '
                                          f'{(nccrecord_out[lattmp, lontmp, deptmp] - ex) / math.sqrt(sig2_db):.6f}  '
                                          f'{(nccrecord_out[lattmp, lontmp, deptmp] - ex) / math.sqrt(sig2):.6f}  '
                                          f'{nccrecord_out[lattmp, lontmp, deptmp] / len(wave1_4):.6f}  '
                                          f'{nccrecord_out[lattmp, lontmp, deptmp]:.6f}', file=nccmap)
                        nccmap.close()

                    # Output indices to plot waveforms
                    # Here, output the arrival corresponding to the max NCC, and also the picking from polarity check!
                    if sta_wv_out_tag == 1:
                        wvindfn = f'Logs/GS_hypo/wvindex.{iupdown:.0f}.{iitr:.0f}.{i1:.0f}.{i2:.0f}.dat'
                        wvind = open(wvindfn, 'w')
                        for iwv in range(len(wave1_4)):
                            if iupdown == 0:
                                print(f'{iwv:.0f}  {npthead:.0f}  {nptwave:.0f}  {extdrec[iwv]:.0f}  '
                                      f'{arrival[iwv, 0] - npthead:.0f}  {arrival[iwv, 1] - npthead:.0f}  '
                                      f'{arrival[iwv, 0] - npthead + extdrec[iwv]:.0f}  '
                                      f'{itdifrec[iwv] + arrival[iwv, 1] + arrival[iwv, 2] - arrival[iwv, 3] - npthead + extdrec[iwv]:.0f}  '
                                      f'{arrival[iwv, 0] + arrival[iwv, 5]:.0f}  {arrival[iwv, 1] + arrival[iwv, 6]:.0f}'
                                      , file=wvind)
                            elif iupdown == 1:
                                print(f'{iwv:.0f}  {npthead:.0f}  {nptwave:.0f}  {extdrec[iwv]:.0f}  '
                                      f'{arrival[iwv, 1] - npthead:.0f}  {arrival[iwv, 0] - npthead:.0f}  '
                                      f'{arrival[iwv, 1] - npthead + extdrec[iwv]:.0f}  '
                                      f'{itdifrec[iwv] + arrival[iwv, 0] + arrival[iwv, 3] - arrival[iwv, 2] - npthead + extdrec[iwv]:.0f}  '
                                      f'{arrival[iwv, 1] + arrival[iwv, 6]:.0f}  {arrival[iwv, 0] + arrival[iwv, 5]:.0f}'
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

                    print(f'Signal:     L1: {ex:.4f}  L2: {ex2:.4f}  var.: {sig2:.4f}  ', file=nccgsout)
                    print(f'Pre-signal: L1: {ex_db:.4f}  L2: {ex2_db:.4f}  var.: {sig2_db:.4f}  ', file=nccgsout)
                    print(f'{mla:.0f}  {mlo:.0f}  {mdp:.0f}  {mdt:.0f}  '
                          f'{rla:.5f}  {rlo:.5f}  {rdp:.5f}  {rdt:.3f}  '
                          f'{evla:.5f}  {evlo:.5f}  {evdp:.5f}  {evdt:.3f}  '
                          f'{nccmax:.5f}  {nccmax_std:.5f}  {nccmax_std_db:.5f}  {nccmax / len(wave1_4):.5f}\n'
                          , file=nccgsout)

                # Save things for up vs down
                if iupdown == 0:
                    rla_u = rla
                    rlo_u = rlo
                    rdp_u = rdp
                    rdt_u = rdt
                    nccmax_u = nccmax
                    nccmax_std_u = nccmax_std
                    nccmax_std_db_u = nccmax_std_db
                    iitr_u = iitr

            # Output to nccgs.tbl
            ddif = math.sqrt((rla_u + rla) ** 2 + (rlo_u + rlo) ** 2 + (rdp_u + rdp) ** 2)
            print(f'{i1:4.0f}  {i2:4.0f}  {rla_u:9.4f}  {rlo_u:9.4f}  {rdp_u:9.4f}  {rdt_u:7.3f}  '
                  f'{nccmax_std_db_u:8.4f}  {ddif:8.4f}  {iitr_u:2.0f}  '
                  f'{nccmax_u / len(wave1_4):7.4f}  {len(wave1_4):3.0f}  {nccmax_std_u:8.4f}', file=nccgstbl)
            print(f'{i2:4.0f}  {i1:4.0f}  {rla:9.4f}  {rlo:9.4f}  {rdp:9.4f}  {rdt:7.3f}  '
                  f'{nccmax_std_db:8.4f}  {ddif:8.4f}  {iitr:2.0f}  '
                  f'{nccmax / len(wave1_4):7.4f}  {len(wave1_4):3.0f}  {nccmax_std:8.4f}', file=nccgstbl)

            # Save station info
            if sta_wv_out_tag == 1:
                stalogfn = f'Logs/GS_hypo/stainfo.{i1:.0f}.{i2:.0f}.dat'
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

    print(f'\n>>>> Finishes NccPy_GS_Hy.py <<<<')
    return None
