# Auxillary subroutines for the main routine

# V0.6.0 (2022.01)
# Author: Ta-Wei Chang  (twchangsci@gmail.com)

import numpy as np
from numba import jit
import math
import os.path
import shutil


# Output complete parameter file
def para_custom(cat_in):

    print(f'===== Custom parameter mode: output a template input file for parameter modification =====\n')

    # The parameter file
    fname = f'para_input_full.txt'
    para_file = open(fname, 'w')

    # Line 0: catalog file name
    # Auto-fill catalog name also!
    if cat_in:
        print(f'{cat_in}', file=para_file)
    else:
        print(f'<Catalog file name>', file=para_file)
        cat_in = []

    # Line 1, 2: path to data directory & tag of manual edit
    print(f'<Path to data directory>', file=para_file)
    print(f'0', file=para_file)

    # Line 3: path to travel-time table
    trvpath = get_trvtbl()
    print(f'{trvpath}\n', file=para_file)

    # Fill in the latter half
    para_generate(para_file, cat_in)
    para_file.close()

    return None


# Check to see if appending is requested; do so if so
def para_autoextend(inputt):

    # File name of the parameter file (after necessary appendding) to be used
    para_file_pass = []

    # First: load the input parameter file, and determine whether to append full parameter list
    filee = open(inputt)
    inputt_master = filee.readlines()

    # Append needed
    if int(inputt_master[2].rstrip()) == 1:

        print(f'===== Generate complete parameter file from the simplified =====')
        print(f'*** Note: this will shadow any input parameters set below the first 4 lines ***\n')

        # The parameter file
        fname = f'para_input_full.txt'
        para_file = open(fname, 'w')

        # 0. For entire package: copy from inputt
        print(f'{inputt_master[0].rstrip()}', file=para_file)
        print(f'{inputt_master[1].rstrip()}', file=para_file)
        print(f'{inputt_master[2].rstrip()}', file=para_file)

        # The path to the travel-time table
        trvpath = get_trvtbl()
        print(f'{trvpath}\n', file=para_file)

        # If the catalog file is openable:
        if os.access(inputt_master[0].rstrip(), os.R_OK):

            # Fill in the latter half
            para_generate(para_file, inputt_master[0].rstrip())
            para_file.close()

            # To pass the freshly-appended parameter file on
            para_file_pass = fname

        # The catalog file is not openable
        else:
            print(f'<Error> The catalog file is not readable: {inputt_master[0].rstrip()}\n')

    # Says the appending is not needed...
    elif int(inputt_master[2].rstrip()) == 0:
        # Says append is not needed, but parameter file is obviously too short
        if len(inputt_master) < 25:
            print(f'<Error> Error in parameter setting in file: {inputt} :\n'
                  f'Parameters not supplied sufficiently while set in custom mode!\n'
                  f'Either modify to automatic mode, or supply the parameters\n')

        # Use the custom parameter file
        else:
            print(f'===== Use the custom complete parameter file =====')
            para_file_pass = inputt

    return para_file_pass


# Sub-routine to set quick-and-dirty parameter default values
def para_generate(para_file, cat_in):

    # If the catalog is provided: update the grid sizes
    griddist = 8
    if cat_in:
        # Load the catalog
        cat_in_array = np.genfromtxt(cat_in)

        # Search for grid size that spans the lot
        griddist = maxsep(cat_in_array)

    # 1. For grid search (both):
    # #############################################################################################
    ala1 = 10.0  # Model space (half): coarse; lat
    alo1 = 10.0  # Model space (half): coarse; lon
    adp1 = 10.0  # Model space (half): coarse; dep
    adt1 = 1.0  # Model space (half): coarse; time

    # Update using catalog information if available
    if cat_in:
        ala1 = griddist
        alo1 = griddist
        adp1 = griddist

    ala2 = 2.0  # Model space (half): fine; lat
    alo2 = 2.0  # Model space (half): fine; lon
    adp2 = 2.0  # Model space (half): fine; dep
    adt2 = 0.5  # Model space (half): fine; time

    dla1 = 0.2  # Sampling: coarse; lat
    dlo1 = 0.2  # Sampling: coarse; lon
    ddp1 = 0.2  # Sampling: coarse; dep
    ddt1 = 0.01  # Sampling: coarse; time

    dla2 = 0.1  # Sampling: fine; lat
    dlo2 = 0.1  # Samplinag: fine; lon
    ddp2 = 0.1  # Sampling: fine; dep
    ddt2 = 0.01  # Sampling: fine; time

    print(f'{ala1:<7.2f}  {alo1:<7.2f}  {adp1:<7.2f}  {adt1:<7.2f}  '
          f'{dla1:<7.2f}  {dlo1:<7.2f}  {ddp1:<7.2f}  {ddt1:<7.2f}', file=para_file)
    print(f'{ala2:<7.2f}  {alo2:<7.2f}  {adp2:<7.2f}  {adt2:<7.2f}  '
          f'{dla2:<7.2f}  {dlo2:<7.2f}  {ddp2:<7.2f}  {ddt2:<7.2f}\n', file=para_file)

    # 2. For grid search (centroid)
    # #############################################################################################
    # <Usage of program>
    sta_wv_out_tag = 1  # 1: Output station log and arrival time index
    nccmap_out_tag = 0  # 1: Output nccmap files!
    namefix_tag = 1     # 1: Deem that NGUH & N.NGUH are aactually the same station

    iconv = 1      # Whether to apply convolution to adjust for duration difference (1: yes; 0: no)
    strdrop = 10   # Stress drop for the convolution (MPa)
    vrup = 2500    # Rupture velocity for the convolution (m/s)

    # <Settings and restrictions>
    twin = 4   # Window length to CC, including presignal time
    thead = 1  # Presignal length
    minwv = 8  # Min allowed number of waveforms available
    maxwv = 200  # Max allowed number of waveforms available (Cent: unlimited data not necessary)
    maxstadist = 1000  # Max allowed distance to the stations (Cent: travel-time table limits!) (km)
    maxdist = [8, 8]   # Max allowed distance between events [horizontal, vertical] (km)

    # Update using catalog information if available
    if cat_in:
        maxdist = [griddist, griddist]

    # <S/N>
    noscritsn = 5  # S/N using standard deviation ratio between signal and noise
    pre_window_sn = 1  # Separation of noise & signal: time before catalog-predited arrival (s)
    post_window_sn = 4  # Duration of segment for S/N calculation
    noscritmean = 0.1  # remove signal when (mean(signal)-mean(noise))/max(sig)-mean(noise)> this

    print(f'{sta_wv_out_tag:<2.0f}  {nccmap_out_tag:<2.0f}  {namefix_tag:<2.0f}  {iconv:<2.0f}  {strdrop:<.1f}  '
          f'{vrup:<.1f}', file=para_file)
    print(f'{twin:<5.1f}  {thead:<5.1f}  {minwv:<4.0f}  {maxwv:<4.0f}  {maxstadist:<7.1f}  '
          f'{maxdist[0]:<6.1f}  {maxdist[1]:<6.1f}', file=para_file)
    print(f'{noscritsn:<4.1f}  {pre_window_sn:<4.1f}  {post_window_sn:<4.1f}  {noscritmean:<6.2f}\n', file=para_file)

    # 3. For grid search (hypocenter)
    # #############################################################################################
    # <Usage of program>
    sta_wv_out_tag = 1    # 1: Output station log and arrival time index
    nccmap_out_tag = 0    # 1: Output nccmap files!
    namefix_tag = 1       # 1: Deem that NGUH & N.NGUH are aactually the same station

    extdwin = 1.0  # Duration (each direction) of additional window to look for best window to CC (s)
    db_sep = 10  # To do a separate grid search using "noise" for real std. dev., with timing signal-(*) (s)

    # <Settings and restrictions>
    twin = 0.3  # Window length to CC, including presignal time
    thead = 0.2  # Presignal length
    minwv = 8  # Min allowed number of waveforms available
    maxwv = 20  # Max allowed number of waveforms available (Hypo: attenuation issue)
    maxstadist = 150  # Max allowed distance to the stations (Hypo: attenuation issue) (km)
    maxdist = [8, 8]  # Max allowed separation between events [horizontal, vertical] (km)

    # Update using catalog information if available
    if cat_in:
        maxdist = [griddist, griddist]

    # <S/N>
    noscritsn = 10  # S/N using standard deviation ratio between signal and noise
    pre_window_sn = 1  # Separation of noise & signal: time before catalog-predited arrival (s)
    post_window_sn = 4  # Duration of segment for S/N calculation
    noscritmean = 0.1  # remove signal when (mean(signal)-mean(noise))/max(sig)-mean(noise)> this
    preratio = 10  # Remove waveform when presignal std. dev. level differs > this (hypocenter only: amp info used!)

    # <For 1st motion picking>
    uddur = 1.0  # Duration (pos/ neg) to pick ((*)pre - (*)post catalog-predicted arrival)!
    iud1 = 1  # Whether to pick using method 1: amplitude at pick vs std. dev. of pre-signal
    ud1bg = 2.0  # The background length for LTA (s)
    ud1crit = 3.0  # Criterion for picking using method 1: STA/LTA!
    iud2 = 1  # Whether to pick using method 2: gradient of kurtosis
    ud2seg = 0.5  # Duration of kurtosis calculation (s)
    ud2crit = 1.5  # Criterion for picking using method 2: gradient of kurtosis!
    iud3 = 1  # Whether to pick using the method 3: immediate S/N post/ pre pick
    ud3seg = 0.2  # Segmant pre/ post (above-) picked arrival time (s)
    ud3crit = 2.0  # Criterion for picking using method 3: immediate S/N at pick!

    print(f'{sta_wv_out_tag:<2.0f}  {nccmap_out_tag:<2.0f}  {namefix_tag:<2.0f}  {extdwin:<4.1f}  {db_sep:<5.1f}',
          file=para_file)
    print(f'{twin:<5.1f}  {thead:<5.1f}  {minwv:<4.0f}  {maxwv:<4.0f}  {maxstadist:<7.1f}  '
          f'{maxdist[0]:<6.1f}  {maxdist[1]:<6.1f}', file=para_file)
    print(f'{noscritsn:<4.1f}  {pre_window_sn:<4.1f}  {post_window_sn:<4.1f}  {noscritmean:<6.2f}  '
          f'{preratio:<4.1f}', file=para_file)
    print(f'{uddur:<4.1f}  {iud1:<4.1f}  {ud1bg:<4.1f}  {ud1crit:<4.1f}  '
          f'{iud2:<4.1f}  {ud2seg:<4.1f}  {ud2crit:<4.1f}  '
          f'{iud3:<4.1f}  {ud3seg:<4.1f}  {ud3crit:<4.1f}\n', file=para_file)

    # 4. For inversions
    # #############################################################################################
    hl_exception_tag = [1, 1]  # 1: include relative location when one go has NCC extremely higher than the other
    centcc_tag = [1, 0]  # 1: (Hypo part) not include data if mag(both) >= magsep && NCC(cent) > centccthres
    centccthres = 0.95  # Remove pairs of M>=magsep repeating events from hypo data with cent NCC > this
    magsep = 3.2  # Magnitude threshold: if magnitude >= this, separate to hypocenter and centroid!
    ddifmax = [0.3, 0.3]  # (Hypo/ Cent) [Precision limit] Max allowed deviation between 1-->2 and 2-->1
    minwv = [8, 8]   # (Hypo/ Cent) Max allowed number of waveforms available (re-adjustable for inversions)

    maxdist = [8, 8]  # Max allowed separation between events [horizontal, vertical] (km)
    # Update using catalog information if available
    if cat_in:
        maxdist = [griddist, griddist]

    # Parameters for weighting
    pnc_in = 0.1  # Threshold of PN for inversion
    pnch_in = 0.00001  # Exception to allow one side: include the side with high significance
    pncl_in = 0.9  # Exception to exclude one side: remove the side with low significance
    smax = 15  # Maximum sigmas to calculate
    ds = 0.01  # Sampling density of PN; not normally modified

    # For bootstrapping
    bsnum = 5000  # Number of times to bootstrap
    detail_output = 0  # 1: Output details during the bootstrap

    # For plottng
    strdrop = 10  # Stress drop for sizing centroid circles (MPa)
    color_mag = [4.5, 3.2, 2.0]   # Plot events with magnitudes >= these with different colors (descending)
    color_def = ['red', 'green', 'blue']  # Set the colors for events in each magnitude range

    print(f'{hl_exception_tag[0]:<2.0f}  {centcc_tag[0]:<2.0f}  {centccthres:<5.3f}  {magsep:<4.1f}  '
          f'{minwv[0]:<4.0f}  {ddifmax[0]:<4.1f}  {maxdist[0]:<7.2f}  {maxdist[1]:<7.2f}', file=para_file)
    print(f'{hl_exception_tag[1]:<2.0f}  {centcc_tag[1]:<2.0f}  {centccthres:<5.3f}  {magsep:<4.1f}  '
          f'{minwv[1]:<4.0f}  {ddifmax[1]:<4.1f}  {maxdist[0]:<7.2f}  {maxdist[1]:<7.2f}', file=para_file)
    print(f'{pnc_in:<8.6f}  {pnch_in:<8.6f}  {pncl_in:<8.6f}  {smax:<4.1f}  {ds:<5.3f}', file=para_file)
    print(f'{bsnum:<6.0f}  {detail_output:<2.0f}\n', file=para_file)

    print(f'{strdrop:<.1f}', file=para_file)
    print(f'{color_mag[0]:<4.1f}  {color_mag[1]:<4.1f}  {color_mag[2]:<4.1f}', file=para_file)
    print(f'{color_def[0]:<12s}  {color_def[1]:<12s}  {color_def[2]:<12s}', file=para_file)

    # 5. Display of parameters
    # #############################################################################################
    print(f'\n-----------------------------------------------------------------------------', file=para_file)
    print(f'=======================  Corresponding parameters  ==========================', file=para_file)
    # <For entire package>
    print(f'<Catalog file name>', file=para_file)
    print(f'<Path to data directory>', file=para_file)
    print(f'0', file=para_file)
    print(f'<Path to Source>/Travel_table/AK135_reg\n', file=para_file)

    # <For grid search (both)>
    print(f'ala1  alo1  adp1  adt1  dla1  dlo1  ddp1  ddt1', file=para_file)
    print(f'ala2  alo2  adp2  adt2  dla2  dlo2  ddp2  ddt2\n', file=para_file)

    # For grid search (centroid)
    print(f'sta_wv_out_tag(cent)  nccmap_out_tag(cent)  iconv  strdrop  vrup', file=para_file)
    print(f'twin  thead  minwv  maxwv  maxstadist  maxdist[0]  maxdist[1]', file=para_file)
    print(f'noscritsn  pre_window_sn  post_window_sn  noscritmean\n', file=para_file)

    # For grid search (hypocenter)
    print(f'sta_wv_out_tag(hypo)  nccmap_out_tag(hypo)  extdwin  db_sep', file=para_file)
    print(f'twin  thead  minwv  maxwv  maxstadist  maxdist[0]  maxdist[1]', file=para_file)
    print(f'noscritsn  pre_window_sn  post_window_sn  noscritmean  preratio', file=para_file)
    print(f'uddur  iud1  ud1bg  ud1crit  iud2  ud2seg  ud2crit  iud3  ud3seg  ud3crit\n', file=para_file)

    # For inversion
    print(f'hl_exception_tag(hy/hyce)  centcc_tag  centccthres  magsep  minwv(hy)  ddifmax(hy)  maxdist[0] maxdist[1]\n'
          f'hl_exception_tag(ce)       centcc_tag  centccthres  magsep  minwv(ce)  ddifmax(ce)  maxdist[0] maxdist[1]\n'
          f'pnc_in  pnch_in  pncl_in  smax  ds', file=para_file)
    print(f'bsnum  detail_output\n', file=para_file)

    # For plotting
    print(f'strdrop', file=para_file)
    print(f'color_mag[0]  color_mag[1]  color_mag[2]', file=para_file)
    print(f'color_def[0]  color_def[1]  color_def[2]\n', file=para_file)

    # 6. Details of parameters
    # #############################################################################################
    print(f'-----------------------------------------------------------------------------', file=para_file)
    print(f'============  Full description: often-edited marked with arrows  ============', file=para_file)
    # <For entire package>
    print(f'--> Name of catalog', file=para_file)
    print(f'--> Path to data directory', file=para_file)
    print(f'--> Automatically set all parameters? (0: no; 1: yes)', file=para_file)
    print(f'Path to travel time table directory (veloity structure specified)', file=para_file)

    # <For grid search (both)>
    print(f'\n<For grid search (both)>', file=para_file)
    print(f'a(la/lo/dp/dt)1: Model space (0.5 edge length) (lat/ lon/ dep/ time); coarse', file=para_file)
    print(f'a(la/lo/dp/dt)2: Model space (0.5 edge length) (lat/ lon/ dep/ time); fine', file=para_file)
    print(f'd(la/lo/dp/dt)(1/2): Grid spacing (lat/ lon/ dep/ time); (coarse/ fine)', file=para_file)

    # <For grid search (centroid)>
    print(f'\n<For grid search (centroid)>', file=para_file)
    print(f'### Usage of program', file=para_file)
    print(f'--> sta_wv_out_tag: 1: Output station log and arrival time index', file=para_file)
    print(f'--> nccmap_out_tag: 1: Output nccmap files', file=para_file)
    print(f'--> iconv: 1: Apply convolution to adjust for rupture duration (magnitude) difference', file=para_file)
    print(f'--> strdrop: Stress drop for the convolution (MPa)', file=para_file)
    print(f'vrup: Rupture velocity for the convolution (m/s)', file=para_file)

    print(f'\n### Settings and restrictions', file=para_file)
    print(f'--> twin/ thead: Window length/ pre-signal time (included in twin) to CC (s)', file=para_file)
    print(f'minwv/ maxwv: Min/ max allowed available waveforms', file=para_file)
    print(f'maxstadist: Max allowed distance to stations (travel-time table limits/ attenuation) (km)', file=para_file)
    print(f'maxdist: Max allowed distance between events [horizontal, vertical] (km)', file=para_file)

    print(f'\n### S/N', file=para_file)
    print(f'noscritsn: S/N using standard deviation ratio between signal and noise', file=para_file)
    print(f'pre_window_sn: Separation of noise & signal: time before catalog-predited arrival (s)', file=para_file)
    print(f'post_window_sn: Duration of segment for S/N calculation (s)', file=para_file)
    print(f'noscritmean: remove signal when (mean(signal)-mean(noise))/max(sig)-mean(noise)> this', file=para_file)

    # <For grid search (hypocenter)>
    print(f'\n<For grid search (hypocenter)>   (if same as centroid: skipped)', file=para_file)
    print(f'### Usage of program  &  S/N', file=para_file)
    print(f'extdwin: Duration (each direction) of additional window to look for best window to CC (s)', file=para_file)
    print(f'db_sep: To do a separate grid search using "noise" for real std. dev., '
          f'with timing signal-(*) (s)', file=para_file)
    print(f'preratio: Remove waveform when presignal std. dev. level differs > this '
          f'(hypo only: amp info used)', file=para_file)

    print(f'\n<For 1st motion picking>', file=para_file)
    print(f'uddur: Duration (pos/ neg) to pick ((*)pre - (*)post catalog-predicted arrival) (s)', file=para_file)
    print(f'iud1: 1: pick using method 1: amplitude at pick vs std. dev. of pre-signal', file=para_file)
    print(f'ud1bg: Duration of "background" (s) '
          f'(background: length ud1bg prior to catalog onset - uddur)', file=para_file)
    print(f'ud1crit: Criterion for picking using method 1 (How many std. dev.)', file=para_file)
    print(f'iud2: 1: pick using method 2: gradient of kurtosis', file=para_file)
    print(f'ud2seg: Duration of kurtosis calculation (s)', file=para_file)
    print(f'ud2crit: Criterion for picking using method 2', file=para_file)
    print(f'iud3: 1: pick using method 3: immediate S/N post/ pre pick', file=para_file)
    print(f'ud3seg: Segmant pre/ post pick (s)', file=para_file)
    print(f'ud3crit: Criterion for picking using method 3', file=para_file)

    # <For inversions>
    print(f'\n<For inversions>', file=para_file)
    print(f'hl_exception_tag: 1: include relative location when one go has NCC extremely higher than the other'
          f'[Hypo/ Cent (use Hypo for HyCe)]', file=para_file)
    print(f'centcc_tag: 1: (Hypo part) not include data if mag(both) >= magsep && NCC(cent) > centccthres'
          f'[Hypo/ Cent (always 0)]', file=para_file)
    print(f'centccthres: Remove pairs of M>=magsep repeating events from hypo data when their cent '
          f'is determined with NCC > this',
          file=para_file)
    print(f'--> magsep: Magnitude threshold: if magnitude >= this, separate to hypocenter and centroid!',
          file=para_file)
    print(f'minwv: Min allowed available waveforms (re-adjustable for inversion)', file=para_file)
    print(f'ddifmax: [Precision limit] Max allowed deviation between 1-->2 and 2-->1 (km)',
          file=para_file)
    print(f'maxdist: Max allowed separation between events [horizontal, vertical] (km) (re-adjustable for inversion)',
          file=para_file)
    print(f'pnc_in: (weighting) Threshold of PN for inversion', file=para_file)
    print(f'pnch_in: (weighting) Exception to allow one side: include the side with high significance', file=para_file)
    print(f'pncl_in: (weighting) Exception to exclude one side: remove the side with low significance', file=para_file)
    print(f'smax: (weighting) Maximum sigmas to calculate', file=para_file)
    print(f'ds: (weighting) Sampling density of PN', file=para_file)
    print(f'\n<For bootstrap of inversion>', file=para_file)
    print(f'--> bsnum: Number of times to bootstrap', file=para_file)
    print(f'detail_output: 1: Output details of bootstrap', file=para_file)

    # <For plotting>
    print(f'\n<For plotting>', file=para_file)
    print(f'strdrop: Stress drop for sizing centroid circles (MPa)', file=para_file)
    print(f'color_mag: plot events with magnitudes >= these with different colors (descending)', file=para_file)
    print(f'color_def: Set the colors for events in each magnitude range', file=para_file)

    return None


# Display description for help/ introductory mode
def help_display():

    print(f'===== Enter initializing mode: no parameter file specified =====\n')
    print(f'Simple usage of program: just prepare the following')
    print(f'    1. Waveform data: structured as (Some path)/(Vel/Acc)/(YYYYMMDDhhmmss)/(SAC files)  (ss: floor-ed sec)')
    print(f'    2. Original catalog in format:  YYYY MM DD hh mm ss lat lon dep mag')
    print(f'    3. Input parameter file that specifies the path/ file name to the above:\n')

    print(f'    For your convenience, an example simplified parameter file is generated (para_input_light.txt):')
    print(f'        L1: path to original catalog')
    print(f'        L2: path to waveform data files (the parent directory of (Vel/Acc)/~)')
    print(f'        L3: automatically set all parameters? (0: no; 1: yes)')
    print(f'            (0: require complete parameter list)\n')

    print(f'To run the program:')
    print(f'    I:  Standard run mode: run everything (complete parameter file auto generated if necessary)')
    print(f'        "NccPy_main.py -run_all (parameter file: simplified or complete, required)"\n')
    print(f'    II: Custom parameter mode: output complete parameter file for customization')
    print(f'        "NccPy_main.py -para_custom (catalog file: optional but recommended)"\n')
    print(f'    III: Custom run mode: run selected subroutine only (requires full parameter file)')
    print(f'        "NccPy_main.py -run_single (subroutine) (parameter file: complete, required)"')
    print(f'            (For full subroutine list:)')
    print(f'            "NccPy_main.py -run_single"\n')

    return None


# Output simplified parameter file
def para_light():

    # The parameter file for input
    fname = f'para_input_light.txt'

    # If one exists...
    if os.access(fname, os.R_OK):
        print(f'***** The simplified parameter file exists ({fname}): not newly-generated here *****\n')
    else:
        para_file = open(fname, 'w')

        print(f'<Catalog file name>', file=para_file)
        print(f'<Path to data directory>', file=para_file)
        print(f'1', file=para_file)

        para_file.close()

    return None


# List and describe contents of package
def package_content_display():

    print(f'===== For reference: the subroutines of the package =====\n')
    print(f'I.       nccpy_gs_ce: relative  centroid  location between event pairs via grid search')
    print(f'II.      nccpy_gs_hy: relative hypocenter location between event pairs via grid search')
    print(f'III.  nccpy_inv_hyce: final location of hypocenter & centroid (jointly) (all events) (joint inv)')
    print(f'IV.   nccpy_inv_solo: final location of hypocenter & centroid (independently) (all events) (separate inv)')
    print(f'V. nccpy_inv_hyce_bs: bootstrap for uncertainties of the joint inversion results in III.')
    print(f'VI.   nccpy_plot_map: output a quick-and-dirty plot of results on map\n')

    print(f'     To run, type:')
    print(f'     "NccPy_main.py -run_single (subroutine) (parameter file: complete, required)"\n')

    return None


# Grid dimension calculater: automatically set grid size as the round(max inter-event distance) + 1 km
@jit(nopython=True)
def maxsep(cat_in):
    sepmax = 0
    for i1 in range(len(cat_in) - 1):
        for i2 in range(i1 + 1, len(cat_in)):
            evloc1 = (cat_in[i1, 6], cat_in[i1, 7])
            evloc2 = (cat_in[i2, 6], cat_in[i2, 7])
            evtdist = vincenty_custom(evloc1, evloc2)
            depsep = np.abs(cat_in[i1, 8] - cat_in[i2, 8])
            sep = np.sqrt(evtdist ** 2 + depsep ** 2)
            if sep > sepmax:
                sepmax = sep

    sepmax_output = round(sepmax + 1)

    # Avoid setting a grid too tiny (or too large):
    if sepmax_output <= 3:
        sepmax_output = 4
    # elif sepmax_output >= 16:
    #     sepmax_output = 16

    return sepmax_output


# Fetches the directory of the pre-calculated travel-time table
def get_trvtbl():
    trvdir = os.path.dirname(shutil.which(f'NccPy_main.py'))    # Directory of the NccPy package
    trvpath = f'{trvdir}/Travel_table/AK135_reg'                # Specify the path to the travel-time table

    return trvpath


# Calculation of geodesic distance: modified from the vincenty package (small case, remove some features)
@jit(nopython=True)
def vincenty_custom(point1, point2):
    # WGS 84
    a = 6378137  # meters
    f = 1 / 298.257223563
    b = 6356752.314245  # meters; b = (1 - f)a

    max_iterations = 200
    convergence_threshold = 1e-12  # .000,000,000,001

    """
    Vincenty's formula (inverse method) to calculate the distance (in
    kilometers or miles) between two points on the surface of a spheroid
    Doctests:
    >>> vincenty((0.0, 0.0), (0.0, 0.0))  # coincident points
    0.0
    >>> vincenty((0.0, 0.0), (0.0, 1.0))
    111.319491
    >>> vincenty((0.0, 0.0), (1.0, 0.0))
    110.574389
    >>> vincenty((0.0, 0.0), (0.5, 179.5))  # slow convergence
    19936.288579
    >>> vincenty((0.0, 0.0), (0.5, 179.7))  # failure to converge
    >>> boston = (42.3541165, -71.0693514)
    >>> newyork = (40.7791472, -73.9680804)
    >>> vincenty(boston, newyork)
    298.396057
    >>> vincenty(boston, newyork, miles=True)
    185.414657

    author: Maurycy Pietrzak
    author_email: github.com@wayheavy.com
    package url: https://github.com/maurycyp/vincenty
    package: vincenty
    license: Unlicense
    """

    # short-circuit coincident points
    if point1[0] == point2[0] and point1[1] == point2[1]:
        return 0.0

    u1 = math.atan((1 - f) * math.tan(math.radians(point1[0])))
    u2 = math.atan((1 - f) * math.tan(math.radians(point2[0])))
    ll = math.radians(point2[1] - point1[1])
    llambda = ll

    sinu1 = math.sin(u1)
    cosu1 = math.cos(u1)
    sinu2 = math.sin(u2)
    cosu2: float = math.cos(u2)

    for iteration in range(max_iterations):
        sinlambda = math.sin(llambda)
        coslambda = math.cos(llambda)
        sinsigma = math.sqrt((cosu2 * sinlambda) ** 2 +
                             (cosu1 * sinu2 - sinu1 * cosu2 * coslambda) ** 2)
        if sinsigma == 0:
            return 0.0  # coincident points
        cossigma = sinu1 * sinu2 + cosu1 * cosu2 * coslambda
        sigma = math.atan2(sinsigma, cossigma)
        sinalpha = cosu1 * cosu2 * sinlambda / sinsigma
        cossqalpha = 1 - sinalpha ** 2
        # try:
        #     cos2sigmam = cossigma - 2 * sinu1 * sinu2 / cossqalpha
        # except ZeroDivisionError:
        #     cos2sigmam = 0
        if cossqalpha != 0:
            cos2sigmam = cossigma - 2 * sinu1 * sinu2 / cossqalpha
        elif cossqalpha == 0:
            cos2sigmam = 0
        ccc = f / 16 * cossqalpha * (4 + f * (4 - 3 * cossqalpha))
        lambdaprev = llambda
        llambda = ll + (1 - ccc) * f * sinalpha * (sigma + ccc * sinsigma *
                                                   (cos2sigmam + ccc * cossigma *
                                                    (-1 + 2 * cos2sigmam ** 2)))
        if abs(llambda - lambdaprev) < convergence_threshold:
            break  # successful convergence
    else:
        return None  # failure to converge

    usq = cossqalpha * (a ** 2 - b ** 2) / (b ** 2)
    aa = 1 + usq / 16384 * (4096 + usq * (-768 + usq * (320 - 175 * usq)))
    bb = usq / 1024 * (256 + usq * (-128 + usq * (74 - 47 * usq)))
    deltasigma = bb * sinsigma * (cos2sigmam + bb / 4 * (cossigma *
                                                         (-1 + 2 * cos2sigmam ** 2) - bb / 6 * cos2sigmam *
                                                         (-3 + 4 * sinsigma ** 2) * (-3 + 4 * cos2sigmam ** 2)))
    s = b * aa * (sigma - deltasigma)
    s /= 1000  # meters to kilometers

    return round(s, 6)
