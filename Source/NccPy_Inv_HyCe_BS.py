# This program determines the uncertainty of final joint hypocenter and centroid locations through bootstrap

# Major requirements:
#   nccgs.hypo.tbl & nccgs.cent.tbl: the relative location files
#   [Parameter file]
#   [Original catalog]: specified in parameter file
# Major outputs:
#   bs.joint.(hypo/cent).std: standard deviation of lat(km), lon(km), dep(km), time(s)
#   bs.joint.(hypo/cent).errorellipse: Error ellipse (major-, minor-axis, rotation from top (N/ +dep) of error ellipse)
#   bs.joint.(hypo/cent).ci: confidence intervals (90%, 95%, 99%) in lat, lon, dep, time
# Minor outputs:
#   Logs/Inv_BS_HyCe/bs.joint.log: the master log
#   Logs/Inv_BS_HyCe/bs.joint.(hypo/cent).cov: Variance/ covariance (variance of 1, 2, covariance matrix [1, 2])
#   Logs/Inv_BS_HyCe/bs.joint.(hypo/cent).eig: Eigenvector/ eigenvalue (larger eigenvector, eigenvalues of both)
#   Logs/Inv_BS_HyCe/bs.joint.(hypo/cent).all_bs.(lat, etc): The final location of every BS realizations!
#   Logs/Inv_BS_HyCe/bs.joint.(hypo/cent).nccinv.pass: consistant pairs to be used in inversion
#   Logs/Inv_BS_HyCe/rand_table.txt: table of data to use in .tbl in each BS realizations
#   Logs/Inv_BS_HyCe/bs_{ii}.tbl: output FAKE, BUT .tbl-like files to try each BS

# Matrix dimensions (as in NccPy_Inv_HyCe.py):
#     data (nmax = n1 + n2 + m1 + m1 + m2): hypo, relative (rvec_hy, n1) + cent, relative (rvec_ce, n2)
#                                   + catalog, L (ovec_L, m1) + catalog, L (ovec_L, m1) + catalog, S (ovec_S, m2)
#            # Note on ovec: original order and size (ovec, m1 + m2) -> L only (ovec_L, m1) + S only (ovec_S, m2)
#     model (mmax = m1 + m1 + m2): hypo, Large (m1) + cent, Large (m1) + hyce, small (m2)
#     GF; kernal (G, size = nmax * mmax): appropriately set

# V0.6.0 (2022.01)
# Author: Ta-Wei Chang  (twchangsci@gmail.com)


import sys
import os
import shutil
import numpy as np
import time
import datetime
import pandas
import copy
import scipy.stats

from NccPy_Inv_aux import noiseprob_lookup, noiseprob_table, vincenty_custom, abic_core


def nccpy_inv_hyce_bs(para_file_pass):

    # Time the calculation
    start_time = time.time()

    print(f'\n>>>> This is NccPy_Inv_HyCe_BS.py  (Bootstrap for joint inversion uncertainty) <<<<')

    # ## I. Initialize the program: input, generate, and log parameters #####
    # Load the parameter file
    filee = open(para_file_pass)
    para_master = filee.readlines()

    # Catalog, data, travel-time table
    oricatfn = f'{para_master[0].rstrip()}'  # Original catalog
    vtbroot = f'{para_master[3].rstrip()}'  # Directory and specification of travel-time table

    # Grid spacings
    [ala1, alo1, adp1, adt1, dla1, dlo1, ddp1, ddt1] = [float(ii) for ii in para_master[5].split()[0:8]]
    [ala2, alo2, adp2, adt2, dla2, dlo2, ddp2, ddt2] = [float(ii) for ii in para_master[6].split()[0:8]]

    # On usages of the program
    [hl_exception_tag, centcc_tag] = [int(ii) for ii in para_master[17].split()[0:2]]
    [centccthres, magsep] = [float(ii) for ii in para_master[17].split()[2:4]]
    [minwv_hy, minwv_ce] = [int(para_master[17].split()[4]), int(para_master[18].split()[4])]
    [ddifmax_hy, ddifmax_ce] = [float(para_master[17].split()[5]), float(para_master[18].split()[5])]
    [maxdist_hor, maxdist_ver] = [float(ii) for ii in para_master[17].split()[6:8]]

    # On weighting
    [pnc_in, pnch_in, pncl_in, smax, ds] = [float(ii) for ii in para_master[19].split()[0:5]]

    # On bootstrap
    [bsnum, detail_output] = [int(ii) for ii in para_master[20].split()[0:2]]

    # Load the relative location files from grid searches
    nccgstbl_hy = pandas.read_csv(f'nccgs.hypo.tbl', delim_whitespace=True, header=None).values
    nccgstbl_ce = pandas.read_csv(f'nccgs.cent.tbl', delim_whitespace=True, header=None).values

    # Load the catalog, and round depth sing spacing in travel-time table (sampling interval issues)
    vtbpara = pandas.read_csv(f'{vtbroot}.para.txt', delim_whitespace=True, header=None).values  # Parameters
    vtbdepstep = vtbpara[0][5]  # Depth: interval (km)

    master_cat = np.genfromtxt(oricatfn)
    master_cat[:, 8] = np.round(master_cat[:, 8], decimals=int(np.log10(round(1 / vtbdepstep, 4))))

    # Model space and grid spacings
    nla1 = int(round(ala1 / dla1, 0))  # Number of grids: entire, lat
    nlo1 = int(round(alo1 / dlo1, 0))  # Number of grids: entire, lon
    ndp1 = int(round(adp1 / ddp1, 0))  # Number of grids: entire, dep
    ndt1 = int(round(adt1 / ddt1, 0))  # Number of grids: entire, time

    nla2 = int(round(ala2 / dla2, 0))  # Number of grids: fine, lat
    nlo2 = int(round(alo2 / dlo2, 0))  # Number of grids: fine, lon
    ndp2 = int(round(adp2 / ddp2, 0))  # Number of grids: fine, dep
    ndt2 = int(round(adt2 / ddt2, 0))  # Number of grids: fine, time

    ncar1 = (2 * nla1 + 1) * (2 * nlo1 + 1) * (2 * ndp1 + 1) * (2 * ndt1 + 1)  # Number of grid points: entire

    # Other parameters
    earth_rad = (2 * 6378137 + 6356752.314245) / 3 / 1000    # In km (same as geopy.distance.EARTH_RADIUS)
    trivial = 1e-15    # Trivial value used to judge if float is really zero!

    complist = ['lat', 'lon', 'dep', 'tim']     # Component names for naming in looping

    # Directories of output files
    if os.path.exists('Logs/Inv_BS_HyCe'):  # Remove result directory if existing
        shutil.rmtree('Logs/Inv_BS_HyCe')
    os.mkdir('Logs/Inv_BS_HyCe')

    nccinvlog = open('Logs/Inv_BS_HyCe/bs.joint.log', 'w')    # log file

    # Probability table
    smax = int(round(smax))
    ptable = noiseprob_table(ds, smax, ncar1)  # Threshold for whether to include data!

    # Assign the probability and NCC/std thresholds
    (nccc, pnc) = noiseprob_lookup(ds, smax, ptable, pnc_in)      # The (discretized) pass threshold of NCC/std and PN
    (ncccl, pncl) = noiseprob_lookup(ds, smax, ptable, pncl_in)   # The (discretized) exception threshold (the low NCC)
    (nccch, pnch) = noiseprob_lookup(ds, smax, ptable, pnch_in)   # The (discretized) exception threshold (the high NCC)

    # Output them
    print(f'Number of Bootstraps: {bsnum}\n')
    print(f'Min number of stations: hypo: {minwv_hy:4.0f}  cent: {minwv_ce:4.0f}')
    print(f'Max relative locaiton difference (precision): hypo: {ddifmax_hy:7.3f}  cent: {ddifmax_ce:7.3f}')
    print(f'Max event separation (absolute): horizontal: {maxdist_hor:4.1f}  depth: {maxdist_ver:4.1f}\n')
    print(f'NCC and PN thresholds:')
    print(f'PN cutoff: pass: {pnc:8.6f} ,  exception: {pncl:8.6f}  {pnch:8.6f}')
    print(f'NCC cutoff: pass: {nccc:6.3f} ,  exception: {ncccl:6.3f}  {nccch:6.3f}\n')

    print(f'Number of Bootstraps: {bsnum}\n', file=nccinvlog)
    print(f'Min number of stations: hypo: {minwv_hy:4.0f}  cent: {minwv_ce:4.0f}', file=nccinvlog)
    print(f'Max relative locaiton difference (precision): hypo: {ddifmax_hy:7.3f}  cent: {ddifmax_ce:7.3f}', file=nccinvlog)
    print(f'Max event separation (absolute): horizontal: {maxdist_hor:4.1f}  depth: {maxdist_ver:4.1f}\n', file=nccinvlog)
    print(f'NCC and PN thresholds:', file=nccinvlog)
    print(f'PN cutoff: pass: {pnc:8.6f} ,  exception: {pncl:8.6f}  {pnch:8.6f}', file=nccinvlog)
    print(f'NCC cutoff: pass: {nccc:6.3f} ,  exception: {ncccl:6.3f}  {nccch:6.3f}\n', file=nccinvlog)

    if centcc_tag == 1:
        print(f'Remove M >= {magsep} NCC(hypo) data when NCC(cent) > {centccthres}\n')
        print(f'Remove M >= {magsep} NCC(hypo) data when NCC(cent) > {centccthres}\n', file=nccinvlog)

    # ## II. Begin the major loop of calcualtion #####
    # ## II. 1. The original location: ovec #####
    # Event count of mag >= magsep
    m1 = 0
    m2 = 0
    for i1 in range(len(master_cat)):
        if master_cat[i1, 9] + trivial >= magsep:
            m1 += 1
        else:
            m2 += 1

    ovec = np.zeros((len(master_cat), 7))  # Lat, lon, dep, time (0), mag, L(1) / S(0), index in ovec_L/S
    ovec_L = np.zeros((m1, 6))  # Lat, lon, dep, time (0), mag, index in ovec
    ovec_S = np.zeros((m2, 6))  # Lat, lon, dep, time (0), mag, index in ovec

    tmpm1 = -1  # tmp parameter to log down index in m1  (set to -1 to initialize from indice 0)
    tmpm2 = -1  # tmp parameter to log down index in m2  (set to -1 to initialize from indice 0)

    for i1 in range(len(master_cat)):

        ovec[i1, 0] = master_cat[i1, 6]  # Lat
        ovec[i1, 1] = master_cat[i1, 7]  # Lon
        ovec[i1, 2] = master_cat[i1, 8]  # Dep
        ovec[i1, 4] = master_cat[i1, 9]  # Mag

        # Large events
        if master_cat[i1, 9] + trivial >= magsep:
            tmpm1 += 1
            ovec[i1, 5] = 1                       # 1: Large
            ovec[i1, 6] = tmpm1                   # index in m1 (ovec_L)

            ovec_L[tmpm1, 0] = master_cat[i1, 6]  # Lat
            ovec_L[tmpm1, 1] = master_cat[i1, 7]  # Lon
            ovec_L[tmpm1, 2] = master_cat[i1, 8]  # Dep
            ovec_L[tmpm1, 4] = master_cat[i1, 9]  # Mag
            ovec_L[tmpm1, 5] = i1                 # index in original (ovec)

        # Small events
        else:
            tmpm2 += 1
            ovec[i1, 5] = 0                       # 0: small
            ovec[i1, 6] = tmpm2                   # index in m2 (ovec_S)

            ovec_S[tmpm2, 0] = master_cat[i1, 6]  # Lat
            ovec_S[tmpm2, 1] = master_cat[i1, 7]  # Lon
            ovec_S[tmpm2, 2] = master_cat[i1, 8]  # Dep
            ovec_S[tmpm2, 4] = master_cat[i1, 9]  # Mag
            ovec_S[tmpm2, 5] = i1                 # index in original (ovec)

    # Log it down!
    print(f'\nThe original catalog location (ovec), magnitude, L(1) / S(0), index in L / S:', file=nccinvlog)
    for i1 in range(len(ovec)):
        print(f'{ovec[i1, 0]:10.4f}  {ovec[i1, 1]:10.4f}  {ovec[i1, 2]:6.2f}  '
              f'{ovec[i1, 3]:3.1f}  {ovec[i1, 4]:4.1f}  {ovec[i1, 5]:2.0f}  {ovec[i1, 6]:4.0f}', file=nccinvlog)

    ovecllog = open('Logs/Inv_BS_HyCe/ovec_L.log', 'w')    # log down ovec_L
    for i1 in range(len(ovec_L)):
        print(f'{ovec_L[i1, 0]:10.4f}  {ovec_L[i1, 1]:10.4f}  {ovec_L[i1, 2]:6.2f}  '
              f'{ovec_L[i1, 3]:3.1f}  {ovec_L[i1, 4]:4.1f}  {ovec_L[i1, 5]:4.0f}', file=ovecllog)
    ovecllog.close()

    ovecslog = open('Logs/Inv_BS_HyCe/ovec_S.log', 'w')    # log down ovec_L
    for i1 in range(len(ovec_S)):
        print(f'{ovec_S[i1, 0]:10.4f}  {ovec_S[i1, 1]:10.4f}  {ovec_S[i1, 2]:6.2f}  '
              f'{ovec_S[i1, 3]:3.1f}  {ovec_S[i1, 4]:4.1f}  {ovec_S[i1, 5]:4.0f}', file=ovecslog)
    ovecslog.close()

    mmax = 2 * m1 + m2   # The model size (put it here to mimic fortran)
    avela = np.mean(np.r_[ovec_L[:, 0], ovec_L[:, 0], ovec_S[:, 0]])    # Average latitude (Weight L twice, as final)

    # ## II. 2. The relative locations from NccPy/ NccInvPy: rvec ################
    # Initialize some arrays:

    rvec_hy = np.array(())         # lat, lon, dep, dt: The relative location pairs pass screening (hypo)
    rvec_hy_supp = np.array(())    # ev0, ev1, sig, iter: supplementary for the above pairs

    rvec_ce = np.array(())         # lat, lon, dep, dt: The relative location pairs pass screening (cent)
    rvec_ce_supp = np.array(())    # ev0, ev1, sig, iter: supplementary for the above pairs

    # ## II. 2. 1. For hypocenter relative location ################
    for i2 in range(0, len(nccgstbl_hy), 2):

        # Some error messages
        # Quit program if some relative locations are not "in pairs"
        if int(nccgstbl_hy[i2, 0]) != int(nccgstbl_hy[i2 + 1, 1]) or \
                int(nccgstbl_hy[i2, 1]) != int(nccgstbl_hy[i2 + 1, 0]) or \
                abs(nccgstbl_hy[i2, 7] - nccgstbl_hy[i2 + 1, 7]) > trivial:
            print(f'Error in nccgs.hypo.tbl: pair unbalenced: '
                  f'{nccgstbl_hy[i2, 0]:.0f} {nccgstbl_hy[i2 + 1, 1]:.0f} '
                  f'{nccgstbl_hy[i2, 1]:.0f} {nccgstbl_hy[i2 + 1, 0]:.0f} '
                  f'{nccgstbl_hy[i2, 7]:.4f} {nccgstbl_hy[i2 + 1, 7]:.4f}')
            print(f'Error in nccgs.hypo.tbl: pair unbalenced: '
                  f'{nccgstbl_hy[i2, 0]:.0f} {nccgstbl_hy[i2 + 1, 1]:.0f} '
                  f'{nccgstbl_hy[i2, 1]:.0f} {nccgstbl_hy[i2 + 1, 0]:.0f} '
                  f'{nccgstbl_hy[i2, 7]:.4f} {nccgstbl_hy[i2 + 1, 7]:.4f}', file=nccinvlog)
            sys.exit(0)

        # Loop over the centroid tbl, and fetch its CC
        cctmpp = [0, 0]  # If not found in centroid, proceed!

        # Look for the same pair of event
        for iii2 in range(0, len(nccgstbl_ce), 2):
            if nccgstbl_hy[i2, 0] == nccgstbl_ce[iii2, 0] and nccgstbl_hy[i2, 1] == nccgstbl_ce[iii2, 1]:
                cctmpp = [nccgstbl_ce[iii2, 9], nccgstbl_ce[iii2 + 1, 9]]

        # Condition of removing large events with high centroid NCC
        if centcc_tag == 1 and cctmpp[0] > centccthres and cctmpp[1] > centccthres \
                and ovec[int(nccgstbl_hy[i2, 0]), 4] + trivial >= magsep \
                and ovec[int(nccgstbl_hy[i2, 1]), 4] + trivial >= magsep:
            print(f'Pair unused (hypo): {nccgstbl_hy[i2, 0]:.0f} {nccgstbl_hy[i2, 1]:.0f}: large repeaters!  '
                  f'Mag: {ovec[int(nccgstbl_hy[i2, 0]), 4]:4.1f} {ovec[int(nccgstbl_hy[i2, 1]), 4]:4.1f}  '
                  f'CC(cent): {cctmpp[0]:7.4f}  {cctmpp[1]:7.4f}')
            print(f'Pair unused (hypo): {nccgstbl_hy[i2, 0]:.0f} {nccgstbl_hy[i2, 1]:.0f}: large repeaters!  '
                  f'Mag: {ovec[int(nccgstbl_hy[i2, 0]), 4]:4.1f} {ovec[int(nccgstbl_hy[i2, 1]), 4]:4.1f}  '
                  f'CC(cent): {cctmpp[0]:7.4f}  {cctmpp[1]:7.4f}', file=nccinvlog)

        else:
            # Number of waveform limits
            if int(nccgstbl_hy[i2, 10]) >= minwv_hy:

                # Separation of events in catalog
                evtdist_hor = vincenty_custom((ovec[int(nccgstbl_hy[i2, 0]), 0], ovec[int(nccgstbl_hy[i2, 0]), 1]),
                                              (ovec[int(nccgstbl_hy[i2, 1]), 0], ovec[int(nccgstbl_hy[i2, 1]), 1]))
                evtdist_ver = np.abs(ovec[int(nccgstbl_hy[i2, 0]), 2] - ovec[int(nccgstbl_hy[i2, 1]), 2])
                if evtdist_hor <= maxdist_hor and evtdist_ver <= maxdist_ver:

                    # Consistency check (within each pair of relative location!)
                    if nccgstbl_hy[i2, 7] - trivial <= ddifmax_hy:

                        # NCC significance check: ev0
                        if nccgstbl_hy[i2, 6] + trivial >= nccc:

                            if len(rvec_hy) == 0:
                                rvec_hy = np.array([[nccgstbl_hy[i2, 2], nccgstbl_hy[i2, 3],
                                                     nccgstbl_hy[i2, 4], nccgstbl_hy[i2, 5]]])
                                rvec_hy_supp = np.array([[nccgstbl_hy[i2, 0], nccgstbl_hy[i2, 1],
                                                          nccgstbl_hy[i2, 6], nccgstbl_hy[i2, 8]]])
                            else:
                                rvec_hy = np.r_[rvec_hy, [[nccgstbl_hy[i2, 2], nccgstbl_hy[i2, 3],
                                                           nccgstbl_hy[i2, 4], nccgstbl_hy[i2, 5]]]]
                                rvec_hy_supp = np.r_[rvec_hy_supp, [[nccgstbl_hy[i2, 0], nccgstbl_hy[i2, 1],
                                                                     nccgstbl_hy[i2, 6], nccgstbl_hy[i2, 8]]]]

                        # NCC significance check: ev1
                        if nccgstbl_hy[i2 + 1, 6] + trivial >= nccc:

                            if len(rvec_hy) == 0:
                                rvec_hy = np.array([[nccgstbl_hy[i2 + 1, 2], nccgstbl_hy[i2 + 1, 3],
                                                     nccgstbl_hy[i2 + 1, 4], nccgstbl_hy[i2 + 1, 5]]])
                                rvec_hy_supp = np.array([[nccgstbl_hy[i2 + 1, 0], nccgstbl_hy[i2 + 1, 1],
                                                          nccgstbl_hy[i2 + 1, 6], nccgstbl_hy[i2 + 1, 8]]])
                            else:
                                rvec_hy = np.r_[rvec_hy, [[nccgstbl_hy[i2 + 1, 2], nccgstbl_hy[i2 + 1, 3],
                                                           nccgstbl_hy[i2 + 1, 4], nccgstbl_hy[i2 + 1, 5]]]]
                                rvec_hy_supp = np.r_[rvec_hy_supp, [[nccgstbl_hy[i2 + 1, 0], nccgstbl_hy[i2 + 1, 1],
                                                                     nccgstbl_hy[i2 + 1, 6], nccgstbl_hy[i2 + 1, 8]]]]

                    # Exception: allow the exceptionally good one (n2) if the other (n2 + 1) is exceptionally bad
                    elif hl_exception_tag == 1 and \
                            nccgstbl_hy[i2, 6] + trivial >= nccch and nccgstbl_hy[i2 + 1, 6] - trivial < ncccl:

                        if len(rvec_hy) == 0:
                            rvec_hy = np.array([[nccgstbl_hy[i2, 2], nccgstbl_hy[i2, 3],
                                                 nccgstbl_hy[i2, 4], nccgstbl_hy[i2, 5]]])
                            rvec_hy_supp = np.array([[nccgstbl_hy[i2, 0], nccgstbl_hy[i2, 1],
                                                      nccgstbl_hy[i2, 6], nccgstbl_hy[i2, 8]]])
                        else:
                            rvec_hy = np.r_[rvec_hy, [[nccgstbl_hy[i2, 2], nccgstbl_hy[i2, 3],
                                                       nccgstbl_hy[i2, 4], nccgstbl_hy[i2, 5]]]]
                            rvec_hy_supp = np.r_[rvec_hy_supp, [[nccgstbl_hy[i2, 0], nccgstbl_hy[i2, 1],
                                                                 nccgstbl_hy[i2, 6], nccgstbl_hy[i2, 8]]]]

                    # Exception: allow the exceptionally good one (n2 + 1) if the other (n2) is exceptionally bad
                    elif hl_exception_tag == 1 and \
                            nccgstbl_hy[i2 + 1, 6] + trivial >= nccch and nccgstbl_hy[i2, 6] - trivial < ncccl:

                        if len(rvec_hy) == 0:
                            rvec_hy = np.array([[nccgstbl_hy[i2 + 1, 2], nccgstbl_hy[i2 + 1, 3],
                                                 nccgstbl_hy[i2 + 1, 4], nccgstbl_hy[i2 + 1, 5]]])
                            rvec_hy_supp = np.array([[nccgstbl_hy[i2 + 1, 0], nccgstbl_hy[i2 + 1, 1],
                                                      nccgstbl_hy[i2 + 1, 6], nccgstbl_hy[i2 + 1, 8]]])
                        else:
                            rvec_hy = np.r_[rvec_hy, [[nccgstbl_hy[i2 + 1, 2], nccgstbl_hy[i2 + 1, 3],
                                                       nccgstbl_hy[i2 + 1, 4], nccgstbl_hy[i2 + 1, 5]]]]
                            rvec_hy_supp = np.r_[rvec_hy_supp, [[nccgstbl_hy[i2 + 1, 0], nccgstbl_hy[i2 + 1, 1],
                                                                 nccgstbl_hy[i2 + 1, 6], nccgstbl_hy[i2 + 1, 8]]]]

                else:
                    print(f'Pair unused (hypo): {nccgstbl_hy[i2, 0]:.0f} {nccgstbl_hy[i2 + 1, 1]:.0f}: '
                          f'Catalog separation (hor/ ver) ({evtdist_hor:.3f} / {evtdist_ver:.3f}) '
                          f'exceeding limit ({maxdist_hor:.3f} / {maxdist_ver:.3f}', file=nccinvlog)
            else:
                print(f'Pair unused (hypo): {nccgstbl_hy[i2, 0]:.0f} {nccgstbl_hy[i2 + 1, 1]:.0f}: '
                      f'Number of waveform ({nccgstbl_hy[i2, 10]:.0f}) < {minwv_hy:.0f}'
                      , file=nccinvlog)

    # ## II. 2. 2. For centroid relative location #####
    for i2 in range(0, len(nccgstbl_ce), 2):

        # Some error messages
        # Quit program if some relative locations are not "in pairs"
        if int(nccgstbl_ce[i2, 0]) != int(nccgstbl_ce[i2 + 1, 1]) or \
                int(nccgstbl_ce[i2, 1]) != int(nccgstbl_ce[i2 + 1, 0]) or \
                abs(nccgstbl_ce[i2, 7] - nccgstbl_ce[i2 + 1, 7]) > trivial:
            print(f'Error in nccgs.cent.tbl: pair unbalenced: '
                  f'{nccgstbl_ce[i2, 0]:.0f} {nccgstbl_ce[i2 + 1, 1]:.0f} '
                  f'{nccgstbl_ce[i2, 1]:.0f} {nccgstbl_ce[i2 + 1, 0]:.0f} '
                  f'{nccgstbl_ce[i2, 7]:.4f} {nccgstbl_ce[i2 + 1, 7]:.4f}')
            print(f'Error in nccgs.cent.tbl: pair unbalenced: '
                  f'{nccgstbl_ce[i2, 0]:.0f} {nccgstbl_ce[i2 + 1, 1]:.0f} '
                  f'{nccgstbl_ce[i2, 1]:.0f} {nccgstbl_ce[i2 + 1, 0]:.0f} '
                  f'{nccgstbl_ce[i2, 7]:.4f} {nccgstbl_ce[i2 + 1, 7]:.4f}', file=nccinvlog)
            sys.exit(0)

        else:
            # Number of waveform limits
            if int(nccgstbl_ce[i2, 10]) >= minwv_ce:

                # Separation of events in catalog
                evtdist_hor = vincenty_custom((ovec[int(nccgstbl_ce[i2, 0]), 0], ovec[int(nccgstbl_ce[i2, 0]), 1]),
                                              (ovec[int(nccgstbl_ce[i2, 1]), 0], ovec[int(nccgstbl_ce[i2, 1]), 1]))
                evtdist_ver = np.abs(ovec[int(nccgstbl_ce[i2, 0]), 2] - ovec[int(nccgstbl_ce[i2, 1]), 2])
                if evtdist_hor <= maxdist_hor and evtdist_ver <= maxdist_ver:

                    # Consistency check (within each pair of relative location!)
                    if nccgstbl_ce[i2, 7] - trivial <= ddifmax_ce:

                        # NCC significance check: ev0
                        if nccgstbl_ce[i2, 6] + trivial >= nccc:

                            if len(rvec_ce) == 0:
                                rvec_ce = np.array([[nccgstbl_ce[i2, 2], nccgstbl_ce[i2, 3],
                                                     nccgstbl_ce[i2, 4], nccgstbl_ce[i2, 5]]])
                                rvec_ce_supp = np.array([[nccgstbl_ce[i2, 0], nccgstbl_ce[i2, 1],
                                                          nccgstbl_ce[i2, 6], nccgstbl_ce[i2, 8]]])
                            else:
                                rvec_ce = np.r_[rvec_ce, [[nccgstbl_ce[i2, 2], nccgstbl_ce[i2, 3],
                                                           nccgstbl_ce[i2, 4], nccgstbl_ce[i2, 5]]]]
                                rvec_ce_supp = np.r_[rvec_ce_supp, [[nccgstbl_ce[i2, 0], nccgstbl_ce[i2, 1],
                                                                     nccgstbl_ce[i2, 6], nccgstbl_ce[i2, 8]]]]

                        # NCC significance check: ev1
                        if nccgstbl_ce[i2 + 1, 6] + trivial >= nccc:

                            if len(rvec_ce) == 0:
                                rvec_ce = np.array([[nccgstbl_ce[i2 + 1, 2], nccgstbl_ce[i2 + 1, 3],
                                                     nccgstbl_ce[i2 + 1, 4], nccgstbl_ce[i2 + 1, 5]]])
                                rvec_ce_supp = np.array([[nccgstbl_ce[i2 + 1, 0], nccgstbl_ce[i2 + 1, 1],
                                                          nccgstbl_ce[i2 + 1, 6], nccgstbl_ce[i2 + 1, 8]]])
                            else:
                                rvec_ce = np.r_[rvec_ce, [[nccgstbl_ce[i2 + 1, 2], nccgstbl_ce[i2 + 1, 3],
                                                           nccgstbl_ce[i2 + 1, 4], nccgstbl_ce[i2 + 1, 5]]]]
                                rvec_ce_supp = np.r_[rvec_ce_supp, [[nccgstbl_ce[i2 + 1, 0], nccgstbl_ce[i2 + 1, 1],
                                                                     nccgstbl_ce[i2 + 1, 6], nccgstbl_ce[i2 + 1, 8]]]]

                    # Exception: allow the exceptionally good one (n2) if the other (n2 + 1) is exceptionally bad
                    elif hl_exception_tag == 1 and \
                            nccgstbl_ce[i2, 6] + trivial >= nccch and nccgstbl_ce[i2 + 1, 6] - trivial < ncccl:

                        if len(rvec_ce) == 0:
                            rvec_ce = np.array([[nccgstbl_ce[i2, 2], nccgstbl_ce[i2, 3],
                                                 nccgstbl_ce[i2, 4], nccgstbl_ce[i2, 5]]])
                            rvec_ce_supp = np.array([[nccgstbl_ce[i2, 0], nccgstbl_ce[i2, 1],
                                                      nccgstbl_ce[i2, 6], nccgstbl_ce[i2, 8]]])
                        else:
                            rvec_ce = np.r_[rvec_ce, [[nccgstbl_ce[i2, 2], nccgstbl_ce[i2, 3],
                                                       nccgstbl_ce[i2, 4], nccgstbl_ce[i2, 5]]]]
                            rvec_ce_supp = np.r_[rvec_ce_supp, [[nccgstbl_ce[i2, 0], nccgstbl_ce[i2, 1],
                                                                 nccgstbl_ce[i2, 6], nccgstbl_ce[i2, 8]]]]

                    # Exception: allow the exceptionally good one (n2 + 1) if the other (n2) is exceptionally bad
                    elif hl_exception_tag == 1 and \
                            nccgstbl_ce[i2 + 1, 6] + trivial >= nccch and nccgstbl_ce[i2, 6] - trivial < ncccl:

                        if len(rvec_ce) == 0:
                            rvec_ce = np.array([[nccgstbl_ce[i2 + 1, 2], nccgstbl_ce[i2 + 1, 3],
                                                 nccgstbl_ce[i2 + 1, 4], nccgstbl_ce[i2 + 1, 5]]])
                            rvec_ce_supp = np.array([[nccgstbl_ce[i2 + 1, 0], nccgstbl_ce[i2 + 1, 1],
                                                      nccgstbl_ce[i2 + 1, 6], nccgstbl_ce[i2 + 1, 8]]])
                        else:
                            rvec_ce = np.r_[rvec_ce, [[nccgstbl_ce[i2 + 1, 2], nccgstbl_ce[i2 + 1, 3],
                                                       nccgstbl_ce[i2 + 1, 4], nccgstbl_ce[i2 + 1, 5]]]]
                            rvec_ce_supp = np.r_[rvec_ce_supp, [[nccgstbl_ce[i2 + 1, 0], nccgstbl_ce[i2 + 1, 1],
                                                                 nccgstbl_ce[i2 + 1, 6], nccgstbl_ce[i2 + 1, 8]]]]

                else:
                    print(f'Pair unused (cent): {nccgstbl_ce[i2, 0]:.0f} {nccgstbl_ce[i2 + 1, 1]:.0f}: '
                          f'Catalog separation (hor/ ver) ({evtdist_hor:.3f} / {evtdist_ver:.3f}) '
                          f'exceeding limit ({maxdist_hor:.3f} / {maxdist_ver:.3f}', file=nccinvlog)
            else:
                print(f'Pair unused (cent): {nccgstbl_ce[i2, 0]:.0f} {nccgstbl_ce[i2 + 1, 1]:.0f}: '
                      f'Number of waveform ({nccgstbl_ce[i2, 10]:.0f}) < {minwv_ce:.0f}'
                      , file=nccinvlog)

    # ## II. 2. 3. Save & judge ################
    # Save these to nccinv.pass (but only for viewing, the rvec is already given above)
    nccinvhypopass = open('Logs/Inv_BS_HyCe/nccinv.hypo.pass', 'w')    # log file
    for i2 in range(len(rvec_hy)):
        print(f'{rvec_hy_supp[i2, 0]:4.0f}  {rvec_hy_supp[i2, 1]:4.0f}  {rvec_hy[i2, 0]:9.4f}  {rvec_hy[i2, 1]:9.4f}  '
              f'{rvec_hy[i2, 2]:9.4f}  {rvec_hy[i2, 3]:7.3f}  {rvec_hy_supp[i2, 2]:8.4f}  {rvec_hy_supp[i2, 3]:2.0f}'
              , file=nccinvhypopass)
    nccinvhypopass.close()

    nccinvcentpass = open('Logs/Inv_BS_HyCe/nccinv.cent.pass', 'w')    # log file
    for i2 in range(len(rvec_ce)):
        print(f'{rvec_ce_supp[i2, 0]:4.0f}  {rvec_ce_supp[i2, 1]:4.0f}  {rvec_ce[i2, 0]:9.4f}  {rvec_ce[i2, 1]:9.4f}  '
              f'{rvec_ce[i2, 2]:9.4f}  {rvec_ce[i2, 3]:7.3f}  {rvec_ce_supp[i2, 2]:8.4f}  {rvec_ce_supp[i2, 3]:2.0f}'
              , file=nccinvcentpass)
    nccinvcentpass.close()

    # Stop program if no effective pair is available
    if len(rvec_hy) + len(rvec_ce) == 0:
        print(f'Error in nccgs.tbl: no relative location available after screening!')
        print(f'Error in nccgs.tbl: no relative location available after screening!', file=nccinvlog)
        sys.exit(0)
    elif len(rvec_hy) == 0:
        print(f'Warning in nccgs.tbl: centroid inversion mode! no nccgs.hypo.tbl!')
        print(f'\nWarning in nccgs.tbl: centroid inversion mode! no nccgs.hypo.tbl!\n', file=nccinvlog)
    elif len(rvec_ce) == 0:
        print(f'Warning in nccgs.tbl: hypocenter inversion mode! no nccgs.cent.tbl!')
        print(f'\nWarning in nccgs.tbl: hypocenter inversion mode! no nccgs.cent.tbl!\n', file=nccinvlog)

    # Declare arrays and dimensions
    # rvec = np.r_[rvec_hy, rvec_ce]
    rvec_supp = np.r_[rvec_hy_supp, rvec_ce_supp]

    n1 = len(rvec_hy)         # relative location available for hypo
    n2 = len(rvec_ce)         # relative location available for cent
    nmax = mmax + n1 + n2     # The data size, including the constrian
    print(f'Model: {mmax:.0f}  Data: {nmax:.0f}')
    print(f'm1 (# Large evt): {m1:.0f}  m2 (# small evt): {m2:.0f}  n1 (# hypo tbl): {n1:.0f} n2 (# cent tbl): {n2:.0f}\n')
    print(f'\nModel: {mmax:.0f}  Data: {nmax:.0f}', file=nccinvlog)
    print(f'm1 (# Large evt): {m1:.0f}  m2 (# small evt): {m2:.0f}  n1 (# hypo tbl): {n1:.0f} n2 (# cent tbl): {n2:.0f}\n'
          , file=nccinvlog)
    nccinvlog.flush()

    # ## II. 3. The assignment of random seeds, and the choosing of files#####
    # The "random table": #BS * len(tbl), sorted in each BS realization
    randmas_hy = np.sort(np.random.default_rng(0).integers(len(rvec_hy), size=(bsnum, len(rvec_hy))), axis=1)
    randmas_ce = np.sort(np.random.default_rng(0).integers(len(rvec_ce), size=(bsnum, len(rvec_ce))), axis=1)

    # Save this randmas table!
    if detail_output == 1:
        randmassave_hy = open(f'Logs/Inv_BS_HyCe/rand_table.hypo.txt', 'w')
        randmassave_ce = open(f'Logs/Inv_BS_HyCe/rand_table.cent.txt', 'w')
        np.savetxt(randmassave_hy, randmas_hy, fmt='%6.0f')
        np.savetxt(randmassave_ce, randmas_ce, fmt='%6.0f')
        randmassave_hy.close()
        randmassave_ce.close()

    # Initialize parameter for entire BS
    bs_results = np.zeros((mmax, bsnum, 4))         # Final location matrix in #evt * #BS * (lat, lon, dep, time)
    bs_results_hy = np.zeros((m1 + m2, bsnum, 4))   # Final location matrix of hypocenter, in ovec sequence
    bs_results_ce = np.zeros((m1 + m2, bsnum, 4))   # Final location matrix of centroid, in ovec sequence

    # For each #BS:
    for ii in range(bsnum):

        print(f'\n======= This is BS # {ii} =======')
        print(f'\n======= This is BS # {ii} =======', file=nccinvlog)

        # Extract the random vector for this #BS
        rvec_hy_bs = rvec_hy[randmas_hy[ii, :]]
        rvec_ce_bs = rvec_ce[randmas_ce[ii, :]]

        rvec_hy_supp_bs = rvec_hy_supp[randmas_hy[ii, :]]
        rvec_ce_supp_bs = rvec_ce_supp[randmas_ce[ii, :]]
        rvec_supp_bs = np.r_[rvec_hy_supp_bs, rvec_ce_supp_bs]

        if detail_output == 1:
            bstblsave_hy = open(f'Logs/Inv_BS_HyCe/bs_{ii}.hypo.tbl', 'w')
            bstblsave_ce = open(f'Logs/Inv_BS_HyCe/bs_{ii}.cent.tbl', 'w')

            for jj in range(len(rvec_hy)):
                print(f'{rvec_hy_supp_bs[jj, 0]:4.0f}  {rvec_hy_supp_bs[jj, 1]:4.0f}  {rvec_hy_bs[jj, 0]:9.4f}  '
                      f'{rvec_hy_bs[jj, 1]:9.4f}  {rvec_hy_bs[jj, 2]:9.4f}  {rvec_hy_bs[jj, 3]:7.3f}  '
                      f'{rvec_hy_supp_bs[jj, 2]:8.4f}  {0:8.4f}  {rvec_hy_supp_bs[jj, 3]:2.0f}  '
                      f'{0.8:7.4f}  {20:3.0f}', file=bstblsave_hy)

            for jj in range(len(rvec_ce)):
                print(f'{rvec_ce_supp_bs[jj, 0]:4.0f}  {rvec_ce_supp_bs[jj, 1]:4.0f}  {rvec_ce_bs[jj, 0]:9.4f}  '
                      f'{rvec_ce_bs[jj, 1]:9.4f}  {rvec_ce_bs[jj, 2]:9.4f}  {rvec_ce_bs[jj, 3]:7.3f}  '
                      f'{rvec_ce_supp_bs[jj, 2]:8.4f}  {0:8.4f}  {rvec_ce_supp_bs[jj, 3]:2.0f}  '
                      f'{0.8:7.4f}  {20:3.0f}', file=bstblsave_ce)

            bstblsave_hy.close()
            bstblsave_ce.close()

        # ## II. 4. Declare MASSIVE amount of parameters after sizes are determined ################
        mvec = np.zeros((mmax, 4))   # The model vector
        merr = np.zeros((mmax, 4))   # The model error vector
        dvec = np.zeros((nmax, 4))   # The data vector

        G = np.zeros((nmax, mmax))        # The Green's function matrix
        Gt = np.zeros((mmax, nmax))       # The transpose of G
        GtW = np.zeros((mmax, nmax))      # The weighted transpose of G
        GtWG = np.zeros((mmax, mmax))     # GtW * G
        iGtWG = np.zeros((mmax, mmax))    # The inverse of GtWG
        Gg = np.zeros((mmax, nmax))       # iGtWG * GtW
        Ggt = np.zeros((nmax, mmax))      # transpose of Gg
        Ggcovd = np.zeros((mmax, nmax))   # Gg / weighting
        covm = np.zeros((mmax, mmax))     # Ggcovd * Ggt, covariance matrix

        pN = np.zeros(n1 + n2)       # probability of noise vector
        sig2_N = 0             # variance of noise: model-space-dependent, i.e. always size of iitr0
        sig2_S = np.zeros(2)   # variance of signal: quantization error, grid-spacing-dependent
        wf = np.zeros(nmax)    # weighting vector
        d = np.zeros(nmax)     # data vector (observed), for each component
        m = np.zeros(mmax)     # calculated model vector, for each component; Gg * d
        Gm = np.zeros(nmax)    # data vector (calculated), for each component; G * m
        e = np.zeros(nmax)     # error vector,            for each component; d - Gm
        Wd = np.zeros(nmax)    # weighted data vector, for each component; wf * d
        We = np.zeros(nmax)    # weighted error vector , for each component; wf * e

        iv = np.zeros(nmax)    # name tag for evt 0
        jv = np.zeros(nmax)    # name tag for evt 1

        c = np.zeros(mmax)           # eigenvalues of GtWG
        v = np.zeros((mmax, mmax))   # eigenvectors of GtWG

        # ## III. 5. Fill the kernal matrix (G) ################
        # The parts of data (relative locations)
        # For the hypocenters:
        for i3 in range(n1):
            # Event 0
            if ovec[int(rvec_supp_bs[i3, 0]), 4] + trivial >= magsep:         # Block L, hypo
                G[i3, int(ovec[int(rvec_supp_bs[i3, 0]), 6])] = -1           # Index in L (in m1)
            else:                                                         # Block S, both
                G[i3, int(ovec[int(rvec_supp_bs[i3, 0]), 6] + 2 * m1)] = -1  # Index in S (in m2), indent by 2 * m1 (skip 2m1 of L)

            # Event 1
            if ovec[int(rvec_supp_bs[i3, 1]), 4] + trivial >= magsep:         # Block L, hypo
                G[i3, int(ovec[int(rvec_supp_bs[i3, 1]), 6])] = 1            # Index in L (in m1)
            else:                                                         # Block S, both
                G[i3, int(ovec[int(rvec_supp_bs[i3, 1]), 6] + 2 * m1)] = 1   # Index in S (in m2), indent by 2 * m1 (skip 2m1 of L)

            iv[i3] = int(rvec_supp_bs[i3, 0])
            jv[i3] = int(rvec_supp_bs[i3, 1])

        # For the centroids:
        for i3 in range(n1, n1 + n2):
            # Event 0
            if ovec[int(rvec_supp_bs[i3, 0]), 4] + trivial >= magsep:         # Block L, cent
                G[i3, int(ovec[int(rvec_supp_bs[i3, 0]), 6] + m1)] = -1      # Index in L (in m1), indent by m1 (skip m1 of hypo)
            else:                                                         # Block S, both
                G[i3, int(ovec[int(rvec_supp_bs[i3, 0]), 6] + 2 * m1)] = -1  # Index in S (in m2), indent by 2 * m1 (skip 2m1 of L)

            # Event 1
            if ovec[int(rvec_supp_bs[i3, 1]), 4] + trivial >= magsep:         # Block L, cent
                G[i3, int(ovec[int(rvec_supp_bs[i3, 1]), 6] + m1)] = 1       # Index in L (in m1), indent by m1 (skip m1 of hypo)
            else:                                                         # Block S, both
                G[i3, int(ovec[int(rvec_supp_bs[i3, 1]), 6] + 2 * m1)] = 1   # Index in S (in m2), indent by 2 * m1 (skip 2m1 of L)

            iv[i3] = int(rvec_supp_bs[i3, 0])
            jv[i3] = int(rvec_supp_bs[i3, 1])

        # The parts of constrains (catalog locations)
        for i3 in range(n1 + n2, nmax):   # nmax = n1 + n2 + mmax = n1 + n2 + 2 * m1 + m2
            for j3 in range(mmax):
                if i3 - n1 - n2 == j3:
                    G[i3, j3] = 1

            iv[i3] = i3 - n1 - n2
            jv[i3] = i3 - n1 - n2

        Gt = np.transpose(G)

        # ## II. 6. Fill the data matrix (dvec) ################

        # The parts of data: hypocenter relative location
        for i4 in range(n1):
            dvec[i4, 0] = rvec_hy_bs[i4, 0]
            dvec[i4, 1] = rvec_hy_bs[i4, 1]
            dvec[i4, 2] = rvec_hy_bs[i4, 2]
            dvec[i4, 3] = rvec_hy_bs[i4, 3]

        # The parts of data: centroid relative location
        for i4 in range(n1, n1 + n2):
            dvec[i4, 0] = rvec_ce_bs[i4 - n1, 0]
            dvec[i4, 1] = rvec_ce_bs[i4 - n1, 1]
            dvec[i4, 2] = rvec_ce_bs[i4 - n1, 2]
            dvec[i4, 3] = rvec_ce_bs[i4 - n1, 3]

        # The parts of constrains: Large, catalog locations (for hypocenter), change unit to km if necessary
        for i4 in range(n1 + n2, n1 + n2 + m1):
            dvec[i4, 0] = ovec_L[i4 - n1 - n2, 0] * earth_rad * np.pi / 180
            dvec[i4, 1] = ovec_L[i4 - n1 - n2, 1] * earth_rad * np.pi / 180 * np.cos(avela / 180 * np.pi)
            dvec[i4, 2] = ovec_L[i4 - n1 - n2, 2]
            dvec[i4, 3] = ovec_L[i4 - n1 - n2, 3]

        # The parts of constrains: Large, catalog locations (for centroid), change unit to km if necessary
        for i4 in range(n1 + n2 + m1, n1 + n2 + 2 * m1):
            dvec[i4, 0] = ovec_L[i4 - n1 - n2 - m1, 0] * earth_rad * np.pi / 180
            dvec[i4, 1] = ovec_L[i4 - n1 - n2 - m1, 1] * earth_rad * np.pi / 180 * np.cos(avela / 180 * np.pi)
            dvec[i4, 2] = ovec_L[i4 - n1 - n2 - m1, 2]
            dvec[i4, 3] = ovec_L[i4 - n1 - n2 - m1, 3]

        # The parts of constrains: small, catalog locations, change unit to km if necessary
        for i4 in range(n1 + n2 + 2 * m1, n1 + n2 + 2 * m1 + m2):
            dvec[i4, 0] = ovec_S[i4 - n1 - n2 - 2 * m1, 0] * earth_rad * np.pi / 180
            dvec[i4, 1] = ovec_S[i4 - n1 - n2 - 2 * m1, 1] * earth_rad * np.pi / 180 * np.cos(avela / 180 * np.pi)
            dvec[i4, 2] = ovec_S[i4 - n1 - n2 - 2 * m1, 2]
            dvec[i4, 3] = ovec_S[i4 - n1 - n2 - 2 * m1, 3]

        # Log the data vector down:
        # print(f'\nThe data vectors (lat, lon, dep, dt):', file=nccinvlog)
        # for i4 in range(nmax):
        #     print(f'{dvec[i4, 0]:12.4f} {dvec[i4, 1]:12.4f} {dvec[i4, 2]:10.4f} {dvec[i4, 3]:8.2f}', file=nccinvlog)

        # ## II. 7. The weighting for data (upper half of wf) #####
        # The variance of signal/ noise
        sig2_N = ((2 * ala1) ** 2 + (2 * alo1) ** 2 + (2 * adp1) ** 2) / 12

        # sig2_S[0] = (dla1 ** 2 + dlo1 ** 2 + ddp1 ** 2) / 12
        # sig2_S[1] = (dla2 ** 2 + dlo2 ** 2 + ddp2 ** 2) / 12

        # Adhere to the fortran version...
        sig2_S[0] = (dla1 ** 2) / 12
        sig2_S[1] = (dla2 ** 2) / 12

        for i5 in range(n1 + n2):
            stdtmp = int(round(rvec_supp_bs[i5, 2] / ds))

            # Look for the clostest in table
            if 0 <= stdtmp <= int(round(smax / ds)):
                pN[i5] = ptable[stdtmp, 1]
            elif stdtmp < 0:
                pN[i5] = 1
                # print(f'Error: NCC / std < 0: {i5} {stdtmp}', file=nccinvlog)
            elif stdtmp > int(round(smax / ds)):
                pN[i5] = 0
                # print(f'Note: NCC / std > max at table (set pN = 0): {i5} {stdtmp}', file=nccinvlog)

            wf[i5] = 1 / (pN[i5] * sig2_N + (1 - pN[i5]) * sig2_S[int(round(rvec_supp_bs[i5, 3]))])

        # # Log the weighting vector down:
        # print(f'\nThe weighting vectors (NCC / sig, pN, wf, iiter):', file=nccinvlog)
        # for i5 in range(n1 + n2):
        #     print(f'{rvec_supp_bs[i5, 2]:8.4f} {pN[i5]:12.10f} {wf[i5]:15.6f} {rvec_supp_bs[i5, 3]:2.0f}',
        #     file=nccinvlog)

        # ## III. Bayesian Inversion using ABIC #####
        # ## III. 1. The weighting for constrains (lower half of wf) by ABIC #####
        for comp in range(4):     # For lat, lon, dep, dt
            ABICmin = 1 / trivial
            kmin = 0
            ABICrange = 100         # The window to search for min(ABIC)

            # numba version for time saving
            (kmin, abicmat) = abic_core(n1 + n2, nmax, mmax, comp, ABICrange, ABICmin, kmin, wf, G, Gt, GtW, dvec, Wd, We)

            # # Output the matrix
            # abiclog = open(f'Logs/Inv_BS_HyCe/abic.{complist[comp]}', 'w')   # ABIC log
            # for k in range(-ABICrange, ABICrange + 1):
            #     print(f'{abicmat[k, 0]:3.0f}  {abicmat[k, 1]:12.6f}  {abicmat[k, 2]:.8f}', file=abiclog)
            # abiclog.close()

            # Now assign the weighting when min ABIC is found
            alpha2 = 10 ** (kmin * 0.05)
            for i6 in range(n1 + n2, nmax):
                wf[i6] = alpha2

        # ## III. 2. The actual inversion #####
            # Again, least squares inversion (this may look a lot like the above, naturally)
            for i7 in range(mmax):
                for j7 in range(nmax):
                    GtW[i7, j7] = Gt[i7, j7] * wf[j7]

            GtWG = np.matmul(GtW, G)
            iGtWG = np.linalg.inv(GtWG)
            Gg = np.matmul(iGtWG, GtW)

            # Comment out the model covariance matrix (not logged here)
            # for i7 in range(mmax):
            #     for j7 in range(nmax):
            #         Ggcovd[i7, j7] = Gg[i7, j7] / wf[j7]
            #
            # Ggt = np.transpose(Gg)
            # covm = np.matmul(Ggcovd, Ggt)
            #
            # for i7 in range(mmax):
            #     merr[i7, comp] = np.sqrt(covm[i7, i7])

            d = dvec[:, comp]
            m = np.matmul(Gg, d)
            Gm = np.matmul(G, m)
            e = d - Gm
            mvec[:, comp] = m

            for i7 in range(nmax):
                Wd[i7] = wf[i7] * d[i7]
                We[i7] = wf[i7] * e[i7]

            Res = np.dot(e, We)
            Res0 = np.dot(d, Wd)

            if Res < trivial and Res0 < trivial:
                VR = 1
            else:
                VR = 1 - (Res / Res0)

            # # Log down inversion outputs
            # invout = open(f'Logs/Inv_BS_HyCe/invout.{complist[comp]}', 'w')  # inversion output log
            # print(f'ev0, ev1, d, Gm, eWe, wf', file=invout)
            # for i7 in range(nmax):
            #     print(f'{iv[i7]:4.0f} {jv[i7]:4.0f} {d[i7]:15.6f} {Gm[i7]:15.6f} '
            #           f'{We[i7] * e[i7]:15.6f} {wf[i7]:15.6f}', file=invout)
            # invout.close()

            # Log down major outputs, VR, etc
            print(f'### {complist[comp]} ###')
            print(f'Weighting of constrains (alpha2): {alpha2:12.6f}')
            print(f'L2-norm (reloc) = {np.sqrt(Res):15.5f}   L2-norm (ori) = {np.sqrt(Res0):15.5f}')
            print(f'Variance reduction = {VR:10.5f}')

            print(f'\n### {complist[comp]} ###', file=nccinvlog)
            print(f'Weighting of constrains (alpha2): {alpha2:12.6f}', file=nccinvlog)
            print(f'L2-norm (reloc) = {np.sqrt(Res):15.5f}   L2-norm (ori) = {np.sqrt(Res0):15.5f}', file=nccinvlog)
            print(f'Variance reduction = {VR:10.5f}', file=nccinvlog)

        # ## III. 3. Final output of files, parameters, etc #####
        mvec[:, 0] /= (earth_rad * np.pi / 180)
        mvec[:, 1] /= (earth_rad * np.pi / 180 * np.cos(avela / 180 * np.pi))

        # merr[:, 0] /= (earth_rad * np.pi / 180)
        # merr[:, 1] /= (earth_rad * np.pi / 180 * np.cos(np.mean(mvec[:, 0]) / 180 * np.pi))

        # Assign results to matrixes:
        bs_results[:, ii, 0] = mvec[:, 0]
        bs_results[:, ii, 1] = mvec[:, 1]
        bs_results[:, ii, 2] = mvec[:, 2]
        bs_results[:, ii, 3] = mvec[:, 3]

        nccinvlog.flush()

        # Time the execution
        end_time = time.time()
        total_time = end_time - start_time

        print(f'Finished BS # {ii}', str(datetime.timedelta(seconds=total_time)))

    # ## IV. BS finished: calculation of covariance, etc, and output #####
    # ## IV. 1. Initialize: sort the original mmax size to m1 + m2 size #####
    for i8 in range(m1 + m2):
        if ovec[i8, 5] == 1:     # Large: hypocenters and centroids are separated!
            bs_results_hy[i8, :, :] = bs_results[int(ovec[i8, 6]), :, :]    # hypocenter
            bs_results_ce[i8, :, :] = bs_results[int(ovec[i8, 6] + m1), :, :]  # centroid

        else:                    # Small: hypocenters and centroids are the same!
            bs_results_hy[i8, :, :] = bs_results[int(ovec[i8, 6] + 2 * m1), :, :]   # hypocenter
            bs_results_ce[i8, :, :] = bs_results[int(ovec[i8, 6] + 2 * m1), :, :]   # centroid

    # Change unit to km
    bs_results_km_hy = copy.copy(bs_results_hy)   # "real" copy in python
    bs_results_km_hy[:, :, 0] *= (earth_rad * np.pi / 180)
    bs_results_km_hy[:, :, 1] *= (earth_rad * np.pi / 180 * np.cos(avela / 180 * np.pi))

    bs_results_km_ce = copy.copy(bs_results_ce)   # "real" copy in python
    bs_results_km_ce[:, :, 0] *= (earth_rad * np.pi / 180)
    bs_results_km_ce[:, :, 1] *= (earth_rad * np.pi / 180 * np.cos(avela / 180 * np.pi))

    # Initialize some arrays: ("3 pairs": 3 slices (lon-lat, lon-dep, lat-dep))
    rec_main_hy = np.zeros((m1 + m2, 9))   # major-, minor-axis, rotation from top (N/ +dep) of error ellipse for the 3
    rec_cov_hy = np.zeros((m1 + m2, 9))    # variance/ covariance matrix of the 3 pairs
    rec_eig_hy = np.zeros((m1 + m2, 12))   # larger eigenvector, eigenvalues
    rec_std_hy = np.zeros((m1 + m2, 4))    # standard deviation of the 4 directions
    rec_ci_hy = np.zeros((m1 + m2, 12))    # confidence intervals (90%, 95%, 99%)

    rec_main_ce = np.zeros((m1 + m2, 9))   # major-, minor-axis, rotation from top (N/ +dep) of error ellipse for the 3
    rec_cov_ce = np.zeros((m1 + m2, 9))    # variance/ covariance matrix of the 3 pairs
    rec_eig_ce = np.zeros((m1 + m2, 12))   # larger eigenvector, eigenvalues
    rec_std_ce = np.zeros((m1 + m2, 4))    # standard deviation of the 4 directions
    rec_ci_ce = np.zeros((m1 + m2, 12))    # confidence intervals (90%, 95%, 99%)

    # ## IV. 2. Calculation of std, cov, error ellipses, etc: hypocenter #####
    for i9 in range(m1 + m2):
        # Standard deviation first!
        std_lat = np.std(bs_results_km_hy[i9, :, 0], ddof=1)
        std_lon = np.std(bs_results_km_hy[i9, :, 1], ddof=1)
        std_dep = np.std(bs_results_km_hy[i9, :, 2], ddof=1)
        std_tim = np.std(bs_results_km_hy[i9, :, 3], ddof=1)

        rec_std_hy[i9] = [std_lat, std_lon, std_dep, std_tim]

        # Error ellipse: Lon - Lat (same as plotting direction!)
        cov = np.cov(bs_results_km_hy[i9, :, 1], bs_results_km_hy[i9, :, 0], bias=False)    # Covariance matrix of lon, lat
        eigval, eigvec = np.linalg.eigh(cov)    # eigenvalues & eigenvectors (cov is, of course, symmetric!)
        seq = np.argsort(eigval)[::-1]          # The sequence to sort eigenvalues (flip it to descending!)
        eigvalsort = eigval[seq]                # Sort eigenvalues
        eigvecsort = eigvec[:, seq]             # Sort eigenvectors (which are in columns)

        if eigvecsort[1, 0] != 0:
            rot = np.degrees(np.arctan(eigvecsort[0, 0] / eigvecsort[1, 0]))   # Degrees in clockwise from 0 degree (North)
        else:
            rot = 90

        lenmajor = 2 * np.sqrt(scipy.stats.chi2.ppf(0.95, 2) * eigvalsort[0])    # Length of major axis (like diameter)
        lenminor = 2 * np.sqrt(scipy.stats.chi2.ppf(0.95, 2) * eigvalsort[1])    # Length of major axis (like diameter)

        rec_main_hy[i9, 0: 3] = [lenmajor, lenminor, rot]
        rec_cov_hy[i9, 0: 3] = [cov[0, 0], cov[1, 1], cov[0, 1]]
        rec_eig_hy[i9, 0: 4] = [eigvecsort[0, 0], eigvecsort[1, 0], eigvalsort[0], eigvalsort[1]]

        # Error ellipse: Lat - Dep (same as plotting direction!)
        cov = np.cov(bs_results_km_hy[i9, :, 0], bs_results_km_hy[i9, :, 2], bias=False)  # Covariance matrix of lat, dep
        eigval, eigvec = np.linalg.eigh(cov)  # eigenvalues & eigenvectors (cov is, of course, symmetric!)
        seq = np.argsort(eigval)[::-1]  # The sequence to sort eigenvalues (flip it to descending!)
        eigvalsort = eigval[seq]  # Sort eigenvalues
        eigvecsort = eigvec[:, seq]  # Sort eigenvectors (which are in columns)

        if eigvecsort[1, 0] != 0:
            rot = np.degrees(np.arctan(eigvecsort[0, 0] / eigvecsort[1, 0]))  # Degrees in clockwise from 0 degree (+ Depth)
        else:
            rot = 90

        lenmajor = 2 * np.sqrt(scipy.stats.chi2.ppf(0.95, 2) * eigvalsort[0])  # Length of major axis (like diameter)
        lenminor = 2 * np.sqrt(scipy.stats.chi2.ppf(0.95, 2) * eigvalsort[1])  # Length of major axis (like diameter)

        rec_main_hy[i9, 3: 6] = [lenmajor, lenminor, rot]
        rec_cov_hy[i9, 3: 6] = [cov[0, 0], cov[1, 1], cov[0, 1]]
        rec_eig_hy[i9, 4: 8] = [eigvecsort[0, 0], eigvecsort[1, 0], eigvalsort[0], eigvalsort[1]]

        # Error ellipse: Lon - Dep (same as plotting direction!)
        cov = np.cov(bs_results_km_hy[i9, :, 1], bs_results_km_hy[i9, :, 2], bias=False)  # Covariance matrix of lon, dep
        eigval, eigvec = np.linalg.eigh(cov)  # eigenvalues & eigenvectors (cov is, of course, symmetric!)
        seq = np.argsort(eigval)[::-1]  # The sequence to sort eigenvalues (flip it to descending!)
        eigvalsort = eigval[seq]  # Sort eigenvalues
        eigvecsort = eigvec[:, seq]  # Sort eigenvectors (which are in columns)

        if eigvecsort[1, 0] != 0:
            rot = np.degrees(np.arctan(eigvecsort[0, 0] / eigvecsort[1, 0]))  # Degrees in clockwise from 0 degree (+ Depth)
        else:
            rot = 90

        lenmajor = 2 * np.sqrt(scipy.stats.chi2.ppf(0.95, 2) * eigvalsort[0])  # Length of major axis (like diameter)
        lenminor = 2 * np.sqrt(scipy.stats.chi2.ppf(0.95, 2) * eigvalsort[1])  # Length of major axis (like diameter)

        rec_main_hy[i9, 6: 9] = [lenmajor, lenminor, rot]
        rec_cov_hy[i9, 6: 9] = [cov[0, 0], cov[1, 1], cov[0, 1]]
        rec_eig_hy[i9, 8: 12] = [eigvecsort[0, 0], eigvecsort[1, 0], eigvalsort[0], eigvalsort[1]]

        # Confidence intervals
        rec_ci_hy[i9, 0: 12] = \
            [np.percentile(bs_results_km_hy[i9, :, 0], 95) - np.percentile(bs_results_km_hy[i9, :, 0], 5),
             np.percentile(bs_results_km_hy[i9, :, 1], 95) - np.percentile(bs_results_km_hy[i9, :, 1], 5),
             np.percentile(bs_results_km_hy[i9, :, 2], 95) - np.percentile(bs_results_km_hy[i9, :, 2], 5),
             np.percentile(bs_results_km_hy[i9, :, 3], 95) - np.percentile(bs_results_km_hy[i9, :, 3], 5),
             np.percentile(bs_results_km_hy[i9, :, 0], 97.5) - np.percentile(bs_results_km_hy[i9, :, 0], 2.5),
             np.percentile(bs_results_km_hy[i9, :, 1], 97.5) - np.percentile(bs_results_km_hy[i9, :, 1], 2.5),
             np.percentile(bs_results_km_hy[i9, :, 2], 97.5) - np.percentile(bs_results_km_hy[i9, :, 2], 2.5),
             np.percentile(bs_results_km_hy[i9, :, 3], 97.5) - np.percentile(bs_results_km_hy[i9, :, 3], 2.5),
             np.percentile(bs_results_km_hy[i9, :, 0], 99.5) - np.percentile(bs_results_km_hy[i9, :, 0], 0.5),
             np.percentile(bs_results_km_hy[i9, :, 1], 99.5) - np.percentile(bs_results_km_hy[i9, :, 1], 0.5),
             np.percentile(bs_results_km_hy[i9, :, 2], 99.5) - np.percentile(bs_results_km_hy[i9, :, 2], 0.5),
             np.percentile(bs_results_km_hy[i9, :, 3], 99.5) - np.percentile(bs_results_km_hy[i9, :, 3], 0.5)]

    # ## IV. 3. Calculation of std, cov, error ellipses, etc: centroid #####
    for i9 in range(m1 + m2):
        # Standard deviation first!
        std_lat = np.std(bs_results_km_ce[i9, :, 0], ddof=1)
        std_lon = np.std(bs_results_km_ce[i9, :, 1], ddof=1)
        std_dep = np.std(bs_results_km_ce[i9, :, 2], ddof=1)
        std_tim = np.std(bs_results_km_ce[i9, :, 3], ddof=1)

        rec_std_ce[i9] = [std_lat, std_lon, std_dep, std_tim]

        # Error ellipse: Lon - Lat (same as plotting direction!)
        cov = np.cov(bs_results_km_ce[i9, :, 1], bs_results_km_ce[i9, :, 0], bias=False)    # Covariance matrix of lon, lat
        eigval, eigvec = np.linalg.eigh(cov)    # eigenvalues & eigenvectors (cov is, of course, symmetric!)
        seq = np.argsort(eigval)[::-1]          # The sequence to sort eigenvalues (flip it to descending!)
        eigvalsort = eigval[seq]                # Sort eigenvalues
        eigvecsort = eigvec[:, seq]             # Sort eigenvectors (which are in columns)

        if eigvecsort[1, 0] != 0:
            rot = np.degrees(np.arctan(eigvecsort[0, 0] / eigvecsort[1, 0]))   # Degrees in clockwise from 0 degree (North)
        else:
            rot = 90

        lenmajor = 2 * np.sqrt(scipy.stats.chi2.ppf(0.95, 2) * eigvalsort[0])    # Length of major axis (like diameter)
        lenminor = 2 * np.sqrt(scipy.stats.chi2.ppf(0.95, 2) * eigvalsort[1])    # Length of major axis (like diameter)

        rec_main_ce[i9, 0: 3] = [lenmajor, lenminor, rot]
        rec_cov_ce[i9, 0: 3] = [cov[0, 0], cov[1, 1], cov[0, 1]]
        rec_eig_ce[i9, 0: 4] = [eigvecsort[0, 0], eigvecsort[1, 0], eigvalsort[0], eigvalsort[1]]

        # Error ellipse: Lat - Dep (same as plotting direction!)
        cov = np.cov(bs_results_km_ce[i9, :, 0], bs_results_km_ce[i9, :, 2], bias=False)  # Covariance matrix of lat, dep
        eigval, eigvec = np.linalg.eigh(cov)  # eigenvalues & eigenvectors (cov is, of course, symmetric!)
        seq = np.argsort(eigval)[::-1]  # The sequence to sort eigenvalues (flip it to descending!)
        eigvalsort = eigval[seq]  # Sort eigenvalues
        eigvecsort = eigvec[:, seq]  # Sort eigenvectors (which are in columns)

        if eigvecsort[1, 0] != 0:
            rot = np.degrees(np.arctan(eigvecsort[0, 0] / eigvecsort[1, 0]))  # Degrees in clockwise from 0 degree (+ Depth)
        else:
            rot = 90

        lenmajor = 2 * np.sqrt(scipy.stats.chi2.ppf(0.95, 2) * eigvalsort[0])  # Length of major axis (like diameter)
        lenminor = 2 * np.sqrt(scipy.stats.chi2.ppf(0.95, 2) * eigvalsort[1])  # Length of major axis (like diameter)

        rec_main_ce[i9, 3: 6] = [lenmajor, lenminor, rot]
        rec_cov_ce[i9, 3: 6] = [cov[0, 0], cov[1, 1], cov[0, 1]]
        rec_eig_ce[i9, 4: 8] = [eigvecsort[0, 0], eigvecsort[1, 0], eigvalsort[0], eigvalsort[1]]

        # Error ellipse: Lon - Dep (same as plotting direction!)
        cov = np.cov(bs_results_km_ce[i9, :, 1], bs_results_km_ce[i9, :, 2], bias=False)  # Covariance matrix of lon, dep
        eigval, eigvec = np.linalg.eigh(cov)  # eigenvalues & eigenvectors (cov is, of course, symmetric!)
        seq = np.argsort(eigval)[::-1]  # The sequence to sort eigenvalues (flip it to descending!)
        eigvalsort = eigval[seq]  # Sort eigenvalues
        eigvecsort = eigvec[:, seq]  # Sort eigenvectors (which are in columns)

        if eigvecsort[1, 0] != 0:
            rot = np.degrees(np.arctan(eigvecsort[0, 0] / eigvecsort[1, 0]))  # Degrees in clockwise from 0 degree (+ Depth)
        else:
            rot = 90

        lenmajor = 2 * np.sqrt(scipy.stats.chi2.ppf(0.95, 2) * eigvalsort[0])  # Length of major axis (like diameter)
        lenminor = 2 * np.sqrt(scipy.stats.chi2.ppf(0.95, 2) * eigvalsort[1])  # Length of major axis (like diameter)

        rec_main_ce[i9, 6: 9] = [lenmajor, lenminor, rot]
        rec_cov_ce[i9, 6: 9] = [cov[0, 0], cov[1, 1], cov[0, 1]]
        rec_eig_ce[i9, 8: 12] = [eigvecsort[0, 0], eigvecsort[1, 0], eigvalsort[0], eigvalsort[1]]

        # Confidence intervals
        rec_ci_ce[i9, 0: 12] = \
            [np.percentile(bs_results_km_ce[i9, :, 0], 95) - np.percentile(bs_results_km_ce[i9, :, 0], 5),
             np.percentile(bs_results_km_ce[i9, :, 1], 95) - np.percentile(bs_results_km_ce[i9, :, 1], 5),
             np.percentile(bs_results_km_ce[i9, :, 2], 95) - np.percentile(bs_results_km_ce[i9, :, 2], 5),
             np.percentile(bs_results_km_ce[i9, :, 3], 95) - np.percentile(bs_results_km_ce[i9, :, 3], 5),
             np.percentile(bs_results_km_ce[i9, :, 0], 97.5) - np.percentile(bs_results_km_ce[i9, :, 0], 2.5),
             np.percentile(bs_results_km_ce[i9, :, 1], 97.5) - np.percentile(bs_results_km_ce[i9, :, 1], 2.5),
             np.percentile(bs_results_km_ce[i9, :, 2], 97.5) - np.percentile(bs_results_km_ce[i9, :, 2], 2.5),
             np.percentile(bs_results_km_ce[i9, :, 3], 97.5) - np.percentile(bs_results_km_ce[i9, :, 3], 2.5),
             np.percentile(bs_results_km_ce[i9, :, 0], 99.5) - np.percentile(bs_results_km_ce[i9, :, 0], 0.5),
             np.percentile(bs_results_km_ce[i9, :, 1], 99.5) - np.percentile(bs_results_km_ce[i9, :, 1], 0.5),
             np.percentile(bs_results_km_ce[i9, :, 2], 99.5) - np.percentile(bs_results_km_ce[i9, :, 2], 0.5),
             np.percentile(bs_results_km_ce[i9, :, 3], 99.5) - np.percentile(bs_results_km_ce[i9, :, 3], 0.5)]

    # ## IV. 4. Output & save! #####
    # Standard deviation (standard deviation of lat(km), lon(km), dep(km), time(s))
    bs_std_hy = open(f'bs.joint.hypo.std', 'w')
    bs_std_ce = open(f'bs.joint.cent.std', 'w')
    for i9 in range(m1 + m2):
        print(f'{rec_std_hy[i9, 0]:10.6f} {rec_std_hy[i9, 1]:10.6f} '
              f'{rec_std_hy[i9, 2]:10.6f} {rec_std_hy[i9, 3]:10.6f}', file=bs_std_hy)
        print(f'{rec_std_ce[i9, 0]:10.6f} {rec_std_ce[i9, 1]:10.6f} '
              f'{rec_std_ce[i9, 2]:10.6f} {rec_std_ce[i9, 3]:10.6f}', file=bs_std_ce)
    bs_std_hy.close()
    bs_std_ce.close()

    # Error ellipse (major-, minor-axis, rotation from top (N/ +dep) of error ellipse of the 3 pairs)
    bs_errorellipse_hy = open(f'bs.joint.hypo.errorellipse', 'w')
    bs_errorellipse_ce = open(f'bs.joint.cent.errorellipse', 'w')
    for i9 in range(m1 + m2):
        print(f'{rec_main_hy[i9, 0]:12.7f} {rec_main_hy[i9, 1]:12.7f} {rec_main_hy[i9, 2]:9.4f} '
              f'{rec_main_hy[i9, 3]:12.7f} {rec_main_hy[i9, 4]:12.7f} {rec_main_hy[i9, 5]:9.4f} '
              f'{rec_main_hy[i9, 6]:12.7f} {rec_main_hy[i9, 7]:12.7f} {rec_main_hy[i9, 8]:9.4f}', file=bs_errorellipse_hy)
        print(f'{rec_main_ce[i9, 0]:12.7f} {rec_main_ce[i9, 1]:12.7f} {rec_main_ce[i9, 2]:9.4f} '
              f'{rec_main_ce[i9, 3]:12.7f} {rec_main_ce[i9, 4]:12.7f} {rec_main_ce[i9, 5]:9.4f} '
              f'{rec_main_ce[i9, 6]:12.7f} {rec_main_ce[i9, 7]:12.7f} {rec_main_ce[i9, 8]:9.4f}', file=bs_errorellipse_ce)
    bs_errorellipse_hy.close()
    bs_errorellipse_ce.close()

    # Variance/ covariance (variance of [1], [2],  covariance matrix [1, 2] of the 3 pairs)
    bs_cov_hy = open(f'Logs/Inv_BS_HyCe/bs.joint.hypo.cov', 'w')
    bs_cov_ce = open(f'Logs/Inv_BS_HyCe/bs.joint.cent.cov', 'w')
    for i9 in range(m1 + m2):
        print(f'{rec_cov_hy[i9, 0]:12.7f} {rec_cov_hy[i9, 1]:12.7f} {rec_cov_hy[i9, 2]:12.7f} '
              f'{rec_cov_hy[i9, 3]:12.7f} {rec_cov_hy[i9, 4]:12.7f} {rec_cov_hy[i9, 5]:12.7f} '
              f'{rec_cov_hy[i9, 6]:12.7f} {rec_cov_hy[i9, 7]:12.7f} {rec_cov_hy[i9, 8]:12.7f}', file=bs_cov_hy)
        print(f'{rec_cov_ce[i9, 0]:12.7f} {rec_cov_ce[i9, 1]:12.7f} {rec_cov_ce[i9, 2]:12.7f} '
              f'{rec_cov_ce[i9, 3]:12.7f} {rec_cov_ce[i9, 4]:12.7f} {rec_cov_ce[i9, 5]:12.7f} '
              f'{rec_cov_ce[i9, 6]:12.7f} {rec_cov_ce[i9, 7]:12.7f} {rec_cov_ce[i9, 8]:12.7f}', file=bs_cov_ce)
    bs_cov_hy.close()
    bs_cov_ce.close()

    # Eigenvector/ eigenvalue (larger eigenvector, eigenvalues of both)
    bs_eig_hy = open(f'Logs/Inv_BS_HyCe/bs.joint.hypo.eig', 'w')
    bs_eig_ce = open(f'Logs/Inv_BS_HyCe/bs.joint.cent.eig', 'w')
    for i9 in range(m1 + m2):
        print(f'{rec_eig_hy[i9, 0]:13.8f} {rec_eig_hy[i9, 1]:13.8f} {rec_eig_hy[i9, 2]:11.8f} '
              f'{rec_eig_hy[i9, 3]:11.8f} {rec_eig_hy[i9, 4]:13.8f} {rec_eig_hy[i9, 5]:13.8f} '
              f'{rec_eig_hy[i9, 6]:11.8f} {rec_eig_hy[i9, 7]:11.8f} {rec_eig_hy[i9, 8]:13.8f} '
              f'{rec_eig_hy[i9, 9]:13.8f} {rec_eig_hy[i9, 10]:11.8f} {rec_eig_hy[i9, 11]:11.8f}', file=bs_eig_hy)
        print(f'{rec_eig_ce[i9, 0]:13.8f} {rec_eig_ce[i9, 1]:13.8f} {rec_eig_ce[i9, 2]:11.8f} '
              f'{rec_eig_ce[i9, 3]:11.8f} {rec_eig_ce[i9, 4]:13.8f} {rec_eig_ce[i9, 5]:13.8f} '
              f'{rec_eig_ce[i9, 6]:11.8f} {rec_eig_ce[i9, 7]:11.8f} {rec_eig_ce[i9, 8]:13.8f} '
              f'{rec_eig_ce[i9, 9]:13.8f} {rec_eig_ce[i9, 10]:11.8f} {rec_eig_ce[i9, 11]:11.8f}', file=bs_eig_ce)
    bs_eig_hy.close()
    bs_eig_ce.close()

    # Confidence intervals
    bs_ci_hy = open(f'bs.joint.hypo.ci', 'w')
    bs_ci_ce = open(f'bs.joint.cent.ci', 'w')
    for i9 in range(m1 + m2):
        print(f'{rec_ci_hy[i9, 0]:10.6f} {rec_ci_hy[i9, 1]:10.6f} {rec_ci_hy[i9, 2]:10.6f} '
              f'{rec_ci_hy[i9, 3]:10.6f} {rec_ci_hy[i9, 4]:10.6f} {rec_ci_hy[i9, 5]:10.6f} '
              f'{rec_ci_hy[i9, 6]:10.6f} {rec_ci_hy[i9, 7]:10.6f} {rec_ci_hy[i9, 8]:10.6f} '
              f'{rec_ci_hy[i9, 9]:10.6f} {rec_ci_hy[i9, 10]:10.6f} {rec_ci_hy[i9, 11]:10.6f} ', file=bs_ci_hy)
        print(f'{rec_ci_ce[i9, 0]:10.6f} {rec_ci_ce[i9, 1]:10.6f} {rec_ci_ce[i9, 2]:10.6f} '
              f'{rec_ci_ce[i9, 3]:10.6f} {rec_ci_ce[i9, 4]:10.6f} {rec_ci_ce[i9, 5]:10.6f} '
              f'{rec_ci_ce[i9, 6]:10.6f} {rec_ci_ce[i9, 7]:10.6f} {rec_ci_ce[i9, 8]:10.6f} '
              f'{rec_ci_ce[i9, 9]:10.6f} {rec_ci_ce[i9, 10]:10.6f} {rec_ci_ce[i9, 11]:10.6f} ', file=bs_ci_ce)
    bs_ci_hy.close()
    bs_ci_ce.close()

    # The final location of every realization
    if detail_output == 1:
        # Hypocenter
        bs_all_lat_hy = open(f'Logs/Inv_BS_HyCe/all_bs.hypo.lat', 'w')
        bs_all_lon_hy = open(f'Logs/Inv_BS_HyCe/all_bs.hypo.lon', 'w')
        bs_all_dep_hy = open(f'Logs/Inv_BS_HyCe/all_bs.hypo.dep', 'w')
        bs_all_tim_hy = open(f'Logs/Inv_BS_HyCe/all_bs.hypo.tim', 'w')

        np.savetxt(bs_all_lat_hy, bs_results_hy[:, :, 0], fmt='%11.6f')
        np.savetxt(bs_all_lon_hy, bs_results_hy[:, :, 1], fmt='%11.6f')
        np.savetxt(bs_all_dep_hy, bs_results_hy[:, :, 2], fmt='%9.4f')
        np.savetxt(bs_all_tim_hy, bs_results_hy[:, :, 3], fmt='%7.4f')

        bs_all_lat_hy.close()
        bs_all_lon_hy.close()
        bs_all_dep_hy.close()
        bs_all_tim_hy.close()

        # Centroid
        bs_all_lat_ce = open(f'Logs/Inv_BS_HyCe/all_bs.cent.lat', 'w')
        bs_all_lon_ce = open(f'Logs/Inv_BS_HyCe/all_bs.cent.lon', 'w')
        bs_all_dep_ce = open(f'Logs/Inv_BS_HyCe/all_bs.cent.dep', 'w')
        bs_all_tim_ce = open(f'Logs/Inv_BS_HyCe/all_bs.cent.tim', 'w')

        np.savetxt(bs_all_lat_ce, bs_results_ce[:, :, 0], fmt='%11.6f')
        np.savetxt(bs_all_lon_ce, bs_results_ce[:, :, 1], fmt='%11.6f')
        np.savetxt(bs_all_dep_ce, bs_results_ce[:, :, 2], fmt='%9.4f')
        np.savetxt(bs_all_tim_ce, bs_results_ce[:, :, 3], fmt='%7.4f')

        bs_all_lat_ce.close()
        bs_all_lon_ce.close()
        bs_all_dep_ce.close()
        bs_all_tim_ce.close()

    # Time the execution
    end_time = time.time()
    total_time = end_time - start_time

    print(f'Time of execution', str(datetime.timedelta(seconds=total_time)))
    print(f'\nTime of execution', str(datetime.timedelta(seconds=total_time)), file=nccinvlog)

    # Put these at the very end!
    nccinvlog.close()

    return None
