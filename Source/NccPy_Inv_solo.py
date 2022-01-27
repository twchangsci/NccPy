# This program determines the final hypocenter and centroid locations of all events independently
#   through inversions using relative locations from grid searches with original catalog as prior constrains
# It is rewritten and modified from nccinv.f90 in the hyponcc package (by Dr. Satoshi Ide & Kazuaki Ohta)

# Major requirements:
#   nccgs.hypo.tbl & nccgs.cent.tbl: the relative location files
#   [Parameter file]
#   [Original catalog]: specified in parameter file
# Major outputs:
#   reloc.solo.{hy_ce_tag}.full: time (original catalog) + original location + relocated location
#   reloc.solo.{hy_ce_tag}.catalog: time (original catalog) + relocated location
#   reloc.solo.{hy_ce_tag}.equsage: number of used connections each event has (pair: 2)
# Minor outputs:
#   Logs/Inv_{hy_ce_tag}/nccinv.solo.{hy_ce_tag}.log: the master log of everything importent
#   Logs/Inv_{hy_ce_tag}/reloc.solo.{hy_ce_tag}.compare: original + relocated location
#   Logs/Inv_{hy_ce_tag}/probnoise.txt: the probability of noise PN matrix
#   Logs/Inv_{hy_ce_tag}/nccinv.pass: consistant pairs to be used in inversion
#   Logs/Inv_{hy_ce_tag}/abic.{complist[comp]}: intermediate products of ABIC
#   Logs/Inv_{hy_ce_tag}/invout.{complist[comp]}: Inversion products of inversion
#   Logs/Inv_{hy_ce_tag}/reloc.errout: relocated location + error

# Matrix dimensions:
#     data (dvec, len = nmax (= n + mmax)): relative loc (rvec, len = n) + original loc (ovec, len = mmax)
#     model (mvec, len = mmax): final location
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

from NccPy_Inv_aux import noiseprob_lookup, noiseprob_table, vincenty_custom, abic_core


def nccpy_inv_solo(para_file_pass):

    # Time the calculation
    start_time = time.time()

    print(f'\n>>>> This is NccPy_Inv_solo.py  (Independent inversions: hypocenter & centroid) <<<<')

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
    hy_ce_tag_mas = ['hypo', 'cent']  # DO NOT EXCHANGE (the cent is loaded for hypo case for centcc!) (rename allowed)
    hl_exception_tag_mas = [int(para_master[17].split()[0]), int(para_master[18].split()[0])]
    centcc_tag_mas = [int(para_master[17].split()[1]), int(para_master[18].split()[1])]
    [centccthres, magsep] = [float(ii) for ii in para_master[17].split()[2:4]]
    minwv_mas = [int(para_master[17].split()[4]), int(para_master[18].split()[4])]
    ddifmax_mas = [float(para_master[17].split()[5]), float(para_master[18].split()[5])]
    [maxdist_hor, maxdist_ver] = [float(ii) for ii in para_master[17].split()[6:8]]

    # On weighting
    [pnc_in, pnch_in, pncl_in, smax, ds] = [float(ii) for ii in para_master[19].split()[0:5]]

    # Load the catalog, and round depth sing spacing in travel-time table (sampling interval issues)
    vtbpara = pandas.read_csv(f'{vtbroot}.para.txt', delim_whitespace=True, header=None).values  # Parameters
    vtbdepstep = vtbpara[0][5]  # Depth: interval (km)

    master_cat = np.genfromtxt(oricatfn)
    master_cat[:, 8] = np.round(master_cat[:, 8], decimals=int(np.log10(round(1 / vtbdepstep, 4))))

    for i_mas in range(2):

        # On usage of program: #####
        hy_ce_tag = hy_ce_tag_mas[i_mas]
        centcc_tag = centcc_tag_mas[i_mas]
        hl_exception_tag = hl_exception_tag_mas[i_mas]
        minwv = minwv_mas[i_mas]
        ddifmax = ddifmax_mas[i_mas]

        # Relative locations between event pairs ##############
        nccgstbl = pandas.read_csv(f'nccgs.{hy_ce_tag}.tbl', delim_whitespace=True, header=None).values

        if i_mas == 0 and centcc_tag == 1:
            nccgstbl_ce = pandas.read_csv(f'nccgs.{hy_ce_tag_mas[1]}.tbl', delim_whitespace=True, header=None).values

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
        if os.path.exists(f'Logs/Inv_{hy_ce_tag}'):  # Remove result directory if existing
            shutil.rmtree(f'Logs/Inv_{hy_ce_tag}')
        os.mkdir(f'Logs/Inv_{hy_ce_tag}')

        nccinvlog = open(f'Logs/Inv_{hy_ce_tag}/nccinv.solo.{hy_ce_tag}.log', 'w')    # log file

        # Probability table
        smax = int(round(smax))
        ptable = noiseprob_table(ds, smax, ncar1)  # Threshold for whether to include data!

        ptablelog = open(f'Logs/Inv_{hy_ce_tag}/probnoise.txt', 'w')
        for i0 in range(int(round(smax / ds)) + 1):
            print(f'{ptable[i0, 0]:.3f} {ptable[i0, 1]:10.8f}', file=ptablelog)

        ptablelog.close()

        # Assign the probability and NCC/std thresholds
        (nccc, pnc) = noiseprob_lookup(ds, smax, ptable, pnc_in)      # The (discretized) pass thres of NCC/std and PN
        (ncccl, pncl) = noiseprob_lookup(ds, smax, ptable, pncl_in)   # The (discretized) exception thres (the low NCC)
        (nccch, pnch) = noiseprob_lookup(ds, smax, ptable, pnch_in)   # The (discretized) exception thres (the high NCC)

        # Output them
        print(f'Min number of stations: {minwv:3.0f}')
        print(f'Max event separation (absolute): horizontal: {maxdist_hor:4.1f}  depth: {maxdist_ver:4.1f}\n')
        print(f'NCC and PN thresholds:')
        print(f'PN cutoff: pass: {pnc:8.6f} ,  exception: {pncl:8.6f}  {pnch:8.6f}')
        print(f'NCC cutoff: pass: {nccc:6.3f} ,  exception: {ncccl:6.3f}  {nccch:6.3f}\n')

        print(f'Min number of stations: {minwv:3.0f}', file=nccinvlog)
        print(f'Max event separation (absolute): horizontal: {maxdist_hor:4.1f}  depth: {maxdist_ver:4.1f}\n',
              file=nccinvlog)
        print(f'NCC and PN thresholds:', file=nccinvlog)
        print(f'PN cutoff: pass: {pnc:8.6f} ,  exception: {pncl:8.6f}  {pnch:8.6f}', file=nccinvlog)
        print(f'NCC cutoff: pass: {nccc:6.3f} ,  exception: {ncccl:6.3f}  {nccch:6.3f}\n', file=nccinvlog)

        if i_mas == 0 and centcc_tag == 1:
            print(f'Remove M >= {magsep} NCC(hypo) data when NCC(cent) > {centccthres}\n')
            print(f'Remove M >= {magsep} NCC(hypo) data when NCC(cent) > {centccthres}\n', file=nccinvlog)

        # ## II. Begin the major loop of calcualtion #####
        # ## II. 1. The original location: ovec #####
        ovec = np.zeros((len(master_cat), 5))  # Lat, lon, dep, time (0), mag
        for i1 in range(len(master_cat)):
            ovec[i1, 0] = master_cat[i1, 6]    # Lat
            ovec[i1, 1] = master_cat[i1, 7]    # Lon
            ovec[i1, 2] = master_cat[i1, 8]    # Dep
            ovec[i1, 4] = master_cat[i1, 9]    # Mag

        # Log it down!
        print(f'\nThe original catalog location (ovec) + magnitude:', file=nccinvlog)
        for i1 in range(len(ovec)):
            print(f'{ovec[i1, 0]:10.4f}  {ovec[i1, 1]:10.4f}  {ovec[i1, 2]:6.2f}  '
                  f'{ovec[i1, 3]:3.1f}  {ovec[i1, 4]:4.1f}', file=nccinvlog)

        mmax = len(master_cat)   # The model size (put it here to mimic fortran)

        # ## II. 2. The relative locations from NccPy/ NccInvPy: rvec #####
        # Initialize some arrays:
        rvec = np.array(())         # lat, lon, dep, dt: The relative location pairs pass screening
        rvec_supp = np.array(())    # ev0, ev1, sig, iter: supplementary for the above pairs

        for i2 in range(0, len(nccgstbl), 2):

            # Some error messages
            # Quit program if some relative locations are not "in pairs"
            if int(nccgstbl[i2, 0]) != int(nccgstbl[i2 + 1, 1]) or \
                    int(nccgstbl[i2, 1]) != int(nccgstbl[i2 + 1, 0]) or \
                    abs(nccgstbl[i2, 7] - nccgstbl[i2 + 1, 7]) > trivial:
                print(f'Error in nccgs.tbl: pair unbalenced: '
                      f'{nccgstbl[i2, 0]:.0f} {nccgstbl[i2 + 1, 1]:.0f} {nccgstbl[i2, 1]:.0f} {nccgstbl[i2 + 1, 0]:.0f} '
                      f'{nccgstbl[i2, 7]:.4f} {nccgstbl[i2 + 1, 7]:.4f}')
                print(f'Error in nccgs.tbl: pair unbalenced: '
                      f'{nccgstbl[i2, 0]:.0f} {nccgstbl[i2 + 1, 1]:.0f} {nccgstbl[i2, 1]:.0f} {nccgstbl[i2 + 1, 0]:.0f} '
                      f'{nccgstbl[i2, 7]:.4f} {nccgstbl[i2 + 1, 7]:.4f}', file=nccinvlog)
                sys.exit(0)

            # Loop over the centroid tbl, and fetch its CC
            cctmpp = [0, 0]     # If not found in centroid, proceed

            # Look for the same pair of event
            for iii2 in range(0, len(nccgstbl_ce), 2):
                if nccgstbl[i2, 0] == nccgstbl_ce[iii2, 0] and nccgstbl[i2, 1] == nccgstbl_ce[iii2, 1]:
                    cctmpp = [nccgstbl_ce[iii2, 9], nccgstbl_ce[iii2 + 1, 9]]

            # Condition of removing large events with high centroid NCC
            if centcc_tag == 1 and cctmpp[0] > centccthres and cctmpp[1] > centccthres \
                    and ovec[int(nccgstbl[i2, 0]), 4] + trivial >= magsep \
                    and ovec[int(nccgstbl[i2, 1]), 4] + trivial >= magsep:

                print(f'Pair unused (hypo): {nccgstbl[i2, 0]:.0f} {nccgstbl[i2, 1]:.0f}: large repeaters!  '
                      f'Mag: {ovec[int(nccgstbl[i2, 0]), 4]:4.1f} {ovec[int(nccgstbl[i2, 1]), 4]:4.1f}  '
                      f'CC(cent): {cctmpp[0]:7.4f}  {cctmpp[1]:7.4f}')
                print(f'Pair unused (hypo): {nccgstbl[i2, 0]:.0f} {nccgstbl[i2, 1]:.0f}: large repeaters!  '
                      f'Mag: {ovec[int(nccgstbl[i2, 0]), 4]:4.1f} {ovec[int(nccgstbl[i2, 1]), 4]:4.1f}  '
                      f'CC(cent): {cctmpp[0]:7.4f}  {cctmpp[1]:7.4f}', file=nccinvlog)

            else:
                # Number of waveform limits
                if int(nccgstbl[i2, 10]) >= minwv:

                    # Separation of events in catalog
                    evtdist_hor = vincenty_custom((ovec[int(nccgstbl[i2, 0]), 0], ovec[int(nccgstbl[i2, 0]), 1]),
                                                  (ovec[int(nccgstbl[i2, 1]), 0], ovec[int(nccgstbl[i2, 1]), 1]))
                    evtdist_ver = np.abs(ovec[int(nccgstbl[i2, 0]), 2] - ovec[int(nccgstbl[i2, 1]), 2])
                    if evtdist_hor <= maxdist_hor and evtdist_ver <= maxdist_ver:

                        # Consistency check (within each pair of relative location!)
                        if nccgstbl[i2, 7] - trivial <= ddifmax:

                            # NCC significance check: ev0
                            if nccgstbl[i2, 6] + trivial >= nccc:

                                if len(rvec) == 0:
                                    rvec = np.array(
                                        [[nccgstbl[i2, 2], nccgstbl[i2, 3], nccgstbl[i2, 4], nccgstbl[i2, 5]]])
                                    rvec_supp = np.array(
                                        [[nccgstbl[i2, 0], nccgstbl[i2, 1], nccgstbl[i2, 6], nccgstbl[i2, 8]]])
                                else:
                                    rvec = np.r_[rvec, [[nccgstbl[i2, 2], nccgstbl[i2, 3],
                                                         nccgstbl[i2, 4], nccgstbl[i2, 5]]]]
                                    rvec_supp = np.r_[rvec_supp, [[nccgstbl[i2, 0], nccgstbl[i2, 1],
                                                                   nccgstbl[i2, 6], nccgstbl[i2, 8]]]]

                            # NCC significance check: ev1
                            if nccgstbl[i2 + 1, 6] + trivial >= nccc:

                                if len(rvec) == 0:
                                    rvec = np.array([[nccgstbl[i2 + 1, 2], nccgstbl[i2 + 1, 3],
                                                      nccgstbl[i2 + 1, 4], nccgstbl[i2 + 1, 5]]])
                                    rvec_supp = np.array([[nccgstbl[i2 + 1, 0], nccgstbl[i2 + 1, 1],
                                                           nccgstbl[i2 + 1, 6], nccgstbl[i2 + 1, 8]]])
                                else:
                                    rvec = np.r_[rvec, [[nccgstbl[i2 + 1, 2], nccgstbl[i2 + 1, 3],
                                                         nccgstbl[i2 + 1, 4], nccgstbl[i2 + 1, 5]]]]
                                    rvec_supp = np.r_[rvec_supp, [[nccgstbl[i2 + 1, 0], nccgstbl[i2 + 1, 1],
                                                                   nccgstbl[i2 + 1, 6], nccgstbl[i2 + 1, 8]]]]

                        # Exception: allow the exceptionally good one (n2) if the other (n2 + 1) is exceptionally bad
                        elif hl_exception_tag == 1 and \
                                nccgstbl[i2, 6] + trivial >= nccch and nccgstbl[i2 + 1, 6] - trivial < ncccl:

                            if len(rvec) == 0:
                                rvec = np.array(
                                    [[nccgstbl[i2, 2], nccgstbl[i2, 3], nccgstbl[i2, 4], nccgstbl[i2, 5]]])
                                rvec_supp = np.array(
                                    [[nccgstbl[i2, 0], nccgstbl[i2, 1], nccgstbl[i2, 6], nccgstbl[i2, 8]]])
                            else:
                                rvec = np.r_[rvec, [[nccgstbl[i2, 2], nccgstbl[i2, 3],
                                                     nccgstbl[i2, 4], nccgstbl[i2, 5]]]]
                                rvec_supp = np.r_[rvec_supp, [[nccgstbl[i2, 0], nccgstbl[i2, 1],
                                                               nccgstbl[i2, 6], nccgstbl[i2, 8]]]]

                        # Exception: allow the exceptionally good one (n2 + 1) if the other (n2) is exceptionally bad
                        elif hl_exception_tag == 1 and \
                                nccgstbl[i2 + 1, 6] + trivial >= nccch and nccgstbl[i2, 6] - trivial < ncccl:

                            if len(rvec) == 0:
                                rvec = np.array([[nccgstbl[i2 + 1, 2], nccgstbl[i2 + 1, 3],
                                                  nccgstbl[i2 + 1, 4], nccgstbl[i2 + 1, 5]]])
                                rvec_supp = np.array([[nccgstbl[i2 + 1, 0], nccgstbl[i2 + 1, 1],
                                                       nccgstbl[i2 + 1, 6], nccgstbl[i2 + 1, 8]]])
                            else:
                                rvec = np.r_[rvec, [[nccgstbl[i2 + 1, 2], nccgstbl[i2 + 1, 3],
                                                     nccgstbl[i2 + 1, 4], nccgstbl[i2 + 1, 5]]]]
                                rvec_supp = np.r_[rvec_supp, [[nccgstbl[i2 + 1, 0], nccgstbl[i2 + 1, 1],
                                                               nccgstbl[i2 + 1, 6], nccgstbl[i2 + 1, 8]]]]

                    else:
                        print(f'Pair unused: {nccgstbl[i2, 0]:.0f} {nccgstbl[i2 + 1, 1]:.0f}: '
                              f'Catalog separation (hor/ ver) ({evtdist_hor:.3f} / {evtdist_ver:.3f}) '
                              f'exceeding limit ({maxdist_hor:.3f} / {maxdist_ver:.3f}', file=nccinvlog)
                else:
                    print(f'Pair unused: {nccgstbl[i2, 0]:.0f} {nccgstbl[i2 + 1, 1]:.0f}: '
                          f'Number of waveform ({nccgstbl[i2, 10]:.0f}) < {minwv:.0f}', file=nccinvlog)

        # Save these to nccinv.pass, as fortran's nccinv.data (but only for viewing, the rvec is already given above)
        nccinvpass = open(f'Logs/Inv_{hy_ce_tag}/nccinv.pass', 'w')    # log file
        for i2 in range(len(rvec)):
            print(f'{rvec_supp[i2, 0]:4.0f}  {rvec_supp[i2, 1]:4.0f}  {rvec[i2, 0]:9.4f}  {rvec[i2, 1]:9.4f}  '
                  f'{rvec[i2, 2]:9.4f}  {rvec[i2, 3]:7.3f}  {rvec_supp[i2, 2]:8.4f}  {rvec_supp[i2, 3]:2.0f}',
                  file=nccinvpass)

        nccinvpass.close()

        # Stop program if no effective pair is available
        if len(rvec) == 0:
            print(f'Error in nccgs.tbl: no relative location available after screening!')
            print(f'Error in nccgs.tbl: no relative location available after screening!', file=nccinvlog)
            sys.exit(0)

        n = len(rvec)             # The data size (put it here to mimic fortran)
        nmax = mmax + len(rvec)   # The data size, including the constrian
        print(f'Model: {mmax:.0f}  Data: {nmax:.0f}\n')
        print(f'\nModel: {mmax:.0f}  Data: {nmax:.0f}', file=nccinvlog)

        # ## II. 3. Declare MASSIVE amount of parameters after sizes are determined ################
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

        pN = np.zeros(n)       # probability of noise vector
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

        # ## III. 4. Fill the kernal matrix (G) ################
        # The parts of data (relative locations)
        for i3 in range(n):
            G[i3, int(rvec_supp[i3, 0])] = -1
            G[i3, int(rvec_supp[i3, 1])] = 1

            iv[i3] = int(rvec_supp[i3, 0])
            jv[i3] = int(rvec_supp[i3, 1])

        # The parts of constrains (catalog locations)
        for i3 in range(n, nmax):
            for j3 in range(mmax):
                if i3 - n == j3:
                    G[i3, j3] = 1

            iv[i3] = i3 - n
            jv[i3] = i3 - n

        Gt = np.transpose(G)

        # ## II. 5. Fill the data matrix (dvec) ################
        # The parts of data (relative locations)
        for i4 in range(n):
            dvec[i4, 0] = rvec[i4, 0]
            dvec[i4, 1] = rvec[i4, 1]
            dvec[i4, 2] = rvec[i4, 2]
            dvec[i4, 3] = rvec[i4, 3]

        # The parts of constrains (catalog locations), change unit to km if necessary
        for i4 in range(n, nmax):
            dvec[i4, 0] = ovec[i4 - n, 0] * earth_rad * np.pi / 180
            dvec[i4, 1] = ovec[i4 - n, 1] * earth_rad * np.pi / 180 * np.cos(np.mean(ovec[:, 0]) / 180 * np.pi)
            dvec[i4, 2] = ovec[i4 - n, 2]
            dvec[i4, 3] = ovec[i4 - n, 3]

        # Log the data vector down:
        print(f'\nThe data vectors (lat, lon, dep, dt):', file=nccinvlog)
        for i4 in range(nmax):
            print(f'{dvec[i4, 0]:12.4f} {dvec[i4, 1]:12.4f} {dvec[i4, 2]:10.4f} {dvec[i4, 3]:8.2f}', file=nccinvlog)

        # ## II. 6. The weighting for data (upper half of wf) ################
        # The variance of signal/ noise
        sig2_N = ((2 * ala1) ** 2 + (2 * alo1) ** 2 + (2 * adp1) ** 2) / 12

        # sig2_S[0] = (dla1 ** 2 + dlo1 ** 2 + ddp1 ** 2) / 12
        # sig2_S[1] = (dla2 ** 2 + dlo2 ** 2 + ddp2 ** 2) / 12

        # Adhere to the fortran version
        sig2_S[0] = (dla1 ** 2) / 12
        sig2_S[1] = (dla2 ** 2) / 12

        print(f'\nMessages at pN assignment:', file=nccinvlog)

        for i5 in range(n):
            stdtmp = int(round(rvec_supp[i5, 2] / ds))

            # Look for the clostest in table
            if 0 <= stdtmp <= int(round(smax / ds)):
                pN[i5] = ptable[stdtmp, 1]
            elif stdtmp < 0:
                pN[i5] = 1
                print(f'Error: NCC / std < 0: {i5} {stdtmp}', file=nccinvlog)
            elif stdtmp > int(round(smax / ds)):
                pN[i5] = 0
                print(f'Note: NCC / std > max at table (set pN = 0): {i5} {stdtmp}', file=nccinvlog)

            wf[i5] = 1 / (pN[i5] * sig2_N + (1 - pN[i5]) * sig2_S[int(round(rvec_supp[i5, 3]))])

        # Log the weighting vector down:
        print(f'\nThe weighting vectors (NCC / sig, pN, wf, iiter):', file=nccinvlog)
        for i5 in range(n):
            print(f'{rvec_supp[i5, 2]:8.4f} {pN[i5]:12.10f} {wf[i5]:15.6f} {rvec_supp[i5, 3]:2.0f}', file=nccinvlog)

        # Time the execution
        end_time = time.time()
        total_time = end_time - start_time
        print(f'All set before ABIC calculation', str(datetime.timedelta(seconds=total_time)))

        # ## III. Bayesian Inversion using ABIC #####
        # ## III. 1. The weighting for constrains (lower half of wf) by ABIC #####
        for comp in range(4):     # For lat, lon, dep, dt
            ABICmin = 1 / trivial
            kmin = 0
            ABICrange = 100          # The window to search for min(ABIC)

            (kmin, abicmat) = abic_core(n, nmax, mmax, comp, ABICrange, ABICmin, kmin, wf, G, Gt, GtW, dvec, Wd, We)

            # Output the matrix
            abiclog = open(f'Logs/Inv_{hy_ce_tag}/abic.{complist[comp]}', 'w')   # ABIC log
            for k in range(-ABICrange, ABICrange + 1):
                print(f'{abicmat[k, 0]:3.0f}  {abicmat[k, 1]:12.6f}  {abicmat[k, 2]:.8f}', file=abiclog)
            abiclog.close()

            # Now assign the weighting when min ABIC is found!
            alpha2 = 10 ** (kmin * 0.05)
            for i6 in range(n, nmax):
                wf[i6] = alpha2

        # ## III. 2. The actual inversion #####
            # Again, least squares inversion (this may look a lot like the above, naturally)
            for i7 in range(mmax):
                for j7 in range(nmax):
                    GtW[i7, j7] = Gt[i7, j7] * wf[j7]

            GtWG = np.matmul(GtW, G)
            iGtWG = np.linalg.inv(GtWG)
            Gg = np.matmul(iGtWG, GtW)

            for i7 in range(mmax):
                for j7 in range(nmax):
                    Ggcovd[i7, j7] = Gg[i7, j7] / wf[j7]

            Ggt = np.transpose(Gg)
            covm = np.matmul(Ggcovd, Ggt)

            for i7 in range(mmax):
                merr[i7, comp] = np.sqrt(covm[i7, i7])

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

            # Log down inversion outputs
            invout = open(f'Logs/Inv_{hy_ce_tag}/invout.{complist[comp]}', 'w')  # inversion output log
            print(f'ev0, ev1, d, Gm, eWe, wf', file=invout)
            for i7 in range(nmax):
                print(f'{iv[i7]:4.0f} {jv[i7]:4.0f} {d[i7]:15.6f} {Gm[i7]:15.6f} '
                      f'{We[i7] * e[i7]:15.6f} {wf[i7]:15.6f}', file=invout)
            invout.close()

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
        mvec[:, 1] /= (earth_rad * np.pi / 180 * np.cos(np.mean(mvec[:, 0]) / 180 * np.pi))

        merr[:, 0] /= (earth_rad * np.pi / 180)
        merr[:, 1] /= (earth_rad * np.pi / 180 * np.cos(np.mean(mvec[:, 0]) / 180 * np.pi))

        # The mass center check:
        print(f'\nMass center (original):', file=nccinvlog)
        print(f'{np.mean(ovec[:, 0]):12.6f} {np.mean(ovec[:, 1]):12.6f} '
              f'{np.mean(ovec[:, 2]):12.6f} {np.mean(ovec[:, 3]):12.6f}', file=nccinvlog)
        print(f'Mass center (relocated):', file=nccinvlog)
        print(f'{np.mean(mvec[:, 0]):12.6f} {np.mean(mvec[:, 1]):12.6f} '
              f'{np.mean(mvec[:, 2]):12.6f} {np.mean(mvec[:, 3]):12.6f}', file=nccinvlog)

        # reloc.errout: results + error
        relocerrorout = open(f'Logs/Inv_{hy_ce_tag}/reloc.errout', 'w')
        for i8 in range(mmax):
            print(f'{mvec[i8, 0]:12.6f} {mvec[i8, 1]:12.6f} {mvec[i8, 2]:12.6f} {mvec[i8, 3]:12.6f} '
                  f'{merr[i8, 0]:12.6f} {merr[i8, 1]:12.6f} {merr[i8, 2]:12.6f} {merr[i8, 3]:12.6f}',
                  file=relocerrorout)
        relocerrorout.close()

        # reloc.compare: catalog + relocated location;  full: include time also!
        reloccompare = open(f'Logs/Inv_{hy_ce_tag}/reloc.solo.{hy_ce_tag}.compare', 'w')
        relocfull = open(f'reloc.solo.{hy_ce_tag}.full', 'w')
        for i8 in range(mmax):
            print(f'{ovec[i8, 0]:10.5f} {ovec[i8, 1]:10.5f} {ovec[i8, 2]:7.3f} '
                  f'{mvec[i8, 0]:10.5f} {mvec[i8, 1]:10.5f} {mvec[i8, 2]:7.3f} {mvec[i8, 3]:7.3f}', file=reloccompare)

            print(f'{master_cat[i8, 0]:4.0f} {master_cat[i8, 1]:2.0f} {master_cat[i8, 2]:2.0f} '
                  f'{master_cat[i8, 3]:2.0f} {master_cat[i8, 4]:2.0f} {master_cat[i8, 5]:5.2f} '
                  f'{ovec[i8, 0]:10.5f} {ovec[i8, 1]:10.5f} {ovec[i8, 2]:7.3f}  '
                  f'{mvec[i8, 0]:10.5f} {mvec[i8, 1]:10.5f} {mvec[i8, 2]:7.3f}  {master_cat[i8, 9]:4.1f}', file=relocfull)
        reloccompare.close()
        relocfull.close()

        # reloc.catalog: original time & relocated location: full catalog
        reloccat = open(f'reloc.solo.{hy_ce_tag}.catalog', 'w')
        for i8 in range(mmax):
            print(f'{master_cat[i8, 0]:4.0f} {master_cat[i8, 1]:2.0f} {master_cat[i8, 2]:2.0f} '
                  f'{master_cat[i8, 3]:2.0f} {master_cat[i8, 4]:2.0f} {master_cat[i8, 5]:5.2f} '
                  f'{mvec[i8, 0]:10.5f} {mvec[i8, 1]:10.5f} {mvec[i8, 2]:7.3f} {master_cat[i8, 9]:4.1f}', file=reloccat)
        reloccat.close()

        # reloc.equsage: number of consistant connections each event has
        equsage = np.zeros(mmax)
        for i8 in range(n):
            equsage[int(rvec_supp[i8, 0])] += 1
            equsage[int(rvec_supp[i8, 1])] += 1

        relocequsage = open(f'reloc.solo.{hy_ce_tag}.equsage', 'w')
        for i8 in range(mmax):
            print(f'{i8:4.0f} {equsage[i8]:4.0f}', file=relocequsage)
        relocequsage.close()

        # Put these at the very end
        nccinvlog.close()

        # Time the execution
        end_time = time.time()
        total_time = end_time - start_time

        print(f'Time of execution', str(datetime.timedelta(seconds=total_time)))

    return None
