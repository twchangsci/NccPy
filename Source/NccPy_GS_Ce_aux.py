# Auxillary subroutines called by NccPy_GS_Ce.py

# In this subroutine file:
# 1. noiseprob, noiseprobcore: noise probability calculation!
# 2. cc_cent_custom: cross-correlation (12X faster after nopython=True)
# 3. vincenty_custom: distance calculation
#    To be able to use it with numba, we need to calculate our own (geodesic) distance
#    There are three packages available:
#   -geopy.distance.geodesic: best choice, accurate till 1 degree from antipole (still outputs after; but not accurate)
#   -pymap3d.vincenty.vdist: same author as vdist in Matlab
#       accurate ~1 degree from antipole (still outputs after; not accurate) (We don't relocate antipolar points anyway)
#       also gives vdist(0.0, 0.0, 0, 1) = (nan, nan, nan)! This strange bug at exactly 0.5, 1 degree, etc! (Harmful)
#   -vincenty.vincenty: equal (~mm scale) to the above 2 till 1 degree from antinodal, no output after that
#       It doesn't have the bug at particular angles as vdist!
#       It is also simple (<=100 lines, only use math module), making it the best choice to adopt using numba!
# 4. ncc_gridsearch: grid search!  (spin off and call gridsearch_maxsearch to save time)

# V0.6.0 (2022.01)
# Author: Ta-Wei Chang  (twchangsci@gmail.com)


import numpy as np
import math
from numba import jit, prange


# ## I. PROBABILITY OF NOISE CALCULATION #######################
# @jit(nopython=True)     # Numba doesn't support formatted write!
def noiseprob(ds, smax, ncar1, pthre):
    # Takes variable pthre and ncar1 from outside
    if pthre < 1:
        p = np.zeros([smax])
        aaac = open('probnoise.txt', 'w')
        for i in range(smax):
            s = i * ds
            p[i] = noiseprobcore(s, ncar1)
            print(f'{i} {s:.3f} {p[i]:10.6f}', file=aaac)

        nccthrein = 0
        for i in range(smax - 1):
            if p[i] >= pthre > p[i + 1]:
                nccthrein = (i * ds + (i + 1) * ds) / 2

                print('', file=aaac)
                print(f'{i} {pthre:.3f} {p[i]:10.6f} {nccthrein:10.6f}', file=aaac)
                aaac.close()

        if nccthrein == 0:
            print(f'Error!! Increase "smax"!')

    else:
        nccthrein = 0

    return nccthrein


# Run using numba to save time
@jit(nopython=True)
def noiseprobcore(xxo, n):
    # Written using math instead of numpy for speed!
    m = 100000
    c = 1 / math.sqrt(2 * np.pi)
    dx = 100 / m
    sum0 = 0

    for i in range(m):
        x0 = xxo + (i - 1) * dx
        x1 = xxo + i * dx
        f0 = math.exp(-0.5 * x0 * x0)
        f1 = math.exp(-0.5 * x1 * x1)
        sum0 = sum0 + 0.5 * (f0 + f1) * dx

    summ = c * sum0
    p1 = 1 - summ
    pn = 1 - math.pow(p1, n)

    return pn


# ## II. Cross correlation #######################
@jit(nopython=True)
def cc_cent_custom(waveone, wavetwo, nptccln, nptwave):
    # Calculates cross-correlation between 2 waveforms

    cc_calc = np.zeros((int(nptccln - nptwave + 1), 1))

    for idt in range(int(nptccln - nptwave + 1)):
        cctmp1 = (waveone - np.mean(waveone))
        cctmp2 = (wavetwo[idt:idt + nptwave] - np.mean(wavetwo[idt:idt + nptwave]))

        amp1 = np.sqrt(np.dot(cctmp1, cctmp1))
        amp2 = np.sqrt(np.dot(cctmp2, cctmp2))

        if amp1 != 0 and amp2 != 0:
            cc_calc[idt] = np.dot(cctmp1, cctmp2) / amp1 / amp2
            # cc2_save_db[idt][iwv] = np.dot(cctmp1, cctmp2) / (np.max(amp1, amp2) ** 2)
        else:
            cc_calc[idt] = 0

    return cc_calc


# ## III. Geodesic distance #######################
# Calculation of geodesic distance: modified from the vincenty package (small case, remove some features)
# (~175 times faster than geopy.distance after numba usage) (verfied against geopy.distance ~179 degrees apart)
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


# ## IV. Grid Search #######################
# To parallel, use parallel=True, and use range --> prange somewhere
# Need to remove assigning any parameters in separate loops
# In smaller #grid setting and fewer cores used, the un-paralleled is likely faster instead
@jit(nopython=True, parallel=True)
def ncc_gridsearch(passvec1, passvec2, passvec3, wvvec, cc, vtbarrp, vtbarrs):
    # Recover values
    (evla, evlo, evdp, evdt, nla, nlo, ndp, ndt, dla, dlo, ddp, ddt) = passvec1[0:12]
    (iitr, adp, ncar, sample_mas, nptccln, nptwave, nptgrid, gdla, gdlo, ex, ex2) = passvec2[0:11]
    (vtbdistini, vtbdiststep, vtbdepini, vtbdepstep) = passvec3[0:4]
    stla = wvvec[:, 0]
    stlo = wvvec[:, 1]
    ips = wvvec[:, 2]
    tp = wvvec[:, 3]
    o_b = wvvec[:, 4]

    earth_rad = (2 * 6378137 + 6356752.314245) / 3 / 1000   # Radius of Earth, from the above & (2a+b)/3 (Wiki)

    nccrecord = np.zeros((int(2 * nla + 1), int(2 * nlo + 1), int(2 * ndp + 1), int(2 * ndt + 1)))

    ex2record = np.zeros((int(2 * nla + 1), int(2 * nlo + 1), int(2 * ndp + 1), int(2 * ndt + 1)))

    distwvmat = np.zeros((int(2 * nla + 1), int(2 * nlo + 1), int(len(tp))))
    gridstatrvmat = np.zeros((int(2 * nla + 1), int(2 * nlo + 1), int(2 * ndp + 1), int(len(tp))))

    for ila in prange(int(-nla), int(nla + 1)):

        for ilo in range(int(-nlo), int(nlo + 1)):

            for idp in range(int(-ndp), int(ndp + 1)):

                for iwv in range(len(tp)):

                    # For parallel: run dist here!
                    distwvmat[int(ila + nla), int(ilo + nlo), iwv] = \
                        vincenty_custom((stla[iwv], stlo[iwv]), (ila * gdla + evla, ilo * gdlo + evlo)) \
                        / earth_rad / np.pi * 180

                    # Station- grid travel-time!
                    if ips[iwv] == 0:

                        if evdp < adp:  # Grid goes above sea level!
                            gridstatrvmat[int(ila + nla), int(ilo + nlo), int(idp + ndp), iwv] = \
                                vtbarrp[int(np.floor((distwvmat[int(ila + nla), int(ilo + nlo), iwv] - vtbdistini) / vtbdiststep))][int(np.round((((idp + ndp) * ddp) - vtbdepini) / vtbdepstep))] * \
                                (-distwvmat[int(ila + nla), int(ilo + nlo), iwv] + (vtbdistini + (int(np.floor((distwvmat[int(ila + nla), int(ilo + nlo), iwv] - vtbdistini) / vtbdiststep)) + 1) * vtbdiststep)) / vtbdiststep + \
                                vtbarrp[int(np.floor((distwvmat[int(ila + nla), int(ilo + nlo), iwv] - vtbdistini) / vtbdiststep)) + 1][int(np.round((((idp + ndp) * ddp) - vtbdepini) / vtbdepstep))] * \
                                (distwvmat[int(ila + nla), int(ilo + nlo), iwv] - (vtbdistini + int(np.floor((distwvmat[int(ila + nla), int(ilo + nlo), iwv] - vtbdistini) / vtbdiststep)) * vtbdiststep)) / vtbdiststep
                        elif evdp >= adp:  # Grid does not go above sea level
                            gridstatrvmat[int(ila + nla), int(ilo + nlo), int(idp + ndp), iwv] = \
                                vtbarrp[int(np.floor((distwvmat[int(ila + nla), int(ilo + nlo), iwv] - vtbdistini) / vtbdiststep))][int(np.round(((idp * ddp + evdp) - vtbdepini) / vtbdepstep))] * \
                                (-distwvmat[int(ila + nla), int(ilo + nlo), iwv] + (vtbdistini + (int(np.floor((distwvmat[int(ila + nla), int(ilo + nlo), iwv] - vtbdistini) / vtbdiststep)) + 1) * vtbdiststep)) / vtbdiststep + \
                                vtbarrp[int(np.floor((distwvmat[int(ila + nla), int(ilo + nlo), iwv] - vtbdistini) / vtbdiststep)) + 1][int(np.round(((idp * ddp + evdp) - vtbdepini) / vtbdepstep))] * \
                                (distwvmat[int(ila + nla), int(ilo + nlo), iwv] - (vtbdistini + int(np.floor((distwvmat[int(ila + nla), int(ilo + nlo), iwv] - vtbdistini) / vtbdiststep)) * vtbdiststep)) / vtbdiststep

                    elif ips[iwv] == 1:

                        if evdp < adp:  # Grid goes above sea level!
                            gridstatrvmat[int(ila + nla), int(ilo + nlo), int(idp + ndp), iwv] = \
                                vtbarrs[int(np.floor((distwvmat[int(ila + nla), int(ilo + nlo), iwv] - vtbdistini) / vtbdiststep))][int(np.round((((idp + ndp) * ddp) - vtbdepini) / vtbdepstep))] * \
                                (-distwvmat[int(ila + nla), int(ilo + nlo), iwv] + (vtbdistini + (int(np.floor((distwvmat[int(ila + nla), int(ilo + nlo), iwv] - vtbdistini) / vtbdiststep)) + 1) * vtbdiststep)) / vtbdiststep + \
                                vtbarrs[int(np.floor((distwvmat[int(ila + nla), int(ilo + nlo), iwv] - vtbdistini) / vtbdiststep)) + 1][int(np.round((((idp + ndp) * ddp) - vtbdepini) / vtbdepstep))] * \
                                (distwvmat[int(ila + nla), int(ilo + nlo), iwv] - (vtbdistini + int(np.floor((distwvmat[int(ila + nla), int(ilo + nlo), iwv] - vtbdistini) / vtbdiststep)) * vtbdiststep)) / vtbdiststep
                        elif evdp >= adp:  # Grid does not go above sea level
                            gridstatrvmat[int(ila + nla), int(ilo + nlo), int(idp + ndp), iwv] = \
                                vtbarrs[int(np.floor((distwvmat[int(ila + nla), int(ilo + nlo), iwv] - vtbdistini) / vtbdiststep))][int(np.round(((idp * ddp + evdp) - vtbdepini) / vtbdepstep))] * \
                                (-distwvmat[int(ila + nla), int(ilo + nlo), iwv] + (vtbdistini + (int(np.floor((distwvmat[int(ila + nla), int(ilo + nlo), iwv] - vtbdistini) / vtbdiststep)) + 1) * vtbdiststep)) / vtbdiststep + \
                                vtbarrs[int(np.floor((distwvmat[int(ila + nla), int(ilo + nlo), iwv] - vtbdistini) / vtbdiststep)) + 1][int(np.round(((idp * ddp + evdp) - vtbdepini) / vtbdepstep))] * \
                                (distwvmat[int(ila + nla), int(ilo + nlo), iwv] - (vtbdistini + int(np.floor((distwvmat[int(ila + nla), int(ilo + nlo), iwv] - vtbdistini) / vtbdiststep)) * vtbdiststep)) / vtbdiststep

                for idt in range(int(-ndt), int(ndt + 1)):

                    for iwv in range(len(tp)):

                        # In case there isn't an arrival time available
                        if int(round(gridstatrvmat[int(ila + nla), int(ilo + nlo), int(idp + ndp), iwv])) != -1:

                            # When appropriate CC is available
                            if -nptgrid <= \
                                    int(round(o_b[iwv] +
                                              (gridstatrvmat[int(ila + nla), int(ilo + nlo), int(idp + ndp), iwv]
                                               + idt * ddt + evdt) * sample_mas)) - int(round(o_b[iwv] + tp[iwv])) \
                                    <= nptgrid:
                                nccrecord[int(ila + nla), int(ilo + nlo), int(idp + ndp), int(idt + ndt)] += \
                                    cc[int(round(
                                        int(round(o_b[iwv] +
                                                  (gridstatrvmat[int(ila + nla), int(ilo + nlo), int(idp + ndp), iwv]
                                                   + idt * ddt + evdt) * sample_mas)) - int(round(o_b[iwv] + tp[iwv]))
                                        + 0.5 * (nptccln - nptwave))), iwv]

                            # In case the duration of waveform with CC values calculated is too short
                            elif int(round(o_b[iwv] +
                                           (gridstatrvmat[int(ila + nla), int(ilo + nlo), int(idp + ndp), iwv]
                                               + idt * ddt + evdt) * sample_mas)) - int(round(o_b[iwv] + tp[iwv])) \
                                    < -nptgrid:
                                print('Warning: waveform to CC too short! (itdif < -nptgrid)', ila, ilo, idp, idt, iwv,
                                      int(round(o_b[iwv] +
                                                (gridstatrvmat[int(ila + nla), int(ilo + nlo), int(idp + ndp), iwv]
                                                 + idt * ddt + evdt) * sample_mas)) - int(round(o_b[iwv] + tp[iwv])))

                            elif int(round(o_b[iwv] +
                                           (gridstatrvmat[int(ila + nla), int(ilo + nlo), int(idp + ndp), iwv]
                                               + idt * ddt + evdt) * sample_mas)) - int(round(o_b[iwv] + tp[iwv])) \
                                    > nptgrid:
                                print('Warning: waveform to CC too short! (itdif > nptgrid)', ila, ilo, idp, idt, iwv,
                                      int(round(o_b[iwv] +
                                                (gridstatrvmat[int(ila + nla), int(ilo + nlo), int(idp + ndp), iwv]
                                                 + idt * ddt + evdt) * sample_mas)) - int(round(o_b[iwv] + tp[iwv])))

                        elif int(round(gridstatrvmat[int(ila + nla), int(ilo + nlo), int(idp + ndp), iwv])) == -1:
                            if evdp < adp:  # Grid goes above sea level!
                                print('Warning: travel-time not avalable!!', ila, ilo, idp, idt, iwv,
                                      stla[iwv], stlo[iwv], ila * gdla + evla, ilo * gdlo + evlo, (idp + ndp) * ddp,
                                      distwvmat[int(ila + nla), int(ilo + nlo), iwv])
                            elif evdp >= adp:  # Grid does not go above sea level
                                print('Warning: travel-time not avalable!!', ila, ilo, idp, idt, iwv,
                                      stla[iwv], stlo[iwv], ila * gdla + evla, ilo * gdlo + evlo, idp * ddp + evdp,
                                      distwvmat[int(ila + nla), int(ilo + nlo), iwv])

                    # Save down parameters
                    ex2record[int(ila + nla), int(ilo + nlo), int(idp + ndp), int(idt + ndt)] = \
                        nccrecord[int(ila + nla), int(ilo + nlo), int(idp + ndp), int(idt + ndt)] ** 2

    # After the grid search: look for the max ncc value, and also save to output (without idt axis)!
    # The external version:
    (mla, mlo, mdp, mdt, nccmax, nccrecord_out) = gridsearch_maxsearch(nla, nlo, ndp, ndt, nccrecord)

    # Prepare the parameters to be outputted
    # L1 and L2 norm (for standard deviation) (for the coarser one only)
    if iitr == 0:
        ex = np.sum(nccrecord)
        # ex = np.sum(exrecord)
        ex2 = np.sum(ex2record)

    # CC values and where the max is found for each waves at the max
    itdifrec = np.zeros(int(len(tp)))
    ccmaxrec = np.zeros(int(len(tp)))
    distwv = np.zeros(int(len(tp)))
    gridstatrv = np.zeros(int(len(tp)))

    for iwv in range(len(tp)):

        # (Recover here the travel time from grid to each staiton)
        stalocg = (stla[iwv], stlo[iwv])
        grdlocg = (mla * gdla + evla, mlo * gdlo + evlo)
        # grdlocg = (gla, glo)
        distg = vincenty_custom(stalocg, grdlocg)
        distwv[iwv] = distg / earth_rad / np.pi * 180

        if evdp < adp:  # Grid goes above sea level!
            # gdp = mdp * ddp
            gdp = (mdp + ndp) * ddp
        elif evdp >= adp:  # Grid does not go above sea level
            gdp = mdp * ddp + evdp

        disthere = int(np.floor((distwv[iwv] - vtbdistini) / vtbdiststep))
        x0 = (distwv[iwv] - (vtbdistini + disthere * vtbdiststep)) / vtbdiststep
        x1 = (-distwv[iwv] + (vtbdistini + (disthere + 1) * vtbdiststep)) / vtbdiststep

        dephere = int(np.round((gdp - vtbdepini) / vtbdepstep))

        # Station- grid travel-time!
        if ips[iwv] == 0:
            gridstatrv[iwv] = vtbarrp[disthere][dephere] * x1 + vtbarrp[disthere + 1][dephere] * x0
        elif ips[iwv] == 1:
            gridstatrv[iwv] = vtbarrs[disthere][dephere] * x1 + vtbarrs[disthere + 1][dephere] * x0

        # The itdifrec: the initial of RAW itself, hence no correction for indices!
        timee = int(round(o_b[iwv] + (gridstatrv[iwv] + mdt * ddt + evdt) * sample_mas)) - \
            int(round(o_b[iwv] + tp[iwv]))

        itdifrec[iwv] = timee
        ccmaxrec[iwv] = cc[int(round(timee + 0.5 * (nptccln - nptwave))), iwv]

    # Create output vectors
    ovec = (nccmax, mla, mlo, mdp, mdt, ex, ex2)

    return ovec, nccrecord_out, itdifrec, ccmaxrec


# Saves time for the above
@jit(nopython=True)
def gridsearch_maxsearch(nla, nlo, ndp, ndt, nccrecord):
    nccrecord_out = np.zeros((int(2 * nla + 1), int(2 * nlo + 1), int(2 * ndp + 1)))

    nccmax = 0  # Memorize down the max ncc value
    for ila in range(int(-nla), int(nla + 1)):
        ila_i = int(ila + nla)  # The value for indexing

        for ilo in range(int(-nlo), int(nlo + 1)):
            ilo_i = int(ilo + nlo)  # The value for indexing

            for idp in range(int(-ndp), int(ndp + 1)):  # We're only getting index here!
                idp_i = int(idp + ndp)  # The value for indexing

                nccrecord_out[ila_i, ilo_i, idp_i] = np.max(nccrecord[ila_i, ilo_i, idp_i, :])

                for idt in range(int(-ndt), int(ndt + 1)):
                    idt_i = int(idt + ndt)  # The value for indexing

                    if nccrecord[ila_i, ilo_i, idp_i, idt_i] >= nccmax:
                        nccmax = nccrecord[ila_i, ilo_i, idp_i, idt_i]
                        mla = ila
                        mlo = ilo
                        mdp = idp
                        mdt = idt

    return mla, mlo, mdp, mdt, nccmax, nccrecord_out
