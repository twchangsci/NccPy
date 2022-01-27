# Auxillary subroutines called by NccPy_GS_Hy.py

# In this sub-routine file:
# 1. noiseprob, noiseprobcore: noise probability calculation!
# 2. vincenty_custom: distance calculation
#    To be able to use it with numba, we need to calculate our own (geodesic) distance
#    There are three packages available:
#   -geopy.distance.geodesic: best choice, accurate till 1 degree from antipole (still outputs after; but not accurate)
#   -pymap3d.vincenty.vdist: same author as vdist in Matlab
#       accurate ~1 degree from antipole (still outputs after; not accurate) (We don't relocate antipolar points anyway)
#       also gives vdist(0.0, 0.0, 0, 1) = (nan, nan, nan)! This strange bug at exactly 0.5, 1 degree, etc! (Harmful)
#   -vincenty.vincenty: equal (~mm scale) to the above 2 till 1 degree from antinodal, no output after that
#       It doesn't have the bug at particular angles as vdist!
#       It is also simple (<=100 lines, only use math module), making it the best choice to adopt using numba!
# 3. cc_hypo_full: calculates cross-correlation (to expand from just CC to the whole array)
#    cc_hypo_max_declare: does the search for the best CC-ing time in advance to save time
# 4. udcheck: 1st-motion picking
# 5. Grid searche:
#    -ncc_gridsearch_hypo_pre: construction of arrays
#    -ncc_gridsearch_hypo_core/ ncc_gridsearch_hypo_core_insider: the actual grid search
#    -ncc_gridsearch_hypo_post: recovery of indices, parameters, etc

# V0.6.0 (2022.01)
# Author: Ta-Wei Chang  (twchangsci@gmail.com)


import numpy as np
import math
from numba import jit, prange


# ## I. PROBABILITY OF NOISE CALCULATION  #######################
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


# ## II. Geodesic distance #######################
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


# ## III. Cross correlation <FULL> HYPOCENTER #######################
# Calculates cross-correlation for all waveforms, and returns:
# 1. the whole CC matrix cc1_save, cc2_save for [idt][iextd][iwv]
# 2. the CC matrix of presignal time cc1_save_db, cc2_save_db in [idt][iwv]
@jit(nopython=True)
def cc_hypo_full(passvec_cc, arrival, wavelist1, wavelist2):

    # Recover parameters:
    (npthead, nptccln, nptwave, nptgrid, nptdb, nptextd) = passvec_cc[0:6]

    # Declare output matrices:
    # For 1. The CC matrix, including extended dimension!
    cc1_save = np.zeros((int(nptccln - nptwave + 1), int(2 * nptextd + 1), len(wavelist1)))
    cc2_save = np.zeros((int(nptccln - nptwave + 1), int(2 * nptextd + 1), len(wavelist1)))

    # For 2. The CC matrix for pre-signal to retrieve standard deviation!
    cc1_save_db = np.zeros((int(nptccln - nptwave + 1), len(wavelist1)))
    cc2_save_db = np.zeros((int(nptccln - nptwave + 1), len(wavelist1)))

    # 1. The CC calculation for signal, in [idt][iextd][iwv]
    for iwv in range(len(wavelist1)):

        # The extended loop:
        for iextd in range(int(-nptextd), int(nptextd + 1)):

            iextd_i = int(iextd + nptextd)  # The value for indexing

            # Case1 (UP): Event i1 as head (templete)
            # Time setting: one(template) + two(longer one for CC)
            onestart_u = int(round(arrival[iwv, 0] - npthead + iextd))
            oneend_u = int(round(arrival[iwv, 0] - npthead + nptwave + iextd))

            twostart_u = int(round(arrival[iwv, 1] + (arrival[iwv, 2] - arrival[iwv, 3]) - npthead - nptgrid + iextd))
            twoend_u = int(round(arrival[iwv, 1] + (arrival[iwv, 2] - arrival[iwv, 3])
                                 - npthead - nptgrid + nptccln + iextd))

            waveone_u = wavelist1[iwv][onestart_u:oneend_u]
            wavetwo_u = wavelist2[iwv][twostart_u:twoend_u]

            # The cross-correlation
            # For hypocenter: use a different normalization because both shape and AMPLITUDE likeliness are valued here
            for idt in range(int(nptccln - nptwave + 1)):
                cctmp1 = (waveone_u - np.mean(waveone_u))
                cctmp2 = (wavetwo_u[idt:idt + nptwave] - np.mean(wavetwo_u[idt:idt + nptwave]))

                amp1 = np.sqrt(np.dot(cctmp1, cctmp1))
                amp2 = np.sqrt(np.dot(cctmp2, cctmp2))

                if amp1 != 0 and amp2 != 0:
                    cc1_save[idt][iextd_i][iwv] = np.dot(cctmp1, cctmp2) / (max(amp1, amp2) ** 2)
                    # cc1_save[idt][iextd_i][iwv] = np.dot(cctmp1, cctmp2) / (np.max([amp1, amp2]) ** 2)
                    # cc1_save[idt][iextd_i][iwv] = np.dot(cctmp1, cctmp2) / (np.max(amp1, amp2) ** 2)
                else:
                    cc1_save[idt][iextd_i][iwv] = 0

            # # For centroid:
            # for idt in range(int(nptccln - nptwave + 1)):
            #     cctmp1 = (waveone_u - np.mean(waveone_u)) / \
            #              np.sqrt(np.dot(waveone_u - np.mean(waveone_u), waveone_u - np.mean(waveone_u)))
            #     cctmp2 = (wavetwo_u[idt:idt + nptwave] - np.mean(wavetwo_u[idt:idt + nptwave])) / \
            #              np.sqrt(np.dot(wavetwo_u[idt:idt + nptwave] - np.mean(wavetwo_u[idt:idt + nptwave]),
            #                             wavetwo_u[idt:idt + nptwave] - np.mean(wavetwo_u[idt:idt + nptwave])))
            #
            #     cc1_save[idt][iwv] = np.dot(cctmp1, cctmp2)

            # Case2 (DOWN): Event i2 as head (templete)
            # Time setting: one(template) + two(longer one for CC)
            onestart_d = int(round(arrival[iwv, 1] - npthead + iextd))
            oneend_d = int(round(arrival[iwv, 1] - npthead + nptwave + iextd))

            twostart_d = int(round(arrival[iwv, 0] + (arrival[iwv, 3] - arrival[iwv, 2]) - npthead - nptgrid + iextd))
            twoend_d = int(round(arrival[iwv, 0] + (arrival[iwv, 3] - arrival[iwv, 2])
                                 - npthead - nptgrid + nptccln + iextd))

            waveone_d = wavelist2[iwv][onestart_d:oneend_d]
            wavetwo_d = wavelist1[iwv][twostart_d:twoend_d]

            # The cross-correlation
            # For hypocenter: use a different normalization because both shape and AMPLITUDE likeliness are valued here
            for idt in range(int(nptccln - nptwave + 1)):
                cctmp1 = (waveone_d - np.mean(waveone_d))
                cctmp2 = (wavetwo_d[idt:idt + nptwave] - np.mean(wavetwo_d[idt:idt + nptwave]))

                amp1 = np.sqrt(np.dot(cctmp1, cctmp1))
                amp2 = np.sqrt(np.dot(cctmp2, cctmp2))

                if amp1 != 0 and amp2 != 0:
                    cc2_save[idt][iextd_i][iwv] = np.dot(cctmp1, cctmp2) / (max(amp1, amp2) ** 2)
                    # cc2_save[idt][iextd_i][iwv] = np.dot(cctmp1, cctmp2) / (np.max([amp1, amp2]) ** 2)
                    # cc2_save[idt][iextd_i][iwv] = np.dot(cctmp1, cctmp2) / (np.max(amp1, amp2) ** 2)
                else:
                    cc2_save[idt][iextd_i][iwv] = 0

            # # For centroid:
            # for idt in range(int(nptccln - nptwave + 1)):
            #     cctmp1 = (waveone_d - np.mean(waveone_d)) / \
            #         np.sqrt(np.dot(waveone_d - np.mean(waveone_d), waveone_d - np.mean(waveone_d)))
            #     cctmp2 = (wavetwo_d[idt:idt + nptwave] - np.mean(wavetwo_d[idt:idt + nptwave])) / \
            #         np.sqrt(np.dot(wavetwo_d[idt:idt + nptwave] - np.mean(wavetwo_d[idt:idt + nptwave]),
            #                        wavetwo_d[idt:idt + nptwave] - np.mean(wavetwo_d[idt:idt + nptwave])))
            #
            #     cc2_save[idt][iwv] = np.dot(cctmp1, cctmp2)

    # 2. The CC calculation for pre-signal, in [idt][iwv]
    for iwv in range(len(wavelist1)):

        # Case1 (UP): Event i1 as head (templete)
        # Time setting: one(template) + two(longer one for CC)
        onestart_u = int(round(arrival[iwv, 0] - npthead - nptdb))
        oneend_u = int(round(arrival[iwv, 0] - npthead + nptwave - nptdb))

        twostart_u = int(round(arrival[iwv, 1] + (arrival[iwv, 2] - arrival[iwv, 3]) - npthead - nptgrid - nptdb))
        twoend_u = int(round(arrival[iwv, 1] + (arrival[iwv, 2] - arrival[iwv, 3])
                             - npthead - nptgrid + nptccln - nptdb))

        waveone_u = wavelist1[iwv][onestart_u:oneend_u]
        wavetwo_u = wavelist2[iwv][twostart_u:twoend_u]

        # The cross-correlation
        for idt in range(int(nptccln - nptwave + 1)):
            cctmp1 = (waveone_u - np.mean(waveone_u))
            cctmp2 = (wavetwo_u[idt:idt + nptwave] - np.mean(wavetwo_u[idt:idt + nptwave]))

            amp1 = np.sqrt(np.dot(cctmp1, cctmp1))
            amp2 = np.sqrt(np.dot(cctmp2, cctmp2))

            if amp1 != 0 and amp2 != 0:
                cc1_save_db[idt][iwv] = np.dot(cctmp1, cctmp2) / (max(amp1, amp2) ** 2)
                # cc1_save_db[idt][iwv] = np.dot(cctmp1, cctmp2) / (np.max(amp1, amp2) ** 2)
            else:
                cc1_save_db[idt][iwv] = 0

        # Case2 (DOWN): Event i2 as head (templete)
        # Time setting: one(template) + two(longer one for CC)
        onestart_d = int(round(arrival[iwv, 1] - npthead - nptdb))
        oneend_d = int(round(arrival[iwv, 1] - npthead + nptwave - nptdb))

        twostart_d = int(round(arrival[iwv, 0] + (arrival[iwv, 3] - arrival[iwv, 2]) - npthead - nptgrid - nptdb))
        twoend_d = int(round(arrival[iwv, 0] + (arrival[iwv, 3] - arrival[iwv, 2])
                             - npthead - nptgrid + nptccln - nptdb))

        waveone_d = wavelist2[iwv][onestart_d:oneend_d]
        wavetwo_d = wavelist1[iwv][twostart_d:twoend_d]

        # The cross-correlation
        for idt in range(int(nptccln - nptwave + 1)):
            cctmp1 = (waveone_d - np.mean(waveone_d))
            cctmp2 = (wavetwo_d[idt:idt + nptwave] - np.mean(wavetwo_d[idt:idt + nptwave]))

            amp1 = np.sqrt(np.dot(cctmp1, cctmp1))
            amp2 = np.sqrt(np.dot(cctmp2, cctmp2))

            if amp1 != 0 and amp2 != 0:
                cc2_save_db[idt][iwv] = np.dot(cctmp1, cctmp2) / (max(amp1, amp2) ** 2)
                # cc2_save_db[idt][iwv] = np.dot(cctmp1, cctmp2) / (np.max(amp1, amp2) ** 2)
            else:
                cc2_save_db[idt][iwv] = 0

    return cc1_save, cc2_save, cc1_save_db, cc2_save_db


# Pre-calculate the max and argmax in the CC to save time
@jit(nopython=True)
def cc_hypo_max_declare(cc1_save, cc2_save, nptextd):
    # Output declare
    cc1_max = np.zeros((len(cc1_save[:, 0, 0]), len(cc1_save[0, 0, :])))
    cc1_argmax = np.zeros((len(cc1_save[:, 0, 0]), len(cc1_save[0, 0, :])))
    cc2_max = np.zeros((len(cc1_save[:, 0, 0]), len(cc1_save[0, 0, :])))
    cc2_argmax = np.zeros((len(cc1_save[:, 0, 0]), len(cc1_save[0, 0, :])))

    for idt in range(len(cc1_save[:, 0, 0])):
        for iwv in range(len(cc1_save[0, 0, :])):
            cc1_max[idt, iwv] = np.max(cc1_save[idt, :, iwv])
            cc1_argmax[idt, iwv] = np.argmax(cc1_save[idt, :, iwv]) - int(nptextd)

            cc2_max[idt, iwv] = np.max(cc2_save[idt, :, iwv])
            cc2_argmax[idt, iwv] = np.argmax(cc2_save[idt, :, iwv]) - int(nptextd)

    return cc1_max, cc1_argmax, cc2_max, cc2_argmax


# ## IV. The up-/ down-going 1st motion check subroutine #########
@jit(nopython=True)
def udcheck(udprm, udwave1, udwave2):
    # Recover parameters:
    (uddur, ud1bg, ud1crit, ud2seg, ud2crit, ud3seg, ud3crit, sample_mas, iud1, iud2, iud3) = udprm[0:11]

    # Declare time series of STA/LTA and kurtosis to pick:
    stalta1 = np.zeros((int(round(2 * uddur * sample_mas + 1))))
    stalta2 = np.zeros((int(round(2 * uddur * sample_mas + 1))))
    kurt1 = np.zeros((int(round(2 * uddur * sample_mas + 1))))
    kurt2 = np.zeros((int(round(2 * uddur * sample_mas + 1))))

    # 1. The STA/LTA calculation:
    premean1 = np.mean(udwave1[0:int(round(ud1bg * sample_mas))])
    prestd1 = std_custom(udwave1[0:int(round(ud1bg * sample_mas))])

    premean2 = np.mean(udwave2[0:int(round(ud1bg * sample_mas))])
    prestd2 = std_custom(udwave2[0:int(round(ud1bg * sample_mas))])

    for ii in range(len(stalta1)):
        stalta1[ii] = (udwave1[int(round(ud1bg * sample_mas + ii))] - premean1) / prestd1
        stalta2[ii] = (udwave2[int(round(ud1bg * sample_mas + ii))] - premean2) / prestd2

    # 2. The kurtosis calculation: (kurtosis, then gradient)
    for ii in range(len(stalta1)):
        kurt1[ii] = kurtosis_custom(udwave1[int(round((ud1bg - ud2seg) * sample_mas + ii + 1)):
                                            int(round(ud1bg * sample_mas + ii + 1))])
        kurt2[ii] = kurtosis_custom(udwave2[int(round((ud1bg - ud2seg) * sample_mas + ii + 1)):
                                            int(round(ud1bg * sample_mas + ii + 1))])

    gradkurt1 = gradient_custom(kurt1)
    gradkurt2 = gradient_custom(kurt2)

    # 3. The pick!
    # Initialize paraameters!
    pick1 = -1   # The indices of wave 1
    pick2 = -1   # The indices of wave 2

    # (1, 1, 1) Use all criterions
    if iud1 == 1 and iud2 == 1 and iud3 == 1:
        for jj in range(len(stalta1)):
            if np.abs(stalta1[jj]) >= ud1crit:    # iud1: STA/LTA
                if gradkurt1[jj] >= ud2crit:      # iud2: diff(kurtosis)
                    if std_custom(udwave1[int(round(ud1bg * sample_mas + jj)):
                                          int(round((ud1bg + ud3seg) * sample_mas + jj))]) / \
                            std_custom(udwave1[int(round((ud1bg - ud3seg) * sample_mas + jj)):
                                               int(round(ud1bg * sample_mas + jj))]) >= ud3crit:   # iud3: immediate S/N
                        pick1 = jj
                        break

        for jj in range(len(stalta1)):
            if np.abs(stalta2[jj]) >= ud1crit:    # iud1: STA/LTA
                if gradkurt2[jj] >= ud2crit:      # iud2: diff(kurtosis)
                    if std_custom(udwave2[int(round(ud1bg * sample_mas + jj)):
                                          int(round((ud1bg + ud3seg) * sample_mas + jj))]) / \
                            std_custom(udwave2[int(round((ud1bg - ud3seg) * sample_mas + jj)):
                                               int(round(ud1bg * sample_mas + jj))]) >= ud3crit:   # iud3: immediate S/N
                        pick2 = jj
                        break

    # (1, 1, 0) Use criterions 1, 2
    elif iud1 == 1 and iud2 == 1 and iud3 == 0:
        for jj in range(len(stalta1)):
            if np.abs(stalta1[jj]) >= ud1crit:  # iud1: STA/LTA
                if gradkurt1[jj] >= ud2crit:  # iud2: diff(kurtosis)
                    pick1 = jj
                    break

        for jj in range(len(stalta1)):
            if np.abs(stalta2[jj]) >= ud1crit:  # iud1: STA/LTA
                if gradkurt2[jj] >= ud2crit:  # iud2: diff(kurtosis)
                    pick2 = jj
                    break

    # (1, 0, 1) Use criterions 1, 3
    if iud1 == 1 and iud2 == 0 and iud3 == 1:
        for jj in range(len(stalta1)):
            if np.abs(stalta1[jj]) >= ud1crit:  # iud1: STA/LTA
                if std_custom(udwave1[int(round(ud1bg * sample_mas + jj)):
                                      int(round((ud1bg + ud3seg) * sample_mas + jj))]) / \
                        std_custom(udwave1[int(round((ud1bg - ud3seg) * sample_mas + jj)):
                                           int(round(ud1bg * sample_mas + jj))]) >= ud3crit:   # iud3: immediate S/N
                    pick1 = jj
                    break

        for jj in range(len(stalta1)):
            if np.abs(stalta2[jj]) >= ud1crit:  # iud1: STA/LTA
                if std_custom(udwave2[int(round(ud1bg * sample_mas + jj)):
                                      int(round((ud1bg + ud3seg) * sample_mas + jj))]) / \
                        std_custom(udwave2[int(round((ud1bg - ud3seg) * sample_mas + jj)):
                                           int(round(ud1bg * sample_mas + jj))]) >= ud3crit:  # iud3: immediate S/N
                    pick2 = jj
                    break

    # (1, 0, 0) Use criterion 1
    if iud1 == 1 and iud2 == 0 and iud3 == 0:
        for jj in range(len(stalta1)):
            if np.abs(stalta1[jj]) >= ud1crit:  # iud1: STA/LTA
                pick1 = jj
                break

        for jj in range(len(stalta1)):
            if np.abs(stalta2[jj]) >= ud1crit:  # iud1: STA/LTA
                pick2 = jj
                break

    # (0, 1, 1) Use criterions 2, 3
    if iud1 == 0 and iud2 == 1 and iud3 == 1:
        for jj in range(len(stalta1)):
            if gradkurt1[jj] >= ud2crit:  # iud2: diff(kurtosis)
                if std_custom(udwave1[int(round(ud1bg * sample_mas + jj)):
                                      int(round((ud1bg + ud3seg) * sample_mas + jj))]) / \
                        std_custom(udwave1[int(round((ud1bg - ud3seg) * sample_mas + jj)):
                                           int(round(ud1bg * sample_mas + jj))]) >= ud3crit:  # iud3: immediate S/N
                    pick1 = jj
                    break

        for jj in range(len(stalta1)):
            if gradkurt2[jj] >= ud2crit:  # iud2: diff(kurtosis)
                if std_custom(udwave2[int(round(ud1bg * sample_mas + jj)):
                                      int(round((ud1bg + ud3seg) * sample_mas + jj))]) / \
                        std_custom(udwave2[int(round((ud1bg - ud3seg) * sample_mas + jj)):
                                           int(round(ud1bg * sample_mas + jj))]) >= ud3crit:  # iud3: immediate S/N
                    pick2 = jj
                    break

    # (0, 1, 0) Use criterion 2
    if iud1 == 0 and iud2 == 1 and iud3 == 0:
        for jj in range(len(stalta1)):
            if gradkurt1[jj] >= ud2crit:  # iud2: diff(kurtosis)
                pick1 = jj
                break

        for jj in range(len(stalta1)):
            if gradkurt2[jj] >= ud2crit:  # iud2: diff(kurtosis)
                pick2 = jj
                break

    # (0, 0, 1) Use all criterions
    if iud1 == 0 and iud2 == 0 and iud3 == 1:
        for jj in range(len(stalta1)):
            if std_custom(udwave1[int(round(ud1bg * sample_mas + jj)):
                                  int(round((ud1bg + ud3seg) * sample_mas + jj))]) / \
                    std_custom(udwave1[int(round((ud1bg - ud3seg) * sample_mas + jj)):
                                       int(round(ud1bg * sample_mas + jj))]) >= ud3crit:  # iud3: immediate S/N
                pick1 = jj
                break

        for jj in range(len(stalta1)):
            if std_custom(udwave2[int(round(ud1bg * sample_mas + jj)):
                                  int(round((ud1bg + ud3seg) * sample_mas + jj))]) / \
                    std_custom(udwave2[int(round((ud1bg - ud3seg) * sample_mas + jj)):
                                       int(round(ud1bg * sample_mas + jj))]) >= ud3crit:  # iud3: immediate S/N
                pick2 = jj
                break

    # 4. Polarity of pick!
    # Initialize paraameters! (-1: not found; 0: down; 1: up)
    ud1 = -1     # Final output of up/ down/ not found for wave 1
    ud2 = -1     # Final output of up/ down/ not found for wave 2

    # Event 1
    if iud1 == 0 and iud2 == 0 and iud3 == 0:
        ud1 = 1    # The selection is not used at all, set ud1 = ud2 to pass anyway!
    elif pick1 == -1:
        ud1 = -1   # Not found
    elif udwave1[int(round(ud1bg * sample_mas + pick1))] - udwave1[int(round(ud1bg * sample_mas + pick1)) - 1] > 0:
        ud1 = 1    # Up
    elif udwave1[int(round(ud1bg * sample_mas + pick1))] - udwave1[int(round(ud1bg * sample_mas + pick1)) - 1] < 0:
        ud1 = 0    # Down
    elif udwave1[int(round(ud1bg * sample_mas + pick1))] - udwave1[int(round(ud1bg * sample_mas + pick1)) - 1] == 0:
        if udwave1[int(round(ud1bg * sample_mas + pick1))] - udwave1[int(round(ud1bg * sample_mas + pick1)) - 2] > 0:
            ud1 = 1  # Up (If derivative happens to be zero, compare with two points before!)
        elif udwave1[int(round(ud1bg * sample_mas + pick1))] - udwave1[int(round(ud1bg * sample_mas + pick1)) - 2] < 0:
            ud1 = 0  # Down (If derivative happens to be zero, compare with two points before!)
        elif udwave1[int(round(ud1bg * sample_mas + pick1))] - udwave1[int(round(ud1bg * sample_mas + pick1)) - 2] == 0:
            ud1 = -1  # The data looks like: ... 100 100 100 ... which is unnatural, especially close to signal section!

    # Event 2
    if iud1 == 0 and iud2 == 0 and iud3 == 0:
        ud2 = 1  # The selection is not used at all, set ud1 = ud2 to pass anyway!
    elif pick2 == -1:
        ud2 = -1  # Not found
    elif udwave2[int(round(ud1bg * sample_mas + pick2))] - udwave2[int(round(ud1bg * sample_mas + pick2)) - 1] > 0:
        ud2 = 1  # Up
    elif udwave2[int(round(ud1bg * sample_mas + pick2))] - udwave2[int(round(ud1bg * sample_mas + pick2)) - 1] < 0:
        ud2 = 0  # Down
    elif udwave2[int(round(ud1bg * sample_mas + pick2))] - udwave2[int(round(ud1bg * sample_mas + pick2)) - 1] == 0:
        if udwave2[int(round(ud1bg * sample_mas + pick2))] - udwave2[int(round(ud1bg * sample_mas + pick2)) - 2] > 0:
            ud2 = 1  # Up (If derivative happens to be zero, compare with two points before!)
        elif udwave2[int(round(ud1bg * sample_mas + pick2))] - udwave2[int(round(ud1bg * sample_mas + pick2)) - 2] < 0:
            ud2 = 0  # Down (If derivative happens to be zero, compare with two points before!)
        elif udwave2[int(round(ud1bg * sample_mas + pick2))] - udwave2[int(round(ud1bg * sample_mas + pick2)) - 2] == 0:
            ud2 = -1  # The data looks like: ... 100 100 100 ... which is unnatural, especially close to signal section!

    return ud1, ud2, pick1 - uddur * sample_mas, pick2 - uddur * sample_mas


# The un-biased standard deviation calculation (np.std(~, ddof=1) that numba couldn't manage)
@jit(nopython=True)
def std_custom(wave_in):
    stdd = np.sqrt(np.sum(np.abs(wave_in - np.mean(wave_in)) ** 2) / (len(wave_in) - 1))
    return stdd


# The kurtosis calculation (verified with scipy.stats.kurtosis)
@jit(nopython=True)
def kurtosis_custom(wave_in):
    kurt = (np.sum((wave_in - np.mean(wave_in)) ** 4) / len(wave_in)) / (np.var(wave_in) ** 2)
    return kurt


# The gradient calculation (verified with np.gradient)
@jit(nopython=True)
def gradient_custom(sig_in):
    diff = np.zeros((len(sig_in)))
    for kk in range(len(sig_in)):
        if kk == 0:
            diff[kk] = sig_in[1] - sig_in[0]
        elif kk == len(sig_in) - 1:
            diff[kk] = sig_in[len(sig_in) - 1] - sig_in[len(sig_in) - 2]
        else:
            diff[kk] = 0.5 * (sig_in[kk + 1] - sig_in[kk - 1])

    return diff


# ## V. Grid Search #######################
# ## V. 1. Declaration of array, and travel-time calculation #######################
@jit(nopython=True)
def ncc_gridsearch_hypo_pre(passvec1, passvec2, passvec3, wvvec, vtbarrp, vtbarrs):
    # Recover values
    (evla, evlo, evdp, evdt, nla, nlo, ndp, ndt, dla, dlo, ddp, ddt) = passvec1[0:12]
    (iitr, adp, ncar, sample_mas, nptccln, nptwave, nptgrid, nptdb, nptextd, gdla, gdlo, ex, ex2, ex_db, ex2_db) = \
        passvec2[0:15]
    (vtbdistini, vtbdiststep, vtbdepini, vtbdepstep) = passvec3[0:4]
    stla = wvvec[:, 0]
    stlo = wvvec[:, 1]
    ips = wvvec[:, 2]
    tp = wvvec[:, 3]

    earth_rad = (2 * 6378137 + 6356752.314245) / 3 / 1000   # Radius of Earth, from the above & (2a+b)/3 (Wiki)

    nccrecord = np.zeros((int(2 * nla + 1), int(2 * nlo + 1), int(2 * ndp + 1), int(2 * ndt + 1)))
    nccrecord_db = np.zeros((int(2 * nla + 1), int(2 * nlo + 1), int(2 * ndp + 1), int(2 * ndt + 1)))

    extdrecord = np.zeros((int(2 * nla + 1), int(2 * nlo + 1), int(2 * ndp + 1), int(2 * ndt + 1), int(len(tp))))

    ex2record = np.zeros((int(2 * nla + 1), int(2 * nlo + 1), int(2 * ndp + 1), int(2 * ndt + 1)))
    ex2record_db = np.zeros((int(2 * nla + 1), int(2 * nlo + 1), int(2 * ndp + 1), int(2 * ndt + 1)))

    distwvmat = np.zeros((int(2 * nla + 1), int(2 * nlo + 1), int(len(tp))))
    gridstatrvmat = np.zeros((int(2 * nla + 1), int(2 * nlo + 1), int(2 * ndp + 1), int(len(tp))))

    indices_mat_assign = np.zeros((int(2 * nla + 1), int(2 * nlo + 1), int(2 * ndp + 1), int(2 * ndt + 1), 4))

    for ila in range(int(-nla), int(nla + 1)):
        for ilo in range(int(-nlo), int(nlo + 1)):

            for iwv in range(len(tp)):
                # Distance calculation
                distwvmat[int(ila + nla), int(ilo + nlo), iwv] = \
                    vincenty_custom((stla[iwv], stlo[iwv]), (ila * gdla + evla, ilo * gdlo + evlo)) \
                    / earth_rad / np.pi * 180

            for idp in range(int(-ndp), int(ndp + 1)):
                for iwv in range(len(tp)):

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

                    # Travel-time not found (moved here from core)
                    if int(round(gridstatrvmat[int(ila + nla), int(ilo + nlo), int(idp + ndp), iwv])) == -1:
                        if evdp < adp:  # Grid goes above sea level!
                            print('Warning: travel-time not avalable!!', ila, ilo, idp, iwv,
                                  stla[iwv], stlo[iwv], ila * gdla + evla, ilo * gdlo + evlo, (idp + ndp) * ddp,
                                  distwvmat[int(ila + nla), int(ilo + nlo), iwv])
                        elif evdp >= adp:  # Grid does not go above sea level
                            print('Warning: travel-time not avalable!!', ila, ilo, idp, iwv,
                                  stla[iwv], stlo[iwv], ila * gdla + evla, ilo * gdlo + evlo, idp * ddp + evdp,
                                  distwvmat[int(ila + nla), int(ilo + nlo), iwv])

                for idt in range(int(-ndt), int(ndt + 1)):
                    indices_mat_assign[int(ila + nla), int(ilo + nlo), int(idp + ndp), int(idt + ndt), 0] = ila + nla
                    indices_mat_assign[int(ila + nla), int(ilo + nlo), int(idp + ndp), int(idt + ndt), 1] = ilo + nlo
                    indices_mat_assign[int(ila + nla), int(ilo + nlo), int(idp + ndp), int(idt + ndt), 2] = idp + ndp
                    indices_mat_assign[int(ila + nla), int(ilo + nlo), int(idp + ndp), int(idt + ndt), 3] = idt + ndt

    indices_mat = np.reshape(indices_mat_assign,
                             (int(2 * nla + 1) * int(2 * nlo + 1) * int(2 * ndp + 1) * int(2 * ndt + 1), 4))

    return nccrecord, nccrecord_db, extdrecord, ex2record, ex2record_db, gridstatrvmat, indices_mat


# ## V. 2. The kernal of the main grid-search #######################
@jit(nopython=True, parallel=True)
def ncc_gridsearch_hypo_core(passvec1, passvec2, wvvec, cc_max, cc_db,
                             nccrecord, nccrecord_db, extdrecord, gridstatrvmat, indices_mat):
    # Recover values
    (evla, evlo, evdp, evdt, nla, nlo, ndp, ndt, dla, dlo, ddp, ddt) = passvec1[0:12]
    (iitr, adp, ncar, sample_mas, nptccln, nptwave, nptgrid, nptdb, nptextd, gdla, gdlo, ex, ex2, ex_db, ex2_db) = \
        passvec2[0:15]
    # stla = wvvec[:, 0]
    # stlo = wvvec[:, 1]
    # ips = wvvec[:, 2]
    tp = wvvec[:, 3]
    o_b = wvvec[:, 4]

    for ind in prange(len(indices_mat)):

        (nccrecord[int(round(indices_mat[ind, 0])), int(round(indices_mat[ind, 1])),
                   int(round(indices_mat[ind, 2])), int(round(indices_mat[ind, 3]))],
         nccrecord_db[int(round(indices_mat[ind, 0])), int(round(indices_mat[ind, 1])),
                      int(round(indices_mat[ind, 2])), int(round(indices_mat[ind, 3]))]) = \
            ncc_gridsearch_hypo_core_insider(gridstatrvmat
                                             [int(round(indices_mat[ind, 0])), int(round(indices_mat[ind, 1])),
                                              int(round(indices_mat[ind, 2])), :], tp, o_b, cc_max, cc_db,
                                             indices_mat[ind, :], ndt, ddt, evdt, nptgrid, nptccln, nptwave, sample_mas)

    return nccrecord, nccrecord_db, extdrecord


# ## V. 2. 1. The spun-off of above #######################
@jit(nopython=True)
def ncc_gridsearch_hypo_core_insider(gridstatrv, tp, o_b, cc_max, cc_db,
                                     indices, ndt, ddt, evdt, nptgrid, nptccln, nptwave, sample_mas):

    # Initialize outputs
    nccrecord = 0
    nccrecord_db = 0

    for iwv in range(len(tp)):

        # The time is just so messy, that we assign it to a parameter here
        # It should be alright to do so, as the parallel is not done within this sub-sub-routine!
        timee = int(round(o_b[iwv] + (gridstatrv[iwv] + int(round(indices[3] - ndt)) * ddt + evdt) * sample_mas)) - \
                int(round(o_b[iwv] + tp[iwv]))

        # When appropriate CC is available
        if -nptgrid <= timee <= nptgrid:

            # Assign/ append the max in the extended axis to NCC matrix
            nccrecord += \
                cc_max[int(round(timee + 0.5 * (nptccln - nptwave))), iwv]

            # For the double case!
            nccrecord_db += \
                cc_db[int(round(timee + 0.5 * (nptccln - nptwave))), iwv]

        # In case the duration of waveform with CC values calculated is too short
        elif timee < -nptgrid:
            print('Warning: waveform to CC too short! (itdif < -nptgrid)', indices[:], iwv, timee)

        elif timee > nptgrid:
            print('Warning: waveform to CC too short! (itdif > nptgrid)', indices[:], iwv, timee)

    return nccrecord, nccrecord_db


# ## V. 3. The rest of the parameter determination! #######################
@jit(nopython=True)
def ncc_gridsearch_hypo_post(passvec1, passvec2, passvec3, wvvec, cc_max, cc_argmax, vtbarrp, vtbarrs,
                             nccrecord, nccrecord_db, ex2record, ex2record_db):
    # Recover values
    (evla, evlo, evdp, evdt, nla, nlo, ndp, ndt, dla, dlo, ddp, ddt) = passvec1[0:12]
    (iitr, adp, ncar, sample_mas, nptccln, nptwave, nptgrid, nptdb, nptextd, gdla, gdlo, ex, ex2, ex_db, ex2_db) = \
        passvec2[0:15]
    (vtbdistini, vtbdiststep, vtbdepini, vtbdepstep) = passvec3[0:4]
    stla = wvvec[:, 0]
    stlo = wvvec[:, 1]
    ips = wvvec[:, 2]
    tp = wvvec[:, 3]
    o_b = wvvec[:, 4]

    earth_rad = (2 * 6378137 + 6356752.314245) / 3 / 1000  # Radius of Earth, from the above & (2a+b)/3 (Wiki)

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

                    ex2record[int(ila_i), int(ilo_i), int(idp_i), int(idt_i)] = \
                        nccrecord[int(ila_i), int(ilo_i), int(idp_i), int(idt_i)] ** 2
                    ex2record_db[int(ila_i), int(ilo_i), int(idp_i), int(idt_i)] = \
                        nccrecord_db[int(ila_i), int(ilo_i), int(idp_i), int(idt_i)] ** 2

                    if nccrecord[ila_i, ilo_i, idp_i, idt_i] >= nccmax:
                        nccmax = nccrecord[ila_i, ilo_i, idp_i, idt_i]
                        mla = ila
                        mlo = ilo
                        mdp = idp
                        mdt = idt

    # Prepare the parameters to be outputted
    # L1 and L2 norm (for standard deviation) (for the coarser one only)
    if iitr == 0:
        ex = np.sum(nccrecord)
        ex2 = np.sum(ex2record)

        ex_db = np.sum(nccrecord_db)
        ex2_db = np.sum(ex2record_db)

    # CC values and where the max is found for each waves at the max
    itdifrec = np.zeros(int(len(tp)))
    ccmaxrec = np.zeros(int(len(tp)))
    extdrec = np.zeros(int(len(tp)))

    distwv = np.zeros(int(len(tp)))
    gridstatrv = np.zeros(int(len(tp)))

    for iwv in range(len(tp)):

        # (Recover here the travel time from grid to each staiton)
        stalocg = (stla[iwv], stlo[iwv])
        grdlocg = (mla * gdla + evla, mlo * gdlo + evlo)
        distg = vincenty_custom(stalocg, grdlocg)
        distwv[iwv] = distg / earth_rad / np.pi * 180

        if evdp < adp:  # Grid goes above sea level!
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

        # The itdifrec: the initial of RAW itself, hence no correction for indices (we want both -+)!
        timee = int(round(o_b[iwv] + (gridstatrv[iwv] + mdt * ddt + evdt) * sample_mas)) - \
            int(round(o_b[iwv] + tp[iwv]))
        itdifrec[iwv] = timee
        extdrec[iwv] = cc_argmax[int(round(timee + 0.5 * (nptccln - nptwave))), iwv]
        ccmaxrec[iwv] = cc_max[int(round(timee + 0.5 * (nptccln - nptwave))), iwv]

    # Create output vectors
    ovec = (nccmax, mla, mlo, mdp, mdt, ex, ex2, ex_db, ex2_db)

    return ovec, nccrecord_out, itdifrec, ccmaxrec, extdrec
