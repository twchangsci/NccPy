# Auxillary subroutines for the NccPy_Inv series codes

# V0.6.0 (2022.01)
# Author: Ta-Wei Chang  (twchangsci@gmail.com)


import numpy as np
import math
from numba import jit


# ## I. PROBABILITY OF NOISE CALCULATION SUBROUTINES #######################
# @jit(nopython=True)     # Numba doesn't support formatted write!
def noiseprob_table(ds, smax_in, ncar1):
    # Takes variable pthre and ncar1 from outside

    smax = int(round(smax_in / ds)) + 1
    p = np.zeros((smax, 2))

    for i in range(smax):
        s = i * ds
        p[i, 0] = s
        p[i, 1] = noiseprobcore(s, ncar1)

    return p


# Just look it up! Do not need to re-calculate the table!
@jit(nopython=True)     # Numba doesn't support formatted write!!
def noiseprob_lookup(ds, smax_in, ptable, pcc):
    # Takes variable pthre and ncar1 from outside

    smax = int(round(smax_in / ds)) + 1

    if pcc < 1:
        nccthreout = 0
        for i in range(smax - 1):
            if ptable[i, 1] >= pcc > ptable[i + 1, 1]:
                nccthreout = (i * ds + (i + 1) * ds) / 2
                pccout = (ptable[i, 1] + ptable[i + 1, 1]) / 2
                break

        if nccthreout == 0:
            print(f'Error!! Increase "smax"!')

    else:
        nccthreout = 0
        pccout = 1

    return nccthreout, pccout


# Run using numba to save time
@jit(nopython=True)
def noiseprobcore(xxo, n):
    # Written using math instead of numpy for speed
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


# ## II. Geodesic distance SUBROUTINE #######################
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


# ## III. THE ABIC CALCULATION #######################
# Run this here with numba to accelerate the calculation!
@jit(nopython=True)
def abic_core(n, nmax, mmax, comp, abicrange, abicmin, kmin, wf, g, gt, gtw, dvec, wd, we):

    abicmat = np.zeros((int(round(2 * abicrange + 1)), 3))

    for k in range(-abicrange, abicrange + 1):
        alpha2 = 10 ** (k * 0.05)

        # Tentative weighting
        for i6 in range(n, nmax):
            wf[i6] = alpha2

        # Least squares inversion
        for i6 in range(mmax):
            for j6 in range(nmax):
                gtw[i6, j6] = gt[i6, j6] * wf[j6]

        gtwg = gtw @ g
        igtwg = np.linalg.inv(gtwg)
        gg = igtwg @ gtw

        c, v = np.linalg.eigh(gtwg)    # gtwg, product by its transpose, is symmetric
        # c, v = np.linalg.eig(gtwg)   # may fail because of tiny complex values in output...
        d = dvec[:, comp]
        m = gg @ d
        gm = g @ m
        e = d - gm

        for i6 in range(nmax):
            wd[i6] = wf[i6] * d[i6]
            we[i6] = wf[i6] * e[i6]

        res = np.dot(e, we)

        # To calculate the ABIC:
        x = nmax * np.log(res)
        y = mmax * np.log(alpha2)
        z = 0
        for i6 in range(mmax):
            z += np.log(abs(c[i6]))

        abic = x - y + z

        # Log it down!
        abicmat[k, 0] = k
        abicmat[k, 1] = alpha2
        abicmat[k, 2] = abic
        # print(f'{k:3.0f}  {alpha2:12.6f}  {abic:.8f}', file=abiclog)

        # Assign smallest
        if abic < abicmin:
            abicmin = abic
            kmin = k

    return kmin, abicmat
