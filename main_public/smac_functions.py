#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2022/07/08
# version ='1.0'
# ---------------------------------------------------------------------------

'''
This code copiles all functions that will be called later by other codes. For the sake of clarity
these functions are defined in this separated piece of code.
'''


import numpy as np
import math
def pixelToWorldCoords(pX, pY, geoTransform):
    ''' Input image pixel coordinates and get world coordinates according to geotransform using gdal
    '''

    def applyGeoTransform(inX, inY, geoTransform):
        outX = geoTransform[0] + inX * geoTransform[1] + inY * geoTransform[2]
        outY = geoTransform[3] + inX * geoTransform[4] + inY * geoTransform[5]
        return outX, outY

    mX, mY = applyGeoTransform(pX, pY, geoTransform)
    return mX, mY

def worldToPixelCoords(wX, wY, geoTransform, dtype='int'):
    ''' Input world coordinates and get pixel coordinates according to reverse geotransform using gdal
    '''
    reverse_transform = ~ affine.Affine.from_gdal(*geoTransform)
    px, py = reverse_transform * (wX, wY)
    if dtype == 'int':
        px, py = int(px + 0.5), int(py + 0.5)
    else:
        px, py = px + 0.5, py + 0.5
    return px, py


def xyval(A):
    """
    Function to list all pixel coords including their associated value starting from the  upper left corner (?)
    :param A: raster band as numpy array
    :return: x and y of each pixel and the associated value
    """
    import numpy as np
    x, y = np.indices(A.shape)
    return x.ravel(), y.ravel(), A.ravel()




def to_numpy2(transform):
    return np.array([transform.a,
                     transform.b,
                     transform.c,
                     transform.d,
                     transform.e,
                     transform.f, 0, 0, 1], dtype='float64').reshape((3,3))

def xy_np(transform, rows, cols, offset='center'):
    if isinstance(rows, int) and isinstance(cols, int):
        pts = np.array([[rows, cols, 1]]).T
    else:
        assert len(rows) == len(cols)
        pts = np.ones((3, len(rows)), dtype=int)
        pts[0] = rows
        pts[1] = cols

    if offset == 'center':
        coff, roff = (0.5, 0.5)
    elif offset == 'ul':
        coff, roff = (0, 0)
    elif offset == 'ur':
        coff, roff = (1, 0)
    elif offset == 'll':
        coff, roff = (0, 1)
    elif offset == 'lr':
        coff, roff = (1, 1)
    else:
        raise ValueError("Invalid offset")

    _transnp = to_numpy2(transform)
    _translt = to_numpy2(transform.translation(coff, roff))
    locs = _transnp @ _translt @ pts
    return locs[0].tolist(), locs[1].tolist()

def latlon_to_utm32n_series(lat_deg, lon_deg):
    """
    Convert geographic coordinates (lat, lon in degrees, WGS84)
    to UTM Zone 32N (EPSG:32632) using series expansions.

    Returns:
      (easting, northing) in meters.
    """
    # WGS84 constants
    a = 6378137.0                     # semi-major axis
    f = 1 / 298.257223563             # flattening
    e2 = 2*f - f*f                    # eccentricity squared
    e = math.sqrt(e2)

    # UTM parameters for Zone 32N
    k0 = 0.9996
    lambda0_deg = 9.0  # central meridian
    E0 = 500000.0
    N0 = 0.0           # for northern hemisphere

    # 1) Compute n, A, alpha_i
    n = f / (2 - f)

    # A factor (truncated series for (1 + n^2/4 + n^4/64 + ...))
    # We'll keep up to n^4 for typical accuracy.
    A_bar = (1 + (n**2)/4 + (n**4)/64)
    A = (a / (1 + n)) * A_bar

    # Some alpha_i expansions (truncated to 3 terms)
    alpha_1 = 0.5*n - (2/3)*n**2 + (37/96)*n**3
    alpha_2 = (1/48)*n**2 + (1/15)*n**3
    alpha_3 = (17/480)*n**3  # can add more if needed

    # 2) Convert lat, lon to radians; shift by central meridian
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lambda0 = math.radians(lambda0_deg)

    # 3) Compute "isometric latitude" psi (one version of the formula)
    #   For small lat, the simpler series expansions are also used, but let's do the direct approach:
    #   psi = asinh(tan(lat)) - e * atanh(e * sin(lat)), etc.
    #   We'll use a commonly cited formula for isometric latitude:
    t = math.tan(math.pi/4 + lat/2)
    part_e = ((1 - e*math.sin(lat)) / (1 + e*math.sin(lat)))**(e/2)
    psi = math.log(t) - math.log(part_e)

    # 4) xi' and eta'
    #   Delta lambda
    dlon = lon - lambda0

    xi_prime = math.atan2(math.sinh(psi), math.cos(dlon))
    eta_prime = math.atanh(math.sin(dlon) / math.cosh(psi))

    # 5) Apply the alpha_i sums for x, y
    def sincosh_sum(i):
        # sin(2 i xi'), cosh(2 i eta')
        return math.sin(2*i*xi_prime) * math.cosh(2*i*eta_prime)

    def sinsinh_sum(i):
        # sin(2 i xi'), sinh(2 i eta')
        return math.sin(2*i*xi_prime) * math.sinh(2*i*eta_prime)

    # Summation for x (eta') and y (xi')
    # x = k0 * A [ eta' + Σ alpha_i sin(2 i xi') cosh(2 i eta') ] + E0
    # y = k0 * A [ xi' + Σ alpha_i sin(2 i xi') sinh(2 i eta') ] + N0

    # We'll just sum up i=1..3 for demonstration:
    sum_x = (alpha_1 * sincosh_sum(1)
           + alpha_2 * sincosh_sum(2)
           + alpha_3 * sincosh_sum(3))

    sum_y = (alpha_1 * sinsinh_sum(1)
           + alpha_2 * sinsinh_sum(2)
           + alpha_3 * sinsinh_sum(3))

    x = E0 + k0 * A * (eta_prime + sum_x)
    y = N0 + k0 * A * (xi_prime + sum_y)

    return (x, y)