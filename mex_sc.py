import os
import spiceypy
import numpy as np
import scipy as sp
import mex
import celsius
import glob

from os import getenv
from os.path import expanduser

import time as py_time
import pickle

import pylab as plt
import matplotlib as mpl

output_directories = [
    '/Users/dave/Documents/Mars/',
    '/homelocal/data_maris/mex_results/'
]

class MEXException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# class AISOrbitInfo(object):
#     """docstring for AISOrbitInfo"""
#     def __init__(self, orbit=None):
#         super(AISOrbitInfo, self).__init__()

class AISOrbitInfo(): pass

def locate_data_directory():
    """Decide where to get data from"""
    return os.getenv('SC_DATA_DIR') + 'mex/'

def locate_output_directory():
    """Decide where to put results"""
    for directory in output_directories:
        if os.path.exists(directory):
            return directory
    raise MEXException("No output directory found")

def data_directory_is_complete(directory):
    if directory == valid_directories[-1]:
        return False
    return True

def check_create_file(f):
    d = os.path.dirname(f)
    if d:
        if not os.path.exists(d):
            os.makedirs(d)

    if not os.path.exists(f):
        ff = open(f, 'w')

# Some constants
# Seidelmann 2007
mars_mean_radius_km = 3389.50
mars_radius_km = mars_mean_radius_km
mars_eq_radius_km = 3396.19
mars_polar_radius_km = 3376.20

mars_orbital_period_days = 686.971
mars_rotation_period_days = 1.025957
mars_mass_kg = 6.4185e23

spice_furnsh_done = False
last_spice_time_window = 'NONE_INITIALIZED'

REQUIRED_KERNELS = ['/lsk/NAIF*.TLS',
                    '/pck/DE*-MASSES.TPC',
                    '/pck/MARS_IAU2000_V*.TPC',
                    '/pck/PCK*.TPC',
                    '/pck/EARTH_*.BPC',
                    '/fk/EARTH_TOPO_050714.TF',
                    '/fk/EARTHFIXEDIAU.TF',
                    '/fk/EARTHFIXEDITRF93.TF',
                    '/fk/ESTRACK_V*.TF',
                    '/fk/MEX_V*.TF',
                    '/fk/NEW_NORCIA_TOPO.TF',
                    '/fk/RSSD*.TF',
                    '/spk/DE*.BSP',
                    '/spk/ORMF______*.BSP']

def check_spice_furnsh(*args, **kwargs):
    load_kernels(*args, **kwargs)

def load_kernels(time=None, force=False, verbose=False,
                load_all=False, keep_previous=False):
    """Load spice kernels, with a stateful thing to prevent multiple calls"""
    # global last_spice_time_window
    last_spice_time_window = getattr(spiceypy, 'last_spice_time_window', 'MEX:NONE')

    if load_all:
        # Launch to now + 10 yrs
        start = celsius.spiceet("2003-06-01T00:00")
        finish = celsius.spiceet(mex.now() + 10.*86400.*365.)

    if time is None:
        start = None
        finish = None
        start_str = 'NO_TIME_SET'
        finish_str = ''
        start_int=-999999
        finish_int=-999999
    else:
        if hasattr(time, '__len__'):
            start = time[0]
            finish = time[-1]

        else:
            start = time
            finish = time
        start_str = celsius.utcstr(start, 'ISOC')
        finish_str = celsius.utcstr(finish, 'ISOC')
        start_int = int(start_str[2:4] + start_str[5:7] + '01')
        finish_int = int(finish_str[2:4] + finish_str[5:7] + '01')
        start_str = '%06d' % start_int
        finish_str = '%06d' % finish_int

    this_spice_time_window = start_str + finish_str

    if not 'NONE' in last_spice_time_window:
        if last_spice_time_window == this_spice_time_window:
            if verbose:
                print('LOAD_KERNELS: Interval unchanged')
            return

        if keep_previous:
            if verbose:
                print('LOAD_KERNELS: Keeping loaded kernels')
            return

    spiceypy.last_spice_time_window = 'MEX:'+this_spice_time_window

    spiceypy.kclear()

    try:
        kernel_directory = mex.data_directory + 'spice'
        if verbose:
            print('LOAD_KERNELS: Registering kernels:')

        for k in REQUIRED_KERNELS:

            if '*' in k:
                files = glob.glob(kernel_directory + k)
                m = -1
                file_to_load = ''
                for f in files:
                    t = os.path.getmtime(f)
                    if t > m:
                        m = t
                        file_to_load = f
                if verbose:
                    print(file_to_load)
                spiceypy.furnsh(file_to_load)

            else:
                spiceypy.furnsh(kernel_directory + k)
                if verbose: print(kernel_directory + k)

        if start_int > -999999:
            # Load time-sensitive kenrels
            for f in glob.iglob(kernel_directory + '/spk/ORMM_T19_*.BSP'):
                this_int = int(f.split('_T19_')[1][:6])
                if this_int < start_int: continue
                if this_int > finish_int: continue
                spiceypy.furnsh(f)
                if verbose: print(f)

    except Exception as e:
        spiceypy.kclear()
        spiceypy.last_spice_time_window = 'MEX:NONE_ERROR'
        raise
    
    # print('LOAD_KERNELS: Loaded %s' % last_spice_time_window)

def unload_kernels():
    """Unload kernels"""

    # last_spice_time_window

    try:
        spiceypy.kclear()

        # But, we always want the LSK loaded.  This should be safe provided
        # a) celsius was loaded first (safe assertion, this code won't work
        # without it), meaning that the latest.tls was updated if needs be
        # b) uptime for this instance is less than the lifetime of a tls kernel
        # (years?)
        spiceypy.furnsh(
            getenv("SC_DATA_DIR", default=expanduser('~/data/')) + \
            'latest.tls'
        )

        spiceypy.last_spice_time_window = 'MEX:NONE_UNLOADED'
    except RuntimeError as e:
        spiceypy.last_spice_time_window = 'MEX:NONE_ERROR'
        raise e

load_spice_kernels = load_kernels # Synonym, innit
unload_spice_kernels     = unload_kernels

def position(time, frame='IAU_MARS', target='MEX', observer='MARS',
            correction='NONE'):
    """Wrapper around spiceypy.spkpos that handles array inputs, and provides useful defaults"""

    check_spice_furnsh(time)

    def f(t):
        try:
            pos, lt = spiceypy.spkpos(target, t, frame, correction, observer)
        except spiceypy.support_types.SpiceyError:
            return np.empty(3) + np.nan
        return np.array(pos)

    if hasattr(time, '__iter__'):
        x = np.array([f(t) for t in time]).T

    else:
        x = f(time)

    # unload_kernels()
    return x

def iau_mars_position(time):
    """An alias: position(time, frame='IAU_MARS')"""
    return position(time, frame='IAU_MARS')

def mso_position(time):
    """An alias: position(time, frame='MSO')"""
    return position(time, frame='MSO')

def reclat(pos):
    """spiceypy.reclat with a wrapper for ndarrays"""

    check_spice_furnsh(keep_previous=True)

    if isinstance(pos, np.ndarray):
        if len(pos.shape) > 1:
            return np.array([spiceypy.reclat(p) for p in pos.T]).T
    return spiceypy.reclat(pos)

def recpgr(pos, body="MARS"):
    """spiceypy.recpgr for mars, with a wrapper for ndarrays"""

    check_spice_furnsh(keep_previous=True)

    if body == "MARS":
       r, e = 3396.2, 0.005888934691714269
    else:
       raise NotImplemented("Unknown body: " + body)

    def f(p):
        return spiceypy.recpgr(body, p, r, e)
    if isinstance(pos, np.ndarray):
        if len(pos.shape) > 1:
            return np.array([f(p) for p in pos.T]).T
    return f(pos)

# Convert "WEST" longitudes to "EAST", which seems to be more commonly used.
# This is the reason for the non-multiplication of the longitude here,
# and the -1 in the corresponding iau_pgr_alt_lat_lon_position function
def iau_r_lat_lon_position(time, **kwargs):
    """" Return the position of MEX at `time`, in Radial Distance/Latitude/EAST Longitude.
    No accounting for the oblate spheroid is done, hence returning radial distance [km]"""
    tmp = reclat(position(time, frame = 'IAU_MARS', **kwargs))
    out = np.empty_like(tmp)
    out[0] = tmp[0]
    out[1] = np.rad2deg(tmp[2])
    out[2] = np.rad2deg(tmp[1])
    return out

# Convert "WEST" longitudes to "EAST", which seems to be more commonly used.
# This is the reason for the -1 * multiplication of the longitude here, and the *1 in the
# corresponding iau_pgr_alt_lat_lon_position function
def iau_pgr_alt_lat_lon_position(time, **kwargs):
    """ Return the position of MEX at `time`, in Altitude/Latitude/EAST Longitude.
    This accounts also for the oblate-spheriod of Mars"""
    tmp = recpgr(position(time, frame = 'IAU_MARS', **kwargs))
    out = np.empty_like(tmp)
    out[0] = tmp[2]
    out[1] = np.rad2deg(tmp[1])
    out[2] = 360. - 1. * np.rad2deg(tmp[0]) #Convert to EAST LONGITUDE, RECPGR returns in [0, 2pi], so add 360.
    return out

# Some notes on longitudes:
# 2014-08-11
# To give consistent results, outputs of recpgr need to be -ve

def mso_r_lat_lon_position(time, mso=False, sza=False, **kwargs):
    """Returns radial distance [km], lat and lon (EAST) in degrees.
    With `mso' set, return [r/lat/lon], [mso x/y/z [km]].
    With `sza' set, return [r/lat/lon], [sza [deg]].
    With both, return return [r/lat/lon], [mso x/y/z [km]], [sza [deg]]."""

    if sza:
        pos = position(time, frame = 'MSO', **kwargs)
        sza = np.rad2deg(np.arctan2(np.sqrt(pos[1]**2 + pos[2]**2), pos[0]))
        if isinstance(sza, np.ndarray):
            inx = sza < 0.
            if np.any(inx):
                sza[inx] = 180. + sza[inx]
        elif sza < 0.0:
            sza = 180. + sza

        tmp = reclat(pos)
        tmp_out = np.empty_like(tmp)
        tmp_out[0] = tmp[0]
        tmp_out[1] = np.rad2deg(tmp[2])
        tmp_out[2] = np.rad2deg(tmp[1])
        if mso:
            return tmp_out, pos, sza
        return tmp_out, sza

    else:
        pos = position(time, frame = 'MSO', **kwargs)
        tmp = reclat(pos)
        tmp_out = np.empty_like(tmp)
        tmp_out[0] = tmp[0]
        tmp_out[1] = np.rad2deg(tmp[2])
        tmp_out[2] = np.rad2deg(tmp[1])
        if mso:
            return tmp_out, pos
        return tmp_out

def solar_longitude(t, body="MARS", correction="NONE"):
    """Calculate solar longitude, default Mars. Nb. PlanetoCENTRIC coordinates, and this is NOT a synonym for sub-solar longitude (PlanetoGRAPHIC longitude).  Defined as Ls = solar_longitude = 0 deg at NH vernal equinox, 90 deg a NH summer solstice etc."""

    check_spice_furnsh(t)

    def f(_t):
        return spiceypy.lspcn(body, _t, correction)

    if hasattr(t, '__iter__'):
        return np.array([f(_t) for _t in t])
    else:
        return f(t)

def sub_solar_longitude(et):

    check_spice_furnsh(et)

    def f(t):
        pos, lt = spiceypy.spkpos("SUN", t, 'IAU_MARS', "NONE", "MARS")
        return np.array(pos)

    def func(time):
        if hasattr(time, '__iter__'):
            return np.array([f(t) for t in time]).T
        else:
            return f(time)

    tmp = recpgr(func(et))
    return np.rad2deg(tmp[0])

def sub_solar_latitude(et, body='MARS'):
    check_spice_furnsh(et)

    def f(t):
        pos, lt = spiceypy.spkpos("SUN", t, 'IAU_' + body, "NONE", body)
        return np.array(pos)

    def func(time):
        if hasattr(time, '__iter__'):
            return np.array([f(t) for t in time]).T
        else:
            return f(time)

    tmp = recpgr(func(et))
    return np.rad2deg(tmp[1])

def modpos(x, radians=False, min=0.):
    if radians:
        return (x % (2. * np.pi) + 2. * np.pi) % (2. * np.pi)
    return (x % (360.) + 360.) % 360.

def describe_func_call(f):
    def fint(*args, **kwargs):
        print('~'*5)
        print('Function: ' + f.__name__)
        print('Inputs:')
        for a in args:
            print(a)
        for k, v in kwargs.items():
            print(str(k) + ' = ' + str(v))

        rval = f(*args, **kwargs)
        print('Returned:')
        print(rval)
        print('~'*5)
        return rval
    return fint

def mex_mission_phase(time, long_name=False):
    """Return the mission phase designator for time"""
    orbit = mex.orbits[time]
    if not orbit:
        return

    # not sure about 7689 - could also be 7669.  Seems to be some missing data or sc dead time between extension 2 and 3?
    starts = [0, 2540, 4800, 7690, 8320]
    names = ['', 'EXT1', 'EXT2', 'EXT3', 'EXT4*']
    if long_name:
        names = ['PRIME', 'EXTENSION 1', 'EXTENSION 2', 'EXTENSION 3', 'EXTENSION 4*']

    result = None
    for o, n in zip(starts, names):
        if o < orbit.number:
            result = n
    return result

def read_mex_orbits(fname):
    """docstring for read_mex_orbits"""

   # 11  2004 JAN 11 02:23:03    1/0021867775.48376  2004 JAN 11 07:24:05   238.65   -11.59   270.18    -2.24      274.35    86.64   0.717   228.71   357.75   222497088.5     12965.68


    utc_date = np.dtype([('Year','int'), ('MonthStr','a',3),
        ('Day','int'),('Time','a',8)])

    print('Reading %s ... ' % fname, end=' ')

    orbit_list = celsius.OrbitDict()
    lines = 0
    with open(fname) as f:
        f.readline()
        f.readline()

        for line in f.readlines():
            try:
                parts = line.lstrip().split('  ')
                n = int(parts[0])
                tmp = celsius.Orbit(
                        n,
                        celsius.utcstr_to_spiceet(parts[4]),
                        celsius.utcstr_to_spiceet(parts[1]),
                        name='MEX')

                orbit_list[n] = tmp
                lines += 1
            except Exception as e:
                if 'Unable to determine' in parts[4]:
                    break
                else:
                    raise

    for k in orbit_list:
        if k > 1:
            orbit_list[k].start = orbit_list[k-1].apoapsis
        else:
            orbit_list[k].start = orbit_list[k].periapsis - 1.0
        orbit_list[k].finish = orbit_list[k].apoapsis

    print(' Read %d lines' % lines)



    return orbit_list


def read_all_mex_orbits(recompute=False, allow_write=True, verbose=False):

    require_write = False
    fname = mex.data_directory + 'orbits.pck'

    if not recompute:
        try:
            age = (py_time.clock() - os.path.getctime(fname)) / 86400
            if age > 10:
                print("Pickled orbits file is %f days old - recomputing" % age)
                require_write = True
        except OSError as e:
            print("Pickled orbits not found - generating")
            age = -1
            require_write = True
        else:
            if verbose:
                print('Restoring pickled orbits from %s' % fname)

            try:
                with open(fname, 'rb') as f:
                    orbits = pickle.load(f)
                    return orbits
            except Exception as e:
                print('Encountered error reading stored orbit file:')
                print('\t', e)
                require_write = True
    else:
        require_write = True

    processed = []
    for pattern in ['ORMF_*.ORB', 'ORMM_MERGED_*.ORB']:
        max_version = -1
        max_str = ''
        files = glob.glob(mex.data_directory + 'spice/orbnum/' + pattern)
        for f in files:
            if int(f[-9:-4]) > max_version:
                max_version = int(f[-9:-4])
                max_str = f
        # print max_str, max_version
        processed.append(read_mex_orbits(max_str))

    predicted, real = processed

    if verbose:
        print('Predicted orbits read     : %d - %d (%d)' % (min(predicted.keys()), max(predicted.keys()), len(predicted)))
        print('Reconstructed orbits read: %d - %d (%d)' % (min(real.keys()), max(real.keys()), len(real)))

    overwritten = []
    keys = list(real.keys())
    for number, orbit in predicted.items():

        if number not in keys:
            real[number] = orbit
            overwritten.append(number)

    if verbose:
        print('Inserted %d predicted values (min = %d):' % (len(overwritten), min(overwritten)))

    if allow_write and require_write:

        with open(fname, 'wb') as f:
            pickle.dump(real, f)
            print('Pickled orbits to %s' % fname)

    return real

def plot_mex_orbits_bar(start=None, finish=None, ax=None, height=0.02, number_every=10, sharex=False, labels=True):
    """docstring for plot_mex_orbits_bar"""

    fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    xlims = plt.xlim()

    p = ax.get_position()
    if sharex:
        if isinstance(sharex, bool):
            sharex = ax
    else:
        sharex = None
    new_ax = fig.add_axes([p.x0, p.y1, p.x1 - p.x0, height], sharex=sharex)

    if start is None:
        start = xlims[0]
    if finish is None:
        finish = xlims[1]

    plt.xlim(start, finish)
    plt.ylim(0., 1.)

    x = np.array([1., 0., 0., 1., 1.])
    y = np.array([1., 1., 0., 0., 1.])

    orbit_count = 0
    orbit_list = list(mex.orbits.values())
    for o in orbit_list:
        if o.number % 2 == 0:
            continue
        if (o.start < finish) & (o.finish > start):
            dx = o.finish - o.start
            plt.fill(x * dx + o.start, y, 'k', edgecolor='none')
            orbit_count = orbit_count + 1

    if number_every is None:
        number_every = int(orbit_count / 10)

    if number_every:
        numbered_orbits = [o for o in orbit_list if (o.number % number_every) == 0]

    ticks = [o.periapsis for o in numbered_orbits]
    nums  = [o.number for o in numbered_orbits]
    new_ax.yaxis.set_major_locator(mpl.ticker.NullLocator())

    new_ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    new_ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(nums))
    # new_ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, steps=[1,2,5,10]))
    new_ax.tick_params(axis='x', direction='out', bottom=False, top=True, labelbottom=False, labeltop=True)
    if labels:
        plt.ylabel("MEX", rotation='horizontal')

    plt.sca(ax)
    return new_ax
