import numpy as np
import numpy.linalg

import matplotlib.pylab as plt
import matplotlib as mpl
from scipy.interpolate import griddata

import celsius
import datetime
import os
import subprocess

import sys
import struct
import stat
import copy as copy_module

import mars

import celsius
# import scipy.ndimage.filters as filters
from scipy.signal import detrend
import scipy.ndimage.morphology as morphology

import skimage.morphology
import skimage.measure

import pickle

# import json

import urllib.request, urllib.error, urllib.parse
import tempfile

# from calibration import AISEmpiricalCalibration
# calibrator = None

import mex

__auto_version__ = '1.0'

# data_directory = mex.locate_data_directory()
data_directory = os.getenv('SC_DATA_DIR') + 'mex/'

debug = False

ais_spacing_seconds = 7.543
ais_number_of_delays = 80
ais_delays = (np.arange(80) * 91.4 + 91.4 + 162.5) * 1E-6
ais_max_delay = 7474.5 * 1.0E-6
ais_min_delay = 253.9 * 1.0E-6

# Logarithms here
ais_vmin = -16
ais_vmax = -13
ais_vmax = -9
# ais_vmax_actual = -9

speed_of_light_kms = 299792.458

DIGITIZATION_BANNED_KEYS = ['frequencies', 'delays', 'data', 'ionogram',
                    'threshold_data', '_ion_data', '_cyc_data', '_morphology_fp_locations']
DIGITIZATION_ARRAY_KEYS  = ['traced_delay', 'traced_frequency', 'traced_intensity',
                                        'morphology_fp_local_selected_f', 'td_cyclotron_selected_t']

LAST_KNOWN_GOOD_AIS_ORBIT = 12216

# Frequency ranges where MARSIS radiated power > 30dB, according to Jordan '08
# Tuple of (start, stop) for each band (in MHz)
POWER_BANDS_30DB = ((1.3, 2.25), (2.5, 3.5), (3.5, 4.5), (4.5, 5.1))

# These were wrong up untill 2013-05-02.  Thereafter, corrected myself
FREQUENCY_TABLE_FIXED_AFTER = 185378178.18399999
FREQUENCY_TABLE_FIXED_AFTER_ORBIT = 2366

# Should replace these with physical values not just approximations
def fp_to_ne(fp, error=None):
    if error is not None:
        return ((fp / 8980.0)**2.0, 2.0 * error * fp/8980.0**2.0)
    return (fp / 8980.0)**2.0

def ne_to_fp(ne, error=None):
    if error is not None:
        return np.sqrt(ne) * 8980.0, 8980. * 0.5 * error / np.sqrt(ne)
    return np.sqrt(ne) * 8980.0

def td_to_modb(td, error=None):
    if error is not None:
        return (2.0 * np.pi / (1.758820150E11 * td), 2.0 * np.pi / (1.758820150E11 * td**2.) * error)
    return 2.0 * np.pi / (1.758820150E11 * td)

def modb_to_td(b):
    return 2.0 * np.pi / (1.758820150E11 * b)

ais_fp_local_sensible = list(map(ne_to_fp, (1., 10000.)))
ais_td_cyclotron_sensible = (91.4E-6, ais_max_delay)
ais_auto_cyclotron_max_err = modb_to_td(3E-9)

def arg_peak2(x):
    deriv = x[1:] - x[:-1]
    s = np.argsort(deriv)#[::-1]
    s = s[0:s.shape[0] + 1 // 10]
    return s[np.argmax(x[s])]

def arg_peak3(x):
    y = x.copy()
    y[1:] = x[1:] - x[:-1]
    y[:-1] = y[:-1] - x[1:]
    plt.plot(y-5.,'g-')
    return np.argmax(y)


def arg_peak(x, width=4, no_peaks=1, full_output=False, threshold=None, absolute_threshold=False):
    out = np.zeros_like(x)
    for i in range(width+1, out.shape[0]-width-1):
        out[i] = x[i] - 0.5 * (np.mean(x[i-width:i]) + np.mean(x[i+1:i+width+1]))
        # out[i] = (x[i] - 0.5 * (np.mean(x[i-width:i]) + np.mean(x[i+1:i+width+1])))

    # if threshold is not None:
    #     tmp_out = out.copy()
    #     tmp_out -= np.mean(tmp_out)
    #     tmp_out /= np.std(tmp_out)
    #     vals, = np.where(tmp_out > threshold)
    #     h = [x for x in range(out.shape[0]) if x not in vals]
    #     out -= np.mean(out[h])
    #     out /= np.std(out[h])
    #     if full_output:
    #         return vals, out
    #     return vals

    vals = np.argsort(out)[-no_peaks:]
    h = [x for x in range(out.shape[0]) if x not in vals]
    if not absolute_threshold:
        out -= np.mean(out[h])
        out /= np.std(out[h])

    if threshold is not None:
        vals = np.array([v for v in vals if out[v] > threshold], dtype=np.int32)
    if full_output:
        return vals, out
    return vals

def arg_peak_test(*args, **kwargs):
    plt.close("all")
    plt.figure()
    plt.subplot(211)
    x = np.random.randn(100)
    x = x + (np.arange(100)/100.)**2. * 4.
    sigma = 4.
    x[50] = np.abs(x[50] + sigma)
    x[80] = np.abs(x[80] + sigma)
    # x = x * (np.arange(100)/100.) * 0.3
    plt.plot(x,'k-')

    plt.subplot(212)
    y, out = arg_peak(x, full_output=True, *args, **kwargs)
    print(y)

    dx = 0.5 * fwhm(x, peak_inx=y)
    print('dx = ', dx)
    plt.plot(out)
    plt.plot(y, x[y], 'ro')
    plt.vlines(np.array((y[0]-dx, y[0]+dx)), *plt.ylim())
    plt.hlines(3., *plt.xlim())
    plt.show()

def rebin_peaks(x):
    n = x.shape[0]/2
    out = np.empty(n, dtype=x.dtype)
    for i in range(0, n):
        out[i] = np.amax(x[i:i+2])

    return out

def proc_peaks(x, max_iter=3,edges=False):
    last_out = x.astype(np.float64)
    m = np.amin(last_out)
    if not edges:
        last_out[0] = m
        last_out[-1] = m
    out = None
    for i in range(max_iter):
        del out
        out = last_out.copy()
        for j in range(1,out.shape[0] - 1):
            # print j, (2. * last_out[j]), (last_out[j-1] + last_out[j+1]),  not (2. * last_out[j]) > (last_out[j-1] + last_out[j+1])
            # if not (2. * last_out[j]) > (last_out[j-1] + last_out[j+1]):
            #     out[j] = 0.
            if not ( last_out[j]) > max((last_out[j-1], last_out[j+1])):
                out[j] = 0.
            # This is such a hack, and needs to be done better
            if last_out[j] == last_out[j+1]:
                out[j+1] = last_out[j] * 0.99999
                # out[j] = last_out[j]

        last_out = out
    return last_out

def fwhm(x, y=None, peak_inx=None, *args, **kwargs):
    if y is None:
        y, x = x, np.arange(x.shape[0])
    if x.shape != y.shape:
        raise ValueError("x and y dimensions must match")
    if len(x.shape) > 1:
        raise ValueError("only 1-d arrays please")
    if peak_inx is None:
        peak_inx = arg_peak(y, *args, **kwargs)

    hm = 0.5 * y[peak_inx] + np.mean(y)
    inx = np.hstack((np.arange(0,peak_inx), np.arange(peak_inx+1, x.shape[0])))

    for i in range(1, min((peak_inx, y.shape[0] - peak_inx))):
        if 0.5 * (y[peak_inx + i] + y[peak_inx-i]) < hm:
            tmp = (x[peak_inx + i] - x[peak_inx - i])
            if isinstance(tmp, np.ndarray):
                return tmp[0]
            else:
                return tmp
    return np.inf

def laminated_delays(d, f, fp_local, altitude=None):
    """Do the inversion - Morgan '08 method
    2012-11-07: all inputs in SI: seconds, Hz and m"""
    const_c = 2.99792458E8 #m/s

    # f, fp_local in hz
    # d in sec
    g = np.hstack((fp_local, f[:-1]))

    def func(i,j):
        return 1.0/const_c * np.log(
            (1.0 - np.sqrt(1.0 - (f[i]/f[j])**2.)) / (1.0 + np.sqrt(1.0 - (f[i]/f[j])**2.0)) *
            (1.0 + np.sqrt(1.0 - (g[i]/f[j])**2.)) / (1.0 - np.sqrt(1.0 - (g[i]/f[j])**2.0))
        )

    m = np.zeros((d.shape[0], d.shape[0]))
    for j in range(d.shape[0]):
        for i in eval("range(j+1)"):
            m[j,i] = func(i, j)
    # m = m.T

    # print "DET:", np.linalg.det(m)
    alpha = np.linalg.solve(m, d)**-1.0

    z = np.hstack((1.0/alpha * np.log(f / g))) * 2.0

    z = np.cumsum(z)

    z = np.hstack((0., z))
    f = np.hstack((fp_local, f))

    if altitude is not None:
        return (altitude - z/2.0, fp_to_ne(f))
    else:
        return (-z/2.0, fp_to_ne(f))

def laminated_test():
    d = np.ones(10) * 3. * 1.0E-3
    f = np.linspace(1.0, 3.0, d.shape[0]) * 1.0E6
    fp0 = 0.2 * 1.0E6

    print('-'*10)
    print(d)
    print(f)
    print(fp0)
    print('-'*10)

    print(laminated_delays(d, f, fp0))

def remove_none_edge_intersecting(img, edge=0, width=1, as_list=False):

    mask = np.zeros(img.shape,dtype=int)
    out = np.zeros(img.shape,dtype=int)
    # print '--->', img.sum()

    if edge == 0:
        mask[:,0:0+width] = 1
    elif edge == 1:
        mask[:,-1-width:-1] = 1
    elif edge == 2:
        mask[0:width,:] = 1
    elif edge == 3:
        mask[-1-width:-1,:] = 1
    else:
        raise ValueError('Edge is duff')

    fp_list = []

    s = skimage.measure.label(img.astype(int))
    s_set = np.unique(s * mask)
    if s_set.sum() > 0:
        for v in s_set:
            q = (s == v)
            if np.all(img[q]):
                out[s == v] = 1
                tmp = np.zeros_like(img)
                tmp[s==v] = 1
                fp_list.append((np.mean(tmp.nonzero()[1]), np.mean(tmp.nonzero()[0])))

    # if as_list:
    #     return img, sorted(fp_list, key=lambda x: x[0])

    if as_list:
        # Sort in frequency
        return out, sorted(fp_list, key=lambda x: x[0])

    return out

class AISFileManager(object):
    """docstring for AISFileManager"""
    def __init__(self, remote='None', local='DEFAULT',
                    copy_to_local=True, verbose=False,
                    overwrite=False, brain=True):
        super(AISFileManager, self).__init__()

        self.remote = remote
        self.copy_to_local = copy_to_local
        self.verbose = verbose
        self.overwrite = overwrite
        self.brain = brain

        if os.uname()[1] == 'spis': self.brain = False
        if os.uname()[1] == 'brain': self.brain = False # Hacks!

        self.local = os.getenv('SC_DATA_DIR') + 'mex/marsis/ais/'

        self._known_empty_orbits_file = self.local + 'ais_known_empty_orbits.pk'
        if os.path.exists(self._known_empty_orbits_file):
            with open(self._known_empty_orbits_file, 'rb') as f:
                self._known_empty_orbits = pickle.load(f)
        else:
            print('Creating ais_known_empty_orbits.pk')
            self._known_empty_orbits = []
            with open(self._known_empty_orbits_file, 'wb') as f:
                pickle.dump(self._known_empty_orbits, f)

        if remote.lower() == 'iowa':
            self.remote_url = 'http://www-pw.physics.uiowa.edu/plasma-wave/marsx/restricted/DATA/ACTIVE_IONOSPHERIC_SOUNDER/'
            username, password = os.getenv('MEX_IOWA_USER_PASS').split(':')
            self.passman = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            self.passman.add_password(None, self.remote_url, username, password)
        else:
            self.remote_url = ''

    def get_file(self, time, remote=None, *args, **kwargs):
        if remote is None:
            remote = self.remote

        try:
            f = self._get_local_file(time, *args, **kwargs)
        except IOError as e:
            if not remote:
                raise e
            try:
                f = self._get_remote_file(time, *args, **kwargs)
            except IOError as e:
                print(e)
                raise IOError("Couldn't locate the appropriate file, either here or remotely")
        return f

    def _get_brain_file(self, time, copy_to_local=None):
        if copy_to_local is None:
            copy_to_local = self.copy_to_local

        if (time in self._known_empty_orbits) and (time < LAST_KNOWN_GOOD_AIS_ORBIT):
            if self.verbose:
                print('Orbit %d is known to be empty' % time)
            raise IOError("No data for orbit %d known already" % time)

        brain_fname = 'dja@brain.irfu.se:/data/div/data_maris/mex/marsis/ais/RDR%dX/FRM_AIS_RDR_%d.DAT' % (time // 10, time)
        fd, temp_f_name = tempfile.mkstemp(suffix='_aistmp.dat')
        command = ('scp', brain_fname, temp_f_name)

        if self.verbose:
            print('Fetching %s' % brain_fname)
        try:
            # os.spawnvp(os.P_WAIT, command[0], command)
            subprocess.check_call(command)
        except subprocess.CalledProcessError as e:
            raise IOError("Retrieval from brain failed: %s" % str(e))

        if copy_to_local:
            fname = self.local + 'RDR%dX/FRM_AIS_RDR_%d.DAT' % (time // 10, time)

            if os.path.exists(fname) and not self.overwrite:
                if self.verbose:
                    print('Local file already %s already exists, and not overwriting' % fname)

            d = os.path.dirname(fname)
            if d and not os.path.exists(d):
                if self.verbose:
                    print('Creating %s' % d)
                os.makedirs(d)

            command = ('mv', temp_f_name, fname)

            try:
                # os.spawnvp(os.P_WAIT, command[0], command)
                subprocess.check_call(command)
            except subprocess.CalledProcessError as e:
                print(e)
                raise IOError("Error moving file to %s" % fname)

            return fname
        else:
            return temp_f_name

    def _get_remote_file(self, time, copy_to_local=None):

        if copy_to_local is None:
            copy_to_local = self.copy_to_local

        if self.brain:
            try:
                fname = self._get_brain_file(time, copy_to_local=copy_to_local)
                return fname
            except Exception as e:
                print("Searching SPIS - failed: %s" % str(e))

        if (time in self._known_empty_orbits) and \
                                (time < LAST_KNOWN_GOOD_AIS_ORBIT):
            if self.verbose:
                print('Orbit %d is known to be empty' % time)
            raise IOError("No data for orbit %d known already" % time)

        if self.remote_url:
            url = self.remote_url + 'RDR%dX/FRM_AIS_RDR_%d.DAT' \
                    % (time // 10, time)
            try:
                if self.verbose:
                    print('Trying %s' % url)

                self.authhandler = urllib.request.HTTPBasicAuthHandler(self.passman)
                self.opener = urllib.request.build_opener(self.authhandler)
                urllib.request.install_opener(self.opener)
                pagehandle = urllib.request.urlopen(url)
                print('Fetching %s ... ' % url)
                thepage = pagehandle.read()

            except urllib.error.HTTPError as e:
                if e.code == 404:
                    with open(self._known_empty_orbits_file, 'wb') as f:
                        self._known_empty_orbits.append(time)
                        pickle.dump(self._known_empty_orbits, f)

                raise IOError('Could not read %s' % url)
            if self.verbose:
                print("                     ... done")

            if copy_to_local:
                fname = self.local + 'RDR%dX/FRM_AIS_RDR_%d.DAT' \
                        % (time // 10, time)

                if os.path.exists(fname) and not self.overwrite:
                    if self.verbose:
                        print("""Local file already %s already exists,
and not overwriting""" % fname)

                d = os.path.dirname(fname)
                if d and not os.path.exists(d):
                    if self.verbose:
                        print('Creating %s' % d)
                    os.makedirs(d)

                f = open(fname, 'wb')
                print('Writing %s' % fname)
                f.write(thepage)
                f.close()
                return fname
            else:
                fd, temp_f_name = tempfile.mkstemp(suffix='_aistmp.dat')
                with open(temp_f_name, 'w') as f:
                    f.write(thepage)
                print('Wrote %s to %s' % (url, temp_f_name))
                return temp_f_name
        else:
            raise IOError("Remote file access not set up")


    def _get_local_file(self, time):
        fname = self.local + 'RDR%dX/FRM_AIS_RDR_%d.DAT' % (time // 10, time)
        if os.path.exists(fname):
            stats = os.stat(fname)
            if int(stats[stat.ST_SIZE]) > 0:
                if self.verbose:
                    print("Found: %s" % fname)
                return fname
            else:
                if self.verbose:
                    print("Empty file: %s" % fname)
                raise IOError("Empty local file for orbit %d" % time)

        raise IOError("No local file found for %d" % time)

# Create a default instance
file_manager = AISFileManager(remote='iowa', verbose=True)

def read_ais(start, finish=None, input_format=None, verbose=False):

    start_orbit = mex.orbits[start]
    start_et = start_orbit.start
    finish_et = start_orbit.finish

    if finish is not None:
        finish_orbit = mex.orbits[finish]
        finish_et = finish_orbit.finish
    else:
        finish_orbit = start_orbit

    if debug or verbose:
        print('read_ais: start = %d, finish = %d' % (start_et, finish_et))

    orbits = list(range(start_orbit.number, finish_orbit.number + 1))

    ionograms = []
    for orbit in orbits:
        fname = file_manager.get_file(orbit)
        if fname:
            if debug or verbose:
                print('read_ais: reading ' + fname)
            ionograms.extend( read_ais_file(fname, verbose=verbose) )

    if not isinstance(start, int):
        ionograms = [i for i in ionograms if (i.time >= start_et and i.time <= finish_et)]

    return ionograms

def read_ais_file(file_name, verbose=False, debug=True):
    """Reads an ais orbit file, returns a list of ionograms"""
    # 400 bytes total
    # Refer to AIS_FORMAT.FMT
    # By object:
    # SCLK_SECOND
    # SCLK_PARTITION
    # SCLK_FINE
    # SCET_DAYS
    # SCET_MSEC
    # -- PAD to 25: 8 bytes
    # SCET_STRING (24)
    # PROCESS_ID
    # INSTRUMENT_MODE - NOTE 2 x 4 bits
    # -- Pad to 60: 9 bytes
    # TRANSMIT_POWER
    # FREQUENCY_TABLE_NUMBER
    # FREQUENCY_NUMBER
    # BAND_NUMBER
    # RECEIVER_ATTENUATION
    #  -- Pad to 65 - 77: 12 bytes
    # FREQUENCY
    # SPECTRAL_DENSITY
    ais_fmt = '>IHHII8x24cBB9xBBBBB12xf80f'
    ais_fmt_short = '>I2H2I8x24c2B9x5B12x81f'
    ais_fmt_size = struct.calcsize(ais_fmt_short)
    # ais_directory = '/Users/dave/data/mex/marsis/ais/'

    if ais_fmt_size != 400:
        raise Error("Format size should be 400 bytes.")

    ionogram_list = []

    if debug: verbose = True

    if file_name:
        try:
            f = open(file_name, 'rb')
            stats = os.stat(file_name)
            nsweeps = int(stats[stat.ST_SIZE] / ais_fmt_size)
            tmp_ionogram = None
            freq_inx = 0
            last_frequency = 1.0E99
            pcount = 0
            if verbose:
                print(file_name, nsweeps, nsweeps/160)


            if nsweeps == 0:
                if verbose:
                    print('Empty file detected: %s' % file_name)
                f.close()
                os.remove(file_name)
                return []

            for i in range(nsweeps):
                s = f.read(ais_fmt_size)
                t = struct.unpack(ais_fmt_short, s)
                this_frequency = t[36]

                if t[36] <= last_frequency:
                    if tmp_ionogram != None:
                        # 2012-07-16: Changed this to use the UTC STRING
                        # print 'AIS:' + ''.join(t[5:26])
                        # tmp_ionogram.time = celsius.time_convert(t[3] + t[4] / 86400000.0,
                        #     input_format='AISSCET')
                        ionogram_list.append(tmp_ionogram)
                    tmp_ionogram = Ionogram()
                    tmp_ionogram.time = celsius.spiceet(b''.join(t[5:26]))
                    freq_inx = 0

                tmp_ionogram.frequencies[freq_inx] = this_frequency
                tmp_ionogram.data[:, freq_inx] = t[37:]
                tmp_ionogram.empty = False
                last_frequency = this_frequency
                freq_inx = freq_inx + 1

            # Force storing of the last one
            if ionogram_list[-1].time == tmp_ionogram.time:
                raise RuntimeError()
            ionogram_list.append(tmp_ionogram)

            f.close()
        except IOError as e:
            print(e)
            return []

    return ionogram_list

class Ionogram(object):
    """docstring for Ionogram"""
    def __init__(self, time = None, orbit = None, number = None, data = None):
        super(Ionogram, self).__init__()


        self.time = celsius.spiceet(time)
        self.orbit = orbit
        self.number = number
        self.data = data

        self.empty = False
        self.linear_frequency = False

        self.frequencies = np.empty(160)
        self.delays = (np.arange(80) * 91.4 + (91.4 + 162.5)) * 1E-6  # Cf. Morgan

        if self.data is None:
            self.empty = True
            self.data =np.empty((80, 160))

        self.fp_local = None
        self.fp_local_error = None
        self.fp_local_phase = None
        self.altitude = None
        if self.time is not None:
            self.find_position()
        else:
            self.mex_altitude = np.inf
            self.sza = 0.0
        # automatically load from some place?
        self.digitization = IonogramDigitization()

    def get_save_state(self):
        # Check saved status
        return False

    def find_position(self):
        pos, mso_pos, sza = mex.mso_r_lat_lon_position(float(self.time), sza=True, mso=True) # all in degrees
        self.mso_r_lat_lon_position = pos
        self.mso_pos = mso_pos
        self.sza = sza
        self.iau_pgr_alt_lat_lon_position = mex.iau_pgr_alt_lat_lon_position(float(self.time))
        self.mex_altitude = self.iau_pgr_alt_lat_lon_position[0]

    def check_frequency_table(self, validate=False):
        # print self.time < celsius.utcstr_to_spiceet("2005-03-14")
        if self.time is not None:
            if self.time <  celsius.utcstr_to_spiceet("2005-03-14"): # Replace with 14 Mar 2005 (according to cookbook)
                return False
                # raise NotImplementedError("Frequency table might be dodgy for times prior to 14 MARCH 2005, but the code doesn't do anything yet about this")
                # Validate the frequency table used
            return True
        else:
            return False
            # raise ValueError("Can't check if we don't know what the time is... [self.time is None]")

    def plot(self, ax=None, colorbar=True, vmin=ais_vmin, vmax=ais_vmax,
                    color='blue', overplot_digitization=True, title=None,
                    errors=True, show_thresholded_data=False, no_title=False,
                    overplot_model=True, altitude_range=None, verbose=False,
                    labels=True, overplot_expected_ne_max=False):
        """docstring for Ionogram plot"""

        if not self.linear_frequency:
            if verbose: print('Ionogram.plot: Interpolating')
            self.interpolate_frequencies()
            # raise mex.MEXException('Need linear frequencies')

        if ax == None:
            ax = plt.gca()
        plt.sca(ax)
        ax.set_axis_bgcolor('black')

        if not np.isfinite(self.mex_altitude):
            if verbose: print('Position')
            self.mex_altitude = mex.iau_pgr_alt_lat_lon_position(
                        float(self.time))[0]

        # milliseconds and megahertz
        extent = (
                    min(self.frequencies) / 1.0E6,
                    max(self.frequencies) / 1.0E6,
                    max(self.delays) * 1.0E3,
                    min(self.delays) * 1.0E3
                )

        data = np.log10(self.data)

        if show_thresholded_data:
            if not hasattr(self, '_cyc_data'):
                if verbose: print('Ionogram.plot: Generating binary arrays')
                self.generate_binary_arrays()
            data = self._cyc_data * (vmax-vmin) + vmin

        if verbose: print('Ionogram.plot: Imshow')
        img = plt.imshow(data, vmin=vmin, vmax=vmax,
            interpolation='Nearest', extent=extent, origin='upper',aspect='auto')

        if labels:
            plt.xlabel(r'$f_s$ / MHz')

        if not altitude_range:
            if labels:
                plt.ylabel(r'$\tau_D$ / ms')
            plt.xlim(extent[0], extent[1])
            plt.ylim(extent[2], extent[3])
            plt.ylim(extent[2],0.0)
        else:
            if labels:
                celsius.ylabel('Alt* / km')
        ax.autoscale(enable=False, tight=True)

        if colorbar:
            if verbose: print('Ionogram.plot: Colorbar')
            cbar = plt.colorbar(img, ticks=[-16,-15,-14,-13,-12,-11,-10], cax=celsius.make_colorbar_cax(ax=ax))
            cbar.set_label(r'Log($V^2m^{-2}Hz^{-1}$)')

        tit = 'Ionogram: ' + celsius.utcstr(float(self.time), format='C')[:-4]
        if title is not None:
            tit = tit + '\n' + title

        if overplot_model or overplot_expected_ne_max:
            if verbose: print('Ionogram.plot: Model')

            pos, self.sza = mex.mso_r_lat_lon_position(
                            float(self.time), sza=True)
            self.mex_altitude = \
                mex.iau_pgr_alt_lat_lon_position(float(self.time))[0]

        if overplot_model:
            if verbose: print('Ionogram.plot: Model (2)')

            do_compute = True
            if hasattr(self, '_model_sza'):
                if abs(self.sza - self._model_sza) < 0.1: do_compute = False

            if do_compute:
                self._model_sza = self.sza
                if verbose: print('Ionogram.plot: Model 1')
                if self.sza < 90.:
                    if verbose:
                        print('Ionogram.plot: Model 1.5', self.mex_altitude, \
                                            np.deg2rad(self.sza))

                    self._model_delays, self._model_frequencies = \
                            mars.Morgan2008ChapmanLayer().ais_response(
                                    sc_altitude=self.mex_altitude,
                                    sc_theta=np.deg2rad(self.sza)
                            )
                else:
                    self._model_delays = np.empty(0)
                    self._model_frequencies = np.empty(0)
            else:
                if verbose: print('Ionogram.plot: Model 2')
                plt.plot(self._model_frequencies / 1E6,
                        -1. * self._model_delays * 1E3,
                        color='white', lw=2.)

        if overplot_expected_ne_max:
            if verbose: print('Ionogram.plot: Nemax')

            csza = np.cos(np.deg2rad(self.sza))
            ne_max = 1.58E5 * csza**0.5
            h_max = 133.6 - 8.9 * np.log(csza)

            plt.plot(ne_to_fp(ne_max)/1e6,
                2. * (self.mex_altitude - h_max) / speed_of_light_kms * 1.0E3,
                'wo', zorder=1000)

        if self.digitization and overplot_digitization:
            if verbose: print('Ionogram.plot: Digitization')
            d = self.digitization
            tit = tit + '\nDigitization: ' + celsius.utcstr(
                                float(d.timestamp), format='C')[:-7]

            # Expected ground delay
            x0, x1 = plt.xlim()
            plt.hlines(2.0 * self.mex_altitude / speed_of_light_kms * 1.0E3,
                3, x1, colors=color, linestyle='dotted', lw=2.)

            if np.isfinite(d.fp_local):
                phase = 0.0
                fs = np.arange(1,20) * d.fp_local / 1E6
                plt.vlines(fs, *(plt.ylim()),
                        color=color, linestyle='dashed', lw=2.)

                if errors:
                    plt.vlines(fs + d.fp_local_error/1.0E6, *(plt.ylim()),
                                    color=color, linestyle='dotted', lw=2.)
                    plt.vlines(fs - d.fp_local_error/1.0E6, *(plt.ylim()),
                                    color=color, linestyle='dotted', lw=2.)
                tit += ', Ne0 = %.1f +/- %.1f / cc' % \
                                fp_to_ne(d.fp_local,d.fp_local_error)

            if d.morphology_fp_local_selected_f.size > 0:
                for fs in d.morphology_fp_local_selected_f:
                    plt.vlines(fs / 1.0E6, *plt.ylim(),
                                        color=color, linestyle='solid', lw=2.)

            if d.traced_delay.size > 0:
                plt.plot(d.traced_frequency/1.0E6, d.traced_delay*1.0E3,
                    color=color, marker='+')

            if np.isfinite(d.td_cyclotron):
                xmin, xmax = plt.xlim()
                ymin, ymax = np.sort(plt.ylim())
                td = np.arange(0., ymax, d.td_cyclotron * 1.0E3)
                if td.shape[0] > 1:
                    plt.hlines(td, xmin, xmax, color=color,
                                    linestyle='dashed', lw=2.)
                    if errors:
                        plt.hlines(td + d.td_cyclotron_error * 1.0E3,
                                    xmin, xmax, color=color,
                                    linestyle='dotted', lw=2.)
                        plt.hlines(td - d.td_cyclotron_error * 1.0E3,
                                    xmin, xmax, color=color,
                                    linestyle='dotted', lw=2.)

                td = td_to_modb(d.td_cyclotron)*1.0E9
                tit += ', |B| = %.1f +/- %.1f nT' % (
                    td, td - td_to_modb(
                            d.td_cyclotron + d.td_cyclotron_error)*1.0E9)

            if np.isfinite(d.ground):
                xmin, xmax = plt.xlim()
                plt.hlines(d.ground*1.0E3, 3.0, xmax,
                            color=color, linestyle='dashed', lw=2.)

            # Note that these are stored in 'self', not in the digitization d
            if hasattr(self,'_morphology_fp_locations'):
                for f in self._morphology_fp_locations:
                    fs = np.interp(f[0],
                            np.arange(self.frequencies.shape[0]),
                            self.frequencies) / 1E6
                    plt.plot( (fs,fs), plt.ylim(),
                                color='green', linestyle='solid', lw=2.)

            if hasattr(self, '_intermediate_trace'):
                plt.gca().add_patch(plt.Rectangle(
                    (self._intermediate_trace[1][0]/1E6,
                        self._intermediate_trace[0][0]*1E3),
                    self._intermediate_trace[1][1]/1E6 -
                                    self._intermediate_trace[1][0]/1E6,
                    self._intermediate_trace[0][1]*1E3 -
                                    self._intermediate_trace[0][0]*1E3,
                    color='white',alpha=0.3)
                    )

        if not no_title:
            plt.title(tit)

        if altitude_range:
            if verbose: print('Ionogram.plot: Altitude')
            oldx = ax.get_xlim()
            alt_ax = ax.twinx()
            alt_ax.autoscale(enable=False, tight=True)
            ax.tick_params(labelleft=False, labelright=False,
                    labelbottom=True, labeltop=False)
            ax.set_ylim(bottom=2. / speed_of_light_kms *
                            (self.mex_altitude - altitude_range[0]) * 1E3)
            ax.set_ylim(top=2. / speed_of_light_kms *
                            (self.mex_altitude - altitude_range[1]) * 1E3)
            alt_ax.set_ylim(*altitude_range)
            alt_ax.set_xlim(*oldx)
            alt_ax.tick_params(labelleft=True, labelright=False,
                                    labeltop=False,labelbottom=False)

            # print 'ALT AX:', alt_ax.get_ylim(), alt_ax.get_xlim()
            # print 'Del AX:', ax.get_ylim(), ax.get_xlim()

    def interpolate_frequencies(self):
        """Interpolate such that frequencies are at fixed spacing - no information lost, but the data volume goes up by **** """
        if self.linear_frequency:
            return
        # freq_sep = np.amin(np.diff(self.frequencies))
        freq_sep = 20000 # 20 khz

        # This will drop the highest frequency, but who cares
        # new_frequencies = np.arange(np.amin(self.frequencies),
        #     np.amax(self.frequencies), freq_sep)
        new_frequencies = np.arange(0.,
            np.amax(self.frequencies), freq_sep)
        new_data = np.empty((80, new_frequencies.shape[0]))

        for i in range(80):
            new_data[i,:] = np.interp(new_frequencies, self.frequencies, self.data[i,:], left=10.**ais_vmin)
        #
        # self._old_data = self.data.copy()
        # self._old_frequencies = self.frequencies.copy()

        self.data = new_data
        self.frequencies = new_frequencies
        self.linear_frequency = True

        return self

    def threshold_data(self, threshold=10.**-15):
        # print 'Applying threshold at %e ' % threshold
        self.thresholded_data = np.zeros(self.data.shape, dtype=np.int16)
        threshold = max((np.median(self.data), threshold))
        self.thresholded_data[self.data > threshold] = True
        self.applied_threshold = threshold
        return self

    def generate_binary_arrays(self, *args, **kwargs):
        """This sets up the images for use in feature extraction"""
        if not self.linear_frequency:
            self.interpolate_frequencies()
        if not hasattr(self, 'thresholded_data'):
            self.threshold_data(*args, **kwargs)
        self._fp_data = morphology.binary_opening(self.thresholded_data,
            structure=np.ones((20,1)))
        # self._fp_data = celsius.remove_none_edge_intersecting(self._fp_data, 2)
        self._fp_data, self._morphology_fp_locations = remove_none_edge_intersecting(self._fp_data,
            2, as_list=True, width=5)

        # print '***'
        # for f in self._morphology_fp_locations:
        #     print f, np.interp(f[0],
        #                np.arange(self.frequencies.shape[0]), self.frequencies)  /1E6
        # print '***'

        # Retain only those bits continuous with the top
        #
        # n = self._fp_data.shape[0]
        # self._fp_data[np.logical_not(np.equal(
        #     np.cumsum(self._fp_data,0),
        #     (np.arange(n).reshape(n,1) + 1)))] = 0

        # self._fp_data = np.zeros_like(self.data) + 10.**ais_vmin
        # for i in range(self._fp_data.shape[1]):
        #     # print i, np.median(self.data[0:10,i]), np.median(self.data[:,i]/np.max(self.data[0:10,i]))
        #     if np.median(self.data[:,i]/np.median(self.data[0:10,i])) > 0.1:
        #         self._fp_data[:,i] = self.data[:,i]

        # Label, find COM of each thing, chuck those with COMS in the lower ~25 % of the image
        # Rank by size, up to a maximum of 'one column'
        # highest taken to be first line, then take remaining as harmonics if they are above 50% of first
        s = skimage.measure.label(self._fp_data.astype(int))
        self._morphology_fp_locations = []
        for v in np.unique(s):
            tmp = s == v
            non_zero = np.all(self._fp_data[tmp] != 0)
            # print '--'
            # print v, non_zero
            if not non_zero:
                continue

            tmp_nz = tmp.nonzero()
            # center of mass x, y, size, height,
            line = [np.mean(tmp_nz[1]), np.mean(tmp_nz[0]), np.sum(tmp), np.max(tmp_nz[0]) - np.min(tmp_nz[0])]
            if line[2] > 80:
                line[2] = 80
            # print 'line = ', line
            if line[1] > 53: #66%
                continue
            if np.min(tmp_nz[0]) > 26:
                continue
                # print ' -- rejecting'

            self._morphology_fp_locations.append(line)

        if self._morphology_fp_locations:
            # sort largest to smallest
            self._morphology_fp_locations = sorted(self._morphology_fp_locations, key=lambda x: -x[2])
            threshold = int(self._morphology_fp_locations[0][2] * 0.3)
            # print 'threshold',threshold
            # only those greater than 50% of the largest retained
            self._morphology_fp_locations = [v for v in self._morphology_fp_locations if v[2] >= threshold]
            self._maximum_fp = self._morphology_fp_locations[0]

        if self._morphology_fp_locations:
            f0 = self._morphology_fp_locations[0][0]
            # self._morphology_fp_locations = [v for v in self._morphology_fp_locations if v[0] >= f0]
            self._morphology_fp_locations = sorted(self._morphology_fp_locations, key=lambda x: x[0])

        # print 'all lines:'
        # for v in self._morphology_fp_locations:
            # print v
        # print '/all lines'

        # self._cyc_data = morphology.binary_opening(
        #     np.logical_and(self.thresholded_data, np.logical_not(self._fp_data)),
        #     structure=np.ones((1,4)))

        # self._cyc_data = np.logical_and(self.thresholded_data, np.logical_not(self._fp_data))

        # self._cyc_data = morphology.binary_closing(self.thresholded_data,
        #     structure=np.ones((1,20)))

        structure = np.zeros((2,1))
        structure[1,0] = 1
        self._ion_data = morphology.binary_hit_or_miss(self.thresholded_data, structure1=structure)
        self._cyc_data = morphology.binary_opening(self._ion_data, structure=np.ones((1,3)))
        # mask = np.zeros_like(self._cyc_data)
        # mask[0:3,:] = 1
        # self._cyc_data = morphology.binary_dilation(self._cyc_data, structure=np.ones((1,2)), mask=mask)
        # self._cyc_data = morphology.binary_erosion(self._cyc_data, structure=np.ones((1,2)))

        self.quality_factor = 1. - np.sum(
            np.logical_and(self.thresholded_data,
            np.logical_not(np.logical_or( self._fp_data, self._ion_data)))
        ) / float(np.sum(self.thresholded_data))

        return self

    def delete_binary_arrays(self):
        del self._fp_data
        del self._cyc_data
        del self._ion_data

        return self

    def calculate_fp_local(self, figure_number=False):
        """Take the first processed line, calculate an error based on the remaining.
        Requires that interpolate_frequencies() has been called to make the frequency
        spacing linear.
        If *figure_number* is set, plot the FFT out on that figure."""

        if figure_number:
            fig = plt.figure(figure_number)
            plt.clf()

        self.digitization.delete_fp_local()

        # Compute via the integration method
        density, error = calibrator(self)
        self.digitization.set_integrated_fp_local(*ne_to_fp(density, error))

        if not hasattr(self, '_fp_data'):
            self.generate_binary_arrays()

        if not hasattr(self, '_morphology_fp_locations'):
            return "Auto-fp_local: Failed (No plasma lines detected in morphological processing)"

        if not len(self._morphology_fp_locations):
            return "Auto-fp_local: Failed (No plasma lines detected in morphological processing)"

        scaled = np.sum(self._fp_data, 0).astype(np.float32)
        correction = 0

        det_fp_local = -np.inf
        det_err = np.inf

        if len(self._morphology_fp_locations):
            det_fp_local = np.interp(self._maximum_fp[0],
                            np.arange(self.frequencies.shape[0]), self.frequencies)
            diffs = []
            det_err = np.inf
            if len(self._morphology_fp_locations) > 1:
                for f in self._morphology_fp_locations[1:]:
                    fp = np.interp(f[0], np.arange(self.frequencies.shape[0]),
                                        self.frequencies)
                    diffs.append(np.min(np.abs(fp - np.arange(2,10) * det_fp_local)))
                det_err = np.mean(np.array(diffs))
                del diffs

        if figure_number:
            plt.subplot(111)
            plt.plot(self.frequencies *1E-6, scaled, 'k.',drawstyle='steps-mid')
            plt.xlim(0.,4)
            plt.ylabel('Processed fp_lines')
            plt.xlabel('f/MHz')
            plt.title("Plasma line detection [v%s]" % (__auto_version__))
            plt.ylim(0., 1.1 * np.amax(scaled))
            plt.vlines(det_fp_local*1E-6, *plt.ylim(), color='blue')
            plt.vlines(2. * det_fp_local*1E-6, *plt.ylim(),
                            color='blue', linestyle='dashed')

        if ((det_fp_local < ais_fp_local_sensible[0]) |
                (det_fp_local > ais_fp_local_sensible[1])):
            msg = "Auto-fp_local: Failed (result %f not sensible)" % fp_to_ne(det_fp_local)
            return msg

        msg = 'Auto-fp_local: n_e = %.1f +/- %.1f cc' % \
                                    (fp_to_ne(det_fp_local, det_err))

        # print msg
        # print fp_to_ne(f), fp_to_ne(f + err), fp_to_ne(f - err)
        self.digitization.set_morphology_fp_local(
                        det_fp_local,
                        det_err,
                        method='AUTO - V' + __auto_version__,
                        length=self._morphology_fp_locations[0][3]
                    )
        # self.digitization.quality_factor = self.quality_factor
        # self.digitization.set_velocity_proxy(self._morphology_fp_locations[0][3])
        # self.digitization.maximum_fp_local = self._maximum_fp
        # print 'Velocity proxy:',self.digitization.velocity_proxy

        if figure_number:
            if np.isfinite(self.digitization.fp_local):
                plt.vlines(self.digitization.fp_local*np.arange(100) *1E-6,
                                *plt.ylim(), linestyles='dashed')
            plt.show()

        return msg

    def calculate_td_cyclotron(self, figure_number=False, threshold=3, ax=False):
        self.digitization.delete_td_cyclotron()
        if figure_number:
            f = plt.figure(figure_number)
            plt.clf()

        if ax:
            plt.sca(ax)
            plt.cla()

        if not hasattr(self, '_cyc_data'):
            self.generate_binary_arrays()

        fmax_bin, = np.where((self.frequencies > 0.3E6))
        fmax_bin = fmax_bin[0]
        td_min_bin = 2
        # Ignore the first few ms: 5 bins ignored = delays below 0.62 ms
        # scaled = np.sum(self._cyc_data[td_min_bin:,:fmax_bin], 1) # .3 MHz
        # scaled = np.sum(self.data[td_min_bin:,:fmax_bin], 1) # .3 MHz

        # d = self._ion_data * self.data # 2012-03-08
        # d = self._cyc_data * self.data
        d = self._ion_data
        scaled = np.sum(d[td_min_bin:,:fmax_bin], 1) # .3 MHz
        scaled_delays = self.delays[td_min_bin:]
        scaled = proc_peaks(scaled)
        max_scaled = np.amax(scaled)

        peak_threshold = 0.1 * max_scaled
        if peak_threshold < threshold:
            peak_threshold = threshold
        first_peak_inx, = np.where(scaled > peak_threshold)

        if first_peak_inx.shape[0] < 1:
            msg = "Auto-cyclotron: Failed (first line not detected)"
            return msg

        first_peak_inx = first_peak_inx[0]
        if scaled[first_peak_inx] < threshold:
            msg = "Auto-cyclotron: Failed (first line intensity %e below threshold %e)" %\
                            (scaled[first_peak_inx], threshold)
            return msg

        first_peak_delay = self.delays[first_peak_inx + td_min_bin]

        peak_threshold = 0.5 * scaled[first_peak_inx]
        if peak_threshold < threshold:
            peak_threshold = threshold

        second_peak_inx, = np.where(scaled[first_peak_inx+1:] > peak_threshold)

        if (second_peak_inx.shape[0] < 1):
            err = first_peak_delay
            second_peak_delay = np.nan
            # if (first_peak_delay < (self.delays[-1]/4.)):
            #     msg = "Auto-cyclotron: Failed (Only one line, expected more for this delay %e)" % first_peak_delay
            #     return msg
            # else:
            #     err = first_peak_delay
            #     second_peak_delay = np.nan
        else:
            second_peak_inx = second_peak_inx[0]
            second_peak_delay = self.delays[td_min_bin + first_peak_inx + 1 + second_peak_inx]

            fratio = second_peak_delay / first_peak_delay
            if fratio > 1.6667:
                err = second_peak_delay - 2. * first_peak_delay
                # print 'No correction, 2.0 = %f' % fratio
            elif (fratio <= 1.6667) & (fratio > 1.4):
                err = second_peak_delay - 3./2. * first_peak_delay
                first_peak_delay = first_peak_delay/2.
                # print 'Correction 1.5 ~= %f [2nd harmonic]' % fratio
            elif (fratio <= 1.4) & (fratio > 1.2857):
                err = second_peak_delay - 4./3. * first_peak_delay
                first_peak_delay = first_peak_delay/3.
                # print 'Correction 1.33 ~= %f [3rd harmonic]' % fratio
            else:
                msg = "Auto-cyclotron: Failed (couldn't make sense of second line at %e x first)" % fratio
                return msg

        if ((first_peak_delay < ais_td_cyclotron_sensible[0]) |
                                (first_peak_delay > ais_td_cyclotron_sensible[1])):
            msg = "Auto-cyclotron: Failed (result not sensible %f)" % first_peak_delay
            return msg
        else:
            err = np.abs(err)
            d = td_to_modb(first_peak_delay, err)
            d = (d[0] * 1.0E9, d[1] * 1.0E9)
            msg = "Auto-cylotron: Found lines at %.1f +/- %.1f nT" % d
            self.digitization.set_cyclotron( first_peak_delay, np.abs(err),
                                    method='AUTO - V' + __auto_version__)
            # self.digitization.quality_factor = self.quality_factor

        if figure_number:
            plt.subplot(211)
            plt.plot(scaled_delays * 1E3, scaled,'k-')
            plt.vlines(first_peak_delay*1.E3, *plt.ylim(), colors='blue')
            plt.vlines(second_peak_delay*1.E3, *plt.ylim(),
                colors='blue',linestyle='dashed')
            plt.vlines(np.arange(0., plt.xlim()[1], first_peak_delay * 1.0E3) ,
                *plt.ylim(),  colors='red',linestyle='dotted')

        if ax:
            plt.plot(scaled_delays * 1.0E3, scaled,'k-')
            plt.vlines(first_peak_delay*1.E3, *plt.ylim(), colors='blue')
            plt.vlines(second_peak_delay*1.E3, *plt.ylim(),
                colors='blue',linestyle='dashed')
            plt.vlines(np.arange(0., plt.xlim()[1], first_peak_delay * 1.0E3) ,
                *plt.ylim(),  colors='red',linestyle='dotted')

        return msg

    # def calculate_td_cyclotronx(self, figure_number=False, threshold=1E-14, ax=False):
    #     self.digitization.delete_td_cyclotron()
    #     if figure_number:
    #         f = plt.figure(figure_number)
    #         plt.clf()
    #
    #     if ax:
    #         plt.sca(ax)
    #         plt.cla()
    #
    #     if not hasattr(self, '_cyc_data'):
    #         self.generate_binary_arrays()
    #
    #     fmax_bin, = np.where((self.frequencies > 0.3E6))
    #     fmax_bin = fmax_bin[0]
    #     td_min_bin = 2
    #     # Ignore the first few ms: 5 bins ignored = delays below 0.62 ms
    #     # scaled = np.sum(self._cyc_data[td_min_bin:,:fmax_bin], 1) # .3 MHz
    #     # scaled = np.sum(self.data[td_min_bin:,:fmax_bin], 1) # .3 MHz
    #
    #     # d = self._ion_data * self.data # 2012-03-08
    #     # d = self._cyc_data * self.data
    #     d = self._ion_data
    #     scaled = np.sum(d[td_min_bin:,:fmax_bin], 1) # .3 MHz
    #
    #     scaled = proc_peaks(scaled)
    #
    #     max_scaled, = np.where(scaled > (0.1 * np.amax(scaled)))
    #     if max_scaled.shape[0] < 1:
    #         msg = "Auto-cyclotron: Failed (first line not detected)"
    #         return msg
    #
    #     max_scaled = max_scaled[0]
    #     if scaled[max_scaled] < threshold:
    #         msg = "Auto-cyclotron: Failed (first line intensity %e below threshold %e)" % (scaled[max_scaled], threshold)
    #         return msg
    #
    #     max_scaled_delay = self.delays[max_scaled + td_min_bin]
    #
    #     ds = np.arange(82) * 91.4 # covers the whole delay range including zero.
    #     scaled = np.interp(ds, self.delays[td_min_bin:], scaled, left=0., right=0.)
    #
    #     vals = np.zeros(40)
    #     for i in range(2,40):
    #         vals[i] = np.sum(scaled[i::i]) * i / 82.0 / fmax_bin
    #
    #     # imax = arg_peak(vals, width=1)
    #     # imax = np.argmax(vals)
    #     vals_inx = np.arange(vals.shape[0])
    #     imax = int(round(np.interp(max_scaled_delay, vals_inx*91.4, vals_inx)))
    #
    #     dtx = imax * 91.4 * 1.0E-6
    #     max_vals = np.amax(vals)
    #
    #     if True:
    #         fw = fwhm(np.arange(vals.shape[0]) * 91.4 * 1.0E-6, vals, peak_inx=imax)
    #         if ((dtx < ais_td_cyclotron_sensible[0]) |
    #                                 (dtx > ais_td_cyclotron_sensible[1])):
    #             msg = "Auto-cyclotron: Failed (result not sensible %f)" % dtx
    #         # elif (0.5 * fw) > ais_auto_cyclotron_max_err:
    #         #     msg = "Auto-cyclotron: Failed (error too large %f)" % (fw * 0.5)
    #         else:
    #             d = td_to_modb(dtx, 0.5 * fw)
    #             d = (d[0] * 1.0E9, d[1] * 1.0E9)
    #             msg = "Auto-cylotron: Found lines at %.1f +/- %.1f nT" % d
    #             self.digitization.set_cyclotron( dtx, 0.5 * fw ,
    #                                     method='AUTO - V' + __auto_version__)
    #             self.digitization.quality_factor = self.quality_factor
    #     else:
    #         msg = "Auto-cyclotron: Failed (Threshold %e, val %e)" % (threshold,
    #                                                                         vals[imax])
    #
    #     if figure_number:
    #         plt.subplot(211)
    #         plt.plot(ds * 1.0E-3, scaled,'k-')
    #         # if imax.shape[0] > 0:
    #         # plt.vlines(ds[imax] * 1.0E-3, *plt.ylim())
    #         plt.vlines(dtx*1.E3, *plt.ylim(), colors='blue')
    #
    #         plt.vlines(max_scaled_delay*1.E-3, *plt.ylim(), colors='green')
    #         if not (vals[imax] < max_vals):
    #             plt.vlines(np.arange(0., plt.xlim()[1], dtx * 1.0E3) , *plt.ylim(),
    #                 colors='red')
    #         plt.subplot(212)
    #         plt.plot(np.arange(vals.shape[0]) * 91.4 * 1.0E-3, vals, 'k-')
    #         plt.vlines(ds[imax] * 1.0E-3, *plt.ylim(), colors='k')
    #
    #     if ax:
    #         plt.plot(ds * 1.0E-3, scaled,'r-')
    #         plt.plot(np.arange(vals.shape[0]) * 91.4 * 1.0E-3, vals, 'k-')
    #         plt.vlines(ds[imax] * 1.0E-3, *plt.ylim(), colors='k')
    #
    #     return msg

    # def calculate_reflection(self):
    #     if not hasattr(self, '_ion_data'):
    #         self.generate_binary_arrays()
    #      self._ion_data
    #          s = skimage.measure.label(img.astype(int))
    # s_set = np.unique(s * mask)
    # if s_set.sum() > 0:
    #     for v in s_set:
    #         q = (s == v)
    #         if np.all(img[q]):
    #             out[s == v] = 1
    #             tmp = np.zeros_like(img)
    #             tmp[s==v] = 1
    #             fp_list.append((np.mean(tmp.nonzero()[1]), np.mean(tmp.nonzero()[0])))


    # Old version (as of 2012-07-18 - new above)
    # 2012-10-16 - ??
    def calculate_reflection(self, frequency_range=None, h=None,
            delay_range=(1E-3, 7E-3)):

        d = self.digitization
        d.delete_trace()

        # Using the Morgan 08 model (Chapman theory only)
        pos, sza = mex.mso_r_lat_lon_position(float(self.time), sza = True)
        if frequency_range is None:
            if sza > 90:
                frequency_range = (0.3, 2.0)
            else:
                # frequency_range = (1.0, 4.)# never really higher than 4 Mhz
                frequency_range =(1.0,
                        ne_to_fp(1.5 * 1.58E5 * np.cos(np.deg2rad(sza))**0.5) / 1E6)
                if frequency_range[1] < 2.:
                        frequency_range = (frequency_range[0], 2.)
                # Factor of 1.3 to give upper sensible limit
        if h is None:
            # if sza > 90:
            #     h = 133.6
            # else:
            #     h = 133.6 - 8.9 * np.log(np.cos(np.deg2rad(sza)))
            h = 80.

        pos = mex.iau_pgr_alt_lat_lon_position(float(self.time))
        delay = 2. * (pos[0] - h) / speed_of_light_kms
        ground_delay = 2. * pos[0] / speed_of_light_kms

        # print "Expected delay = %f ms" % delay
        if (delay < delay_range[0]) | (delay > delay_range[1]):
              return "Auto-ionosphere: Echo not expected within %f - %f ms" % (delay_range[0] * 1e3, delay_range[1] * 1e3)

        # Introduce a bias to larger delays at higher f = dispersion effect
        d.set_trace(np.array((delay - (frequency_range[1] - frequency_range[0])/2.77*1E-3,
                                                    delay )),
            np.array(frequency_range) * 1E6,
            method="AUTO-INTERMEDIATE")

        self._intermediate_trace = (np.array(
            (delay - (frequency_range[1] - frequency_range[0])/2.77*1E-3, delay )),
            np.array(frequency_range) * 1E6)

        m = self.refine_trace(width=91.4 * 2. * 1E-6 )

        # Now a whole bunch of checks to try and throw away junk results
        if d.traced_delay.size:
            md = np.mean(d.traced_delay)
            mean_alt = pos[0] - md / 2. * speed_of_light_kms
            md = np.abs(md - delay)
            n = d.traced_delay.shape[0]
            f_space = np.diff(d.traced_frequency)
            d_space = np.diff(d.traced_delay)
            # print 'mean alt', mean_alt
            if md > 5.E-3:
                m = "Auto-ionosphere: Failed (Located trace not sensible %f)" % md
            elif mean_alt < 30.:
                m = "Auto-ionosphere: Failed (Located trace too close to ground: %f km)" % mean_alt
            elif n < 5:
                m = "Auto-ionosphere: Failed (Too few points returned: %d)" % n
            elif (np.amax(d.traced_frequency) - np.amin(d.traced_frequency)) < 0.25E6:
                m = "Auto-ionosphere: Failed (Trace too short)"
            # elif np.median(d_space) < 0.:
            #     # print np.amin(np.diff(d.traced_delay)), np.mean(np.diff(d.traced_delay)),
            #     m = "Auto-ionosphere: Failed (Declining trace)"
            # elif np.std(d.traced_delay) > 0.0004:
            #     m = "Auto-ionosphere: Failed (Crappy delays)"
            elif np.amax(f_space) > 0.3E6:
                m = "Auto-ionosphere: Failed (Non-continuous trace)"
            # elif np.mean(f_space) > 5. * 10681.:
            #     m = "Auto-ionosphere: Failed (Non-continuous trace)"
            if 'Failed' in m:
                d.delete_trace()
                return m
            else:
                d.traced_method = "AUTO - V" + __auto_version__

        return "Auto-ionosphere:" + m

    def calculate_ground_trace(self, threshold=1E-16, *args, **kwargs):
        """docstring for calculate_ground_trace"""

        self.digitization.delete_ground()

        self.interpolate_frequencies()
        # from 3 MHz, ~1400 mus
        d = self.data[10:,271:] > threshold
        scaled = np.sum(d, 1).astype(float)
        if np.amax(scaled) < 0.3 * d.shape[1]:
            return "Auto-ground: Failed (no peak above threshold: %f, %f, %f)" % (np.sum(d), np.amax(scaled), 0.3 * d.shape[1])

        imax = arg_peak(scaled)
        if imax.shape[0] == 1:
            imax = imax[0]
            self.digitization.set_ground(self.delays[imax + 10])

            return "Auto-ground: Found at %.1f ms" % (self.delays[imax + 10] * 1.0E3)

    def refine_trace(self, dig=None, threshold=0.001, width=91.4E-6 * 5., min_delay=1E-6, max_delay='max'):
        """Automagically adjust the current trace to fit peaks in the data, above *threshold*.
        max_delay is numeric to specify some maximum, or 'ground' for the determined ground, or else"""

        # threshold = -1. * np.infty
        if not dig:
            dig = self.digitization

        d_max = ais_max_delay
        if not isinstance(max_delay, str):
            d_max = max_delay
        elif 'ground' in max_delay:
            if np.isfinite(dig.ground):
                d_max = dig.ground

        if not hasattr(self, '_cyc_data'):
            self.generate_binary_arrays()
        # dig.quality_factor = self.quality_factor

        # Needs to handle the edges sensibly
        freqs = dig.traced_frequency
        delays = dig.traced_delay

        subs = (self.frequencies > freqs[0]) & (self.frequencies < freqs[-1])
        new_freqs = self.frequencies[subs]
        new_delays = np.interp(new_freqs, freqs, delays)
        new_intensities = np.zeros_like(new_delays)
        other = np.arange(subs.shape[0])
        other = other[subs]

        d, = np.where( (self.delays > min_delay) & (self.delays < d_max) )

        # print self.delays
        # print min_delay, d_max
        # print d
        data = self._ion_data

        for i, s in enumerate(other):
            v = np.sum(data[d,s].astype(int))
            if v > 3 :
                continue

            distance = (self.delays[d] - new_delays[i]) / (2. * width)
            column = data[d,s] * \
                np.exp(-distance**2.)
            column[np.abs(distance) > 2.] = -1E99

            # celsius.code_interact(locals())

            m = np.argmax(column)
            new_delays[i] = self.delays[d[m]]
            new_intensities[i] = column[m]

        inx, = np.where(new_intensities > threshold)

        if inx.shape[0] < 2:
            dig.delete_trace()
            return "Refine-trace: Failed (Not enough data above threshold %e)" % threshold

        new_freqs = new_freqs[inx]
        new_delays = new_delays[inx]
        inx = np.argsort(new_freqs)

        # Retain only after last absolute change of more than 3 delay increments
        inx, = np.where(np.abs(np.diff(new_delays)) > (91.4E-6 * 3))
        if inx.shape[0] > 1:
            new_freqs = new_freqs[inx[-1]+1:-1]
            new_delays = new_delays[inx[-1]+1:-1]

        dig.traced_frequency = new_freqs
        dig.traced_delay = new_delays
        dig.traced_method = dig.traced_method + '+R'
        return "Refine-trace: Success"

    def extract_frequency(self, frequency):
        """docstring for extract_frequency"""
        inx = np.argmin(np.abs(self.frequencies - frequency))
        return self.data[:,inx]

    # def .digitization = ais_code.IonogramDigitization()(self, dig=None):
    #     if not dig:
    #         dig = IonogramDigitization()
    #     dig.set_time(self.time)
    #     if hasattr(self, 'digitizations'):
    #         self.digitizations.append(dig)
    #     else:
    #         self.digitizations = [dig]
    #     return dig

    # def get_current_digitization(self):
    #     if not self.digitizations:
    #         return self.digitization = ais_code.IonogramDigitization()
    #     return self.digitization

    def __str__(self):
        s = self.__repr__() + '\n'
        ss = [str(k) + ' = ' + str(v) for k, v in self.__dict__.items() if (k not in DIGITIZATION_BANNED_KEYS) and (k[0] != '_')]
        return s + '\n'.join(ss)

# This doesn't actually get used much now
class AISTimeSeries(object):
    """docstring for AISTimeSeries"""
    def __init__(self, ionogram_list=None):
        raise Exception("Candidate for deletion....")


        super(AISTimeSeries, self).__init__()
        self.ionogram_list = ionogram_list
        self.min_time = np.inf
        self.max_time = -np.inf

        for i in self.ionogram_list:
            if i.time is not None:
                if i.time < self.min_time:
                    self.min_time = i.time
                if i.time > self.max_time:
                    self.max_time = i.time

    def plot_spectrum(self, ax=None, time_range=None, vmin=None, vmax=None):
        if self.ionogram_list is None:
            raise mex.MEXException('No data here to plot')

        if ax is None:
            ax = plt.gca()
        plt.sca(ax)

        for ionogram in self.ionogram_list:
            if time_range is not None:
                if (ionogram.time < time_range[0]) or (ionogram.time > time_range[1]):
                    continue

            # Should check for consistency in the frequency table here maybe?
            data = np.log10(np.sum(ionogram.data, axis=0))

            plt.scatter(np.zeros_like(data) + ionogram.time, ionogram.frequencies / 1.0E6,
                c=data, marker='s', hold=True, vmin=-16, vmax=-10, edgecolor='none',
                rasterized=True)

        plt.xlim(self.min_time, self.max_time)
        plt.show()

    def plot_frequency(self, frequency=2.5, ax=None, time_range=None):
        """docstring for plot"""

        if frequency < 5.5:
            frequency *= 1.0E6

        if self.ionogram_list is None:
            raise mex.MEXException('No data here to plot')

        if ax is None:
            ax = plt.gca()
        plt.sca(ax)

        for ionogram in self.ionogram_list:
            if time_range is not None:
                if (ionogram.time < time_range[0]) or (ionogram.time > time_range[1]):
                    continue

            # Should check for consistency in the frequency table here maybe?
            data = np.array(np.log10(ionogram.extract_frequency(frequency)))

            plt.scatter(np.zeros_like(data) + ionogram.time, ionogram.delays * 1E3, c=data,
                marker='s', hold=True, vmin=-16, vmax=-10, edgecolor='none', rasterized=True)

        plt.xlim(self.min_time, self.max_time)
        plt.show()

class AISEmpiricalCalibrationOLD(object):
    """docstring for AISEmpiricalCalibration"""
    def __init__(self, fname=None, orbits=None):
        super(AISEmpiricalCalibration, self).__init__()

        raise RuntimeError("Depreciated code")

        if fname is None:
            fname = os.getenv('SC_DATA_DIR') + 'mex/ais_calfile'

        if orbits is None: orbits =  [
                                        2106,
                                        2630,
                                        2683,
                                        2673,
                                        2912,
                                        4304,
                                        2590,
                                        4989,
                                        4216,
                                        3875,
                                        4469,
                                        5083,
                                        2643,
                                    ]
        self.fname  = fname
        self.orbits = orbits

        if os.path.exists(fname + '.npy'):
            print('Loading calibration from %s' % (fname + '.npy'))
            self.cal = np.load(fname + '.npy')
            self.min_val = np.min(self.cal[1])
        else:
            self.calibrate()

    def _construct_calibration_array(self):
        if os.path.exists(self.fname + '_in.npy'):
            out = np.load(self.fname + '_in.npy')
        else:
            chunks = []

            for o in self.orbits:
                print('Reading %d' % o)
                # dm_data = morgan.read_orbit(o)
                this_orbit = np.empty((dm_data.shape[0], 3)) + np.nan
                this_orbit[:,0] = dm_data[:,0]
                this_orbit[:,1] = dm_data[:,1]

                igs = read_ais(o)

                for inx, i in enumerate(igs):
                    new_n = self._q(i)

                    dt = np.abs(this_orbit[:,0] - i.time)
                    inx = np.argmin(dt)
                    if dt[inx] > ais_spacing_seconds:
                        print('-- out of range')
                    this_orbit[inx, 2] = new_n

                chunks.append(this_orbit)

            out = chunks[0].copy()
            print(out.shape)
            for i in range(1, len(chunks)):
                out = np.vstack((out, chunks[i]))
                print(out.shape)

            out = out.T
            inx, = np.where(np.isfinite(np.sum(out,0)) & (out[1] > 0.01))
            out = out[:,inx]
            out = out[:,np.argsort(out[2])]

            print('Saving', out.shape)
            np.save(self.fname + '_in.npy', out)

        print(out.shape)
        self.t = out[0]
        self.x = out[2]
        self.y = out[1]
        self.ly = np.log10(self.y)


    def calibrate(self, points = 10):
        print('Calibrating...')
        if not hasattr(self, 't'): self._construct_calibration_array()


        n = self.ly.size / points

        self.cal = np.empty((3, points)) + np.nan

        for i in range(points):
            inx = np.arange(i * n, (i+1)*n, 1, dtype=int)
            my = np.median(self.ly[inx])
            mx = np.median(self.x[inx])
            r   = np.std(self.ly[inx])
            dx  = np.std(self.x[inx])

            self.cal[0,i] = mx
            self.cal[1,i] = my
            self.cal[2,i] = r
            plt.plot(mx, my, 'r.')
            plt.plot((mx, mx), (my + r, my-r),'r-')
            plt.plot((mx - dx, mx +dx), (my,my),'r-')

        np.save(self.fname, self.cal)
        self.min_val = np.min(self.cal[1])

    def plot(self):
        if not hasattr(self, 't'): self._construct_calibration_array()
        if not hasattr(self, 'cal'): self.calibrate()

        plt.close('all')
        plt.figure()
        plt.plot(self.x, self.ly, 'k.')

        for i in range(self.cal.shape[1]):
            c = self.cal[:,i]
            print(c)
            plt.plot((c[0], c[0]), (c[1]-c[2], c[1]+c[2]), 'r-')
            plt.plot(c[0], c[1], 'r.')

        plt.figure()

        dists = np.empty_like(self.t)
        sigmas = np.empty_like(self.t)
        for i in range(self.t.shape[0]):
            # if i % 10 != 0: continue
            val = 10.**np.interp(self.x[i], self.cal[0], self.cal[1])
            err = 10.**np.interp(self.x[i], self.cal[0], self.cal[2])
            plt.plot(self.y[i],val,'k.')
            plt.plot((self.y[i],self.y[i]), (val-err, val+err), 'k-')
            dists[i] = self.y[i] - val
            sigmas[i] = np.abs(np.log10(dists[i])/np.log10(err))
            dists[i] /= self.y[i]

        x = np.array((10., 1E4))
        plt.plot(x,x, 'r-')
        plt.plot(x, x*2., 'r-')
        plt.plot(x, x*.5, 'r-')

        plt.yscale('log')
        plt.xscale('log')

        plt.figure()
        plt.hist(np.abs(dists), bins=20, range=(0., 4.))

        # plt.figure()
        # plt.hist(sigmas, bins=20)
        dists = np.abs(dists)
        # some statistics:
        s = 100. / float(self.t.shape[0])
        print()
        print('%f%% with relative error < 0.1' % (np.sum(dists < 0.1) * s))
        print('%f%% with relative error < 0.5' % (np.sum(dists < 0.5) * s))
        print('%f%% with relative error < 1.0' % (np.sum(dists < 1.0) * s))
        print('%f%% with relative error < 2.0' % (np.sum(dists < 2.0) * s))
        print()
        print('%f%% with sigma < 1.0' % (np.sum(sigmas < 1.0) * s))
        print('%f%% with sigma < 2.0' % (np.sum(sigmas < 2.0) * s))
        print('%f%% with sigma < 4.0' % (np.sum(sigmas < 4.0) * s))

        plt.show()

    def _q(self, ig):
        ig.interpolate_frequencies()
        return np.mean(np.max(np.log10(ig.data),1)[0:20])


    def __call__(self, ig):
        if not hasattr(self, 'cal'): self.calibrate()
        x = self._q(ig)

        val = np.interp(x, self.cal[0], self.cal[1])
        err = np.interp(x, self.cal[0], self.cal[2])
        # print 'CAL: ', val, err, val < (self.min_val + 0.01)
        if val < (self.min_val + 0.01):
            val = np.nan
            err = np.nan

        return 10.**val, 10.**val * err / 0.434

class AISEmpiricalCalibration(object):
    """docstring for AISEmpiricalCalibration"""
    def __init__(self):
        super(AISEmpiricalCalibration, self).__init__()

    def calibrate(self, points = 10):
        raise NotImplementedError("Please, code not writings.")

    def plot(self):
         raise NotImplementedError("Please, code not writings.")

    def __call__(self, ig):
        "Take an ionogram, return the corresponding density"
        ig.interpolate_frequencies()
        return self.current(ig)

    def previous_previous(self, ig):
        q = np.mean(np.max(np.log10(ig.data),1)[0:20])

        # Manual fits to the data, for now
        # issue is with fitting the large variation meaningfully - lots of outliers lead
        # to wild distortions
        # This fit (more or less) maximises the peaked-ness of the resulting relative error distribution
        # Maximising the fraction of measurements with |relative error| < 0.5
        ne = 10**(np.exp((q+12.9)/0.4) + 1.2)
        err = 0.362713559939 * ne

        if ne < 20.:
            ne = np.nan
            err = np.inf
        return ne, err

    def previous(self, ig):
        # Prior to 2013-04-17
        q = np.mean(np.max(np.log10(ig.data),1)[0:20])

        if q < -14.5: return 0., np.nan
        # [  5.96849485   3.11080148 -12.46817905   0.82511371]
        x = 5.96849485 + 3.11080148 * np.tanh( (q + 12.46817905) / 0.82511371)
        x = np.exp(x)
        return x, 0.365837992962 * x

        # errs:
        # [ 0.03359992  0.91625332  6.23799479]

    # def current(self, ig):
    #     # 2013-04-17
    #     q = np.mean(np.max(np.log10(ig.data),1)[0:20])
    #     if q < -14.5: return 0., np.nan
    #     x = -0.202724663*(q**4.) -1.08247656e+01*(q**3.) -2.15552479e+02*(q**2.) -1.89534641e+03*q -6.19880273e+03
    #     x = np.exp(x)
    #     return x, 0.377424469584 * x

    def current(self, ig):
        # 2013-04-17
        q = np.mean(np.max(np.log10(ig.data),1)[0:20])
        if q < -14.5: return 0., np.nan

        x = -2.37925862e-01*(q**4.) -1.27267054e+01*(q**3.) -2.53909312e+02*(q**2.)-2.23759838e+03*q  -7.33890229e+03
        x = np.exp(x)
        return x, 0.378173135384 * x

calibrator = AISEmpiricalCalibration()

# curvefit:
# [   0.44805593   13.44865813  103.52077902]
# errs:
# [ 0.01805124  0.47196973  3.08031408]

# -------------
# Statistics:
#
# Ne_local:
# AUTO_N_GOOD =  20561
# MANUAL_N_GOOD =  8192
# BOTH_N_GOOD =  8185
# Correlation 0.947221963974
# Mean delta/N: -0.0329220691361
# Fraction with |delta/N| < 0.5: 0.839462431277
# Mean abs delta/N: 0.277981664682
# Std delta/N 0.479066540492
#
# |B|:
# AUTO_N_GOOD =  14469
# MANUAL_N_GOOD =  575
# BOTH_N_GOOD =  387
# Correlation 0.173033777887
# Fraction with |delta/N| < 0.5: 0.894056847545
# Mean delta/N: 0.0158814652373
# Mean abs delta/N: 0.149775178022
# Std delta/N 0.304047243316



class IonogramDigitization:
    """An object that retains info about parameters derived from an Ionogram"""
    def __init__(self, ionogram=None):
        # super(IonogramDigitization, self).__init__()

        if ionogram is None:
            self.time = np.nan
        else:
            self.time = ionogram.time

        self.timestamp = celsius.now()

        # These also create, of course
        self.delete_trace()
        self.delete_fp_local()
        self.delete_td_cyclotron()
        self.delete_ground()

    def set_timestamp(self):
        self.timestamp = celsius.now()

    def to_dict(self):
        d = {}
        d['time'] = self.time
        d['timestamp'] = self.timestamp

        d['fp_local'] = self.fp_local
        d['fp_local_error'] = self.fp_local_error

        d['td_cyclotron'] = self.td_cyclotron
        d['td_cyclotron_error'] = self.td_cyclotron_error

        d['traced_delay'] = self.traced_delay.tolist()
        d['traced_frequency'] = self.traced_frequency.tolist()

        return d

    def from_dict(self, d):
        self.time = d['time']
        self.timestamp = d['timestamp']

        self.fp_local = d['fp_local']
        self.fp_local_error = d['fp_local_error']

        self.td_cyclotron = d['td_cyclotron']
        self.td_cyclotron_error = d['td_cyclotron_error']

        self.traced_delay = np.array(d['traced_delay'])
        self.traced_frequency = np.array(d['traced_delay'])

    def delete_trace(self):
        self.traced_delay       = np.empty(0)
        self.traced_frequency   = np.empty(0)
        self.traced_intensity   = np.empty(0)
        self.traced_method      = ''

        self.altitude = np.empty(0)
        self.density = np.empty(0)

    def delete_fp_local(self):
        self.fp_local           = np.nan
        self.fp_local_error     = np.nan

        self.morphology_fp_local = np.nan
        self.morphology_fp_local_error = np.nan
        self.morphology_fp_local_selected_f = np.empty(0)
        self.morphology_fp_local_length = np.nan

        self.integrated_fp_local = np.nan
        self.integrated_fp_local_error = np.nan

    def delete_td_cyclotron(self):
        self.td_cyclotron       = np.nan
        self.td_cyclotron_error = np.nan

    def delete_ground(self):
        self.ground = np.nan


    # Threshold = 150.cm -3 in Hz
    def _compute_fp_local(self, threshold=109982.):
        """Compute fp local by comparing the two methods"""
        self.fp_local = self.integrated_fp_local
        self.fp_local_error = self.integrated_fp_local_error

        # integrated always finite unless below its own threshold (low density)
        # both need to be finite to compare,  if integrated is nan, density is low
        # therefore ignore morphology as well
        if np.isfinite(self.morphology_fp_local * self.integrated_fp_local):
            if self.integrated_fp_local > threshold:
                self.fp_local = self.morphology_fp_local
                self.fp_local_error = self.morphology_fp_local_error

        # print '--', self.integrated_fp_local, self.integrated_fp_local > threshold, self.morphology_fp_local, self.fp_local
        # self.set_timestamp()

    def set_morphology_fp_local(self, fp_local, fp_local_error, method,
                                    selected_f=None, length=None):
        """fp_local from morphological processing"""
        self.morphology_fp_local = fp_local
        self.morphology_fp_local_error = fp_local_error

        if isinstance(selected_f, np.ndarray):
            if selected_f.shape[0] > 0:
                self.morphology_fp_local_selected_f = selected_f
        elif selected_f:
            self.morphology_fp_local_selected_f = selected_f
        if length:
            self.morphology_fp_local_length = length

        self._compute_fp_local()

    def set_integrated_fp_local(self, integrated_fp_local, integrated_fp_local_error):
        self.integrated_fp_local       = integrated_fp_local
        self.integrated_fp_local_error = integrated_fp_local_error
        self._compute_fp_local()

    def set_cyclotron(self, td_cyclotron, td_cyclotron_error, method, selected_t=None):
        self.td_cyclotron = td_cyclotron
        self.td_cyclotron_error = td_cyclotron_error
        self.td_cyclotron_method = str(method)
        if selected_t is not None:
            self.td_cyclotron_selected_t = selected_t

    def set_trace(self, traced_delay, traced_frequency, method, traced_intensity=None):
        a = np.argsort(traced_frequency)
        self.traced_delay = traced_delay[a]
        self.traced_frequency = traced_frequency[a]
        self.traced_method = method
        if traced_intensity is not None:
            self.traced_intensity = traced_intensity[a]

    def set_ground(self, ground):
        self.ground = ground

#     def set_velocity_proxy(self, bin):
#         """velocity proxy is dx / dt, where dt is maximum dt of first plasma line,
# dx is the scale size of the local plasma oscillation, taken to be 40 m
# (tip-tip antenna length)."""
#         if bin < 0:
#             self.velocity_proxy = 40E-3 / ais_delays[0] # km/s
#         self.velocity_proxy = 40E-3 / ais_delays[bin]

    def __lt__(self, other):
        return self.time < other.time

    def __eq__(self, other):
        # return ((abs(self.time - other.time) < ais_spacing_seconds)
        #                    and (self.timestamp == other.timestamp))
        return (abs(self.time - other.time) < (ais_spacing_seconds/2.))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __bool__(self):
        if np.isfinite(self.fp_local): return True
        if np.isfinite(self.td_cyclotron): return True
        if np.isfinite(self.morphology_fp_local): return True
        if np.isfinite(self.integrated_fp_local): return True
        if np.isfinite(self.ground): return True

        if self.traced_delay.size > 0: return True

        return False

    def is_invertible(self, strict=False):
        if strict:
            if ~np.isfinite(self.fp_local):
                return False

        if self.traced_delay.size < 4: return False
        if self.traced_delay.shape != self.traced_frequency.shape: return False
        return True

    def invert(self, substitute_fp=False):
        self.altitude = np.empty(0)
        self.density  = np.empty(0)

        if isinstance(substitute_fp, bool):
            strict = not substitute_fp
        else:
            strict = not np.isfinite(substitute_fp)
        if not self.is_invertible(strict=strict):
            print('None-invertible ionogram')
            return False

        fp0 = self.fp_local
        if ~np.isfinite(fp0):
            if substitute_fp:
                fp0 = substitute_fp
            else:
                print('None-finite fp0')
                return False

        try:
            self.altitude, self.density = laminated_delays(
                            self.traced_delay, self.traced_frequency, fp0)
            self.altitude /= 1.0E3 # into km
            # self.density is already in cm^-3
        except np.linalg.LinAlgError as e:
            # print('Linear algebra error, ' + str(e))
            return False

        pos = mex.iau_mars_position(float(self.time))
        self.altitude = np.sqrt(np.sum(pos**2)) - mex.mars_mean_radius_km + \
            self.altitude

        return True

    def get_filename(self):
        if not self.time:
            return ''
        orb = mex.orbits[self.time]
        if orb:
            return (mex.data_directory +
                            "marsis/ais_digitizations/%05d/%05d.dig" %
                                    ((orb.number // 1000) * 1000, orb.number))

    def plot(self, ax=None, clear=False, **kwargs):
        if not ax:
            ax = plt.gca()
        plt.sca(ax)
        if clear:
            plt.cla()
        plt.semilogx(self.density, self.altitude, **kwargs)

    def __str__(self):
        def parse(s):
            if isinstance(s, np.ndarray):
                return str(s.tolist())
            return str(s)

        f = [str(k) + ' = ' + parse(v) for k, v in self.__dict__.items()]
        return repr(self) + '\n' + '\n'.join(f)

class DigitizationDB():
    """Class that keeps track of the digitizations, only allowing one instance per ais frame."""
    def __init__(self, orbit=None, filename=None, verbose=False, load=True):

        self.filename = 'tmp.dat'
        if filename:
            self.filename = filename

        if orbit:
            self.filename = (mex.data_directory +
                                "marsis/ais_digitizations/%05d/%05d.dig" %
                                ((orbit // 1000) * 1000, orbit))

        self._digitization_list = []
        self.verbose = verbose

        if load:
            if os.path.exists(self.filename):
                self.load(self.filename)
            else:
                print('%s does not exist (yet)' % self.filename)

    def write(self, filename=None):

        if not filename:
            filename = self.filename

        mex.check_create_file(filename)

        with open(filename, 'wb') as f:
            pickle.dump( [dg.to_dict() for dg in self._digitization_list], f)

        return self

    def load(self, filename):
        if not filename:
            filename = self.filename

        if not os.path.exists(filename):
            raise IOError("File %s doesn't exist" % filename)

        if os.stat(filename).st_size == 0:
            raise IOError("File %s is empty" % filename)

        try:
            with open(filename, 'rb') as f:
                try:
                    tmp = pickle.load(f)
                except UnicodeDecodeError as e:
                    print("Unable to read %s - old version?" % filename)
                    tmp = []

            # Some quick sanity check
            if tmp:
                if not isinstance(tmp[0], dict):
                    # raise IOError("File %s does not contain a list of dictionaries - old version")
                    print('File %s has an old/unrecognized format: deleting' % \
                            filename)
                    os.remove(filename)

            self._digitization_list = []
            for t in tmp:
                tt = IonogramDigitization()
                tt.from_dict(t)
                self._digitization_list.append(tt)

        except Exception as e:
            self._digitization_list = []
            raise

        if self.verbose:
            print("Read %d digitizations from %s" % (
                                len(self._digitization_list), filename))
        return self

    def __len__(self):
        return len(self._digitization_list)

    def __contains__(self, digitization):
        return digitization in self._digitization_list

    def add(self, digitization):
        if isinstance(digitization, type([])):
            for d in digitization:
                self.add_single(d)
        else:
            self.add_single(digitization)

    def add_single(self, digitization):
        if 'IonogramDigitization' in str(type(digitization)):
            if not digitization in self._digitization_list:
                self._digitization_list.append(digitization)
            else:
                self._digitization_list.remove(digitization)
                self._digitization_list.append(digitization)
                if self.verbose: print('(Replaced 1 digitization)')
        else:
            raise ValueError("Supply IonogramDigitization objects only: %s" % str(type(digitization)))

    def get_all(self, copy=False):
        if copy:
            return copy_module.deepcopy(self._digitization_list)
        return self._digitization_list

    def get_nearest(self, time, copy=False, strict=False):
        """Return the nearest digitization to *time*"""
        # List comprehensions work fine on empty lists
        # shallow copy
        tmp = self._digitization_list[:]

        if not tmp:
            new_ig = IonogramDigitization()
            new_ig.time = time
            self._digitization_list.append(new_ig)
            self._digitization_list.sort(key=lambda x:x.time)
            if copy:
                return copy_module.deepcopy(new_ig)
            return new_ig

        db_t = np.abs(np.array([d.time for d in tmp]) - time)
        imin = np.argmin(db_t)

        if db_t[imin] > ais_spacing_seconds:
            if strict:
                raise ValueError('Time mis-match by %f seconds' % db_t[imin])
            else:
                new_ig = IonogramDigitization()
                new_ig.time = time
                self._digitization_list.append(new_ig)
                self._digitization_list.sort(key=lambda x:x.time)
                if copy:
                    return copy_module.deepcopy(new_ig)
                return new_ig
        else:
            if copy:
                return copy_module.deepcopy(tmp[imin])
            else:
                return tmp[imin]


def compute_all_digitizations(orbit, filename=None, verbose=False):
    db = DigitizationDB(orbit=orbit, filename=filename, load=False)
    ionogram_list = read_ais(orbit)

    fp_local_counter = 0
    td_cyclotron_counter = 0
    ion_counter = 0
    ground_counter = 0
    saved_count = 0
    # print len(db)
    for e, i in enumerate(ionogram_list):
        if verbose: print(i)
        i.digitization = IonogramDigitization(i)
        if not 'failed' in i.calculate_fp_local().lower(): fp_local_counter += 1
        if not 'failed' in i.calculate_ground_trace().lower(): ground_counter += 1
        if not 'failed' in i.calculate_reflection().lower(): ion_counter += 1
        if not 'failed' in i.calculate_td_cyclotron().lower(): td_cyclotron_counter += 1
        i.delete_binary_arrays()

        if i.digitization:
            # print i.digitization.time, i.digitization.timestamp, i.digitization.fp_local
            i.digitization.set_timestamp()
            db.add(i.digitization)

    db.write()
    result = '%d: Processed %d Ionograms: FP = %d, TD = %d, REFL = %d, GND = %d' % (orbit,
        len(db), fp_local_counter, td_cyclotron_counter, ion_counter, ground_counter)
    print(result)
    return result


def produce_ne_b_file(orbits, file_name='DJA_MARSIS_ne_b_$DATE.txt', use_ais_index=False):
    """docstring for produce_ne_b_file"""

    if '$DATE' in file_name:
        file_name = file_name.replace('$DATE', celsius.utcstr(celsius.now(),'ISOC')[:10])

    f = open(file_name, 'w')
    f.write('# <UTC Time> <Ne / cm^-3> <Ne Error / cm^-3> <B / nT> <B Error / nT>\n')
    f.write("""# Note: For Ne < 600 or B > 70, these values should be considered to be upper and lower limits, respectively\n""")
    no_records = 0
    orb_records = 0
    last_orbit = 0
    tmp_orbit_list = []
    for o in orbits:
        try:
            db = DigitizationDB(orbit=o, load=True)
        except Exception as e:
            print("No data for %d" % o)
            continue

        all_digs = db.get_all()
        for d in all_digs:
            write = False
            ne = np.nan
            ne_err = np.nan
            b = np.nan
            b_err = np.nan

            if hasattr(d, 'fp_local'):
                write = True
                ne, ne_err = fp_to_ne(d.fp_local, d.fp_local_error)
                ne_err = np.abs(ne_err)
                # ne_err = ne - fp_to_ne(d.fp_local + d.fp_local_error)

            if hasattr(d, 'td_cyclotron'):
                write = True
                b, b_err = td_to_modb(d.td_cyclotron, d.td_cyclotron_error)
                b *= 1e9
                b_err *= 1e9
                b_err = np.abs(b_err)
                # b_err = b - td_to_modb(d.td_cyclotron + d.td_cyclotron_error) * 1.E9

            if write:
                orb_records += 1
                no_records += 1
                f.write('%s %f %f %f %f\n' % (celsius.spiceet_to_utcstr(d.time, precision=0), ne, ne_err, b, b_err))

        last_orbit += 1
        tmp_orbit_list.append(o)
        if (last_orbit % 10) == 0:
            print(str(tmp_orbit_list) + (': wrote %d records' % orb_records))
            orb_records = 0
            tmp_orbit_list = []

        del db

    f.close()
    print('Wrote %d records' % (no_records))

def write_yearly_ne_b_files(years=None, directory='.'):
    if years is None:
        years = list(range(2005, int(celsius.utcstr(celsius.now())[:4]) + 1))

    for y in years:
        start = mex.orbits[celsius.spiceet("%04d-001T00:00" % y)].number
        finish = mex.orbits[celsius.spiceet("%04d-001T00:00" % (y+1))].number - 1
        fname = directory + '/' + 'DJA_MARSIS_ne_b_%d.txt' % y
        produce_ne_b_file(list(range(start, finish)), file_name=fname)

def _sync_ais_data(start=1840, finish=None, outfile=None):

    fm = AISFileManager(remote='iowa')

    fm.verbose = True
    if finish is None:
        finish = mex.orbits[celsius.now()].number
        # finish = mex.orbits[max((mex.orbits.keys()))].number
    if outfile is None:
        outfile = data_directory + 'ais_coverage.txt'

    if start < 1840:
        raise ValueError("First AIS orbit was 1840")

    results = {}
    for o in range(start, finish + 1):
        print()
        try:
            f = fm.get_file(o, remote=True)
        except IOError as e:
            print("No file for orbit %d" % o)
            continue

        if f and os.path.exists(f):
            stats = os.stat(f)
            n = stats[stat.ST_SIZE] / 400 / 160
        else:
            n = 0
        results[o] = n
        print('%d, %d' % (o, n))

    with open(outfile, 'wb') as f:
        pickle.dump(results, f)

    fm.verbose = False

def _generate_ais_coverage(start=1, finish=None, outfile=None):
    global file_manager

    file_manager.verbose = False
    if finish is None:
        # finish = mex.orbits[celsius.now()- 86400. * 365].number
        finish = mex.orbits[max((list(mex.orbits.keys())))].number
    if outfile is None:
        outfile = data_directory + 'ais_coverage.txt'

    results = {}
    print(finish + 1)
    for o in range(1840, finish + 1):
        # print
        try:
            f = file_manager.get_file(o, remote=False)
        except IOError as e:
            # print "No file for orbit %d" % o
            results[o] = 0
            continue

        if f and os.path.exists(f):
            stats = os.stat(f)
            n = stats[stat.ST_SIZE] / 400 / 160
        else:
            n = 0
        results[o] = n
        # print '%d, %d' % (o, n)

    with open(outfile, 'wb') as f:
        pickle.dump(results, f)

    file_manager.verbose = False


def get_ais_coverage(f=None):

    if f is None:
        f = data_directory + 'ais_coverage.txt'

    with open(f, 'rb') as f:
        return pickle.load(f)



def plot_ais_coverage_bar(start=None, finish=None, ax=None, height=0.02,
            partial=False, color='green', sharex=False):
    """docstring for plot_mex_orbits_bar"""

    fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    xlims = plt.xlim()

    if sharex:
        if isinstance(sharex, bool):
            sharex = ax
    else:
        sharex = None

    p = ax.get_position()
    new_ax = fig.add_axes([p.x0, p.y1, p.x1 - p.x0, height],
        xticks=[], yticks=[], sharex=sharex)

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

    g = get_ais_coverage()

    for o in orbit_list:
        if (o.start < finish) & (o.finish > start):
            dx = o.finish - o.start
            dy = 1.

            if partial:
                if g[o.number] == 0:
                    continue
                elif g[o.number] < partial:
                    dy = 0.5
            else:
                dy = float(g[o.number]) / 317.
                if dy < 0.05:
                    continue

            plt.fill(x * dx + o.start, y * dy, color=color)
            orbit_count = orbit_count + 1

    plt.ylabel("AIS", rotation='horizontal')
    # plt.sca(ax)
    return new_ax


def get_ais_index(fname=None):
    if fname is None:
        fname = data_directory + 'ais_index.pk'
    try:
        with open(fname, 'rb') as f:
            return pickle.load(f)
    except IOError as e:
        print(e)
        _generate_ais_index(fname)

def _generate_ais_index(outfile=None, update=True, start_orbit=None,
    recompute=False):
    import mars

    # g = get_ais_coverage()
    count = 0

    if outfile is None:
        outfile = data_directory + 'ais_index.pk'

    field_model = mars.CainMarsFieldModel()
    out = {}
    if update:
        start_orbit = None
        with open(outfile, 'rb') as f:
            out = pickle.load(f)
            cout = len(out)
        print("Loaded %d records" % count)
        if start_orbit is None:
            start_orbit = max([k for k in out if np.any(np.isfinite(out[k]['ne'])) > 0])
        print("Starting update at orbit %d" % start_orbit)
    else:
        if start_orbit is None:
            start_orbit = 2366

    for o in range(start_orbit, mex.orbits[celsius.now()].number - 10):
        if start_orbit is not None:
            if o < start_orbit: continue

        # if g[o] < 1:
        #     continue
        try:
            i = read_ais(o)
        except IOError as e:
            print(e)
            continue

        try:
            if recompute:
                compute_all_digitizations(o)
            dgs = DigitizationDB(o).get_all()
        except Exception as e:
            print(e)
            continue

        a = dict()
        a['n'] = len(i)
        count += 1 #a['n']
        a['time'] = np.array([float(ig.time) for ig in i])
        a['mso_rll'], a['mso_pos'], a['sza'] = mex.mso_r_lat_lon_position(a['time'], sza=True, mso=True)
        a['iau_pos'] = mex.iau_r_lat_lon_position(a['time'])
        a['mso_rho'] = np.sqrt(a['mso_pos'][1,:]**2. + a['mso_pos'][2,:]**2.)

        a['b_model'] = field_model(a['iau_pos'])

        ne = np.empty_like(a['time']) + np.nan
        ne_max = np.empty_like(a['time']) + np.nan
        b = np.empty_like(a['time']) + np.nan

        if not dgs:
            compute_all_digitizations(o)
            dgs = DigitizationDB(o).get_all()

        for d in dgs:
            inx = np.argmin(np.abs(a['time'] - d.time))
            ne[inx] = fp_to_ne(d.fp_local)
            b[inx] = td_to_modb(d.td_cyclotron)
            if d.traced_frequency.size:
                ne_max[inx] = fp_to_ne(np.max(d.traced_frequency))

        a['ne'] = ne
        a['b']  = b
        a['ne_max'] = ne_max

        out[o] = a

        print('Orbit %d: %d' % (o, a['n']))

    with open(outfile, 'wb') as f:
        pickle.dump(out,f)
    print('Wrote %d records to %s' % (count, outfile))

def get_ais_data(sza=None, altitude=None, time=None, solar_longitude=None,
        sub_solar_longitude=None, leave_nans_in=False,
        latitude=None, longitude=None, model_magnitude=None,
        model_inclination=None, density_range=None,
        description=False):
    """Returns AIS densities that satisfy criteria. Longitudinal criteria are interpreted as being (L0, DL) to give L0 - DL to L0 + DL (around the circle)"""


    if not 'ais_index' in celsius.datastore:
        print('Loading ais data...', end=' ')
        celsius.datastore['ais_index'] = get_ais_index()
        print(' done')

    data = celsius.datastore['ais_index']

    if not time:
        time = (FREQUENCY_TABLE_FIXED_AFTER, celsius.now())

    ne  = np.array(np.hstack([data[d]['ne'] for d in data]))
    ne_max  = np.array(np.hstack([data[d]['ne_max'] for d in data]))
    b  = np.array(np.hstack([data[d]['b'] for d in data]))
    _time  = np.array(np.hstack([data[d]['time'] for d in data]))
    _sza = np.array(np.hstack([data[d]['sza'] for d in data]))
    alt = np.array(np.hstack([data[d]['iau_pos'][0] for d in data]))
    alt -= mex.mars_mean_radius_km

    lat = np.array(np.hstack([data[d]['iau_pos'][1] for d in data]))
    lon = np.array(np.hstack([data[d]['iau_pos'][2] for d in data]))

    calc_incl = lambda x: np.arctan2(x['b_model'][0],
        np.sqrt(x['b_model'][1]**2. + x['b_model'][2]**2.))
    magnitude = lambda x: np.sqrt(
        x['b_model'][0]**2. + x['b_model'][1]**2. + x['b_model'][2]**2.)
    incl = np.array(np.hstack([calc_incl(data[d]) for d in data]))
    incl *= (180. / np.pi)

    mag = np.array(np.hstack([magnitude(data[d]) for d in data]))

    if not leave_nans_in:
        inx = np.isfinite(_sza * ne)
    else:
        inx = np.ones_like(ne, dtype=np.bool)

    if sza:
        inx = inx & (_sza > sza[0]) & (_sza < sza[1])

    if altitude:
        inx = inx & (alt > altitude[0]) & (alt < altitude[1])

    if time:
        inx = inx & (_time > time[0]) & (_time < time[1])

    if solar_longitude:
        sol_lon = mex.solar_longitude(_time) * 180./np.pi
        diff = np.abs(celsius.angle_difference(sol_lon, solar_longitude[0],
            degrees=True))
        inx = inx & (diff < solar_longitude[1])
    else:
        sol_lon = None

    if sub_solar_longitude:
        sub_sol_lon = mex.sub_solar_longitude(_time)
        diff = np.abs(celsius.angle_difference(sub_sol_lon,
                        sub_solar_longitude[0], degrees=True))
        inx = inx & (diff < sub_solar_longitude[1])
    else:
        sub_sol_lon = None

    if latitude:
        inx = inx & (lat > latitude[0]) & (lat < latitude[1])

    if longitude:
        diff = np.abs(celsius.angle_difference(lon, longitude[0], degrees=True))
        inx = inx & (diff < longitude[1])

    if model_magnitude:
        inx = inx & (mag > model_magnitude[0]) & (mag < model_magnitude[1])

    if model_inclination:
        inx = inx & (incl > model_inclination[0]) & \
            (incl < model_inclination[1])

    if density_range:
        inx = inx & (ne > density_range[0]) & (ne < density_range[1])

    ninx = np.sum(inx)
    print("Returning %d of %d (%.1f%%)" % (ninx, inx.shape[0], 100. * ninx/inx.shape[0]))

    ne    = ne[inx]
    ne_max = ne_max[inx]
    _time = _time[inx]
    _sza  = _sza[inx]
    alt   = alt[inx]
    lat   = lat[inx]
    lon   = lon[inx]
    incl  = incl[inx]
    mag   = mag[inx]
    b     = b[inx]

    return_obj = dict(ne=ne, sza=_sza, alt=alt, lat=lat, lon=lon, time=_time,
                b_mag_model=mag, b=b, incl_model=incl, ne_max=ne_max)

    if sol_lon is not None:
        return_obj['sol_lon'] = sol_lon[inx]

    if sub_sol_lon is not None:
        return_obj['sub_sol_lon'] = sub_sol_lon[inx]

    if description:
        d = []
        if sza:
            d.append(r'%d$^\circ$ < $\chi$ < %d$^\circ$' % (sza[0], sza[1]))
        if altitude:
            d.append('%d < h < %d' % (altitude[0], altitude[1]))
        if time:
            f = lambda x: celsius.utcstr(x, 'ISOC')[:10]
            d.append('%s < t < %s' % (f(time[0]), f(time[1])))

        if solar_longitude:
            d.append(r'$L_S$ = %d $\pm$ %d$^\circ' % \
                (solar_longitude[0], solar_longitude[1]))
        if sub_solar_longitude:
            d.append(r'$\lambda_{Sun}$ = %d $\pm$ %d$^\circ' % \
                (sub_solar_longitude[0], sub_solar_longitude[1]))
        if latitude:
            d.append(r'%d$^\circ$ < $\theta$ < %d$^\circ$' % (
                        latitude[0], latitude[1]))
        if longitude:
            d.append(r'$\lambda$ = %d $\pm$ %d$^\circ' % \
                (longitude[0], longitude[1]))
        if model_magnitude:
            d.append(r'%d < $|B_C|$ < %d' % (
                model_magnitude[0], model_magnitude[1]))
        if model_inclination:
            d.append(r'%d$^\circ$ < $\iota_C$ < %d$^\circ$' % \
                (model_inclination[0], model_inclination[1]))
        if density_range:
            d.append(r'%d < $n_e / cm^{-3}$ < %d' % \
                (density_range[0], density_range[1]))

    if description:
        return return_obj, '; '.join(d)

    return return_obj


if __name__ == '__main__':
    # _sync_ais_data(start=13716)
    # raise RuntimeError()
    _generate_ais_index(recompute=False, update=True)
    _generate_ais_coverage()

    # orbits = range(1840, 10300)
    # produce_ne_b_file(orbits)

    # if True:
    #      a = DigitizationDB('tmp.txt')
    #      for i in range(50):
    #          d = IonogramDigitization()
    #          d.set_time(86400 + i)
    #          d.set_timestamp(86400 + np.random.rand() * 50)
    #          d.set_fp_local(100., 10.)
    #          d.set_cyclotron(1., "MANUAL", 1.)
    #          d.set_trace(np.random.randn(10), np.random.randn(10), method='X')
    #          try:
    #              a.add(d)
    #          except mex.MEXException, e:
    #              pass
    #
    #      a.write('tmp.txt')
    #
    #      s = a.get(start_time = 86400 + 10, finish_time = 86400 + 20)
    #      print len(s)
    #
    #      s = a.remove_old()
    #      print len(a)
    #
    #  if False:
    #      plt.close('all')
    #      filename = file_manager.get_file(8020)
    #      allframes = read_orbit(filename)
    #
    #      print 'Read %d ionograms from file %s' % (len(allframes), filename)
    #
    #      allframes = [allframes[219]]
    #
    #      allframes *= 50
    #      # allframes = allframes[200:]
    #
    #      # ts_fig = plt.figure()
    #      # ts = AISTimeSeries(allframes)
    #      # ts.plot_frequency()
    #      #
    #      # plt.figure()
    #      # ts.plot_spectrum()
    #
    #      fig = plt.figure(figsize=(14, 6), dpi=80)
    #
    #      for i, frame in enumerate(allframes):
    #          plt.clf()
    #          ax = plt.subplot(121)
    #          ax2 = plt.subplot(122)
    #          plt.show()
    #
    #          print frame.time
    #          frame.interpolate_frequencies()
    #          frame.calculate_fft(plot=False)
    #
    #          frame.plot(ax=ax)
    #          plt.xlim(0, 3.)
    #          plt.show()
    #          frame.manual_extract_ionosphere()
    #          frame.inversion(plot=True, ax=ax2)
    #          plt.show()
    #
    #          if len(allframes) > 0:
    #              s = raw_input()
    #              if s != '':
    #                  break
    #
