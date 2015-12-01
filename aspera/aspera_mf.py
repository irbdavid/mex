"""
Functions for plotting some ASPERA spectral data from Fraenz, probably now defunct - was
only ever used for brief intervals of data within the '10/'12 campaigns.

"""

import numpy as np
import matplotlib.pylab as plt
import matplotlib

import scipy
import scipy.io
import mex
import celsius
import glob

__author__ = "D ANDREWS"

def plot_aspera_els(start, finish=None, verbose=False, ax=None, colorbar=True,
                        vmin=None, vmax=None, cmap=None, safe=True):
    """docstring for plot_aspera_els"""
    if cmap is None:
        cmap = plt.cm.Spectral_r

    if ax is None:
        ax = plt.gca()
    plt.sca(ax)

    if finish is None:
        finish = start + 86400.

    if vmin is None:
        vmin = 5.

    if vmax is None:
        vmax = 9.

    no_days = (finish - start) / 86400.

    if verbose: print('Plotting ASPERA/ELS between %s and %s...' % (celsius.utcstr(start, 'ISOC'), celsius.utcstr(finish, 'ISOC')))

    directory = mex.data_directory + 'aspera/els/'

    all_files_to_read = []

    for et in np.arange(start - 10., finish + 10., 86400.):
        dt = celsius.spiceet_to_datetime(et)
        f_name = directory + 'MEX_ELS_EFLUX_%4d%02d%02d_*.cef' % (dt.year, dt.month, dt.day)
        all_day_files = glob.glob(f_name)
        if not all_day_files:
            if verbose: print("No files matched %s" % f_name)
        else:
            all_files_to_read.extend(all_day_files)

    success = False
    all_extents = []
    for f_name in all_files_to_read:
        try:
            # Find energy bins:
            with open(f_name, 'r') as f:
                line_no = 0
                while line_no < 43:
                    line_no += 1
                    line = f.readline()
                    if 'DATA = ' in line:
                        energy_bins = np.fromstring(line[7:], sep=',')
                        energy_bins.sort()
                        break
                else:
                    raise IOError("No ENERGY_BINS info found in header")

            data = np.loadtxt(f_name, skiprows = 43, converters={1:lambda x: celsius.utcstr_to_spiceet(x[:-1])})

            if data.shape[1] != (energy_bins.shape[0] + 2):
                raise ValueError("Size of ENERGY_BINS and DATA doesn't match")

            # Check timing:
            dt = np.diff(data[:,1])
            spacing = np.median(dt)
            # checks = abs(dt - spacing) > (spacing/100.)
            # if np.any(checks):
            #     # raise ValueError("Spacing is not constant: %d differ by more than 1%% of %f:" % (np.sum(checks), spacing))
            #     print "Spacing is not constant: %d differ by more than 1%% of %f (Maximum = %f):" % (np.sum(checks), spacing, max(abs(dt - spacing)))
            #
            # if safe and (max(abs(dt - spacing)) > 10.):
            #     print '-- To big spacing - dumping'
            #     continue

            # Interpolate to constant spacing:
            n_records = int((data[-1,1] - data[0,1]) / spacing)
            new_data = np.empty((n_records, data.shape[1])) + np.nan
            new_data[:,1] = np.linspace(data[0,1], data[-1,1], n_records)
            for i in range(3, data.shape[1]):
                new_data[:,i] = np.interp(new_data[:,1],data[:,1], data[:,i], left=np.nan, right=np.nan)

            data = new_data

            extent = (data[0,1], data[-1,1], energy_bins[0], energy_bins[-1])

            if (extent[0] > finish) or (extent[1] < start):
                if verbose:
                    print("This block not within plot range - dumping")
                continue

            all_extents.append(extent)
            if verbose:
                print('Plotting ASPERA ELS block, Time: %s - %s, Energy: %f - %f' % (
                                celsius.utcstr(extent[0],'ISOC'), celsius.utcstr(extent[1],'ISOC'),
                                extent[2], extent[3]))
                print('Shape = ', data.shape)

            plt.imshow(np.log10(data[:,3:].T), interpolation="nearest", aspect='auto', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
            success = True
        except IOError as e:
            if verbose:
                print('Error reading %f' % f_name)
                print('--', e)
            continue

    if success and colorbar:
        plt.xlim(start, finish)
        plt.ylim(max([e[2] for e in all_extents]), min([e[3] for e in all_extents]))
        celsius.ylabel('E / eV')
        plt.yscale('log')
        cmap.set_under('w')
        old_ax = plt.gca()
        plt.colorbar(cax=celsius.make_colorbar_cax(), cmap=cmap, ticks=[5,6,7,8,9])
        plt.ylabel(r'log$_{10}$ D.E.F.')
        plt.sca(old_ax)


def plot_aspera_ima(start, finish=None, verbose=False, ax=None, colorbar=True, cmap=None, safe=True):
    """docstring for plot_aspera_ima"""

    if cmap is None:
        cmap = plt.cm.Spectral_r

    if ax is None:
        ax = plt.gca()
    plt.sca(ax)

    if finish is None:
        finish = start + 86400.

    no_days = (finish - start) / 86400.

    if verbose: print('Plotting ASPERA/IMA between %s and %s...' % (celsius.utcstr(start, 'ISOC'), celsius.utcstr(finish, 'ISOC')))

    directory = mex.data_directory + 'aspera/ima-all/'

    all_files_to_read = []

    for et in np.arange(start - 10., finish + 10., 86400.):
        dt = celsius.spiceet_to_datetime(et)
        f_name = directory + 'MEX_IMA_ALL_COUNTS_%4d%02d%02d_*.cef' % (dt.year, dt.month, dt.day)
        all_day_files = glob.glob(f_name)
        if not all_day_files:
            if verbose: print("No files matched %s" % f_name)
        else:
            all_files_to_read.extend(all_day_files)

    success = False
    all_extents = []

    for f_name in all_files_to_read:
        try:
            # Find energy bins:
            with open(f_name, 'r') as f:
                line_no = 0
                while line_no < 43:
                    line_no += 1
                    line = f.readline()
                    if 'DATA = ' in line:
                        energy_bins = np.fromstring(line[7:], sep=',')
                        break
                else:
                    raise IOError("No ENERGY_BINS info found in header")

            data = np.loadtxt(f_name, skiprows = 43, converters={1:lambda x: celsius.utcstr_to_spiceet(x[:-1])})

            if data.shape[1] != (energy_bins.shape[0] + 2):
                raise ValueError("Size of ENERGY_BINS and DATA doesn't match")

            # Check timing:
            dt = np.diff(data[:,1])
            spacing = np.median(dt)
            # checks = abs(dt - spacing) > (spacing/100.)
            # if np.any(checks):
            #     # raise ValueError("Spacing is not constant: %d differ by more than 1%% of %f:" % (np.sum(checks), spacing))
            #     print "Spacing is not constant: %d differ by more than 1%% of %f (Maximum = %f):" % (np.sum(checks), spacing, max(abs(dt - spacing)))
            #
            # if safe and (max(abs(dt - spacing)) > (12. * 3)):
            #     print '-- To big spacing - dumping'
            #     continue

            # Interpolate to constant spacing:
            n_records = int((data[-1,1] - data[0,1]) / spacing)
            new_data = np.empty((n_records, data.shape[1])) + np.nan
            new_data[:,1] = np.linspace(data[0,1], data[-1,1], n_records)
            for i in range(3, data.shape[1]):
                new_data[:,i] = np.interp(new_data[:,1],data[:,1], data[:,i], left=np.nan, right=np.nan)

            data = new_data

            extent = (data[0,1], data[-1,1], energy_bins[-1], energy_bins[0])
            all_extents.append(extent)

            if (extent[0] > finish) or (extent[1] < start):
                if verbose:
                    print("This block not within plot range - dumping")
                continue

            if verbose:
                print('Plotting ASPERA IMA block, Time: %s - %s, Energy: %f - %f' % (
                                celsius.utcstr(extent[0],'ISOC'), celsius.utcstr(extent[1],'ISOC'),
                                extent[2], extent[3]))
                print('Shape = ', data.shape)

            data[:,3:] += 1.E-9 # better logging


            plt.imshow(np.log10(data[:,3:].T), interpolation="nearest",  aspect='auto', extent=extent, vmin=0, vmax=2.5, cmap=cmap)
            success = True
        except IOError as e:
            if verbose:
                print('Error reading %f' % f_name)
                print('--', e)
            continue

    if success and colorbar:
        plt.xlim(start, finish)
        plt.ylim(min([e[2] for e in all_extents]), max([e[3] for e in all_extents]))
        celsius.ylabel('E / eV')
        plt.yscale('log')
        cmap.set_under('w')
        old_ax = plt.gca()
        plt.colorbar(cax=celsius.make_colorbar_cax(), ticks=[0,1,2], cmap=cmap)
        plt.ylabel(r'log$_{10}$ Counts')
        plt.sca(old_ax)


# class SpectralPlotter(object):
#     """docstring for SpectraPlotter"""
#     def __init__(self, name, verbose=False):
#         super(SpectraPlotter, self).__init__()
#         self.name = name
#         self.verbose = verbose
#
#     def get_filenames(self,et):
#         """Return a list of filenames which correspond to a given et"""
#         raise NotImplementedError('Derived must override')
#
#     def plot(selfstart, finish=None, verbose=False, ax=None, colorbar=True, cmap=None, safe=True):
#         """docstring for plot_aspera_ima"""
#
#         if cmap is None:
#             cmap = plt.cm.Spectral_r
#
#         if ax is None:
#             ax = plt.gca()
#         plt.sca(ax)
#
#         if finish is None:
#             finish = start + 86400.
#
#         no_days = (finish - start) / 86400.
#
#         if verbose: print 'Plotting %s between %s and %s...' % (self.name, celsius.utcstr(start, 'ISOC'), celsius.utcstr(finish, 'ISOC'))
#
#         directory = mex.data_directory + 'aspera/ima-all/'
#
#         all_files_to_read = []
#
#         for et in np.arange(start - 10., finish + 10., 86400.):
#             all_files_to_read.extend(self.get_filenames(et))
#         if not all_day_files:
#             if verbose: print "No files matched"
#             self.finalize()
#
#         success = False
#         all_extents = []
#
#         for f_name in all_files_to_read:
#             try:
#                 # Find energy bins:
#                 with open(f_name, 'r') as f:
#                     line_no = 0
#                     while line_no < 43:
#                         line_no += 1
#                         line = f.readline()
#                         if 'DATA = ' in line:
#                             energy_bins = np.fromstring(line[7:], sep=',')
#                             break
#                     else:
#                         raise IOError("No ENERGY_BINS info found in header")
#
#                 data = np.loadtxt(f_name, skiprows = 43, converters={1:lambda x: celsius.utcstr_to_spiceet(x[:-1])})
#
#                 if data.shape[1] != (energy_bins.shape[0] + 2):
#                     raise ValueError("Size of ENERGY_BINS and DATA doesn't match")
#
#                 # Check timing:
#                 dt = np.diff(data[:,1])
#                 spacing = np.median(dt)
#                 checks = abs(dt - spacing) > (spacing/100.)
#                 if np.any(checks):
#                     # raise ValueError("Spacing is not constant: %d differ by more than 1%% of %f:" % (np.sum(checks), spacing))
#                     print "Spacing is not constant: %d differ by more than 1%% of %f (Maximum = %f):" % (np.sum(checks), spacing, max(abs(dt - spacing)))
#
#                 if safe and (max(abs(dt - spacing)) > (12. * 3)):
#                     print '-- To big spacing - dumping'
#                     continue
#
#                 extent = (data[0,1], data[-1,1], energy_bins[-1], energy_bins[0])
#                 all_extents.append(extent)
#
#                 if (extent[0] > finish) or (extent[1] < start):
#                     if verbose:
#                         print "This block not within plot range - dumping"
#                     continue
#
#                 if verbose:
#                     print 'Plotting ASPERA IMA block, Time: %s - %s, Energy: %f - %f' % (
#                                     celsius.utcstr(extent[0],'ISOC'), celsius.utcstr(extent[1],'ISOC'),
#                                     extent[2], extent[3])
#                     print 'Shape = ', data.shape
#
#                 data[:,3:] += 1.E-9 # better logging
#
#
#                 plt.imshow(np.log10(data[:,3:].T), interpolation="nearest",  aspect='auto', extent=extent, vmin=0, vmax=2.5, cmap=cmap)
#                 success = True
#             except IOError, e:
#                 if verbose:
#                     print 'Error reading %f' % f_name
#                     print '--', e
#                 continue
#
#         if success and colorbar:
#             plt.xlim(start, finish)
#             plt.ylim(min([e[2] for e in all_extents]), max([e[3] for e in all_extents]))
#             celsius.ylabel('E / eV')
#             plt.yscale('log')
#             cmap.set_under('w')
#             old_ax = plt.gca()
#             plt.colorbar(cax=celsius.make_colorbar_cax(), ticks=[0,1,2], cmap=cmap)
#             plt.ylabel(r'log$_{10}$ Counts')
#             plt.sca(old_ax)
#
#
#
#






if __name__ == '__main__':
    plt.close('all')
    fig, ax = plt.subplots(2,1, sharex=True)

    start = mex.orbits[8000].periapsis - 2.5 * 3600
    finish = mex.orbits[8000].periapsis + 2.5 * 3600
    verbose = True

    plt.set_cmap(plt.cm.Spectral_r)
    plt.sca(ax[0])
    plot_aspera_els(start, finish, verbose=verbose)
    plt.sca(ax[1])
    plot_aspera_ima(start, finish, verbose=verbose)
    plt.vlines((start, finish), *plt.ylim())
    plt.xlim(start - 3600., finish + 3600.)
    plt.show()
