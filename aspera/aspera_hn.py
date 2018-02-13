"""
Code for reading ASPERA data prepared in Kiruna by Hans Nilsson & Leif Kalla

"""

import matplotlib.pylab as plt
import numpy as np
import mex

import celsius

from scipy.io import loadmat
from matplotlib.cm import Spectral, Spectral_r, Greens, Greys
import matplotlib.cm

import os
import pickle
import tempfile
import subprocess

aspera_scan_time = 12. # check dis.

# There's some bug in Hans code (see email) that gives one-record gaps.
# Fudge-fix by stretching, but note we can't then claim exact accuracy of plots
# down to ~seconds!
FUDGE_FIX_HANS_IMA_TIME_BUG = True

MAX_FAILED_REMOTE_FILES = 24*10 # Assume all are missing if we find this many errors, to keep things moving

def read_els(start, finish=None, verbose=False):
    """
    Read ASPERA/ELS data into blocks
    """

    if finish is None:
        finish = start + 1.

    directory = mex.data_directory + 'aspera/nilsson/els/mat/'

    et = start - 3600. #one second before

    out = []
    error_files = []

    remote_failed_count = 0
    # allow_remote = True

    while True:
        if et > (finish + 3600.):
            break

        dt = celsius.spiceet_to_datetime(et)
        fname ='%4d/elec%4d%02d%02d%02d.mat' % (dt.year, dt.year, dt.month,
                dt.day, dt.hour)
        remote_path = 'dja@aurora.irf.se:/irf/data/mars/aspera3/mars/elsdata/'

        if not os.path.exists(directory + fname):
            remote_fname = remote_path + '%04d%02d/%4d%02d%02d%02d0000.mat'
            remote_fname = remote_fname % (dt.year, dt.month,
                dt.year, dt.month, dt.day, dt.hour)

            fd, temp_f_name = tempfile.mkstemp(suffix='tmp.mat')
            command = ('scp', remote_fname, temp_f_name)

            if verbose:
                print('Fetching %s' % remote_fname)
            try:
                # os.spawnvp(os.P_WAIT, command[0], command)
                subprocess.check_call(command)
                # reset fail count on success
                remote_failed_count = 0
            except subprocess.CalledProcessError as e:
                remote_failed_count += 1
                # raise IOError("Retrieval from aurora failed: %s" % str(e))
                print("Retrieval of %s from aurora failed" % remote_fname)
                if remote_failed_count > MAX_FAILED_REMOTE_FILES:
                    print('Maximum failed remote tries reached')
                    break
            else:
                d = os.path.dirname(directory + fname)
                if d and not os.path.exists(d):
                    if verbose:
                        print('Creating %s' % d)
                    os.makedirs(d)

                command = ('mv', temp_f_name, directory + fname)

                try:
                    # os.spawnvp(os.P_WAIT, command[0], command)
                    subprocess.check_call(command)
                except subprocess.CalledProcessError as e:
                    print(e)
                    raise IOError("Error moving file to %s" %
                        (directory + fname))

        try:
            b = loadmat(directory + fname)
        except IOError as e:
            error_files.append(fname)
            if verbose:
                print('Missing: %s' % fname)
            et += 60. * 60.
            continue

        # Squeeze arrays
        for o in b:
            if isinstance(b[o], np.ndarray):
                b[o] = np.squeeze(b[o])

        if verbose:
            print('Read:    %s: %s - %s' % (fname,
                    celsius.time_convert(b['elstimes'][0,0],
                        'UTCSTR', 'MATLABTIME'),
                    celsius.time_convert(b['elstimes'][0,-1],
                        'UTCSTR', 'MATLABTIME')))

#         if b['EElec'].size < 2:
#             print("""----------------------
# read_els: %s contains data that is effectively 1-dimensional (i.e. not an image)
# shape is %s. Ignoring this!
# ---------------------""" % (fname, str(b['fElec'].shape)))
#             et += 60. * 60.
#             continue

        et += 60. * 60.
        out.append(b)

    return out

def plot_els_spectra(start, finish, sector='SUM', colorbar=True,
                        ax=None, blocks=None,
                        return_val=None, verbose=False, check_times=True,
                        vmin=None, vmax=None,
                        cmap=None, norm=None, die_on_empty_blocks=False,
                        **kwargs):
    """
    Plot ASPERA/ELS spectra from start - finish.
    sector = int from 0 - 15 to select single sector to plot.
    sector = 'SUM' to integrate over all
    sector = 'MEAN' to average over all

    Shapes of the stuff in each block (215 = time!)
        altElec (1, 215)
        latElec (1, 215)
        zmsoElec (1, 215)
        dEElec (128, 1)
        ymsoElec (1, 215)
        xmsoElec (1, 215)
        tElec (1, 215)
        fElec (16, 128, 215)
        longElec (1, 215)
        EElec (128, 1)

    New versions:
        elslevels (128, 112)
        elsmatrix (128, 16, 783)
        elstimes (1, 783)
        elssectors (1, 16)
        elsleveltimes (1, 112)
        elsdeltatimes (128, 112)
    """

    if not blocks:
        blocks = read_els(start, finish, verbose=verbose, **kwargs)

    if ax is None:
        ax = plt.gca()
    plt.sca(ax)
    ax.set_axis_bgcolor('lightgrey')

    if not cmap:
        cmap = Spectral_r
        cmap.set_bad('white')

    ims = []

    last_finish = -9E99

    # label = r'Flux / $cm^{-2}s^{-1}$'
    label = 'Counts'

    # Make sure we set the ylimits correctly by tracking min and max energies
    min_energy = 1e99
    max_energy = -1e99

    cbar_ticks = np.array((7,8,9,10,11))

    # Establish function to handle the data
    if isinstance(sector, str):
        if sector.lower() == 'mean':
            process = lambda x: np.nanmean(x,1)
            # if vmin is None:
            #     vmin = 7.
            # if vmax is None:
            #     vmax = 11.

        elif sector.lower() == 'sum':
            process = lambda x: np.nansum(x,1)
            # if vmin is None:
            #     vmin = 7. + 1.
            # if vmax is None:
            #     vmax = 11. + 1.
            # cbar_ticks += 1
        else:
            raise ValueError('Unrecognized argument for sector: %s' % sector)
    else:
        process = lambda x: x[:,sector,:]
        # if vmin is None:
        #     vmin = 7.
        # if vmax is None:
        #     vmax = 11.

    if vmin is None: vmin=0
    if vmax is None: vmax=4

    if not norm:
        norm = plt.Normalize(vmin, vmax, clip=False)

    norm = None

    for b in blocks:
        # min(abs()) to account for negative values in E table
        extent = [celsius.matlabtime_to_spiceet(np.min(b['elstimes'])),
                        celsius.matlabtime_to_spiceet(np.max(b['elstimes'])),
                        np.min(np.abs(b['elslevels'])),
                        np.max(b['elslevels'])]

        if extent[2] < min_energy:
            min_energy = extent[2]

        if extent[3] > max_energy:
            max_energy = extent[3]
        print(extent)
        # if check_times:
        #     spacing = 86400.*np.mean(np.diff(b['tElec']))
        #     if spacing > 15.:
        #         raise ValueError("Resolution not good? Mean spacing = %fs " % spacing )

        img = process(b['elsmatrix'])
        # img[img < 10.] = np.min(img)
        img = np.log10(img)

        if verbose:
            print('Fraction good = %2.0f%%' % (np.float(np.sum(np.isfinite(img))) / (img.shape[0] * img.shape[1]) * 100.))
            # print 'Min, mean, max = ', np.nanmin(img), np.nanmean(img), np.nanmax(img)

        if extent[0] < last_finish:
            s = "Blocks overlap: Last finish = %f, this start = %f" % (
                                    last_finish, extent[0])
            if check_times:
                raise ValueError(s)
            else:
                print(s)

        # e = extent[2]
        # extent[2] = extent[3]
        # extent[3] = e

        ims.append( plt.imshow(img, extent=extent, origin='upper',
                    interpolation='nearest',
                    cmap=cmap, norm=norm) )

        last_finish = extent[1]

    number_of_blocks = len(blocks)

    if blocks:
        plt.xlim(start, finish)
        plt.ylim(min_energy, max_energy)

        cbar_im = ims[0]

    else:
        plt.ylim(1E0, 1E4)
        # need to create an empty image so that colorbar functions
        cbar_im = plt.imshow(np.zeros((1,1)), cmap=cmap, norm=norm,
                visible=False)

    plt.ylim(min_energy, max_energy)
    plt.yscale('log')
    # plt.ylabel('E / eV')
    print('ELS PLOT: Time is not accurate to more than ~+/- 4s for now.')

    if colorbar:
        if ims:
            plt.colorbar(cbar_im, cax=celsius.make_colorbar_cax(), ticks=cbar_ticks).set_label(label)

    if return_val:

        if return_val.upper() == 'IMAGES':
            del blocks
            return ims
        elif return_val.upper() == 'BLOCKS':
            del ims
            return blocks
        else:
            print('Unrecognised return_value = ' + str(return_value))



    del blocks
    del ims
    return number_of_blocks

def test_els_spectra(orbit=6000):
    """docstring for test"""

    start  = mex.orbits[orbit].periapsis - 4. * 3600.
    finish = mex.orbits[orbit].periapsis + 4. * 3600.

    blocks = read_els(start, finish, verbose=True)
    # check_blocks(blocks)
    #
    # b = blocks[0]
    # keys = [k for k in b.keys() if k[0] != '_']
    # keys = [k for k in keys if len(b[k].shape) == 3]
    # n = len(keys) + 2

    plt.close('all')
    fig, axs = plt.subplots(3,1,figsize=celsius.paper_sizes['A4'])
    ax = iter(axs)
    # for i, k in enumerate(keys):
    #     plt.subplot(n, 1, i+1)
    #     plot_ima_spectra(start, finish, species=k, blocks=blocks, verbose=(i==0))
    #     celsius.ylabel(k)

    plt.sca(next(ax))
    plot_els_spectra(start, finish, sector=0, blocks=blocks, verbose=True)

    plt.sca(next(ax))
    plot_els_spectra(start, finish, sector='MEAN', blocks=blocks, verbose=True)

    plt.sca(next(ax))
    plot_els_spectra(start, finish, sector='SUM', blocks=blocks, verbose=True)


    # plt.sca(ax.next())
    # plot_els_spectra(start, finish, sector='MEAN', blocks=blocks, verbose=True)
    #
    # ax = iter(axs)
    # for i in range(16):
    #     plt.sca(ax.next())
    #     plot_els_spectra(start, finish, sector=i, blocks=blocks, verbose=True)




    plt.show()

# ---------------------------------------------------------------------------------------
# IMA IMA IMA IMA IMA IMA IMA IMA IMA IMA IMA IMA IMA IMA IMA IMA IMA IMA IMA IMA IMA

def check_blocks(blocks, check_durations=False, raise_errors=False):
    """All attributes are of identical shape for all blocks in the list
(except in time-duration).  Raises ValueError if they're not.  KeyError if they don't have
the same attributes.
For now, with check_durations=False, just checks the size of the energy axis and assumes
the others are OK (pretty fair assumption).

"""

    if not blocks:
        if not raise_errors:
            return False
        raise ValueError("Empty list of data")

    if len(blocks) == 1:
        if not raise_errors:
            return False
        raise ValueError("Single data block only - can't cross-check")

    if check_durations:
        base = blocks[0]
        for k,v in base.items():
            if isinstance(v, np.ndarray):
                for b in blocks[1:]:
                    if v.shape != b[k].shape:
                        if not raise_errors:
                            return False
                        raise ValueError("Shape mis-match")
    else:
        n_e = blocks[0]['dE'].shape[0]
        for b in blocks[1:]:
            if b['dE'].shape[0] != n_e:
                if not raise_errors:
                    return False
                raise ValueError('Shape mis-match (in energy)')

def read_ima(start, finish=None, dataset='FION', verbose=False, aux=False):
    """Read Nilsson's ima files into a list.
    dataset="fion": best for heavy-ions
    dataset="ion": best for protons, but not much in it.
    dataset="aux": ancillary info
note: setting aux=True also reads the aux files, and appends each files contents into the
principal blocks being read by extending each dictionary."""

    if aux:
        if dataset == 'aux': raise ValueError("""Don't append aux data to the aux data. The whole universe will explode""")

    if finish is None:
        finish = start + 1.

    directory = mex.data_directory + 'aspera/nilsson/Mars_mat_files4/'

    remote_path = 'dja@aurora.irf.se:/irf/data/mars/aspera3/Mars_mat_files4/'

    et = start - 3600. #one second before

    out = []
    error_files = []
    remote_failed_count = 0

    while True:
        if et > (finish + 3600.):
            break

        dt = celsius.spiceet_to_datetime(et)
        fname = dataset.lower() + '%4d%02d%02d%02d00.mat' % (dt.year, dt.month,
                dt.day, dt.hour)

        if not os.path.exists(directory + fname):
            remote_fname = remote_path + fname

            fd, temp_f_name = tempfile.mkstemp(suffix='tmp.mat')
            command = ('scp', remote_fname, temp_f_name)

            if verbose:
                print('Fetching %s' % remote_fname)
            try:
                # os.spawnvp(os.P_WAIT, command[0], command)
                subprocess.check_call(command)

                # reset to zero on success
                remote_failed_count = 0

            except subprocess.CalledProcessError as e:
                remote_failed_count += 1
                # raise IOError("Retrieval from aurora failed: %s" % str(e))
                print("Retrieval of %s from aurora failed" % remote_fname)
                if remote_failed_count > MAX_FAILED_REMOTE_FILES:
                    print('Maximum failed remote tries reached')
                    break
            else:
                d = os.path.dirname(directory + fname)
                if d and not os.path.exists(d):
                    if verbose:
                        print('Creating %s' % d)
                    os.makedirs(d)

                command = ('mv', temp_f_name, directory + fname)

                try:
                    # os.spawnvp(os.P_WAIT, command[0], command)
                    subprocess.check_call(command)
                except subprocess.CalledProcessError as e:
                    print(e)
                    raise IOError("Error moving file to %s" %
                        (directory + fname))

        try:
            out.append(loadmat(directory + fname))

            # Squeeze arrays
            for o in out[-1]:
                if isinstance(out[-1][o], np.ndarray):
                    out[-1][o] = np.squeeze(out[-1][o])

            if verbose:
                print('Read: %s: %s - %s' % (fname,
                        celsius.time_convert(out[-1]['tmptime'][0], 'UTCSTR', 'MATLABTIME'),
                        celsius.time_convert(out[-1]['tmptime'][-1], 'UTCSTR', 'MATLABTIME')
                    ))
        except IOError as e:
            error_files.append(fname)
            if verbose:
                print('Missing: %s' % fname)

        et += 60. * 60.

    if verbose:
        print('Read %d files - %d errors' % (len(out), len(error_files)))

    if aux:
        aux_data_blocks = read_ima(start, finish, dataset='aux', verbose=verbose)

        if len(aux_data_blocks) != len(out):
            raise IOError("Number of aux data files doesn't match the actual data files")

        for block, aux_block in zip(out, aux_data_blocks):
            out['aux'] = aux_block

    return out

def plot_ima_spectra(start, finish, species=['H','O','O2'], colorbar=True,
                        ax=None, blocks=None,
                        check_times=True, return_val=None, verbose=False,
                        check_overlap=True, vmin=2, vmax=7., raise_all_errors=False,
                        cmap=None, norm=None, die_on_empty_blocks=False,
                        accept_new_tables=True, inverted=True,
                        **kwargs):

    """ Plot IMA spectra from start - finish, .
    blocks is a list of data blocks to work from, if this is not specified then we'll read them
    species names: [only those with x will function]
        heavy (16, 85, 272) x
        sumions (85, 32)
        CO2 (16, 85, 272) x
        E (96, 1)
        Horig (16, 85, 272) x
        tmptime (1, 272)
        H (16, 85, 272)
        dE (1, 85)
        sumorigions (85, 32)
        O (16, 85, 272) x
        Hsp (16, 85, 272) x
        mass (50, 272)
        alpha (16, 85, 272) x
        O2 (16, 85, 272) x
        He (16, 85, 272) x

    for blocks read with dataset='fion', we'll add an 'f' before
            CO2, O2, O2plus, O, Horig, He, H
    automagically
    """

    if not blocks:
        blocks = read_ima(start, finish, verbose=verbose, **kwargs)

    if ax is None:
        ax = plt.gca()
    plt.sca(ax)
    ax.set_axis_bgcolor('lightgrey')

    if not cmap:
        cmap = matplotlib.cm.Greys_r
        # cmap.set_under('white')
    if not norm:
        norm = plt.Normalize(vmin, vmax, clip=True)

    ims = []

    last_finish = -9E99

    if blocks:
        if 'fH' in list(blocks[0].keys()): #blocks were read with dataset='fion'
            if isinstance(species, str):
                if species in ('O', 'H', 'He', 'alpha', 'Horig', 'O2', 'O2plus', 'CO2'):
                    species = 'f' + species
            else:
                new_species = []
                for s in species:
                    if s in ('O', 'H', 'He', 'alpha', 'Horig', 'O2', 'O2plus', 'CO2'):
                        new_species.append('f' + s)
                species = new_species

    # if isinstance(species, basestring):
    #     if species[0] == 'f':
    #         label =  species[1:] + r' flux \n / $cm^{-2}s^{-1}$'
    #     else:
    #         label = species + '****'
    # else:
    #     label = ''
    #     for s in species:
    #         if s[0] == 'f':
    #             label += s[1:] + ', '
    #         else:
    #             label += s
    #     label += r' flux \n/ $cm^{-2}s^{-1}$'

    label = r'Flux / $cm^{-2}s^{-1}$'

    # Make sure we set the ylimits correctly by tracking min and max energies
    min_energy = 60000.
    max_energy = -10.

    for b in blocks:
        # min(abs()) to account for negative values in E table
        extent = [celsius.matlabtime_to_spiceet(b['tmptime'][0]),
                        celsius.matlabtime_to_spiceet(b['tmptime'][-1]),
                        min(abs(b['E'])),
                        max(b['E'])]
        # Account for the varying size of the Energy table:
        if  b['sumions'].shape[0] == 96:
            extent[2] = b['E'][-1]
            extent[3] = b['E'][0]
        elif b['sumions'].shape[0] == 85: #revised energy table
            extent[2] = b['E'][-11]
            extent[3] = b['E'][0]
            if extent[2] < 0.:
                raise ValueError('Energies should be positive - unrecognised energy table?')
        else:
            if accept_new_tables:
                extent[2] = np.min(np.abs(b['E']))
                extent[3] = b['E'][0]
                print('New table:', extent[2], extent[3], b['E'][-1], b['E'][0], b['sumions'].shape[0])
            else:
                raise ValueError('Unrecognised energy table: E: %e - %e in %d steps?' %
                                    (b['E'][-1], b['E'][-1], b['sumions'].shape[0])
                                )


        if extent[2] < min_energy:
            min_energy = extent[2]

        if extent[3] > max_energy:
            max_energy = extent[3]

        if check_times:
            spacing = 86400.*np.mean(np.diff(b['tmptime']))
            if spacing > 15.:
                if raise_all_errors:
                    raise ValueError("Resolution not good? Mean spacing = %fs " % spacing)
                else:
                    plt.annotate("Resolution warning:\nmean spacing = %.2fs @ %s " % (
                            spacing, celsius.utcstr(np.median(b['tmptime']))), (0.5, 0.5),
                            xycoords='axes fraction', color='red', ha='center')

        if not isinstance(species, str):
            # produce the MxNx3 array for to up 3 values of species (in R, G, B)
            img = np.zeros((b[species[0]].shape[1], b[species[0]].shape[2], 3)) + 1.
            for i, s in enumerate(species):
                im = np.sum(b[s], 0).astype(np.float64)
                # im[im < 0.01] += 0.1e-10

                if inverted:
                    im = 1. - norm(np.log10(im))
                    for j in (0,1,2):
                        if j != i:
                            img[...,j] *= im
                    tot = np.sum(1. - img)
                else:
                    img[...,i] *= norm(np.log10(im))
                    tot = np.sum(img)

        else:
            img = np.sum(b[species], 0)
            img[img < 0.01] += 0.1e-10
            img = np.log10(img)
            tot = np.sum(img)
        if verbose:
            print('Total scaled: %e' % tot)
            # print 'Fraction good = %2.0f%%' % (np.float(np.sum(np.isfinite(img))) / (img.size) * 100.)
            # print 'Min, mean, max = ', np.min(img), np.mean(img), np.max(img)

        if check_overlap and (extent[0] < last_finish):
            raise ValueError(
                "Blocks overlap: Last finish = %f, this start = %f" %
                                (last_finish, extent[0]))

        if FUDGE_FIX_HANS_IMA_TIME_BUG:
            # print extent, last_finish
            if abs(extent[0] - last_finish) < 20.: # if there's 20s or less between blocks
                if verbose:
                    print('\tFudging extent: Adjusting start by %fs' % (extent[0] - last_finish))
                extent[0] = last_finish# squeeze them together

        if inverted:
            name = cmap.name
            if name[-2:] == '_r':
                name = name[:-2]
            else:
                name = name + '_r'
            cmap = getattr(plt.cm, name)

        if extent[1] < start:
            # if verbose:
            print('Dumping block (B)', start - extent[1])
            continue
        if extent[0] > finish:
            print('Dumping block (A)', extent[0] - finish)
            continue

        ims.append( plt.imshow(img, extent=extent, origin='upper',
                    interpolation='nearest',
                    cmap=cmap, norm=norm) )

        last_finish = extent[1]

    if ims:
        plt.xlim(start, finish)
        plt.ylim(min_energy, max_energy)
        cbar_im = ims[0]
    else:
        plt.ylim(10., 60000) # guess
        # invisible image for using with colorbar
        cbar_im = plt.imshow(np.zeros((1,1)), cmap=cmap, norm=norm, visible=False)

    plt.yscale('log')
    celsius.ylabel('E / eV')

    if colorbar:
        ## make_colorbar_cax is doing something weird to following colorbars...
        # if not isinstance(species, basestring):
        #     img = np.zeros((64,len(species),3)) + 1.
        #     cax = celsius.make_colorbar_cax(width=0.03)
        #     for i, s in enumerate(species):
        #         for j in (0,1,2):
        #             if j != i:
        #                 img[:,i,j] = np.linspace(0., 1., 64)
        #     # plt.sca(cax)
        #     plt.imshow(img, extent=(0, 3, vmin, vmax), interpolation='nearest', aspect='auto')
        #     plt.xticks([0.5, 1.5, 2.5], label.split(', '))
        #     plt.xlim(0., 3.)
        #     cax.yaxis.tick_right()
        #     cax.xaxis.tick_top()
        #     plt.yticks = [2,3,4,5,6]
        #     cax.yaxis.set_label_position('right')
        #     plt.ylabel(r'$Flux / cm^{-2}s^{-1}$')
        #     # plt.sca(ax)
        # else:
        ticks = np.arange(int(vmin), int(vmax)+1, dtype=int)
        plt.colorbar(cbar_im, cax=celsius.make_colorbar_cax(), ticks=ticks).set_label(label)

    if return_val:
        if return_val == 'IMAGES':
            del blocks
            return ims
        elif return_val == 'BLOCKS':
            return blocks
        else:
            print('Unrecognised return_value = ' + str(return_value))

    del blocks
    del ims
    return


def test_ima_spectra(orbit=8020):
    """docstring for test"""

    start  = mex.orbits[orbit].periapsis - 2. * 3600.
    finish = mex.orbits[orbit].periapsis + 2. * 3600.

    blocks = read_ima(start, finish, verbose=True)
    # check_blocks(blocks)
    #
    # b = blocks[0]
    # keys = [k for k in b.keys() if k[0] != '_']
    # keys = [k for k in keys if len(b[k].shape) == 3]
    # n = len(keys) + 2

    plt.close('all')
    plt.figure(figsize=celsius.paper_sizes['A4'])
    # for i, k in enumerate(keys):
    #     plt.subplot(n, 1, i+1)
    #     plot_ima_spectra(start, finish, species=k, blocks=blocks, verbose=(i==0))
    #     celsius.ylabel(k)

    plot_ima_spectra(start, finish, species=('H', 'O', 'O2'), blocks=blocks, verbose=True, inverted=False)

    plt.show()

def plot_ima_mass_matrix(start, finish, product=None, colorbar=True, ax=None, blocks=None, **kwargs):
    """ Integrate to produce a mass-matrix from start-finish, processing azimuths"""

    if (finish - start) < aspera_scan_time:
        raise ValueError('Duration too short: %d' % (finish - start))

    if not blocks:
        blocks = read_ima(start, finish, **kwargs)

    if ax is None:
        ax = plt.gca()

    plt.sca(ax)

    products_to_process = ('CO2', 'H', 'O', 'Hsp', 'alpha', 'O2', 'He')
    if product:
        if not hasattr(product, '__iter__'):
            products_to_process = [product]
        else:
            products_to_process = product

    img = np.zeros_like(blocks[0]['sumions'])

    for b in blocks:
        inx, = np.where((b['tmptime']> start) & (b['tmptime'] < finish))
        if inx.shape[0]:
            for p in products_to_process:
                img += np.sum(b[p][:,:,inx], 2)

    plt.imshow(img, origin='lower')
    if colorbar:
        plt.colorbar(cax=celsius.make_colorbar_cax())

    return img

def test_ima_mass_matrix(orbit=8020):
    pass


def els_coverage(recompute=False):
    """Returns a dictionary, key=orbit number, value = size in MB for the ELS data from that orbit"""

    results = {}
    try:
        with open(mex.data_directory + 'nilsson_els_coverage.pck', 'r') as f:
            results = pickle.load(f)
    except Exception as e:
        recompute = True
        print(e)

    if recompute:
        now = celsius.now()
        directory = mex.data_directory + 'aspera/nilsson/els/mat/'
        results = {}
        for o in mex.orbits:
            t = mex.orbits[o].start
            if t > celsius.now(): break
            t1 = mex.orbits[o].finish
            size = 0
            while t < t1:
                dt = celsius.spiceet_to_datetime(t)
                fname = directory + '%4d/elec%4d%02d%02d%02d00.mat' % (dt.year, dt.year, dt.month, dt.day, dt.hour)

                if os.path.exists(fname):
                    size += os.path.getsize(fname)
                t += 3600.
            results[o] = size / (1024 * 1024) #MB
        with open(mex.data_directory + 'nilsson_els_coverage.pck', 'w') as f:
            pickle.dump(results, f)

        return results

def ima_coverage(recompute=False):
    """Returns a dictionary, key=orbit number, value = size in MB for the IMA data from that orbit"""

    results = {}
    try:
        with open(mex.data_directory + 'nilsson_ima_coverage.pck', 'r') as f:
            results = pickle.load(f)
    except Exception as e:
        recompute = True
        print(e)

    if recompute:
        now = celsius.now()
        directory = mex.data_directory + 'aspera/nilsson/Mars_mat_files4/'
        results = {}
        for o in mex.orbits:
            t = mex.orbits[o].start
            if t > celsius.now(): break
            t1 = mex.orbits[o].finish
            size = 0
            while t < t1:
                dt = celsius.spiceet_to_datetime(t)
                fname = directory + 'fion'+ '%4d%02d%02d%02d00.mat' % (dt.year, dt.month, dt.day, dt.hour)

                if os.path.exists(fname):
                    size += os.path.getsize(fname)
                t += 3600.
            results[o] = size / (1024 * 1024) #MB
        with open(mex.data_directory + 'nilsson_ima_coverage.pck', 'w') as f:
            pickle.dump(results, f)
    return results

EMAILS_FROM_HANS = """
Hi David

The data you want can be found on our new server mimas.irf.se

The server contains a copy of my data, I think it will rsync regularly, but if something
is missing, let me know.

You find the data under an "rsync module" named ima. My calibrated energy spectrograms are
in the folder Mars_mat_files4. The filenames begin with ion, fion or aux, and then
yyyymmddhh00.mat

fion means that the masses where obtained by fitting a function to the mass channels. This
is best for the heavy ions. Note that separation of heavy ions is not quite correct at
energies above some 50 eV, so add all heavy ions together for most purposes. I am working
on an improved "in-flight adjusted" mass calibration, but I am not ready yet.

ion means ion species are obtained though a "table look up" approach. No important
difference to the previous, but heavy ions are less reliable and protons slightly better
because the theoretical function somewhat underestimates the width of the proton
distribution (in mass channels) for some situations.

aux contains auxiliary data like position and looking-direction.

The variables in the fion file are (some samples):

fO (sector x energy x time) Total flux of O (in each energy range), therefore suitable to
sum up for total flux. [cm^-2s^-1] energy is 96 steps (200705 - 200911 if i remember
right), 85 after that.

E energy table, which always has 96 values. The last 11 are negative for newer data so
that there is actually 85 energy steps. I will not discuss here and now why it is like
that...

dE is the estimated relative width of the energy spacing between the measurements, used to
obtain the total flux. To get a delta E in eV for differential flux, multiply by E, i.e.
E(1:85).*dE for newer data with 85 energy steps.

For steradians, divide dphidtheta=4*22.5*pi*pi/180/180;

tmptime time in matlab time

Worth to note: the cleaning of proton contamination (called "ghost" in the Aspera team)
has improved with time. I have not re-run the analysis for older data. A good test which I
use is to check for significant fluxes in the O++ channel. If there are such fluxes, then
the heavy ion data are likely contaminated by proton fluxes (I can describe this better
when we meet). Therefore a code line something like:

ghostdetect=(fO2plus>fO/5) % Divide by 5 is an example of adding some margin.
fO(ghostdetect)=0; can be used to remove any data where we expect some contamination from
protons.

Note that the last data point is systematically missing in much of this data (i.e. 12s per
one hour data) due to a bug...

There is also a useful variable sumions which is energy x masschannel. This shows the raw
mass data and it is easy to identify "ghost" and any other problems with the data. It is a
sum over all the data of the one-hour file.

The aux files contains the following important variables:

x,y,z are unit looking vectors for each sector, energy channels and time (MSO coordinate
system). Note that it is looking directions, not direction of motion of the observed ion.

ixmso, iymso, izmso: position of the spacecraft

You may note that these files are not identical to what I sent to Niklas. I think I sent
files where I had added the heavy ion species already, and summed all sectors together, as
well as done some fO2plus related cleaning as well.

Some instructions from Leif:

exempel:

lista rsync moduler
---------------------------------------------------------------------------- rsync
mimas.irf.se:: ima Aspera-3 IMA data

synkronisera en katalog under modulen:
---------------------------------------------------------------------------- rsync -av mimas.irf.se::ima/mex/Mars_mat_files4 /some_place_good

/Hans

--------------------------------------- -- Regarding ELS:
On Oct 11, 2012, at 14:23, David Andrews wrote:

Hej Leif, Hans,

Thanks for this, I'm very impressed at the speed you have helped us out here! I've not had
any trouble at all getting access to the IMA data.

Hi David

The ELS files just made available to you are exported in calibrated format from the cl
software. I have been told that there is something wrong with the calibration implemented
in cl. The data are perfectly useful for general studies anyway, such as identification of
regions. I will investigate the calibration issue further when I find time for it. If you
need truly quantitative electron flux estimates, then I can hurry up with such an
investigation.

/Hans

I see no reason why we will want precisely calibrated data in the near future. I have no
real desire to get involved in fitting distributions and determining moments, for example.
One thing that it would be useful to know about this however would be if there are any
consequences of these calibration issues that we need to be aware of in comparing the
appearance of the data across successive orbits, or successive times? If the calibration
is reasonably accurate though, then that will be more than sufficient for all the purposes
I foresee us using it.

Thanks again,

David

On 11 Oct 2012, at 13:16, Leif Kalla wrote:

Hej,

ELS data in matlab files can be feteched from mimas:

We have data for 2004 2005 2006 2007 2008 2009 2010 2011

rsync -av rsync://mimas.irf.se/els/mat dest_katalog

Leif

"""

if __name__ == '__main__':
    import sys
    o = 8020
    if len(sys.argv) > 1:
        o = int(sys.argv[1])

    fig, axs = plt.subplots(2,1, sharex=True)

    test_ima_spectra(o)
    # test_ima_mass_matrix(o)
