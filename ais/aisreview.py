import numpy as np
import numpy.random
import matplotlib.pylab as plt
import matplotlib as mpl
import matplotlib.gridspec
import matplotlib.ticker
import matplotlib.cm
from matplotlib.collections import LineCollection

import gc
import sys

import mex
import spiceypy
from . import ais_code
import celsius
import mex.sub_surface

import mars
import mars.chapman
# import mars.field_models as field_models
import os

import markus
import mex.aspera
from . import morgan

import scipy.ndimage.morphology as morphology

from mex.ais import IonogramDigitization

__author__ = "David Andrews"
__copyright__ = "Copyright 2015, David Andrews"
__credits__ = ["David Andrews, Olivier Witasse"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "David Andrews"
__email__ = "david.andrews@irfu.se"
__status__ = "Development"

class AISReview(object):
    """docstring for AISReview"""
    def __init__(self, orbit, start_time=None, finish_time=None, debug=False,
                    fig=None, db_filename=None, marker_size=4.,
                    verbose=False, vmin=None, vmax=None):
        super(AISReview, self).__init__()

        self.orbit = orbit
        self.verbose = verbose

        self.debug = debug
        self.fig = fig
        # if self.fig is None:
        #     self.fig = plt.figure()

        self.calibrator = ais_code.AISEmpiricalCalibration()

        self.mex_orbit = mex.orbits[self.orbit]
        self.start_time = self.mex_orbit.periapsis - 200. * \
                        ais_code.ais_spacing_seconds
        self.finish_time = self.mex_orbit.periapsis + 200. * \
                        ais_code.ais_spacing_seconds

        if start_time:
            self.start_time = start_time

        if finish_time:
            self.finish_time = finish_time

        if vmin is None:
            vmin = ais_code.ais_vmin

        if vmax is None:
            vmax = ais_code.ais_vmax

        self.vmin = vmin
        self.vmax = vmax

        self.marker_size = marker_size
        self.cbar_ticks = np.arange(-16,-8, 2)

        # Don't keep the db active
        if db_filename is not None:
            self.digitization_list = ais_code.DigitizationDB(filename=db_filename,
                            verbose=self.verbose).get_all()
        else:
            self.digitization_list = ais_code.DigitizationDB(orbit=self.orbit,
                                verbose=self.verbose).get_all()

        if self.digitization_list:
            self._newest = celsius.utcstr(float(max([d.timestamp for d in self.digitization_list])),format='C')[:-9]
            self._oldest = celsius.utcstr(float(min([d.timestamp for d in self.digitization_list])),format='C')[:-9]
            print("%d digitizations loaded, produced between: %s and %s" % (
                                            len(self.digitization_list), self._oldest, self._newest))
        else:
            print("No digitizations loaded :(")
            self._newest = np.nan
            self._oldest = np.nan

        self.ionogram_list = ais_code.read_ais(self.orbit)

        for i in self.ionogram_list:
            i.interpolate_frequencies()

        no_linear_frequencies = self.ionogram_list[0].data.shape[1]

        # self.extent = [self.ionogram_list[0].time, self.ionogram_list[-1].time,
        #                 min(self.ionogram_list[0].frequencies) / 1.0E6,
        #                 max(self.ionogram_list[0].frequencies) / 1.0E6]

        self.extent = [self.start_time,
                       self.finish_time,
                       min(self.ionogram_list[0].frequencies) / 1.0E6,
                       max(self.ionogram_list[0].frequencies) / 1.0E6 ]

        s0 =self.mex_orbit.periapsis
        print(self.extent[0] - s0, self.extent[1] - s0)
        print(self.ionogram_list[0].time - s0, self.ionogram_list[-1].time - s0)
        print(min([i.time for i in self.ionogram_list]) - s0, max([i.time for i in self.ionogram_list]) - s0)


        if self.ionogram_list[0].time < self.extent[0]:
            print('WARNING: Pre-extending plot range by %s seconds to cover loaded ionograms' % (
                            self.extent[0] - self.ionogram_list[0].time))
            self.extent[0] = self.ionogram_list[0].time

        if self.ionogram_list[-1].time > self.extent[1]:
            print('WARNING: Post-extending plot range by %s seconds to cover loaded ionograms' % (
                        self.ionogram_list[-1].time - self.extent[1]))
            self.extent[1] = self.ionogram_list[-1].time

        no_ionograms_expected = ((self.extent[1] - self.extent[0])
                                                        / ais_code.ais_spacing_seconds + 1)

        no_ionograms_expected = int(round(no_ionograms_expected))
        self.tser_arr_all = np.empty((ais_code.ais_number_of_delays, no_linear_frequencies,
            no_ionograms_expected))

        ilast = None
        empty_count = 0
        for i, ig in enumerate(self.ionogram_list):
            ignum = int( round((ig.time - (self.extent[0] + ais_code.ais_spacing_seconds)) / ais_code.ais_spacing_seconds ))
            if ignum > (no_ionograms_expected-1):
                raise mex.MEXException("Out of range %d, %d, %d"
                    % (len(self.ionogram_list), ignum, no_ionograms_expected))

            ig.interpolate_frequencies()
            self.tser_arr_all[:,:,ignum] = ig.data
            if ilast is not None:
                if (ignum != (ilast + 1)):
                    empty_count += 1
                    self.tser_arr_all[:,:,ilast+1:ignum-1] = -9E99
            ilast = ignum

        if empty_count:
            print('Found %d empty ionograms / missing data' % empty_count)
        errs = np.geterr()
        np.seterr(divide='ignore')
        self.tser_arr = np.log10(np.mean(self.tser_arr_all[::-1,:, :], axis=0))
        self.tser_arr_all = np.log10(self.tser_arr_all)
        np.seterr(**errs)

        # Trajectory info
        # self.trajectory = {}
        # self.trajectory['t'] = np.arange(self.extent[0], self.extent[1], 60.)
        # pos = mex.iau_mars_position(self.trajectory['t'])
        # self.trajectory['pos'] = pos / mex.mars_mean_radius_km

        self.field_model = mars.CainMarsFieldModel(nmax=60)
        # self.quick_field_model = mars.CainMarsFieldModelAtMEX()
        self.generate_position()
        self.ionosphere_model = mars.Morgan2008ChapmanLayer()

    def plot_timeseries(self, ax=None, vmin=None, vmax=None,
            colorbar=False, label=True):

        if vmin is None:
            vmin = self.vmin

        if vmax is None:
            vmax = self.vmax

        if ax is None:
            ax = plt.gca()
        plt.sca(ax)
        plt.cla()
        plt.imshow(self.tser_arr[::-1,:], vmin=vmin, vmax=vmax,
            interpolation='Nearest', extent=self.extent, origin='upper',aspect='auto')
        plt.xlim(self.extent[0], self.extent[1])
        plt.ylim(self.extent[2], self.extent[3])
        # plt.vlines(self.ionogram_list[0].time, self.extent[2], self.extent[3], 'r')
        if label:
           celsius.ylabel('f / MHz')

        if colorbar:
            old_ax = plt.gca()
            plt.colorbar(cax = celsius.make_colorbar_cax(), ticks=self.cbar_ticks).set_label(r"$Log_{10} V^2 m^{-2} Hz^{-1}$")
            plt.sca(old_ax)

    def plot_frequency(self, f=2.0, ax=None, median_filter=False,
        vmin=None, vmax=None, colorbar=False):

        if vmin is None:
            vmin = self.vmin

        if vmax is None:
            vmax = self.vmax

        if ax is None:
            ax = plt.gca()
        plt.sca(ax)
        plt.cla()
        freq_extent = (self.extent[0], self.extent[1],
            ais_code.ais_max_delay*1E3, ais_code.ais_min_delay*1E3)

        i = self.ionogram_list[0]
        inx = 1.0E6* (i.frequencies.shape[0] * f) / (i.frequencies[-1] - i.frequencies[0])
        img = self.tser_arr_all[:,int(inx),:]
        # img -= np.mean(img, 0)
        plt.imshow(img, vmin=vmin, vmax=vmax,
            interpolation='Nearest', extent=freq_extent, origin='upper',aspect='auto')
        # plt.imshow(img, interpolation='Nearest', extent=freq_extent, origin='upper',aspect='auto')

        plt.xlim(freq_extent[0], freq_extent[1])
        plt.ylim(freq_extent[2], freq_extent[3])
        # plt.vlines(i.time,freq_extent[2],freq_extent[3], 'r')
        celsius.ylabel(r'$\tau_D / ms$' '\n' '%.1f MHz' % f)
        # plt.annotate('f = %.1f MHz' % f, (0.02, 0.9), xycoords='axes fraction',
        #             color='grey', verticalalignment='top', fontsize='small')

        if colorbar:
            old_ax = plt.gca()
            plt.colorbar(cax = celsius.make_colorbar_cax(),
                        ticks=self.cbar_ticks).set_label(
                                r"$Log_{10} V^2 m^{-2} Hz^{-1}$")
            plt.sca(old_ax)

    def plot_frequency_range(self, f_min=0., f_max=0.2, ax=None, median=False,
        vmin=None, vmax=None, colorbar=False):

        if vmin is None:
            vmin = self.vmin

        if vmax is None:
            vmax = self.vmax

        if ax is None:
            ax = plt.gca()
        plt.sca(ax)
        plt.cla()
        freq_extent = (self.extent[0], self.extent[1],
            ais_code.ais_max_delay*1E3, ais_code.ais_min_delay*1E3)

        i = self.ionogram_list[0]
        inx, = np.where((i.frequencies > f_min*1E6) & (i.frequencies < f_max*1E6))

        if inx.shape[0] < 2:
            raise ValueError("Only %d frequency bins selected." % inx.shape[0])
        print("Averaging over %d frequency bins" % inx.shape[0])

        if median:
            if inx.shape[0] < 3:
                raise ValueError("Median here only really makes sense for 3 or more bins")
            img = np.median(self.tser_arr_all[:,inx,:],1)
        else:
            img = np.mean(self.tser_arr_all[:,inx,:],1)
        plt.imshow(img, vmin=vmin, vmax=vmax,
            interpolation='Nearest', extent=freq_extent, origin='upper',aspect='auto')

        plt.xlim(freq_extent[0], freq_extent[1])
        plt.ylim(freq_extent[2], freq_extent[3])
        # plt.vlines(i.time,freq_extent[2],freq_extent[3], 'r')
        celsius.ylabel(r'$\tau_D / ms$' '\n' '%.1f-%.1f MHz' % (f_min, f_max))
        # plt.annotate('f = %.1f - %.1f MHz' % (f_min, f_max), (0.02, 0.9), xycoords='axes fraction',
            # color='grey', verticalalignment='top', fontsize='small')

        if colorbar:
            old_ax = plt.gca()
            plt.colorbar(cax = celsius.make_colorbar_cax(), ticks=self.cbar_ticks).set_label(r"$Log_{10} V^2 m^{-2} Hz^{-1}$")
            plt.sca(old_ax)

    def plot_frequency_altitude(self, f=2.0, ax=None, median_filter=False,
        vmin=None, vmax=None, altitude_range=(-99.9, 399.9), colorbar=False, return_image=False):

        if vmin is None:
            vmin = self.vmin

        if vmax is None:
            vmax = self.vmax

        if ax is None:
            ax = plt.gca()

        plt.sca(ax)
        plt.cla()
        freq_extent = (self.extent[0], self.extent[1],
            altitude_range[1], altitude_range[0])

        i = self.ionogram_list[0]
        inx = 1.0E6* (i.frequencies.shape[0] * f) / (i.frequencies[-1] - i.frequencies[0])
        img = self.tser_arr_all[:,int(inx),:]

        new_altitudes = np.arange(altitude_range[0], altitude_range[1], 14.)
        new_img = np.zeros((new_altitudes.shape[0], img.shape[1])) + np.nan

        for i in self.ionogram_list:
            e = int( round((i.time - self.extent[0]) / ais_code.ais_spacing_seconds ))

            pos = mex.iau_r_lat_lon_position(float(i.time))
            altitudes = pos[0] - ais_code.speed_of_light_kms * ais_code.ais_delays * 0.5 - mex.mars_mean_radius_km
            s = np.argsort(altitudes)
            new_img[:, e] = np.interp(new_altitudes, altitudes[s], img[s,e], left=np.nan, right=np.nan)

        plt.imshow(new_img, vmin=vmin, vmax=vmax,
            interpolation='Nearest', extent=freq_extent, origin='upper', aspect='auto')

        plt.xlim(freq_extent[0], freq_extent[1])
        plt.ylim(*altitude_range)

        ax.set_xlim(self.extent[0], self.extent[1])
        ax.xaxis.set_major_locator(celsius.SpiceetLocator())

        celsius.ylabel(r'Alt./km')
        plt.annotate('f = %.1f MHz' % f, (0.02, 0.9), xycoords='axes fraction',
            color='cyan', verticalalignment='top', fontsize='small')
        if colorbar:
            old_ax = plt.gca()
            plt.colorbar(cax = celsius.make_colorbar_cax(), ticks=self.cbar_ticks).set_label(r"$Log_{10} V^2 m^{-2} Hz^{-1}$")
            plt.sca(old_ax)

        if return_image:
            return new_img, freq_extent, new_altitudes

    def plot_mod_b(self, fmt='k.', ax=None,
                    field_model=True, errors=True, field_color='blue',
                    br=True, t_offset=0., label=True, **kwargs):
        if ax is None:
            ax = plt.gca()
        plt.sca(ax)

        sub = [d for d in self.digitization_list if np.isfinite(d.td_cyclotron)]
        if len(sub) == 0:
            print("No digitizations with marked cyclotron frequency lines")
            return

        t = np.array([d.time for d in sub])
        b = np.array([d.td_cyclotron for d in sub])
        e = np.array([d.td_cyclotron_error for d in sub])
        # print b
        # print e

        b, e = ais_code.td_to_modb(b, e)
        b *= 1.E9
        e *= 1.E9

        if errors:
            for tt,bb,ee in zip(t,b,e):
                plt.plot((tt,tt),(bb+ee,bb-ee),
                                color='lightgrey',linestyle='solid',marker='None')
                plt.plot(tt,bb,fmt,ms=self.marker_size, **kwargs)
            # plt.errorbar(t, b, e, fmt=fmt, ms=self.marker_size, **kwargs)
        else:
            plt.plot(t, b, fmt, ms=self.marker_size, **kwargs)

        if field_model:
            self.generate_position()

            if field_color is None: field_color = fmt[0]
            # b = self.quick_field_model(self.t)
            b = self.field_model(self.iau_pos)
            plt.plot(self.t - t_offset, np.sqrt(np.sum(b**2., 0)),
                        color=field_color, ls='-')
            if br:
                plt.plot(self.t - t_offset, b[0], 'r-')
                plt.plot(self.t - t_offset, -1. * b[0], 'r', ls='dashed')

        if label:
            celsius.ylabel(r'$\mathrm{|B|/nT}$')
        plt.ylim(0., 200)

    def plot_ne(self, fmt='k.', ax=None, errors=True, label=True,
        marsis=True, aspera=False, full_marsis=False, **kwargs):
        if ax is None:
            ax = plt.gca()
        plt.sca(ax)

        def parse_error(d):
            if not np.isfinite(d.fp_local):
                return
            f = ais_code.fp_to_ne(d.fp_local)
            f0 = ais_code.fp_to_ne(d.fp_local + d.fp_local_error)
            f1 = ais_code.fp_to_ne(d.fp_local - d.fp_local_error)
            if errors:
                plt.plot((d.time, d.time),(f0,f1),
                    color='lightgrey', linestyle='solid',
                    marker='None', zorder=-1000,**kwargs)
            plt.plot(d.time, f, fmt, ms=self.marker_size, zorder=1000,**kwargs)

            if full_marsis and hasattr(d, 'maximum_fp_local'):
                plt.plot(d.time, ais_code.fp_to_ne(d.maximum_fp_local),
                    'b.', ms=self.marker_size, zorder=900, **kwargs)

            if full_marsis:
                if np.isfinite(d.morphology_fp_local):
                    v, e = ais_code.fp_to_ne(d.morphology_fp_local,
                                                d.morphology_fp_local_error)
                    plt.errorbar(float(d.time), v, yerr=e,
                            marker='x', ms=1.3, color='purple',
                            zorder=1e90, capsize=0., ecolor='plum')

                if np.isfinite(d.integrated_fp_local):
                    v, e = ais_code.fp_to_ne(d.integrated_fp_local,
                                                d.integrated_fp_local_error)
                    plt.errorbar(float(d.time), v, yerr=e,
                            marker='x', ms=1.3, color='blue',
                            zorder=1e99, capsize=0., ecolor='cyan')

            # if hasattr(d, 'fp_local_length'):
            #     if d.fp_local_length > 40.:
            #         plt.scatter(d.time, f, s=(float(d.fp_local_length)/80. *)**2. * 5., color='k')
            # plt.errorbar(d.time, f, df, fmt='k', marker='None')

        list(map(parse_error, self.digitization_list))
        print("FP_LOCAL: %d" % len(
                    [d for d in self.digitization_list if np.isfinite(d.fp_local)]))

        ax.set_yscale('log')
        plt.ylim(11., 1.1E5)

        if label:
            # celsius.ylabel(r'$\mathrm{n_e / cm^{-3}}$')
            plt.ylabel(r'n$_e$ / cm$^{-3}$')

            # plt.twinx()
            # s = np.array([[ 0.,  0., ],
            #                 [ 1.,  1.,  ]]).T
            # for i in self.ionogram_list:
            #     i.threshold_data()
            #     plt.plot(float(i.time), np.sum(morphology.binary_hit_or_miss(i.thresholded_data, s)), 'go',ms=1.3)

    def plot_peak_altitude(self, ax=None):
        if ax is None:
            ax = plt.gca()
        plt.sca(ax)

        for d in self.digitization_list:
            if d.is_invertible():
                if d.altitude.size == 0:
                    try:
                        d.invert(substitute_fp=ais_code.ne_to_fp(4.))
                    except BaseException as e:
                        print(e)
                        continue
                plt.plot(d.time, d.altitude[-1], 'k.', ms=self.marker_size)
                alt = mex.iau_pgr_alt_lat_lon_position(float(d.time))[0]
                plt.plot(d.time, alt - d.traced_delay[-1] * ais_code.speed_of_light_kms / 2., 'rx', ms=self.marker_size)
        celsius.ylabel(r'$h_{max} / km$')
        plt.ylim(0o1, 249)

    def plot_peak_density(self, ax=None):
        if ax is None:
            ax = plt.gca()
        plt.sca(ax)

        for d in self.digitization_list:
            if d.is_invertible():
                if d.altitude.size == 0:
                    try:
                        d.invert(substitute_fp=ais_code.ne_to_fp(4.))
                    except BaseException as e:
                        print(e)
                        continue
                plt.plot(d.time, d.density[-1], 'k.', ms=self.marker_size)

        celsius.ylabel(r'$n_{e,max} / cm^{-3}$')
        ax.set_yscale('log')
        plt.ylim(1E4, 5E5)

    def plot_ground_deltat(self, ax=None):
        if ax is None:
            ax = plt.gca()
        plt.sca(ax)
        t = [d.time for d in self.digitization_list if np.isfinite(d.ground)]
        d = [d.ground for d in self.digitization_list if np.isfinite(d.ground)]
        dnew = []
        for time, delay in zip(t, d):
            mex_pos = mex.iau_mars_position(float(time))
            alt = np.sqrt(np.sum(mex_pos * mex_pos)) - mex.mars_mean_radius_km
            dnew.append( (delay - alt * 2. / ais_code.speed_of_light_kms) * 1.0E3)
        plt.plot(t, dnew)

        celsius.ylabel(r'$\Delta\tau_D$ / ms')

    def plot_profiles(self, ax=None, vmin=4., vmax=5.5, cmap=None,
                                cmticks=None, log=True, substitute_fp=None, **kwargs):
        """
        Inverts the profiles, plots them versus time and altitude, color coded to density
        Does not show the 'first step' between the S/C plasma density and the first reflection
        """
        if ax is None:
            ax = plt.gca()
        plt.sca(ax)
        ax.set_axis_bgcolor("gray")

        if cmap is None:
            cmap = matplotlib.cm.hot
        if cmticks is None:
            cmticks = [3, 4, 5, 6]
        n_profiles = int((self.extent[1] - self.extent[0]) / ais_code.ais_spacing_seconds) + 1
        ranges = np.arange(80., 350., 1.)
        times  = np.arange(self.extent[0], self.extent[1], ais_code.ais_spacing_seconds)
        img = np.zeros((len(times), len(ranges))) + np.nan

        label = r'$n_e$ / cm$^{-3}$'
        if log:
            f = np.log10
            label = r'log ' + label
        else:
            f = lambda x: x

        if substitute_fp is None:
            subf = lambda t: 0.
        else:
            if hasattr(substitute_fp, '__call__'):
                subf = lambda t: substitute_fp(t)
            else:
                subf = lambda t: substitute_fp

        for i, d in enumerate(self.digitization_list):
            if d.is_invertible():
                d.invert(substitute_fp=subf(d.time))
                if d.altitude.size:
                    ii = round((float(d.time) - times[0]) / ais_code.ais_spacing_seconds)
                    img[ii,:] = np.interp(ranges, d.altitude[-1:0:-1],
                                            f(d.density[-1:0:-1]),
                                            right=np.nan, left=np.nan)[::-1]

        extent = (self.extent[0], self.extent[1], np.nanmin(ranges), np.nanmax(ranges))
        plt.imshow(img.T, interpolation='Nearest', extent=extent,
                origin='upper',aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)

        celsius.ylabel("alt / km")

        old_ax = plt.gca()
        plt.colorbar(cax = celsius.make_colorbar_cax(), ticks=cmticks, **kwargs).set_label(label)
        plt.sca(old_ax)

    def plot_profiles_delta(self, ax=None):
        if ax is None:
            ax = plt.gca()
        plt.sca(ax)
        ax.set_axis_bgcolor("gray")

        n_profiles = int((self.extent[1] - self.extent[0]) / ais_code.ais_spacing_seconds) + 1
        ranges = np.arange(80., 300., 1.)
        times  = np.arange(self.extent[0], self.extent[1], ais_code.ais_spacing_seconds)
        img = np.zeros((len(times), len(ranges))) + np.nan

        for i, d in enumerate(self.digitization_list):
            if d.is_invertible():
                d.invert(substitute_fp=ais_code.ne_to_fp(4.))
                if d.altitude.size:
                    # as it comes from invert, local values are first, peaks are last
                    ii = round((float(d.time) - times[0]) / ais_code.ais_spacing_seconds)
                    img[ii,:] = np.interp(ranges, d.altitude[::-1],
                                            np.log10(d.density[::-1]),
                                            right=np.nan, left=np.nan)[::-1]

                    # This gives departures from the curve defined by the extrapolation to the first reflection point
                    # h = -1. * (d.altitude[0] - d.altitude[1]) / (np.log(d.density[0]/d.density[1]))
                    # img[ii, :] = np.log10(10.**img[ii,:] - d.density[1] * np.exp(-1. * (ranges - d.altitude[1]) / h))

                    # This gives departures from the Morgan et al model
                    sza = np.interp(d.time, self.t, self.sza)
                    # if sza < 90.:
                    # chap = mars.ChapmanLayer()
                    # # inx, = np.where(np.isfinite(img[ii,:]))
                    # # chap.fit(ranges[inx], 10**img[ii,inx[::-1]], sza)
                    # try:
                    #     chap.fit(d.altitude[::-1], d.density[::-1], np.deg2rad(sza))
                    #     model_densities = chap(ranges, np.deg2rad(sza))
                    #     print chap
                    #     print 'peak:', d.altitude[-1], d.density[-1], sza
                    # except ValueError, e:
                    #     print e
                    #     continue

                    model_densities = self.ionosphere_model(ranges, np.deg2rad(sza))
                    # img[ii, :] = np.log10(10.**img[ii,:] - model_densities)
                    img[ii, :] = 10.**img[ii,:] - model_densities[::-1]

                    # if True:
                    #      ax = plt.gca()
                    #      plt.figure()
                    #      plt.plot(np.log10(d.density), d.altitude,'k-')
                    #      if sza < 90.:
                    #          plt.plot(np.log10(model_densities), ranges,'r-')
                    #          # plt.plot(np.log10(chap(ranges)), ranges, 'r', ls='dotted')
                    #          # plt.plot(np.log10(self.ionosphere_model(ranges, np.deg2rad(sza))), ranges, 'b-')
                    #          # plt.plot(np.log10(self.ionosphere_model(ranges)), ranges, 'b', ls='dotted')
                    #      plt.xlim(2,6)
                    #      plt.sca(ax)

        extent = (self.extent[0], self.extent[1], np.nanmin(ranges), np.nanmax(ranges))
        plt.imshow(img.T, interpolation='Nearest', extent=extent,
                origin='upper',aspect='auto', cmap=matplotlib.cm.RdBu_r, vmin=-1E5,vmax=1E5)

        celsius.ylabel("alt. / km")

        old_ax = plt.gca()
        plt.colorbar(cax = celsius.make_colorbar_cax(), ticks=[-1E5,0,1E5],format='%.1G')
        plt.ylabel(r"log $\Delta n_e$ / cm$^{-3}$")
        plt.sca(old_ax)


    def plot_tec(self, ax=None, ss=True, verbose=False):
        if ax is None:
            ax = plt.gca()
        plt.sca(ax)

        n_profiles = int((self.extent[1] - self.extent[0]) / ais_code.ais_spacing_seconds) + 1
        ranges = np.arange(80., 300., 1.)
        times  = np.arange(self.extent[0], self.extent[1], ais_code.ais_spacing_seconds)
        img = np.zeros((len(times), len(ranges))) + np.nan

        for i, d in enumerate(self.digitization_list):
            if d.is_invertible():
                d.invert(substitute_fp=False)
                if verbose:
                    print(i, d.altitude.size)
                if d.altitude.size:
                    # This tries to give an estimate of sub-solar TEC
                    # try:
                    #     c = mars.chapman.ChapmanLayer()
                    #     c.fit(d.altitude[::-1], d.density[::-1], np.interp(d.time, self.t, self.sza))
                    #     plt.plot(d.time, c.n0 * c.h* 1E3 * 1e6, 'b.', mew=0.)
                    # except ValueError, e:
                    #     print e

                    # This just estimates sub-spacecraft TEC (no SZA correction)
                    # Scale height is coarsely estimated from the height over which a drop
                    # in density by factor e is observed
                    alt_diff = d.altitude[::-1]
                    alt_diff = alt_diff - alt_diff[0]
                    density = d.density[::-1]
                    inx, = np.where(density < (density[0] / 2.71828))
                    # print '------'
                    # print density
                    # print alt_diff
                    if inx.shape[0] > 0:
                        if verbose:
                            print('>', d.time, density[0] * alt_diff[inx[0]] * 1E3 * 1e6, alt_diff[inx[0]])

                        if alt_diff[inx[0]] > 100.: continue
                        plt.plot(d.time, density[0] * alt_diff[inx[0]] * 1E3 * 1e6, 'k.')
                    # print celsius.utcstr(float(d.time)), c.n0 * c.h * 1E3 * 1e6

                    ## This was an attempt to try and get a more robust scale height, by
                    ## just looking near the peak - gives way too big values though, because
                    ## it doesn't look to the SZA, for one.
                    ## 2012-07-16
                    # s = np.argsort(d.altitude)
                    # dens = d.density[s]
                    # alt = d.altitude[s]
                    # if dens.shape[0] < 10: continue
                    # dens = dens[:10]
                    # alt = alt[:10] - alt[0]
                    # sn = np.sum(dens)
                    # b = (sn * np.sum(alt * dens * np.log(dens))
                    #         - np.sum(alt * dens) * np.sum(dens * np.log(dens)))
                    # b = b / (sn * np.sum(alt**2. * dens) - np.sum(alt * dens)**2.)
                    # plt.plot(d.time, dens[0] * b * 1E9, 'r.', mew=0.)
                    # print '---', b, -1. / b, dens[0] * -1. / b * 1E9, np.max(d.density), dens[0]
                else:
                    pass
            else:
                if verbose:
                    print(i, ' not invertible')
                pass
        if ss:
            ss_tec = mex.sub_surface.read_tec(self.start_time, self.finish_time)
            good = ss_tec['FLAG'] == 1
            plt.plot(ss_tec['EPHEMERIS_TIME'][good], ss_tec['TEC'][good], 'r.', mew=0., mec='r')

        plt.ylim(2e14, 9E16)
        plt.yscale('log')
        celsius.ylabel(r"$TEC / m^{-2}$")

    def generate_position(self):
        if not hasattr(self, 't'):
            print("Generating position information...")
            self.t = np.arange(float(self.extent[0]), float(self.extent[1]), ais_code.ais_spacing_seconds)
            if self.debug: print("%d points..." % self.t.size)
            self.pos, self.mso_pos, self.sza = mex.mso_r_lat_lon_position(self.t, sza=True, mso=True)
            if self.debug: print("...stage 2")
            self.iau_pos = mex.iau_r_lat_lon_position(self.t)
            self.mso_rho = np.sqrt(self.mso_pos[1,:]**2. + self.mso_pos[2,:]**2.)
            if self.debug: print('...done.')

    def plot_r(self, ax=None, label=True, fmt='k-', **kwargs):
        if ax is None:
            ax = plt.gca()
        plt.sca(ax)
        self.generate_position()
        plt.plot(self.t, self.iau_pos[0] / mex.mars_mean_radius_km, fmt, **kwargs)
        celsius.ylabel(r'$r / R_M$')

    def plot_altitude(self, ax=None, label=True, fmt='k-', **kwargs):
        if ax is None:
            ax = plt.gca()
        plt.sca(ax)
        self.generate_position()
        plt.plot(self.t, self.iau_pos[0] - mex.mars_mean_radius_km, fmt, **kwargs)
        if label:
            celsius.ylabel('alt. / km')

    def plot_lat(self, ax=None, label=True, fmt='k-', **kwargs):
        if ax is None:
            ax = plt.gca()
        plt.sca(ax)
        self.make_axis_circular(ax)
        self.generate_position()
        plt.plot(self.t, self.iau_pos[1], fmt, **kwargs)
        if label:
            celsius.ylabel(r'$\lambda$')
        plt.ylim(-90., 90.)

    def plot_lon(self, ax=None, label=True, fmt='k-', **kwargs):
        if ax is None:
            ax = plt.gca()
        plt.sca(ax)
        self.make_axis_circular(ax)
        self.generate_position()
        v = celsius.deg_unwrap(self.iau_pos[2])
        for i in [-1,0,1]:
            plt.plot(self.t, v + i * 360, fmt, **kwargs)
        if label:
            celsius.ylabel(r'$\varphi$')
        plt.ylim(0., 360.)

    def plot_sza(self, ax=None, label=True, fmt='k-', **kwargs):
        if ax is None:
            ax = plt.gca()
        plt.sca(ax)
        self.make_axis_circular(ax)
        self.generate_position()
        plt.plot(self.t, self.sza, fmt, **kwargs)
        if label:
            celsius.ylabel(r'$\theta_{SZ}$')
        plt.ylim(0., 180.)

    def make_axis_circular(self, ax):
        ax.yaxis.set_major_locator(celsius.CircularLocator(nmax=5))

    def density_along_orbit(self, ax=None, annotate=True, min_fp_local_length=0, bg_color='dimgrey', cmap=None, vmin=1., vmax=3.):
        if cmap is None:
            cmap = plt.cm.autumn
            cmap.set_bad('dimgrey',0.)
            cmap.set_under('dimgrey',0.)

        if ax is None:
            ax = plt.gca()

        fp_local_list = [d for d in self.digitization_list if np.isfinite(d.fp_local)]

        plt.sca(ax)
        mex.plot_planet(lw=3.)
        mex.plot_bs(lw=1., ls='dashed', color='k')
        mex.plot_mpb(lw=1., ls='dotted', color='k')
        ax.set_aspect('equal', 'box')
        plt.xlim(2,-2)
        plt.autoscale(False,tight=True)
        plt.ylim(0,1.9999)

        if annotate:
            plt.annotate('%d' % self.orbit, (0.05, 0.85), xycoords='axes fraction', va='top')

        def f_x(pos):
            return pos[0] / mex.mars_mean_radius_km
        def f_y(pos):
            return np.sqrt(pos[1]**2. + pos[2]**2.) / mex.mars_mean_radius_km
        # def f_y(pos):
        #     return pos[2] / mex.mars_mean_radius_km

        plt.plot( f_x(self.mso_pos), f_y(self.mso_pos),
                color=bg_color, lw=1., zorder=-10)

        inx = np.interp(np.array([d.time for d in self.ionogram_list]), self.t, np.arange(self.t.shape[0]))
        inx = inx.astype(int)

        plt.plot( f_x(self.mso_pos[:,inx]), f_y(self.mso_pos[:,inx]),
                color=bg_color, ls='None',marker='o', ms=8.,mew=0., mec=bg_color, zorder=-9)

        if fp_local_list:
            val = np.empty_like(self.t) + np.nan
            # for t, v in [(float(f.time), np.log10(ais_code.fp_to_ne(f.maximum_fp_local))) for f in fp_local_list]:
            #     val[np.abs(self.t - t) < ais_code.ais_spacing_seconds] = v

            for f in fp_local_list:
                t = float(f.time)
                # if (f.fp_local_error / f.fp_local) > 0.3:
                #     v = np.log10(20.)
                # else:
                v = np.log10(ais_code.fp_to_ne(f.fp_local))
                # print t, ais_code.fp_to_ne(f.fp_local), f.fp_local_error/f.fp_local > 0.3
                val[np.abs(self.t - t) < ais_code.ais_spacing_seconds] = v

            points = np.array([f_x(self.mso_pos), f_y(self.mso_pos)]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax, clip=True))
            lc.set_array(val)
            lc.set_linewidth(5)
            plt.gca().add_collection(lc)
        else:
            lc = None

        plt.ylabel(r'$\rho / R_M$')
        # plt.ylabel(r'$z / R_M$')
        plt.xlabel(r'$x / R_M$')
        if lc:
            ticks = [i for i in range(10) if ((float(i)+0.1) > vmin) & ((float(i)-0.1) < vmax)]
            old_ax = plt.gca()
            plt.colorbar(lc, cax = celsius.make_colorbar_cax(offset=0.001, height=0.8),
                            ticks=ticks).set_label(r'$log_{10}\;n_e / cm^{-3}$')
            plt.sca(old_ax)

    def modb_along_orbit(self, ax=None, annotate=True, bg_color='dimgrey', cmap=None, vmin=0., vmax=20.):
        if ax is None:
            ax = plt.gca()

        if cmap is None:
            cmap = plt.cm.autumn
            cmap.set_bad('dimgrey',0.)
            cmap.set_under('dimgrey',0.)

        td_cyclotron_list = [d for d in self.digitization_list if np.isfinite(d.td_cyclotron)]

        plt.sca(ax)
        mex.plot_planet(lw=3.)
        mex.plot_bs(lw=1., ls='dashed', color='k')
        mex.plot_mpb(lw=1., ls='dotted', color='k')
        ax.set_aspect('equal','box')
        plt.xlim(2,-2)
        plt.autoscale(False,tight=True)
        plt.ylim(0., 1.999)

        if annotate:
            plt.annotate('%d' % self.orbit, (0.05, 0.85),
                        xycoords='axes fraction', va='top')

        def f_x(pos):
            return pos[0] / mex.mars_mean_radius_km
        def f_y(pos):
            return np.sqrt(pos[1]**2. + pos[2]**2.) / mex.mars_mean_radius_km
        # def f_y(pos):
        #     return pos[2] / mex.mars_mean_radius_km

        plt.plot( f_x(self.mso_pos), f_y(self.mso_pos),
                color=bg_color, lw=2., zorder=-10)

        inx = np.interp(np.array([d.time for d in self.ionogram_list]),
                                        self.t, np.arange(self.t.shape[0]))
        inx = inx.astype(int)
        plt.plot(f_x(self.mso_pos[:,inx]), f_y(self.mso_pos[:,inx]),
                color=bg_color, ls='None',marker='o', ms=8., mew=0., mec=bg_color, zorder=-9)

        if td_cyclotron_list:
            val = np.empty_like(self.t) + np.nan
            for t, v in [(float(f.time), 1E9 * ais_code.td_to_modb(f.td_cyclotron))
                                                        for f in td_cyclotron_list]:
                val[np.abs(self.t - t) < ais_code.ais_spacing_seconds] = v

            points = np.array([f_x(self.mso_pos), f_y(self.mso_pos)]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap,
                    norm=plt.Normalize(vmin=vmin, vmax=vmax, clip=True))

            lc.set_array(val)
            lc.set_linewidth(5)
            plt.gca().add_collection(lc)
        else:
            lc = None
        plt.ylabel(r'$\rho / R_M$')
        # plt.ylabel(r'$z / R_M$')
        plt.xlabel(r'$x / R_M$')
        if lc:
            old_ax = plt.gca()
            plt.colorbar(lc, cax = celsius.make_colorbar_cax(offset=0.001, height=0.8)
                        ).set_label(r'$|B| / nT$')
            plt.sca(old_ax)

    def plot_aspera_ima(self, ax=None, **kwargs):
        if self.verbose: print('PLOT_ASPERA_IMA:')
        if ax is None:
            ax = plt.gca()
        else:
            plt.sca(ax)
        mex.aspera.plot_ima_spectra(self.extent[0], self.extent[1], ax=ax, verbose=self.verbose, **kwargs)

    def plot_aspera_els(self, ax=None, **kwargs):
        if self.verbose: print('PLOT_ASPERA_ELS:')
        if ax is None:
            ax = plt.gca()
        else:
            plt.sca(ax)
        mex.aspera.plot_els_spectra(self.extent[0], self.extent[1], ax=ax, verbose=self.verbose, **kwargs)

    def main(self, fname=None, show=False,
                figurename=None, save=False, along_orbit=False, set_cmap=True):

        # if len(a.digitization_list) < 100:
        #     return
        if set_cmap:
            plt.hot()
        fig = plt.figure(figsize=(8.27, 11.69), dpi=70)

        n = 8 + 4 + 1 + 1 + 1
        hr = np.ones(n)
        hr[-4:] = 0.5
        g = mpl.gridspec.GridSpec(n,1, hspace=0.1, height_ratios=hr,
                        bottom=0.06, right=0.89)

        axes = []
        prev = None
        for i in range(n):
            axes.append(plt.subplot(g[i], sharex=prev))
            axes[i].set_xlim(self.extent[0], self.extent[1])
            axes[i].yaxis.set_major_locator(
                            mpl.ticker.MaxNLocator(prune='upper', nbins=5,
                            steps=[1,2,5,10]))
            l = celsius.SpiceetLocator()
            axes[i].xaxis.set_major_locator(l)
            axes[i].xaxis.set_major_formatter(
                        celsius.SpiceetFormatter(locator=l))

            prev = axes[-1]

        axit = iter(axes)

        self.plot_aspera_ima(ax=next(axit), inverted=False)
        self.plot_aspera_els(ax=next(axit))
        self.plot_mod_b(ax=next(axit))
        self.plot_ne(ax=next(axit))
        self.plot_timeseries(ax=next(axit))

        self.plot_frequency_range(ax=next(axit), f_min=0.0, f_max=0.2,
                                    colorbar=True)
        # self.plot_frequency_range(ax=axit.next(), f_min=0.2, f_max=0.5)
        self.plot_frequency(ax=next(axit), f=0.5)
        # self.plot_frequency(ax=axit.next(), f=0.75)
        self.plot_frequency(ax=next(axit), f=1.)
        # self.plot_frequency(ax=axit.next(), f=1.52)
        self.plot_frequency(ax=next(axit), f=2.)

        self.plot_tec(ax=next(axit))
        # twx = plt.twinx()
        # t = mex.sub_surface.read_tec(self.start_time, self.finish_time)
        # good = t['FLAG'] == 1
        # plt.plot(t['EPHEMERIS_TIME'][good], t['TEC'][good], 'k.', mew=0.)
        # plt.ylabel(r'$TEC / m^{-2}$')
        # plt.yscale('log')
        # plt.ylim(3E13, 2E16)

        # self.plot_profiles(ax=axit.next())
        # self.plot_profiles_delta(ax=axit.next())
        # self.plot_peak_altitude(ax=axit.next())
        # self.plot_peak_density(ax=axit.next())

        self.plot_profiles(ax=next(axit))
        # self.plot_tec(ax=axit.next())

        self.plot_altitude(ax=next(axit))
        self.plot_lat(ax=next(axit))
        self.plot_lon(ax=next(axit))
        self.plot_sza(ax=next(axit))

        # axes[-1].xaxis.set_major_formatter(celsius.SpiceetFormatter())

        for i in range(n-1):
            plt.setp( axes[i].get_xticklabels(), visible=False )
            axes[i].set_xlim(self.extent[0], self.extent[1])
            l = celsius.SpiceetLocator()
            axes[i].xaxis.set_major_locator(l)
            axes[i].xaxis.set_major_formatter(
                    celsius.SpiceetFormatter(locator=l))

        plt.annotate("Orbit %d, plot start: %s, newest digitization: %s" % (
                self.orbit,
                celsius.spiceet_to_utcstr(self.extent[0],fmt='C')[0:17], self._newest),
            (0.5, 0.93), xycoords='figure fraction', ha='center')

        if save:
            if figurename is None:
                fname = mex.locate_data_directory() + ('ais_plots/v0.9/%05d/%d.pdf' % ((self.orbit // 1000) * 1000, self.orbit))
            else:
                fname = figurename
            print('Writing %s' % fname)
            d = os.path.dirname(fname)
            if not os.path.exists(d) and d:
                os.makedirs(d)
            plt.savefig(fname)



        if show:
            plt.show()
        else:
            plt.close(fig)
            plt.close('all')

        if along_orbit:
            # fig = plt.figure()
            fig, ax = plt.subplots(2, 1, squeeze=True, figsize=(4,4), dpi=70, num=plt.gcf().number + 1)
            plt.subplots_adjust(hspace=0.3,wspace=0.0, right=0.85)

            self.density_along_orbit(ax[0], vmax=4.)
            self.modb_along_orbit(ax[1], vmax=100.)

            if save:
                fname = mex.locate_data_directory() + ('ais_plots/A0_v0.9/%05d/%d.pdf' % ((self.orbit // 1000) * 1000, self.orbit))
                print('Writing %s' % fname)
                d = os.path.dirname(fname)
                if not os.path.exists(d):
                    os.makedirs(d)
                plt.savefig(fname)

            if show:
                plt.show()
            else:
                plt.close(fig)
                plt.close('all')

        # if save:
        #     del self

def main(orbit=8021, close=True, debug=False, fname=None,
        fig=None, verbose=False, output_fname=None, **kwargs):
    if close:
        plt.close('all')

    if fname is None:
        fname = mex.locate_data_directory() + \
                        'marsis/ais_digitizations/%05d/%05d.dig' % (
                            (orbit // 1000) * 1000, orbit)

    try:
        print('DB: ', fname)
        a = AISReview(orbit, debug=debug, db_filename=fname, fig=fig,
            verbose=verbose)
        a.main(**kwargs)
    except Exception as e:
        print('AISReview Failed: ', e)
        raise

def stacked_f_plots(start=7894, finish=None, show=True,
            frequencies=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 2.0, 3.0]):
    gc.enable()
    if finish is None:
        finish = start + 1
    plt.close("all")
    fig = plt.figure(figsize = celsius.paper_sizes['A4'])
    orbits = list(range(start, finish))

    if len(orbits) > 1:
        show = False

    # plt.hot()

    for o in orbits:
        plt.clf()
        gc.collect()
        fname = mex.locate_data_directory() + 'marsis/ais_digitizations/%05d/%05d.dig' % ((o // 1000) * 1000, o)
        try:
            a = AISReview(o, debug=True, db_filename=fname)
        except Exception as e:
            print(e)
            continue


        n = len(frequencies) + 4
        hr = np.ones(n)
        hr[0] = 2.
        g = mpl.gridspec.GridSpec(n,1, hspace=0.1, height_ratios=hr)

        axes = []
        prev = None
        for i in range(n):
            axes.append(plt.subplot(g[i], sharex=prev))
            axes[i].set_xlim(a.extent[0], a.extent[1])
            axes[i].yaxis.set_major_locator(mpl.ticker.MaxNLocator(prune='upper', nbins=5, steps=[1,2,5,10]))
            axes[i].xaxis.set_major_locator(celsius.SpiceetLocator())
            axes[i].xaxis.set_major_formatter(celsius.SpiceetFormatter())
            prev = axes[-1]

        ax = iter(axes)

        a.plot_timeseries(ax=next(ax))
        a.plot_frequency_range(ax=next(ax), f_min=0.0, f_max=0.2)

        for i, f in enumerate(frequencies):
            a.plot_frequency_altitude(ax=next(ax), f=f)

        plt.sca(next(ax))
        b = a.quick_field_model(a.t)
        plt.plot(a.t, np.sqrt(np.sum(b**2., 0)), 'k-')
        plt.plot(a.t, b[0], 'r-')
        plt.plot(a.t, b[1], 'g-')
        plt.plot(a.t, b[2], 'b-')
        celsius.ylabel(r'$B_{SC} / nT$')

        plt.sca(next(ax))
        ion_pos = a.iau_pos
        ion_pos[0,:] = 150.0 + mex.mars_mean_radius_km
        bion = a.field_model(ion_pos)
        plt.plot(a.t, np.sqrt(np.sum(bion**2., 0)), 'k-')
        plt.plot(a.t, bion[0], 'r-')
        plt.plot(a.t, bion[1], 'g-')
        plt.plot(a.t, bion[2], 'b-')
        celsius.ylabel(r'$B_{150} / nT$')

        for i in range(n-1):
            ax = axes[i]
            plt.setp( ax.get_xticklabels(), visible=False )
            ax.xaxis.set_major_formatter(celsius.SpiceetFormatter())

        plt.annotate("Orbit %d, plot start: %s" % (o, celsius.spiceet_to_utcstr(a.extent[0])[0:14]),
            (0.5, 0.93), xycoords='figure fraction', ha='center')

        if show:
            plt.show()

        gc.collect()



def ao_plot(o):
    a = AISReview(o)

    # fig = plt.figure()
    fig, ax = plt.subplots(2, 1, squeeze=True, figsize=(4,4), dpi=70, num=plt.gcf().number + 1)
    plt.subplots_adjust(hspace=0.3,wspace=0.0, right=0.85)

    a.density_along_orbit(ax[0], vmax=4.)
    a.modb_along_orbit(ax[1], vmax=100.)


def mainloop(orbits = list(range(7890, 8200)), show=False):
    import gc
    gc.enable()
    gc.collect()

    for o in orbits:
        try:
            main(o, show=show, save=True)
        except Exception as e:
            print(e)
            del e
        finally:
            plt.close('all')
        gc.collect()
        plt.close('all')
        gc.collect()

if __name__ == '__main__':
    plt.close("all")

    if len(sys.argv) > 1:
        orbit = int(sys.argv[1])
    else:
        orbit = 8021
    if len(sys.argv) > 2:
        fname = str(sys.argv[2])
        save = True
    else:
        fname = None
        save = False

    main(orbit, figurename=fname, save=save, show=True,
        along_orbit=False, verbose=True, debug=True)
    # ao_plot(orbit)
    plt.show()
    # plt.savefig("/Users/dave/Desktop/9747.pdf", dpi=20)
