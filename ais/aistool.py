import numpy as np
import numpy.random
import matplotlib.pylab as plt
import matplotlib as mpl
import matplotlib.cm
import matplotlib.gridspec

import scipy.ndimage.morphology as morphology

import sys
import pickle

import mex
import spiceypy
import mex.ais as ais
# import mex.ais_code as ais_code
import mex.ais.aisreview
import celsius
import celsius.mars

from mex.ais import IonogramDigitization, DigitizationDB
import imp

__author__ = "David Andrews"
__copyright__ = "Copyright 2015, David Andrews"
__credits__ = ["David Andrews"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "David Andrews"
__email__ = "david.andrews@irfu.se"
__status__ = "Development"

ais_tool_instance = None

debug = False

class AISTool(object):
    """docstring for AISTool"""
    def __init__(self, orbit=8020, debug=False, digitization_db=None, load=True,
        auto=True,
        vmin=-16.0, vmax=-11.0, mobile=False, figure_number=1, timeseries_frequency=0.3):

        global ais_tool_instance
        ais_tool_instance = super(AISTool, self).__init__()

        # A few basic parameters
        self.status = None
        self.current_ionogram = None
        self.debug = debug
        self.orbit = None
        self.browsing = False
        self.minimum_interaction_mode = False
        self._initial_digitization_db = digitization_db
        self._digitization_saved = False
        self.load = load

        self.auto = auto

        self.ionospheric_model = celsius.mars.Morgan2008ChapmanLayer()

        np.seterr(all='ignore')

        self.params = dict(auto_refine=False, substitute_fp=ais.ne_to_fp(4.))

        self._bad_keypress = False
        self._messages = []
        self._message_counter = 0
        plt.set_cmap('viridis')
        self.selected_plasma_lines = []
        self.selected_cyclotron_lines = []

        self.timeseries_frequency = timeseries_frequency

        self.vmin = vmin
        self.vmax = vmax

        mex.check_spice_furnsh()

        # Set up the figure
        # self.figure = plt.figure(figsize=(12, 6))
        figsize = (20, 12)
        if mobile:
            figsize = (17, 8)

        if plt.get_backend() == 'nbAgg':
            figsize = (12,7)

        plt.close(figure_number)
        self.figure = plt.figure(figure_number,
                figsize=figsize, facecolor='0.6')
        g = mpl.gridspec.GridSpec(6, 2, width_ratios=[1,0.34],
            height_ratios=[0.001, 0.001, 7,5,2,16], wspace=0.16, hspace=0.1,
            left=0.05, right=0.95, bottom=0.08, top=0.95)

        # self.stat_ax = plt.subplot(g[0,:])
        # self.traj_ax = plt.subplot(g[1,:])
        self.tser_ax = plt.subplot(g[2,:])
        self.freq_ax = plt.subplot(g[3,:])
        self.ig_ax   = plt.subplot(g[5,0])
        self.ne_ax   = plt.subplot(g[5,1])
        self.cbar_ax = plt.gcf().add_axes([0.45,  0.04, 0.3, 0.01])

        # self.fp_local_figure_number = figure_number + 1
        # self.td_cyclotron_figure_number = figure_number + 2

        self.fp_local_figure_number = False
        self.td_cyclotron_figure_number = False

        self.stored_color = 'white'
        self.interactive_color = 'red'

        # All the connections get set up:
        self.cids = []
        self.cids.append(self.figure.canvas.mpl_connect('key_press_event',
                                            self.on_keypress))
        self.cids.append(self.figure.canvas.mpl_connect('button_press_event',
                                            self.on_click))
        self.cids.append(self.figure.canvas.mpl_connect('button_release_event',
                                            self.on_release))
        self.cids.append(self.figure.canvas.mpl_connect('motion_notify_event',
                                            self.on_move))
        self.cids.append(self.figure.canvas.mpl_connect('scroll_event',
                                            self.on_scroll))

        self.message("Initialized")

        plt.show()
        self.set_orbit(orbit)
        self.update()


    def message(self, m):
        if not m:
            return

        print('>> '+ m)
        self._messages.append(str(m))
        if len(self._messages) > 6:
            self._messages = self._messages[-6:]
            self._message_counter += 1

    def disconnect(self):
        for c in self.cids:
            self.figure.canvas.mpl_disconnect(c)
        self.cids = []

    def set_orbit(self, orbit, strict=True):
        orbit = int(orbit)
        print('-----------------\nSetting orbit = %d' % orbit)
        # Now the "science"
        successfully = False
        attempts = 0
        while not successfully:
            new_ionogram_list = []
            try:
                new_ionogram_list = ais.read_ais(orbit)
                new_orbit = orbit
                break
            except IOError as e:
                print('No data available for orbit %d' % orbit)
            if strict or attempts > 10:
                raise mex.MEXException("Orbit not found - no data, missing file, or some other bollocks.")
            orbit = orbit - (self.orbit - orbit) / abs(self.orbit - orbit)
            attempts = attempts + 1

        self.ionogram_list = new_ionogram_list
        self.orbit = new_orbit

        new_data = []

        for i in self.ionogram_list:
            i.interpolate_frequencies()

        # for i in range(len(self.ionogram_list) - 2):
        #     if i == 0: continue
        #     new_data.append( np.mean(
        #         np.dstack([ig.data for ig in self.ionogram_list[i:i+2]]), 2))
        # if new_data:
        #     for i in range(len(new_data)):
        #         self.ionogram_list[i].data = new_data[i]


        # If the user specified one, load it, else get the default for the orbit
        if self.load:
            if self._initial_digitization_db:
                self.digitization_db = DigitizationDB(
                            filename=self._initial_digitization_db, verbose=True)
            else:
                self.digitization_db = DigitizationDB(orbit=self.orbit)
            self._digitization_saved = True
        else:
            self.digitization_db = DigitizationDB(load=False)
            self._digitization_saved = False
        # Now we do some processing, generate a data cube for the orbit
        # and generate the timeseries
        self.ionogram_list[0].interpolate_frequencies()
        no_linear_frequencies = self.ionogram_list[0].data.shape[1]
        self.extent = (self.ionogram_list[0].time, self.ionogram_list[-1].time,
                        min(self.ionogram_list[0].frequencies) / 1.0E6,
                        max(self.ionogram_list[0].frequencies) / 1.0E6)
        no_ionograms_expected = ((self.extent[1] - self.extent[0])
                                                        / ais.ais_spacing_seconds + 1)
        no_ionograms_expected = int(round(no_ionograms_expected))
        self.tser_arr_all = np.empty((ais.ais_number_of_delays, no_linear_frequencies,
            no_ionograms_expected))

        if self.debug:
            print('Creating data cube (filling empties)')
            print('Expected number of ionograms = %d, found = %d' % (
                no_ionograms_expected,len(self.ionogram_list)))
        ilast = None
        empty_count = 0
        for i, ig in enumerate(self.ionogram_list):
            ignum = int( round((ig.time - self.extent[0]) / ais.ais_spacing_seconds ))
            if ignum > no_ionograms_expected:
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

        # Hold the update for now
        self.set_ionogram(self.ionogram_list[0], update=False)

        errs = np.geterr()
        np.seterr(divide='ignore')
        self.tser_arr = np.log10(np.mean(self.tser_arr_all[::-1,:, :], axis=0))
        self.tser_arr_all = np.log10(self.tser_arr_all)
        np.seterr(**errs)

        # Trajectory info
        self.trajectory = {}
        self.trajectory['t'] = np.arange(self.extent[0], self.extent[1], 60.)
        pos = mex.iau_mars_position(self.trajectory['t'])
        self.trajectory['pos'] = pos / mex.mars_mean_radius_km

        self.message("Set orbit to %d" % self.orbit)

        self.status = None
        self.update()
        return self

    def set_ionogram(self, ionogram, update=True, auto=None):

        if auto is None:
            auto = self.auto

        if not isinstance(ionogram, ais.Ionogram):
            if isinstance(ionogram, str):
                if ionogram.lower() == 'next':
                    ig_inc = 1
                elif ionogram.lower() == 'previous':
                    ig_inc = -1
            else:
                ig_inc = ionogram

            for i, ig in enumerate(self.ionogram_list):
                if ig is self.current_ionogram:
                    if ((i + ig_inc) > 0) and ((i + ig_inc) < len(self.ionogram_list)):
                        ionogram = self.ionogram_list[i+ig_inc]
                    else:
                        self.set_orbit(self.orbit + int(ig_inc / abs(ig_inc)), strict=False)

        if ionogram is not self.current_ionogram:
            if not self.digitization_saved():
                # print "Current digitization not saved"
                self.save_current_digitization()
            self.current_ionogram = ionogram

            plt.close(self.fp_local_figure_number)
            plt.close(self.td_cyclotron_figure_number)

            # Try to load from DB, otherwise set up an empty one:
            dig = self.digitization_db.get_nearest(ionogram.time)
            self.current_ionogram.digitization = dig

            # if self.debug: print 'Found %d digitizations' % len(dig)
            if not dig:
                dig = IonogramDigitization()
                dig.time = self.current_ionogram.time
                self.current_ionogram.digitization = dig
                if auto:
                    self.auto_fit(update=update)
                    self._digitization_saved = False
            else:
                # We loaded something, fresh from the DB and therefore:
                self._digitization_saved = True

            self.selected_plasma_lines = []
            self.selected_cyclotron_lines = []

            plt.sca(self.ig_ax)
            plt.cla()
            plt.sca(self.ne_ax)
            plt.cla()

            ig_index = 0
            test_ig = self.ionogram_list[0]
            while test_ig != self.current_ionogram:
                ig_index += 1
                test_ig = self.ionogram_list[ig_index]

            self.message("Set ionogram to %s [%d/%d]" % (
                celsius.utcstr(1. * self.current_ionogram.time, format='C'),
                ig_index,
                len(self.ionogram_list)))

            if update:
                self.set_status(None)
                self.update()
            return self

    def on_click(self, event):
        # if self.debug: print 'on_click
        if event.inaxes == self.ig_ax:
            if (self.status is None):
                return

            elif self.status == 'tracing':
                self.status = 'go_tracing'
                self.traced_delays = []
                self.traced_frequencies = []
                self.traced_delays.append(event.ydata)
                self.traced_frequencies.append(event.xdata)
                self.message('Started tracing...')

            elif self.status == 'plasma_lines':
                self.selected_plasma_lines.append(event.xdata)
                if len(self.selected_plasma_lines) >= 1:
                    arr = np.array(self.selected_plasma_lines, ndmin=1)
                    # arr = np.sort(arr)
                    self.current_ionogram.digitization.set_fp_local_manual(
                        arr * 1.0E6)
                    self._digitization_saved = False
                    self.current_ionogram.digitization.set_timestamp()
                    self.message('Selected Plas. Line. @ %f MHz, Morph Fp_local = %f MHz' % (event.xdata, self.current_ionogram.digitization.fp_local/1E6))
                    self.update()

            elif self.status == 'cyclotron_lines':
                self.selected_cyclotron_lines.append(event.ydata)
                if len(self.selected_cyclotron_lines) > 0:
                    arr = np.array(self.selected_cyclotron_lines)
                    arr = np.abs(np.diff(np.sort(np.hstack((0.0, arr)))))

                    # print 'SETTING CYCLOTRONS:'
                    # print np.mean(arr) / 1.0E3, np.std(arr) / 1.0E3
                    # print ais.td_to_modb(np.mean(arr) / 1.0E3) * 1E9, ais.td_to_modb(np.std(arr) / 1.0E3) * 1E9

                    self.current_ionogram.digitization.set_cyclotron(
                        np.mean(arr) / 1.0E3, np.std(arr) / 1.0E3,
                        selected_t=self.selected_cyclotron_lines, method='MANUAL')
                    self._digitization_saved = False
                    self.current_ionogram.digitization.set_timestamp()
                    self.message('Selected a cyclotron line')
                    self.update()

            elif self.status == 'ground':
                # Could refine this
                self.current_ionogram.digitization.set_ground(event.ydata / 1.0E3)
                self._digitization_saved = False
                self.current_ionogram.digitization.set_timestamp()
                self.set_status(None)
                self.message('Selected the ground line')
                self.update()

            elif self.status == 'editing':
                pass # select nearest point
            elif self.status == 'pick_frequency':
                self.timeseries_frequency = event.xdata
                self.status = None
                self.message('Changed frequency to %f MHz' % event.xdata)
                self.update()

        if event.inaxes == self.tser_ax:
            if self.status == None:
                # Get nearest ionogram from self.timeseries
                for i in self.ionogram_list:
                    if i.time > event.xdata:
                        new_ionogram = i
                        self.set_ionogram(new_ionogram)
                        self.update()
                        break

            elif self.status == 'pick_frequency':
                self.timeseries_frequency = event.ydata
                self.status = None
                self.message('Changed frequency to %f MHz' % event.xdata)
                self.update()

        if event.inaxes == self.freq_ax:
            if self.status == None:
                # Get nearest ionogram from self.timeseries
                for i in self.ionogram_list:
                    if i.time > event.xdata:
                        new_ionogram = i
                        self.set_ionogram(new_ionogram)
                        self.update()
                        break

    def on_move(self, event):
        if event.inaxes == self.ig_ax:
            if (self.status is None):
                return

            if self.status == 'go_tracing':
                self.traced_delays.append(event.ydata)
                self.traced_frequencies.append(event.xdata)
                if plt.gca() != self.ig_ax:
                    plt.sca(self.ig_ax)

                if len(self.traced_delays) > 2:
                    plt.plot(self.traced_frequencies[-2:], self.traced_delays[-2:],
                        color=self.interactive_color)
                    self.figure.canvas.draw()

            elif self.status == 'plasma_lines':
                pass # do nothing - we're clicking per line
            elif self.status == 'cyclotron_lines':
                pass # do nothing - we're clicking per line
            elif self.status == 'editing':
                pass # do nothing - we're clicking per line

    def on_release(self, event):
        # if self.debug: print 'on_release'
        if event.inaxes == self.ig_ax:
            d = self.current_ionogram.digitization

            # Determine whether anything has changed also
            # Update self.ionogram_saved
            if (self.status is None):
                return
            if self.status == 'go_tracing':
                self.message('Finished tracing')
                if self.tracing_status_retain:
                    self.set_status('tracing') # return to start of tracing mode
                else:
                    self.set_status(None)
                self._digitization_saved = False

                d.set_trace(np.array(self.traced_delays) * 1.0E-3,
                            np.array(self.traced_frequencies) * 1.0E6, method='MANUAL')

                if self.params["auto_refine"]:
                    self.message( self.current_ionogram.refine_trace() ) # Will operate on d as well
                self.message("Inversion successful" if d.invert(substitute_fp=self.params['substitute_fp']) else "Inversion failed")
                self.update()

            # elif self.status == 'plasma_lines':
            #     if self.debug and hasattr(d, 'fp_local'):
            #         print 'FP_LOCAL =  ',d.fp_local,d.fp_local_error
            #
            # elif self.status == 'cyclotron_lines':
            #     if self.debug and hasattr(d, 'td_cyclotron'):
            #         print 'TD_CYCLOTRON = ', d.td_cyclotron, d.td_cyclotron_error

            elif self.status == 'editing':
                pass # fix selected point to position

    def on_scroll(self, event):
        return
        print("SCROLL INNIT")
        if self.status == 'plasma_lines':
            fp = self.current_ionogram.digitization.fp_local
            if not np.isfinite(fp):
                self.current_ionogram.digitization.set_morphology_fp_local(
                    0.5e6, np.inf, 'scroll_guess')
            else:
                new_fp = (event.step * 0.005 + 1.) * fp
                print('SCROLL: ', fp, new_fp)
                self.current_ionogram.digitization.set_morphology_fp_local(
                    new_fp, new_fp * 0.01, 'scroll'
                )
            self.update()

        # elif self.status == 'cyclotron_lines':



    def auto_fit(self, plasma_lines=True, cyclotron_lines=True,
                    ionosphere=True, ground=True, new_digitization=False, update=True):
        # self.message('AUTO_FIT disabled!!')
        # return

        if new_digitization:
            self.current_ionogram.digitization = IonogramDigitization()
            self.message('Added new digitization')

        i = self.current_ionogram
        i.threshold_data()
        i.generate_binary_arrays()
        self.message( i.calculate_ground_trace() )
        self.message( i.calculate_fp_local(
                figure_number=self.fp_local_figure_number) )
        self.message(
            i.calculate_td_cyclotron(
                figure_number=self.td_cyclotron_figure_number) )
        self.message( i.calculate_reflection() )

        i.delete_binary_arrays()
        print("Quality factor = ", i.quality_factor)

        if not self.current_ionogram.digitization:
            self._digitization_saved = False
            self.current_ionogram.digitization.set_timestamp()
        # self.message('Ran automatic fit routines')
        if update:
            self.update()
        return self

    def update(self):
        """ This redraws the various axes """
        plt.sca(self.ig_ax)
        plt.cla()

        if debug:
            print('DEBUG: Plotting ionogram...')

        alpha = 0.5
        self.current_ionogram.interpolate_frequencies() # does nothing if not required
        self.current_ionogram.plot(ax=self.ig_ax, colorbar=False,
            vmin=self.vmin, vmax=self.vmax,
            color='white', verbose=debug,
            overplot_digitization=True,alpha=alpha,errors=False,
            overplot_model=False, overplot_expected_ne_max=True)
        if debug:
            print('DEBUG: ... done')
        plt.colorbar(cax=self.cbar_ax, orientation='horizontal',
            ticks=mpl.ticker.MultipleLocator())
        plt.sca(self.cbar_ax)
        plt.xlabel(r'spec. dens. / $V^2m^{-2}Hz^{-1}$')
        plt.sca(self.ig_ax)

        # Plasma and cyclotron lines
        if len(self.selected_plasma_lines) > 0:
            extent = plt.ylim()
            for v in self.selected_plasma_lines:
                plt.vlines(v, extent[0], extent[1], 'red',alpha=alpha)

        if len(self.selected_cyclotron_lines) > 0:
            extent = plt.xlim()
            for v in self.selected_cyclotron_lines:
                plt.hlines(v, extent[0], extent[1], 'red',alpha=alpha)

        f = self.current_ionogram.digitization.morphology_fp_local
        if np.isfinite(f):
            plt.vlines(
                np.arange(1., 5.) * f / 1E6, plt.ylim()[0],
                plt.ylim()[1],
                color='red', lw=1.,alpha=alpha)

        # If current digitization is invertible, do it and plot it
        if self.current_ionogram.digitization:
            if debug:
                print('DEBUG: Inverting, computing model...')

            d = self.current_ionogram.digitization
            plt.sca(self.ne_ax)
            plt.cla()
            if d.is_invertible():
                winning = d.invert()
                if winning & np.all(d.density > 0.) & np.all(d.altitude > 0.):
                    plt.plot(d.density, d.altitude, color='k')
            plt.xlim(5.E1, 5E5)
            plt.ylim(0,499)
            alt = np.arange(0., 499., 5.)
            if self.current_ionogram.sza < 89.9:
                plt.plot(self.ionospheric_model(alt,
                        np.deg2rad(self.current_ionogram.sza)), alt, color='green')
            plt.grid()
            plt.xscale('log')
            plt.xlabel(r'$n_e / cm^{-3}$')
            plt.ylabel('alt. / km')
            fname = self.digitization_db.filename
            if len(fname) > 30: fname = fname[:10] + '...' + fname[-20:]
            plt.title('Database: ' + fname)

        if debug:
            print('DEBUG: Plotting timeseries....')

        # Timeseries integrated bar
        plt.sca(self.tser_ax)
        plt.cla()
        plt.imshow(self.tser_arr[::-1,:], vmin=self.vmin, vmax=self.vmax,
            interpolation='Nearest', extent=self.extent, origin='upper',aspect='auto')
        plt.xlim(self.extent[0], self.extent[1])
        plt.ylim(self.extent[2], self.extent[3])
        plt.ylim(0., 5.5)
        plt.vlines(self.current_ionogram.time,
            self.extent[2], self.extent[3], self.stored_color)
        plt.hlines(self.timeseries_frequency, self.extent[0],  self.extent[1],
            self.stored_color, 'dashed')
        plt.ylabel('f / MHz')

        # Frequency bar
        plt.sca(self.freq_ax)
        plt.cla()
        freq_extent = (self.extent[0], self.extent[1],
            ais.ais_max_delay*1E3, ais.ais_min_delay*1E3)
        inx = 1.0E6 * (self.current_ionogram.frequencies.shape[0] *
            self.timeseries_frequency) /\
            (self.current_ionogram.frequencies[-1] - self.current_ionogram.frequencies[0])

        self._freq_bar_data = self.tser_arr_all[:,int(inx),:]
        plt.imshow(self.tser_arr_all[:,int(inx),:], vmin=self.vmin, vmax=self.vmax,
            interpolation='Nearest', extent=freq_extent, origin='upper',aspect='auto')
        plt.xlim(freq_extent[0], freq_extent[1])
        plt.ylim(freq_extent[2], freq_extent[3])
        plt.vlines(self.current_ionogram.time,
            freq_extent[2],freq_extent[3], self.stored_color)
        plt.ylabel(r'$\tau_D / ms$')

        title = "AISTool v%s, Orbit = %d, Ionogram=%s " % (__version__,
            self.orbit, celsius.spiceet_to_utcstr(self.current_ionogram.time,
            fmt='C'))

        if self.browsing:
            title += '[Browsing] '
        if self.minimum_interaction_mode:
            title += '[Quick] '
        if self._digitization_saved == False:
            title += 'UNSAVED '
        if self.get_status() is not None:
            title += '[Status = %s] ' % self.get_status()

        pos, sza = mex.mso_r_lat_lon_position(float(self.current_ionogram.time),
            sza=True)

        title += '\nMSO: Altitude = %.1f km, Elevation = %.1f, Azimuth = %.1f deg, SZA = %.1f' % (pos[0] - mex.mars_mean_radius_km, mex.modpos(pos[1]), mex.modpos(pos[2]), sza)

        pos = mex.iau_pgr_alt_lat_lon_position(float(self.current_ionogram.time))
        title += '\nIAU: Altitude = %.1f km, Latitude = %.1f, Longitude = %.1f deg' % (
            pos[0], pos[1], mex.modpos(pos[2]))

        plt.sca(self.tser_ax)
        plt.title(title)

        # Message history:
        if len(self._messages):
            txt = ''
            for i, s in enumerate(self._messages):
                txt += str(i + self._message_counter) + ': ' + s + '\n'
            plt.annotate(txt, (0.05, 0.995), xycoords='figure fraction',
                fontsize=8, horizontalalignment='left', verticalalignment='top')

        # Axis formatters need redoing after each cla()
        nf = mpl.ticker.NullFormatter

        loc_f = celsius.SpiceetLocator()
        loc_t = celsius.SpiceetLocator()
        self.freq_ax.xaxis.set_major_formatter(celsius.SpiceetFormatter(loc_f))
        self.tser_ax.xaxis.set_major_formatter(nf())

        self.freq_ax.xaxis.set_major_locator(loc_f)
        self.tser_ax.xaxis.set_major_locator(loc_t)
        if debug:
            print('DEBUG: drawing...')

        self.figure.canvas.draw()
        return self

    def set_status(self, status=None):
        if status in ['tracing', 'plasma_lines', 'cyclotron_lines',
            'editing', 'orbit', 'pick_frequency', 'ground', None]:
            self.status = status
        else:
            raise mex.MEXException("Bad status")
        return self

    def get_status(self):
        return self.status

    def digitization_saved(self):
        if self.browsing:
            # print 'browsing'
            return True
        if self.current_ionogram is None:
            # print 'ig none'
            return True
        # if self.current_ionogram.digitization:
        #     # print 'len 0'
        #     if self.debug: print "Shouldn't have a 0 length list of digitizations"
        #     return False
        if not self.current_ionogram.digitization in self.digitization_db:
            # print 'is empty?'
            return self.current_ionogram.digitization is False
        return self._digitization_saved

    def save_current_digitization(self):
        if self._digitization_saved and self.debug:
            print("Already saved, apparently, but we'll try again anyway")
        if self.digitization_db is not None:
            if self.current_ionogram.digitization:
                try:
                    self.digitization_db.add(
                        self.current_ionogram.digitization
                    )
                    self.digitization_db.write()
                    self._digitization_saved = True
                    self.message("Saved digitizations")
                except mex.MEXException as e:
                    print("Save not successful: " + str(e))
                    self.message("SAVE UNSUCCESSFUL!")
        else:
            raise mex.MEXException("No digitization database loaded")
        return self

    def on_keypress(self, event):
        """AISTool key table:
        (# at the end = not implemented yet)
        r: reload selected ionogram - warn if changed #
        s: save selected ionogram trace etc
            (non - destructive, as we're adding a new version)
        n = r arrow = space: next ionogram in series (warn if changed)
        p = l arrow: previous ionogram in series (..)
        b: browse mode - don't check saved state on changing ionograms
        u: force an update of the display
        o: change orbit - next keys should be one of n/p for next/previous,
            or 4 digits for an orbit number
        a: auto fit trace, plasma, cyclotron, ground position
            * plasma and cyclotron lines are implemented, ground and trace aren't yet
        w: manual plasma freq (click adjacent lines, w again to stop)
        c: manual cyclotron freq (click adjacent lines, c again to stop)
        t: manual trace ionogram (draw near trace, release, peak fitting is then done)
        e: edit trace #
        f: pick a frequency for the constant-frequency plot
        g: ground level
        h: help dialog #
        i: print info on ionogram, digitization
        d: add a new empty digitization
        z,x: show previous, next plasma, cyclotron and traces on top of current#
        enter: step through plasma lines, trace, cyclotron, ground, save, next
            using the minimum possible interaction
        q: process the whole orbit!!
        """

        ignore_keys = ['`']

        if event.key:
            if event.key in ignore_keys:
                return

            if event.key == ' ':
                event.key = 'enter'

            if self.status != 'orbit':
                f = getattr(self, 'key_' + event.key, None)
                if f:
                    if self.minimum_interaction_mode and (not event.key in \
                            ['enter', 'n','right']):
                        self.minimum_interaction_mode = False
                        self.set_status(None)
                    f()
                    self._bad_keypress = False
                    self.update()
                else:
                    print('You pressed: ' + event.key)
                    if self._bad_keypress:
                        print(self.on_keypress.__doc__)
                    else:
                        print("Don't be such a prat.")
                    self._bad_keypress = True

            elif self.status == 'orbit':
                if event.key == 'n':
                    if self.debug: print('Loading next orbit')
                    self.set_orbit(self.orbit + 1, strict=False)
                    self.set_status(None)
                    return
                if event.key == 'p':
                    if self.debug: print('Loading previous orbit')
                    self.set_orbit(self.orbit - 1, strict=False)
                    self.set_status(None)
                    return
                if event.key in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    self.new_orbit.append(event.key)
                else:
                    print("Didn't recognise the orbit number (%s) - n = next, p = previous, otherwise 5 digits" % (''.join(self.new_orbit)))
                    self.set_status(None)
                if len(self.new_orbit) == 5:
                    self.set_status(None)
                    self.set_orbit(int(''.join(self.new_orbit)), strict=False)
                    return

    def key_t(self, retain=False):
        self.tracing_status_retain = retain
        if retain and self.status == 'tracing':
            self.set_status(None)
            return
        self.set_status('tracing')

    def key_w(self):
        if self.status == 'plasma_lines':
            self.set_status(None)
            # if self.debug: print 'Starting plasma lines'
        else:
            self.set_status('plasma_lines')
            self.selected_plasma_lines = []
        self.update()

    def key_c(self):
        if self.status == 'cyclotron_lines':
            self.set_status(None)
        else:
            # if self.debug: print 'Starting cyclotron lines'
            self.set_status('cyclotron_lines')
            self.selected_cyclotron_lines = []
        self.update()

    def key_o(self):
        self.status = 'orbit'
        self.new_orbit = []

    def key_n(self):
        self.set_ionogram('next')

    def key_p(self):
        self.set_ionogram('previous')

    def key_1(self):
        if not hasattr(self.current_ionogram, "_cyc_data"):
            self.current_ionogram.generate_binary_arrays()

        if not hasattr(self.current_ionogram, '_old_data'):
            self.current_ionogram._old_data = self.current_ionogram.data.copy()

        self.current_ionogram.data = 10.** (self.current_ionogram._cyc_data *
                                                (self.vmax-self.vmin) + self.vmin)
        self.update()

    def key_2(self):
        if not hasattr(self.current_ionogram, "_ion_data"):
            self.current_ionogram.generate_binary_arrays()

        if not hasattr(self.current_ionogram, '_old_data'):
            self.current_ionogram._old_data = self.current_ionogram.data.copy()

        self.current_ionogram.data = 10.** (self.current_ionogram._ion_data *
                                                (self.vmax-self.vmin) + self.vmin)
        self.update()

    def key_3(self):
        if not hasattr(self.current_ionogram, "_fp_data"):
            self.current_ionogram.generate_binary_arrays()

        if not hasattr(self.current_ionogram, '_old_data'):
            self.current_ionogram._old_data = self.current_ionogram.data.copy()

        self.current_ionogram.data = 10.** (self.current_ionogram._fp_data *
                                                (self.vmax-self.vmin) + self.vmin)
        # d = celsius.remove_none_edge_intersecting(self.current_ionogram._fp_data, 2)
        # self.current_ionogram.data = 10.** (d * (self.vmax-self.vmin) + self.vmin)
        # self.current_ionogram.data = self.current_ionogram._fp_data
        self.update()

    def key_4(self):
        if hasattr(self.current_ionogram, '_old_data'):
            self.current_ionogram.data = self.current_ionogram._old_data
            del self.current_ionogram._old_data
            self.update()

    def key_5(self):
        if not hasattr(self.current_ionogram, "thresholded_data"):
            self.current_ionogram.threshold_data()

        if not hasattr(self.current_ionogram, '_old_data'):
            self.current_ionogram._old_data = self.current_ionogram.data.copy()

        # s = np.zeros((10,2))
        # s[:,1] = 1
        # self.current_ionogram.data = morphology.binary_hit_or_miss(self.current_ionogram.thresholded_data, s)
        self.current_ionogram.data = self.current_ionogram.thresholded_data

    def key_q(self):
        # self.message('Q disabled!!')
        # return
        filename = self.digitization_db.filename

        del self.digitization_db
        self.digitization_db = DigitizationDB(orbit=self.orbit,
            filename=filename, load=False, verbose=True)

        def p(s):
            print(s)
            if not 'failed' in s.lower():
                return 1
            return 0

        fp, td, ground, reflection = 0, 0, 0, 0


        #     self.message('Added new digitization')
        #
        # i = self.current_ionogram
        # i.threshold_data()
        # i.generate_binary_arrays()
        # self.message( i.calculate_ground_trace() )
        # self.message( i.calculate_fp_local(
        #         figure_number=self.fp_local_figure_number) )
        # self.message(
        #     i.calculate_td_cyclotron(
        #         figure_number=self.td_cyclotron_figure_number) )
        # self.message( i.calculate_reflection() )
        #
        # i.delete_binary_arrays()
        # print("Quality factor = ", i.quality_factor)
        #
        # if not self.current_ionogram.digitization:
        #     self._digitization_saved = False
        #     self.current_ionogram.digitization.set_timestamp()

        for d in self.ionogram_list:
            print('-----')
            d.threshold_data()
            d.generate_binary_arrays()
            d.digitization = IonogramDigitization()
            d.digitization.time = d.time
            fp += p(d.calculate_fp_local())
            td += p(d.calculate_td_cyclotron())
            ground += p(d.calculate_ground_trace())
            reflection += p(d.calculate_reflection())
            d.delete_binary_arrays()
            d.digitization.set_timestamp()
            self.digitization_db.add(d.digitization)
            # print self.current_ionogram.time
        print('-----')
        print('Totals: FP = %d, TD = %d, ground = %d, reflection = %d' % (fp, td, ground, reflection ))
        self.digitization_db.write()

    def key_f(self):
        if self.get_status() is not None:
            if self.debug: print('Waiting for status None')
        else:
            self.status = 'pick_frequency'

    def key_s(self):
        self.save_current_digitization()

    def key_u(self):
        self.update()

    def key_right(self):
        self.key_n()

    def key_left(self):
        self.key_p()

    def key_b(self):
        self.browsing = not self.browsing

    def key_a(self):
        self.auto_fit()

    def key_g(self):
        self.set_status('ground')

    def key_h(self):
        if hasattr(self, '_histogram_figure_number'):
            plt.figure(self._histogram_figure_number)
            plt.clf()
        else:
            fig = plt.figure()
            self._histogram_figure_number = fig.number
        plt.hist(np.log10(self.current_ionogram.data.flatten()), bins=np.arange(-25.,-9.,0.1) - 0.05, fc='none')

    def key_d(self):
        self.current_ionogram.digitization = IonogramDigitization(self.current_ionogram)
        self._digitization_saved = False

    def key_m(self):
        # Toggle on/off
        self.minimum_interaction_mode = not self.minimum_interaction_mode
        print("Minimum interaction mode = %s" % self.minimum_interaction_mode)
        if self.minimum_interaction_mode:
            self.minimum_interaction_mode_counter = 0
            self.key_enter()
        else:
            self.minimum_interaction_mode_counter = -1

    def key_enter(self):
        if not self.minimum_interaction_mode:
            self.minimum_interaction_mode_counter = 0
            self.minimum_interaction_mode = True
        mic = self.minimum_interaction_mode_counter
        if mic == 0: # Just started this mode
            if ~np.isfinite(self.current_ionogram.digitization.fp_local):
                self.auto_fit()
            self.key_w()
        elif mic == 1: # Done plasma lines, start tracing
            self.key_w()
            self.key_t(retain=False)
        elif mic == 2: # Done tracing, start cyclotron
            # self.key_t(retain=False)
            self.key_c()
        # elif mic == 3: # Done cycltron, start ground
        #     self.key_c()
        #     self.key_g()
        elif mic == 3: # Done ground, save and next, set mic = 0
            self.key_c()
            self.key_s()
            try:
                self.key_n()
                self.key_w()
            except mex.MEXException as e:
                print("Caught an exception: " + str(e))
            finally:
                self.key_d()
            if ~np.isfinite(self.current_ionogram.digitization.fp_local):
                self.auto_fit()
        else:
            raise mex.MEXException(
                "minimum_interaction_mode_counter should be between 0 and 5")
        mic = mic + 1
        if mic > 3:
            mic = 1
        self.minimum_interaction_mode_counter = mic
        return self

    def key_space(self):
        self.key_n()

    def key_0(self):
        fig = plt.figure()
        ax = plt.subplot(111)
        self.current_ionogram.plot(ax=ax, overplot_digitization=False, vmin=self.vmin, vmax=self.vmax, overplot_model=False)
        fname = 'Ionogram-O%d_%s.png' % (mex.orbits[float(self.current_ionogram.time)].number,
            celsius.spiceet_to_utcstr(self.current_ionogram.time, fmt='C'))
        plt.title(celsius.spiceet_to_utcstr(self.current_ionogram.time,
                fmt='C'))
        plt.savefig(fname.replace(':',''))
        plt.close(fig)

    def key_r(self):
        self.message( self.current_ionogram.refine_trace() ) # Will operate on d as well

    def key_i(self):
        print('----------------')
        print('Current ionogram:')
        print(self.current_ionogram)
        print('')
        print('Working digitization:')
        print(self.current_ionogram.digitization)
        print('----------------')

    def key_v(self):
        self.key_s()
        mex.ais.aisreview.main(self.orbit, fname=self.digitization_db.filename,
            close=False, along_orbit=True, save=True, figurename='tmp.pdf')

if __name__ == '__main__':
    plt.ion()
    imp.reload(mex.ais.ais_code)
    imp.reload(mex.ais)
    imp.reload(mex)

    orbit = 8020
    if len(sys.argv) > 1:
        orbit = int(sys.argv[1])

    if len(sys.argv) > 2:
        db = sys.argv[2]
    else:
        db = None

    global ais_tool_instance

    ais_tool_instance = AISTool(debug=False, mobile=True,
        orbit=orbit, digitization_db=db)
