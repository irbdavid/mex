# maRS radio sounder data handler

import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
import glob
import mex

maRs_data_directory = mex.locate_data_directory() + '/maRs/'

class maRSSounding(object):
    """docstring for maRSSounding"""
    def __init__(self, time, load=True):
        super(maRSSounding, self).__init__()
        self.loaded = False
        if load:
            self.load(time)

    def plot(self, ax=None, labels=True):
        """docstring for plot"""

        if not self.loaded:
            raise mex.MEXException('Data not loaded')

        if ax == None:
            ax = plt.gca()

        plt.plot(self.density, self.geopotential_height, 'k-')

        if labels:
            plt.xlabel(r'$n_e / cm^{-3}$')
            plt.ylabel(r'$Geopot. Height / km$')
            plt.title('MaRS for orbit %d' % (self.orbit_number))

    # row utcstr spiceet r/km geopot/km lat lon refractivity signaldbm ne nesigma
    # 00015  2010-04-10T13:48:12.450  324179358.635646  6903.697  1733.897  -73.76   54.75    0.000442  -64.5    -777.54336341     520.60429029
    def load(self, time):
        """time is an orbit number, for now"""
        fname = glob.glob(maRs_data_directory + '*%d*/*.TAB' % (time))
        if len(fname) != 1:
            raise mex.MEXException("Found multiple matches for orbit %d" % (time))

        fname = fname[0]

        self.time, self.altitude, self.geopotential_height, self.lat, self.lon, \
            self.refractivity, self.signal_intensity, self.density, self.density_sigma \
            = np.loadtxt(fname, usecols=(2,3,4,5,6,7,8,9,10), unpack=True)

        # TO cm^-3
        # Because initial value is / 10^6 m^-3 == cm^-3
        # self.density *= 1.0
        # self.density_sigma *= 1.0

        self.orbit_number = time
        self.file_name    = fname
        self.loaded       = True


if __name__ == '__main__':
    import matplotlib.ticker as ticker


    orbits = [7940,7964,7988,7999,8026,8033,8037,8051,8058,8082]
    plt.close('all')

    fig, axs = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(8.27,11.69), dpi=60)

    once = True
    i = 0
    for orbit, ax in zip(orbits, axs.flatten()):
        plt.sca(ax)
        ax.set_xscale('Log')

        # if once:
        #     once = False
        #     ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
        #     ax.yaxis.set_major_locator(ticker.MultipleLocator(100.0))

        m = maRSSounding(orbit)
        m.plot(labels = False)
        plt.annotate(m.orbit_number, xy=(0.7,0.7), xycoords='axes fraction')
        plt.ylim(0., 400.)
        plt.xlim(1.E2, 1.E6)

        if (i % 2) == 0:
            plt.ylabel('Alt.* / km')
        if i > 7:
            plt.xlabel(r'$n_e / cm^{-3}$')
        i = i + 1

        print(m.file_name)
        plt.show()









