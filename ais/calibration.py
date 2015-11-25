import numpy as np
import matplotlib.pylab as plt

import mex
from . import ais_code
from . import morgan
# import random

import os
import glob
import random

def get_all_possible_orbits():

    files = glob.glob(mex.data_directory + 'ais_dm/dendata/TRACE*.txt')
    orbits = [int(f[-8:-4]) for f in files]
    return orbits

class AISEmpiricalCalibration(object):
    """docstring for AISEmpiricalCalibration"""
    def __init__(self, fname=None, orbits=None, recalibrate=False):

        raise RuntimeError("Depreciated code")

        super(AISEmpiricalCalibration, self).__init__()
        if fname is None:
            fname = mex.data_directory + 'ais_calfile'

        if orbits is None:
            orbits = get_all_possible_orbits()
            orbits = random.sample(orbits, 50)

        self.fname  = fname
        self.orbits = orbits
        self.recalibrate = recalibrate

        if os.path.exists(fname + '.npy') and (not recalibrate):
            print('Loading calibration from %s' % (fname + '.npy'))
            self.cal = np.load(fname + '.npy')
        else:
            self.calibrate()

    def _construct_calibration_array(self):
        if os.path.exists(self.fname + '_in.npy') and not (self.recalibrate):
            out = np.load(self.fname + '_in.npy')
        else:
            chunks = []

            for o in self.orbits:
                print('Reading %d' % o)


                try:
                    dm_data = morgan.read_orbit(o)
                except Exception as e:
                    print(e)
                    continue

                this_orbit = np.empty((dm_data.shape[0], 3)) + np.nan
                this_orbit[:,0] = dm_data[:,0]
                this_orbit[:,1] = dm_data[:,1]

                igs = ais_code.read_ais(o)

                for inx, i in enumerate(igs):
                    new_n = self._q(i)

                    dt = np.abs(this_orbit[:,0] - i.time)
                    inx = np.argmin(dt)
                    if dt[inx] > ais_code.ais_spacing_seconds:
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

    def plot(self):
        if not hasattr(self, 't'): self._construct_calibration_array()
        if not hasattr(self, 'cal'): self.calibrate()

        # plt.close('all')
        plt.figure()
        plt.plot(self.x, self.ly, 'k.')

        plt.xlabel('Ionogram Signal')
        plt.ylabel(r'$log_10 n_e / cm^{-3}$')
        plt.hlines(np.log10(150.), plt.xlim()[0], plt.xlim()[1], color='green',
                        linestyles='dashed')
        for i in range(self.cal.shape[1]):
            c = self.cal[:,i]
            print(c)
            plt.plot((c[0], c[0]), (c[1]-c[2], c[1]+c[2]), 'r-')
            plt.plot(c[0], c[1], 'r.')


        p = np.polyfit(self.x, self.ly, 10)
        x = np.arange(plt.xlim()[0], plt.xlim()[1], 0.01)
        plt.plot(x, np.poly1d(p)(x), 'b-')


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
        # return np.mean(np.max(np.log10(ig.data),1)[0:20])
        return np.mean(np.max(np.log10(ig.data),1)[0:15])


    def __call__(self, ig, density=True):
        if not hasattr(self, 'cal'): self.calibrate()
        x = self._q(ig)

        val = np.interp(x, self.cal[0], self.cal[1])
        err = np.interp(x, self.cal[0], self.cal[2])

        # if not density:
        #     val = 0.5 * val + np.log10(8980.)
        # Errors transformed as we are switching from logarithm
        return 10.**val, 10.**val * err / 0.434

if __name__ == '__main__':
    import mex.ais.aisreview
    import celsius
    c = AISEmpiricalCalibration()

    # plt.close('all')
    c2 = AISEmpiricalCalibration('test_calibration', recalibrate=True)
    c2.plot()

    # plt.close('all')
    # plt.figure(figsize=reversed(celsius.paper_sizes['A4']))
    # for o in c.orbits:
    #     a = mex.ais.aisreview.AISReview(o)
    #     a.plot_ne()
    #     p = mex.orbits[o].periapsis
    #     plt.xlim(p - 200. * 8, p + 200.*8)
    #     plt.title(str(o))
    #     plt.savefig('AISCalibration_Comparison_%d.pdf' % o)

