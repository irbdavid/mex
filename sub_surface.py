import mex
import numpy as np
import matplotlib.pylab as plt
import os

SS_TEC_DTYPE = np.dtype([
                    ("PULSE_NUMBER", np.int32),
                    ("EPHEMERIS_TIME", np.float64),
                    ("LATITUDE", np.float32),
                    ("LONGITUDE", np.float32),
                    ("LOCAL_TRUE_SOLAR_TIME", np.float32),
                    ("X_SC_MSO", np.float32),
                    ("Y_SC_MSO", np.float32),
                    ("Z_SC_MSO", np.float32),
                    ("SZA", np.float32),
                    ("TEC", np.float32),
                    ("A1", np.float32),
                    ("A2", np.float32),
                    ("A3", np.float32),
                    ("FLAG", np.bool),
                ])

# Lillis 10a: ~1 TECU at SZA=0, ~0.1 at 90 deg, ~0.02 beyond terminator
TECU_electrons_per_m2 = 1E16

def get_tec_filename(orbit):

    f = 'DDR%dX/FRM_TEC_DDR_%d.TAB' % (orbit / 10, orbit)

    if orbit < 1830:
        raise IOError("No data prior to orbit 1830")

    if orbit < 2530:
        d =  mex.data_directory + 'marsis/MEX-M-MARSIS-5-DDR-SS-TEC-V1.0/DATA/'
    elif orbit < 4810:
        d =  mex.data_directory + 'marsis/MEX-M-MARSIS-5-DDR-SS-TEC-EXT1-V1.0/DATA/'
    # elif orbit < 8410:
    #     raise IOError("I have missing data from ~4810 ~8410")
    #     d =  mex.data_directory + 'marsis/MEX-M-MARSIS-5-DDR-SS-TEC-EXT1-V1.0/DATA/'
    elif orbit < 11000:
        d =  mex.data_directory + 'marsis/TEC_FROM_WLODEK_UNPUBLISHED/DATA/'

        # Filename is also mangled here
        if orbit > 10000:
            f = 'DDR%dXX/FRM_TEC_DDR_%d.TAB' % (orbit / 100, orbit)

    else:
        raise IOError("No data installed after orbit 11000")

    if not os.path.exists(d+f):
        raise IOError("File %s doesn't exist" % (d + f))

    return d + f

def read_tec(start, finish=None, params=('EPHEMERIS_TIME', 'TEC', 'SZA')):
    """read_tec from marsis sub-surface dataset.  Start, finish are ets, position=True
    retains the various position columns in the tec files"""
    if finish is None:
        finish = start + 86400.

    start_orbit = mex.orbits[start].number
    finish_orbit = mex.orbits[finish].number
    orbits = list(range(start_orbit, finish_orbit + 1))

    # if position:
    # SS_TEC_DTYPE = np.dtype([
    #                 ("PULSE_NUMBER", np.int32),
    #                 ("EPHEMERIS_TIME", np.float64),
    #                 ("LATITUDE", np.float32),
    #                 ("LONGITUDE", np.float32),
    #                 ("LOCAL_TRUE_SOLAR_TIME", np.float32),
    #                 ("X_SC_MSO", np.float32),
    #                 ("Y_SC_MSO", np.float32),
    #                 ("Z_SC_MSO", np.float32),
    #                 ("SZA", np.float32),
    #                 ("TEC", np.float32),
    #                 ("A1", np.float32),
    #                 ("A2", np.float32),
    #                 ("A3", np.float32),
    #                 ("FLAG", np.bool),
    #             ])
    # else:
    #     SS_TEC_DTYPE = np.dtype([
    #                 ("EPHEMERIS_TIME", np.float64),
    #                 ("SZA", np.float32),
    #                 ("TEC", np.float32),
    #                 ("A1", np.float32),
    #                 ("A2", np.float32),
    #                 ("A3", np.float32),
    #                 ("FLAG", np.bool),
    #             ])

    chunks = []
    for o in orbits:
        try:
            fname = get_tec_filename(o)
            d = np.loadtxt(fname, dtype=SS_TEC_DTYPE)
            chunks.append(d)
        except IOError as e:
            print('Failed to read TEC for orbit %d: %s' % (o, str(e)))

    lengths = [c.shape[0] for c in chunks] + [0]
    n = sum(lengths)

    if n < 1:
        print('No data loaded, returning an empty array')
        return np.zeros(0, dtype=SS_TEC_DTYPE)

    output = np.hstack(chunks)

    output = output[(output['EPHEMERIS_TIME'] > start) & (output['EPHEMERIS_TIME'] < finish)]

    return output

def plot_tec(start=None, finish=None, ax=None, x_axis='EPHEMERIS_TIME',
                fmt='k.', bad_fmt='r.', mew=0., labels=True):
    """Quickie that plots TEC in TECU"""

    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)

    if x_axis == 'EPHEMERIS_TIME':
        if start is None:
            start, finish = plt.xlim()
        else:
            plt.xlim(start, finish)

    t = read_tec(start, finish)

    good = t['FLAG'] == True
    plt.plot(t[x_axis][good], t['TEC'][good] / TECU_electrons_per_m2, fmt, mew=mew)
    plt.plot(t[x_axis][~good], t['TEC'][~good] / TECU_electrons_per_m2, bad_fmt, mew=mew)

    plt.xlabel(x_axis)
    plt.ylabel('TEC / TECU')
    plt.yscale('log')

if __name__ == '__main__':
    # Test

    plt.figure()
    start = mex.orbits[2849].start
    finish = mex.orbits[2859].finish
    t = read_tec(start, finish)
    # plt.plot(t[0], t[1], 'k.')
    plt.plot(t["SZA"], t["TEC"] / TECU_electrons_per_m2, 'k.')
    plt.ylabel('TEC / TECU')
    plt.yscale('log')
    plt.xlabel('SZA / deg')

    plt.figure()
    plot_tec(start, finish)

    plt.show()