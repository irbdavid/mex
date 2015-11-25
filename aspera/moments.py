"""
READ IMA AND ELS MOMENTS FROM MF

Bare in mind - different operation of IMA pre 2008 has an impact here
Don't compare apples and oranges within the various parameters

2013-04-24: Updated ELS moments to use Markus's V9b dataset instead of V9

"""
import mex
import numpy as np
import os

import pylab as plt
import urllib.request, urllib.error, urllib.parse

# import cdf
from spacepy import pycdf

# IMA_MOMENT_TAGS = {
#     1:'Time',
#     2:'MEX_ASP_IMA_H_G_red_max3FM_Density',
#     3:'MEX_ASP_IMA_H_G_red_max3FM0_TVelocity',
#     4:'MEX_ASP_IMA_H_G_FTemperature_all_1',
#     5:'MEX_ASP_IMA_PAC',
# }

# 2013-08-12
IMA_MOMENT_TAGS = {
    1:'Time',
    2:'MEX_ASP_IMA_H_G_Density',
    3:'MEX_ASP_IMA_H_G_FDensity_low',
    4:'MEX_ASP_IMA_HVY_L2Density',
    5:'MEX_ASP_IMA_O_Density',
    6:'MEX_ASP_IMA_O2_Density',
    7:'MEX_ASP_IMA_H_G_TVelocity',
    8:'MEX_ASP_IMA_H_G_FMVelocity_low',
    9:'MEX_ASP_IMA_HVY_L2TVelocity',
    10:'MEX_ASP_IMA_O2_TVelocity',
    11:'MEX_ASP_IMA_H_G_TTemperature',
    12:'MEX_ASP_IMA_H_G_FMTemperature_low',
    13:'MEX_ASP_IMA_HVY_L2TTemperature',
    14:'MEX_ASP_IMA_O_TTemperature',
    15:'MEX_ASP_IMA_O2_TTemperature',
    16:'MEX_ASP_IMA_PAC',
}

# 2013-08-15: not sorted, so who cares
IMA_MOMENT_V10_TAGS = {
      1:'Time',
      2:'MEX_ASP_IMA_H_G_Density',
      3:'MEX_ASP_IMA_O_Density',
      4:'MEX_ASP_IMA_O2_Density',
      5:'MEX_ASP_IMA_H_G_Velocity',
      6:'MEX_ASP_IMA_O_Velocity',
      7:'MEX_ASP_IMA_H_G_TTemperature',
      8:'MEX_ASP_IMA_O_TTemperature',
      9:'MEX_ASP_IMA_PAC',
}

# 2013-09-19: see email from MF dated 16 September 2013
IMA_MOMENT_V8_TAGS = {
        1:'Time',
        2:'MEX_ASP_IMA_H_G_red_max3FM_Density',                  # H+G,Density,300-4000eV,R:3,C:FM
        3:'MEX_ASP_IMA_H_G_red_max3FM0_TVelocity',               # H+G,TVelocity,300-4000eV,R:3,C:FM0       FLOAT    1
        4:'MEX_ASP_IMA_H_G_FTemperature_all_1',                  # H+G,FTemperature,all,300-4000eV,R:3,C:FM0   FLOAT    1
        5:'MEX_ASP_IMA_PAC',                                     # MEX_ASP_IMA_PAC                    FLOAT    1
}

# ['MEX_ASP_IMA_H_G_TTemperature', 'MEX_ASP_IMA_O2_Density', 'MEX_ASP_IMA_O_Density', 'MEX_ASP_IMA_PAC', 'MEX_ASP_IMA_H_G_Velocity', 'MEX_ASP_IMA_O_TTemperature', 'Time', 'MEX_ASP_IMA_H_G_Density', 'MEX_ASP_IMA_O_Velocity']

ELS_MOMENT_TAGS = {
             1  :'Time',
             2  :'MEX_ASP_ELS_Density',
             3  :'MEX_ASP_ELS_e__FDensity_all',
             4  :'MEX_ASP_ELS_e__FDensity_low',
             5  :'MEX_ASP_ELS_e__FDensity_high',
             6  :'MEX_ASP_ELS_e__FDensity_sall',
             7  :'MEX_ASP_ELS_ThTemperature',
             8  :'MEX_ASP_ELS_e__FTemperature_all',
             9  :'MEX_ASP_ELS_e__FTemperature_low',
            10  :'MEX_ASP_ELS_e__FTemperature_high',
            11  :'MEX_ASP_ELS_e__FTemperature_sall',
            12  :'MEX_ASP_ELS_Velocity_X',  #3 components
            13  :'MEX_ASP_ELS_Velocity_Y',  #3 components
            14  :'MEX_ASP_ELS_Velocity_Z',  #3 components
            15  :'MEX_ASP_ELS_SCPotential',
            16  :'MEX_ASP_ELS_e__FPotential_DEF',
            17  :'MEX_ASP_ELS_SteppingMode',
            18  :'MEX_ASP_ELS_Flux_5_20eV',
            19  :'MEX_ASP_ELS_Flux_20_30eV',
            20  :'MEX_ASP_ELS_Flux_20_50eV',
            21  :'MEX_ASP_ELS_Flux_50_100eV',
            22  :'MEX_ASP_ELS_Flux_100_200eV',
            23  :'MEX_ASP_ELS_Intensity',    # New in V9b
}

def sync_els_moments_mf(uname, password, verbose=False, t=None):
    """Iterate though moment files, downloading if necessary"""
    start = 20040209
    # http://www.mps.mpg.de/data/projekte/mars-express/aspera/aspera3_mom_flat/ELS_V9/MEX_ASP_ELS_MOM_V9b_20040209.cef.gz

    if t is None:
        t = celsius.spiceet("2013-04-28T12:00:00")

    tf = celsius.now()

    local_directory = mex.data_directory + 'aspera/ELS_V9/'

    remote_url = 'http://www.mps.mpg.de/data/projekte/mars-express/aspera/aspera3_mom_flat/ELS_V9/'
    passman = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    passman.add_password(None, remote_url, uname, password)

    found = 0
    missing = 0
    downloaded = 0
    attempted = 0

    d = os.path.dirname(local_directory)
    if d and not os.path.exists(d):
        if verbose: print('Creating %s' % d)
        os.makedirs(d)

    while t < tf:
        attempted += 1
        fname = 'MEX_ASP_ELS_MOM_V9b_%s.cef.gz' % celsius.utcstr(t, 'ISOC')[:10].replace('-','')
        url = remote_url + fname
        local_fname = local_directory + fname
        if os.path.exists(local_fname):
            found += 1
            if verbose: "OK: %s" % fname
        else:
            try:
                authhandler = urllib.request.HTTPBasicAuthHandler(passman)
                opener = urllib.request.build_opener(authhandler)
                urllib.request.install_opener(opener)
                pagehandle = urllib.request.urlopen(url)
                thepage = pagehandle.read()
                f = open(local_fname, 'w')
                f.write(thepage)
                f.close()
                downloaded += 1
                if verbose:
                    print('DL: %s' % fname)

            except urllib.error.HTTPError as e:
                missing += 1
                if e.code == 404:
                    if verbose:
                        print('MISSING: %s' % fname)
                # raise IOError('Could not read %s' % url)
        t += 86400.

    print('\nAttempted %d, found %d, downloaded %d, missing %d (check: %s)' % (
            attempted, found, downloaded, missing, str((attempted - (found + downloaded + missing)) == 0)))


def sync_ima_moments_mf(uname, password, verbose=False, t=None):
    """Iterate though IMA moment files, downloading if necessary"""
    # http://www.mps.mpg.de/data/projekte/mars-express/aspera/aspera3_mom_flat/IMA_V9/MEX_ASP_IMA_MOM_V9_20040201.cef
    # MEX_ASP_IMA_MOM_V9_20040201.cef
    if t is None:
        t = celsius.spiceet("2012-12-26T12:00:00")

    tf = celsius.now()

    local_directory = mex.data_directory + 'aspera/IMA_V9/'

    remote_url = 'http://www.mps.mpg.de/data/projekte/mars-express/aspera/aspera3_mom_flat/IMA_V9/'
    passman = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    passman.add_password(None, remote_url, uname, password)

    found = 0
    missing = 0
    downloaded = 0
    attempted = 0

    d = os.path.dirname(local_directory)
    if d and not os.path.exists(d):
        if verbose: print('Creating %s' % d)
        os.makedirs(d)

    while t < tf:
        attempted += 1
        fname = 'MEX_ASP_IMA_MOM_V9_%s.cef' % celsius.utcstr(t, 'ISOC')[:10].replace('-','')
        url = remote_url + fname
        local_fname = local_directory + fname
        if os.path.exists(local_fname):
            found += 1
            if verbose: "OK: %s" % fname
        else:
            try:
                authhandler = urllib.request.HTTPBasicAuthHandler(passman)
                opener = urllib.request.build_opener(authhandler)
                urllib.request.install_opener(opener)
                pagehandle = urllib.request.urlopen(url)
                thepage = pagehandle.read()
                f = open(local_fname, 'w')
                f.write(thepage)
                f.close()
                downloaded += 1
                if verbose:
                    print('DL: %s' % fname)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    missing += 1
                    if verbose:
                        print('MISSING: %s' % fname)
                # raise IOError('Could not read %s' % url)

        t += 86400.

    print('\nAttempted %d, found %d, downloaded %d, missing %d (check: %s)' % (
            attempted, found, downloaded, missing, str((attempted - (found + downloaded + missing)) == 0)))


# asp_ima_v8_dtype = np.dtype()
def read_ima_moments(start, finish=None, verbose=False, usecols=None, nan=True, version=None):
    """
    IMA Moments from markus.  V10 supplied as CDF, V9 as CEF.  V8 as CEF
    """

    if version is None:
        version = 10

    if version not in (8, 9, 10):
        raise ValueError("Only versions 8, 9 and 10 known about")

    if not usecols:
        if version == 9:
            usecols = (1, 2, 4, 7, 11)
        elif version == 10:
            usecols = (1, 2, 3, 4, 5, 7)
        elif version == 8:
            usecols = (1, 2, 3, 4, 5)

    # if verbose:
    #     print "Reading IMA_V%d" % version
    #     print 'Reading... '
    #     for c in usecols:
    #         print IMA_MOMENT_TAGS[c]

    if finish is None:
        finish = start + 7. * 3600.

    if finish < start:
        raise ValueError("Start before finish: Start %s, Finish %s" %
                        (celsius.utcstr(start), celsius.utcstr(finish)))

    d = celsius.spiceet_to_datetime(start)

    if (d.year < 2008) and version == 9:
        raise ValueError('Only able to deal with the 2008+ data for now')

    if version == 9:
        new_dtype = np.dtype([ (IMA_MOMENT_TAGS[k], np.float64) for k in usecols])
    elif version == 10:
        new_dtype = np.dtype([ (IMA_MOMENT_V10_TAGS[k], np.float64) for k in usecols])
    elif version == 8:
        new_dtype = np.dtype([ (IMA_MOMENT_V8_TAGS[k], np.float64) for k in usecols])

    t0 = start - 86400. * 1.1
    chunks = []
    while t0 < (finish + 86400. * 1.1):
        d = celsius.spiceet_to_datetime(t0)
        t0 += 86400.

        if version == 9:
            fname = mex.data_directory + 'aspera/IMA_V9/MEX_ASP_IMA_MOM_V9_'
            fname += '%4d%02d%02d.cef' % (d.year, d.month, d.day)

            if os.path.exists(fname):
                try:
                    data = np.loadtxt(fname, converters={1:lambda x:celsius.spiceet(x[:-1])},
                        skiprows=184, usecols=usecols, dtype=new_dtype).T
                except IOError as e:
                    print('Error: %s - %s' % (fname, str(e)))
                    continue

                chunks.append(data)

                if verbose:
                    print('Read %d records from %s' % (data.shape[0], fname))

            else:
                if verbose:
                    print('%s does not exist' % fname)

        elif version == 8:
            fname = mex.data_directory + 'aspera/IMA_V8/ASP3_IMA_MOM8HG_'
            fname += '%4d%02d%02d.cef.gz' % (d.year, d.month, d.day)

            if os.path.exists(fname):
                try:
                    data = np.loadtxt(fname, converters={1:lambda x:celsius.spiceet(x[:-1])},
                        skiprows=64, usecols=usecols, dtype=new_dtype).T
                except IOError as e:
                    print('Error: %s - %s' % (fname, str(e)))
                    continue

                chunks.append(data)

                if verbose:
                    print('Read %d records from %s' % (data.shape[0], fname))

            else:
                if verbose:
                    print('%s does not exist' % fname)

        elif version == 10:
            fname = mex.data_directory + 'aspera/IMA_V10/MEX_ASP_IMA_MOM_V10_'
            fname += '%4d%02d%02d.cdf' % (d.year, d.month, d.day)
            if os.path.exists(fname):
                f = cdf.archive(fname)
                data = np.empty(len(f['Time']), dtype=new_dtype) # + np.nan

                # for t in f['Time']:
                #     tt = str(t)[1:-1]
                #     print t, str(t), tt, celsius.utcstr_to_spiceet(tt)

                data['Time'] = np.array([celsius.utcstr_to_spiceet(str(t)[1:-1]) for t in f['Time']])

                for cn in usecols[1:]:
                    c = IMA_MOMENT_V10_TAGS[cn]
                    if cn in (5,6): # just keep the X component of the velocities
                        data[c] = np.array([a[0] for a in f[c]])
                    else:
                        data[c] = np.array(f[c])

                chunks.append(data)
                if verbose:
                    print('Read %d records from %s' % (data.shape[0], fname))
            else:
                if verbose:
                    print('%s does not exist' % fname)

    if not chunks:
        raise IOError('No IMA moment data in the interval %s - %s' % (celsius.utcstr(start), celsius.utcstr(finish)))

    out = np.hstack(chunks)
    inx = (out['Time'] > start) & (out['Time'] < finish)

    print('Retaining ', np.mean(inx))

    if nan:
        for n in out.dtype.names:
            out[n][out[n] < -1e30] = np.nan

    return out[inx]

def read_els_moments(start, finish=None, verbose=False, usecols=None, nan=True):
    """
    Variable columns, names (unchecked at read, col=0 is a record number, useless)
    Markus's ELS_V9 moments read in here
    """

    if not usecols:
        usecols = (1,3,8,12)

    # if verbose:
    #     print 'Reading... '
    #     for c in usecols:
    #         print ELS_MOMENT_TAGS[c]

    if finish is None:
        finish = start + 86400.

    if finish < start:
        raise ValueError("Start before finish: Start %s, Finish %s" %
                        (celsius.utcstr(start), celsius.utcstr(finish)))

    d = celsius.spiceet_to_datetime(start)

    new_dtype = np.dtype([ (ELS_MOMENT_TAGS[k], np.float64) for k in usecols])

    t0 = start
    chunks = []
    while t0 < finish:
        # fname = mex.data_directory + 'From Markus/19_Nov_2012/ELS_V9/MEX_ASP_ELS_MOM_V9_'
        fname = mex.data_directory + 'aspera/ELS_V9/MEX_ASP_ELS_MOM_V9b_'
        d = celsius.spiceet_to_datetime(t0)

        t0 += 86400.

        fname += '%4d%02d%02d.cef.gz' % (d.year, d.month, d.day)
        if os.path.exists(fname):

            try:
                data = np.loadtxt(fname, converters={1:lambda x:celsius.spiceet(x[:-1])},
                        skiprows=225, usecols=usecols, dtype=new_dtype).T
            except IOError as e:
                print('Error: %s - %s' % (fname, str(e)))
                continue

            chunks.append(data)
            if verbose:
                print('Read %d records from %s' % (data.shape[0], fname))
        else:
            if verbose:
                print('%s does not exist' % fname)

    if not chunks:
        raise IOError('No ELS moment data in the interval %s - %s' % (celsius.utcstr(start), celsius.utcstr(finish)))

    out = np.hstack(chunks)
    inx = (out['Time'] > start) & (out['Time'] < finish)

    if nan:
        for n in out.dtype.names:
            out[n][out[n] < -1e30] = np.nan


    return out[inx]

def read_combined_moments(start, finish=None, verbose=False, els_cols=None, ima_cols=None, nan=True, ima_version=None):
    """ Reads both sets of moments, combines them to a single np recarray, interpolated at the ELS points"""

    if finish is None:
        finish = start + 86400.

    if finish < start:
        raise ValueError("Start before finish: Start %s, Finish %s" %
                        (celsius.utcstr(start), celsius.utcstr(finish)))

    els = read_els_moments(start, finish, verbose=verbose, usecols=els_cols, nan=nan)
    ima = read_ima_moments(start, finish, verbose=verbose, usecols=ima_cols, nan=nan, version=ima_version)

    all_keys = dict()
    for k in list(els.dtype.fields.keys()): all_keys[k] = np.float64
    for k in list(ima.dtype.fields.keys()): all_keys[k] = np.float64

    new_dt = np.dtype([(k, all_keys[k]) for k in all_keys])
    n = els.shape[0]
    # print new_dt, n
    new = np.empty(n, new_dt)
    for k in all_keys:
        new[k] = np.nan

    # Fill with the ELS info
    for e in list(els.dtype.fields.keys()):
        # print e
        new[e] = els[e]

    # if verbose:
    #     print ima.dtype.fields.keys()
    #     print ima['Time']
    #     print min(ima['Time']), max(ima['Time']), np.mean(ima['Time']), np.mean(np.isfinite(ima['Time']))
    #     print np.median(np.diff(ima['Time']))
    #     print ima['Time']

    if ima['Time'].shape[0] == 0:
        print('NO IMA DATA TO SYNC')
        return new

    for i in list(ima.dtype.fields.keys()):
        # print i
        if i == 'Time': continue

        # print i, np.sum(np.isfinite(ima['Time'])), np.sum(np.isfinite(ima[i])), ima[i].shape
        new[i] = np.interp(new['Time'], ima['Time'], ima[i], left=np.nan, right=np.nan)

    return new

def test_plot(t0, t1=None, verbose=False, ima_version=8):

    from . import aspera_mf
    from . import aspera_hn

    if t1 is None:
        t1 = t0 + 86400. * 3

    plt.figure()

    if ima_version == 8:
        ima_density = 'MEX_ASP_IMA_H_G_red_max3FM_Density'
        ima_velocity = 'MEX_ASP_IMA_H_G_red_max3FM0_TVelocity'
        ima_temperature = 'MEX_ASP_IMA_H_G_FTemperature_all_1'
    elif ima_version == 9:
        ima_density = 'MEX_ASP_IMA_H_G_Density'
        ima_velocity = 'MEX_ASP_IMA_H_G_TVelocity'
        ima_temperature = 'MEX_ASP_IMA_H_G_TTemperature'
    elif ima_version == 10:
        ima_density = 'MEX_ASP_IMA_H_G_Density'
        ima_velocity = 'MEX_ASP_IMA_H_G_Velocity'
        ima_temperature = 'MEX_ASP_IMA_H_G_TTemperature'



    ax = plt.subplot(711)
    data = read_combined_moments(t0, t1, verbose=verbose, ima_version=ima_version)
    plt.plot(data['Time'], data['MEX_ASP_ELS_e__FDensity_all'], 'k.', mew=0., mec='k')
    plt.plot(data['Time'], data[ima_density], 'r.', mew=0., mec='r')
    plt.ylim(0.01, 100.)
    plt.yscale('log')

    plt.subplot(712, sharex=ax)
    plt.plot(data['Time'], data['MEX_ASP_ELS_Velocity'], 'k.', mew=0., mec='k')
    plt.plot(data['Time'], data[ima_velocity], 'r.', mew=0., mec='r')

    plt.ylim(-1000., 1000.)

    plt.subplot(713, sharex=ax)
    plt.plot(data['Time'], data['MEX_ASP_ELS_e__FTemperature_all'], 'k.', mew=0., mec='k')
    plt.plot(data['Time'], data[ima_temperature], 'r.', mew=0., mec='r')
    plt.ylim(0., 200.)

    # plt.subplot(714, sharex=ax)
    # aspera_hn.plot_els_spectra(t0, t1)
    #
    # plt.subplot(715, sharex=ax)
    # aspera_hn.plot_ima_spectra(t0, t1)
    #
    # plt.subplot(716, sharex=ax)
    # aspera_mf.plot_aspera_ima(t0, t1)
    #
    # plt.subplot(717, sharex=ax)
    # aspera_mf.plot_aspera_els(t0, t1)

    plt.xlim(t0, t1)
    mex.setup_time_axis()

    plt.figure()
    plt.subplot(221)
    plt.plot(data['MEX_ASP_ELS_e__FDensity_all'],  data[ima_density], 'k.', mew=0., mec='k')

    plt.subplot(222)
    plt.plot(data['MEX_ASP_ELS_e__FTemperature_all'],  data[ima_velocity], 'k.', mew=0., mec='k')

    plt.subplot(223)
    plt.plot(data['MEX_ASP_ELS_Velocity'],  data[ima_temperature], 'k.', mew=0., mec='k')


    plt.show()

if __name__ == '__main__':
    plt.close('all')

    # print read_ima_moments(mex.orbits[7939].periapsis, mex.orbits[7940].periapsis, verbose=True)
    # raise RuntimeError()
    # sync_els_moments_mf('aspera_team', 'as_pera', verbose=True)
    # sync_ima_moments_mf('aspera_team', 'as_pera', verbose=True)
    # test_plot(mex.orbits[7939].periapsis, mex.orbits[7940].periapsis, verbose=True, ima_version=8)
    t0 = celsius.spiceet("2012-02-20T00:00")
    t1 = celsius.spiceet("2012-03-10T00:00")
    test_plot(t0, t1, verbose=True, ima_version=10)

    # test_plot(mex.orbits[7940].periapsis, mex.orbits[7941].periapsis, verbose=True)
    # test_plot(celsius.spiceet('01 FEB 2012'), celsius.spiceet('4 FEB 2012'))
    # test_plot(celsius.spiceet('10 MAR 2012'), celsius.spiceet('15 MAR 2012'))
