# to read data from DM
import numpy as np
import mex
from . import ais_code

def read_orbit(orbit, return_empty=False):
    """For reading his TRACE_XXXXX.txt data sets (local density, time, and ionospheric traces only)"""
    fname = mex.data_directory + 'ais_dm/dendata/TRACE_%d.txt' % orbit

    try:
        d = np.loadtxt(open(fname, 'r'), converters={1:celsius.utcstr_to_spiceet})
    except IOError as e:
        print('File %s not findings' % fname)
        if return_empty:
            return np.zeros((1,2)) + np.nan
        raise e

    new = np.empty((d.shape[0], 2))

    new[:,0] = d[:,1]
    new[:,1] = ais_code.fp_to_ne(d[:,6] * 1E6)

    return new

def read_orbit_idl(orbit, return_empty=False):
    """Read his IDL data (per-orbit files)"""
    try:
        fname = mex.data_directory + 'ais_dm/idl/TRACE_%d.txt' % orbit
        raise NotImplementedError()
    except IOError as e:
        return read_orbit(orbit, return_empty=return_empty)


def produce_single_dm_ne_file(fname=None):
   if fname is None:
      fname = mex.data_directory + 'ais_dm/dja_generated_file.npy'

   data = []
   for orbit in range(1840, mex.orbits[celsius.now()].number):
      try:
         d = read_orbit(orbit)
      except Exception as e:
         print('%d: Error' % orbit)
         print('\t' + str(e))
         continue
      print('%d: %d, %d' % (orbit, d.shape[0], d.shape[1]))
      data.append(d)

   data = np.vstack(data)

   print('\n\nRead %d records' % data.shape[0])
   np.save(fname, data)
   return data


if __name__ == '__main__':
    # compare()
    produce_single_dm_ne_file()