import matplotlib
matplotlib.use('Agg')
print('MPL Backend: ', matplotlib.get_backend())
import numpy as np

import matplotlib.pyplot as plt

import celsius
import mex
import mex.ais
from mex.ais.aisreview import main

import sys
import gc
import os

import multiprocessing as mp
import subprocess

np.seterr(all='ignore')
save_every = 1

queue=False

def determine_last_processed_orbit():
    max_orbit = -1
    p = os.getenv('SC_DATA_DIR') + 'mex/marsis/ais/'
    for dname in os.listdir(p):
        if dname[:3] != 'RDR': continue

        orbit = int(dname[3:-1]) * 10
        if orbit > max_orbit:
            max_orbit = orbit
            for fname in os.listdir(p + dname):
                orbit = int(fname.split('_')[-1][:-4])
                if orbit > max_orbit:
                    max_orbit = orbit

    return max_orbit

def junk(o):
    return str(o)

def async_worker_computer(o, debug=False, verbose=False):
    result = 'FAILED %d\n' % o
    try:
        mex.ais.compute_all_digitizations(o)
        result = 'SUCCESSFULLY %d\n' % o
    except Exception as e:
        result = result + str(e)
        if debug:
            raise

    global queue
    if queue:
        queue.put(result)
    if verbose:
        print(result)
    return result

def async_worker_review(o, debug=True, verbose=True):
    result = 'FAILED %d\n' % o
    try:
        d = mex.ais.DigitizationDB(o)
        if len(d) == 0:
            mex.ais.compute_all_digitizations(o)
        main(o, show=False, save=True)
        result = 'SUCCESSFULLY %d\n' % o
    except Exception as e:
        result = result + str(e)
        if debug:
            raise

    global queue
    if queue:
        queue.put(result)
    if verbose:
        print(result)
    return result

def queue_writer():
    global queue
    # pass
    fh = open(os.getenv('SC_DATA_DIR') + 'mex/ais_workflow_output.txt','w')
    while True:
        g = queue.get()
        fh.write(str(celsius.utcstr(celsius.now())) + ': '+ str(g)+'\n')
        fh.flush()
        queue.task_done()
    fh.close()

class DeltaTimer(object):
    """docstring for DeltaTimer"""
    def __init__(self):
        super(DeltaTimer, self).__init__()
        self._t = celsius.now()

    def __call__(self, s):
        n = celsius.now()
        print(s + ': %0.3fs' % (n - self._t))
        self._t = n

if __name__ == '__main__':

    verbose = False
    save_every = 1
    np.seterr(all='ignore')
    t0 = celsius.now()
    exception_list = {}
    repeat = True

    start = determine_last_processed_orbit() - 50
    # start = 1840
    # start = 14935
    finish = mex.orbits[celsius.now()].number - 10

    if len(sys.argv) > 1:
        start = int(sys.argv[1])
    if len(sys.argv) > 2:
        finish = int(sys.argv[2])

    orbits = list(range(start, finish))

    completed_orbits = []
    duration_list = []

    # f = lambda o: os.path.exists(mex.data_directory + \
    #                 'marsis/ais_digitizations/%05d/%05d.dig' % (o/1000 * 1000, o))
    #
    # if not repeat:
    #     orbits = [o for o in orbits if f(o)]

    def cb(s):
        print(s)


    queue = mp.JoinableQueue()
    # f = open('output.txt','w')
    # f.write("Starting...\n\n")
    # f.close()

    print('-- Starting writer')
    writer = mp.Process(target=queue_writer)
    writer.start()

    # runner = async_worker_review
    runner = async_worker_computer

    # print('--- test --- ')
    # print(runner(4264, debug=True, verbose=True))
    # print('--- /test --- ')

    processes = 1
    if 'dja' in os.getenv('USER'): #dunno why HOSTNAME doesn't work
        processes = 8

    print('-- Creating pool of %d processes' % processes)
    # pool = mp.Pool(processes, async_worker_init, [the_queue])
    pool = mp.Pool(processes)
    # r = pool.map_async(worker, reversed(orbits), callback=cb)
    # r.wait()
    # r = [pool.apply_async(async_worker, (o,)) for o in reversed(orbits)]

    print('-- Starting work...')
    print('-- orbits = ', orbits)
    r = [pool.apply_async(runner, (o,)) for o in reversed(orbits)]
    print('-- Jobs allocated')

    pool.close()
    print('-- Pool closed, joining pool')

    pool.join()

    queue.close()
    queue.join()
    print('-- Queue closed, joining writer')
    writer.terminate()
    # writer.join()
    # print r[0].get()`

    print("\n" * 5)
    print("Total Duration: %f h" % ((celsius.now() - t0)/ 3600.))
    # print "Exceptions encountered:"
    # for k, v in exception_list.iter_items():
    #     print '%s: %s' % (celsius.utcstr(k), str(v))

    print("Successful orbits, durations:")
    for o, d in zip(completed_orbits, duration_list):
        print("%d: %d" % (o, d))

    mex.ais.ais_code._generate_ais_coverage()
    mex.ais.ais_code._generate_ais_index(recompute=False, update=True)
