import matplotlib
matplotlib.use('Agg')

import multiprocessing as mp
import subprocess

import matplotlib.pyplot as plt
import sys
import gc
from . import aisreview
import mex
import mex.ais
import os

import numpy as np
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

def async_worker_review(o, debug=False, verbose=False):
    result = 'FAILED %d\n' % o
    try:
        mex.ais.compute_all_digitizations(o)
        aisreview.main(o, show=False, save=True)
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
    fh = open(os.path.getenv('SC_DATA_DIR') + 'mex/ais_workflow_output.txt','w')
    while True:
        g = queue.get()
        fh.write(str(g)+'\n')
        fh.flush()
        queue.task_done()
    fh.close()
#
# if __name__ == '__main__':
#     global queue
#     queue = mp.Queue()
#     wproc = mp.Process(target=watcher)
#     wproc.start()
#     n = 0
#     pool = mp.Pool(4, )
#     r = [pool.apply_async(worker, (i,)) for i in range(10)]
#     pool.close()
#     pool.join()
#
#     wproc.join()

class DeltaTimer(object):
    """docstring for DeltaTimer"""
    def __init__(self):
        super(DeltaTimer, self).__init__()
        self._t = celsius.now()

    def __call__(self, s):
        n = celsius.now()
        print(s + ': %0.3fs' % (n - self._t))
        self._t = n

def async_worker(o, debug=False, verbose=False):

    output = '%d - Failed:\n' % o

    try:
        plt.hot()
        now = celsius.now()
        np.seterr(all='ignore')
        # ionogram_list = mex.mex.ais.read_ais(o)
        ionogram_list = mex.ais.read_ais(o)
        n = len(ionogram_list)

        db = mex.ais.DigitizationDB(orbit=o, load=False, verbose=True)

        if save_every:
            dirname = mex.locate_data_directory() + ('marsis/quicklook_igs/%05d/%05d/' % ((o/1000)*1000, o))
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        fp_local_counter = 0
        td_cyclotron_counter = 0
        ion_counter = 0
        ground_counter = 0
        saved_count = 0

        if verbose:
            print('Processing ionograms:')
        for e, i in enumerate(ionogram_list):

            dt = DeltaTimer()

            if verbose:
                print('\n%d' % e)
            i.digitization = mex.ais.IonogramDigitization(i)
            if not 'failed' in i.calculate_fp_local().lower(): fp_local_counter += 1
            if not 'failed' in i.calculate_ground_trace().lower(): ground_counter += 1
            if not 'failed' in i.calculate_reflection().lower(): ion_counter += 1
            if not 'failed' in i.calculate_td_cyclotron().lower(): td_cyclotron_counter += 1
            i.delete_binary_arrays()
            if i.digitization:
                i.digitization.set_timestamp()
                db.add(i.digitization)
            if verbose:
                dt("\tProcessing")
            if save_every:
                if (e % save_every) == 0:
                    fig = plt.figure(figsize=(16,12))
                    ax = plt.subplot(221)
                    i.plot(ax=ax, overplot_digitization=False, overplot_model=False, vmin=-16, vmax=-12)
                    ax = plt.subplot(222)
                    i.plot(ax=ax, overplot_digitization=True, vmin=-16, vmax=-12)
                    ax = plt.subplot(223)
                    i.plot(ax=ax, vmin=-16, vmax=-12, altitude_range=[-300,400],
                        overplot_digitization=False, overplot_model=False)
                    ax = plt.subplot(224)
                    i.plot(ax=ax, overplot_digitization=True, vmin=-16, vmax=-12,
                        altitude_range=[0,300])

                    if verbose:
                        dt("\Plotting")

                    fname = '%05d.png' % saved_count
                    plt.savefig(dirname + fname)
                    # print dirname + fname
                    plt.close(fig)
                    saved_count += 1
                    if verbose:
                        dt("\tPNG write")

        if verbose:
            print('Writing db: %d' % len(db))

        db.write()

        duration = celsius.now() - now
        if verbose:
            print('%f seconds elapsed' % duration)

        if saved_count:
            input_jpgs = dirname + '*.jpg'
            output_file = mex.locate_data_directory() + ('marsis/quicklook_igs/%05d/%05d.avi' % ((o/1000)*1000, o))

            # mencoder "mf://*.jpg" -mf fps=25 -o output.avi -ovc lavc -lavcopts vcodec=mpeg4
            # mencoder mf://*.png -mf fps=20:type=jpg -ovc x264 -x264encopts preset=slow:tune=film:crf=20 -of rawvideo -o ~/01997.264
            commands = [('mogrify',
                            '-format',
                            'jpg',
                            dirname + '*.png'),
                    ('mencoder',
                         'mf://' + input_jpgs,
                         '-mf',
                         'fps=20',
                         '-ovc',
                         'lavc',
                         '-lavcopts',
                         'vcodec=mpeg4:vbitrate=3000000',
                         '-msglevel',
                         'all=-1',
                         '-o',
                         output_file),
                     ('rm',
                        '-rf',
                        dirname)
                        ]

            if saved_count < 1:
                commands = (commands[2],)

            cmd_errors = []
            for command in commands:
                try:
                    os.spawnvp(os.P_WAIT, command[0], command)
                    print("%s\n" % ' '.join(command))
                    subprocess.check_call(command)
                except subprocess.CalledProcessError as e:
                    print("\n\n--- ENCOUNTERED SUBPROCESS ERROR %s\n\n" % str(e))
                    cmd_errors.append("%s\n: ERROR %s" % (' '.join(command),  str(e)))
                    if debug:
                        raise

        review_status = 'SUCCESS'
        try:
            aisreview.main(o, show=False, save=True)
        except Exception as e:
            review_status = 'FAILED: %s' % str(e)
            if debug:
                raise

        output = "COMPLETED ORBIT %d after %d seconds" % (o, duration)
        output += "\n\tFinished at %s" % celsius.utcstr(celsius.now(),"ISOC")
        output += "\n\tSuccessful fits: FP: %d, CYC: %d, ION: %d, GND: %d" % (fp_local_counter,
                                td_cyclotron_counter, ion_counter, ground_counter)
        output += "\n\tReview: " + review_status
        output += "\n\tSaved %d PNGS to %s\n" % (saved_count, dirname)
        for command in commands:
            output += "\tExecuted %s\n" % ' '.join(command)
        for err in cmd_errors:
            output += "\t%s\n" % err
    except Exception as e:
        output += str(e) + '\n'
        output += 'At: ' + celsius.utcstr(celsius.now(),"ISOC")
        if debug:
            raise

    # print output
    global queue
    if queue:
        queue.put(output)

    return output

if __name__ == '__main__':

    verbose = False
    save_every = 1
    np.seterr(all='ignore')
    t0 = celsius.now()
    exception_list = {}
    repeat = True

    start = determine_last_processed_orbit()
    start = 1840
    finish = mex.orbits[celsius.now()].number

    if len(sys.argv) > 1:
        start = int(sys.argv[1])
    if len(sys.argv) > 2:
        finish = int(sys.argv[2])

    coverage = mex.ais.get_ais_coverage()

    done_orbits = []
    orbits = list(range(start, finish))

    completed_orbits = []
    duration_list = []

    orbits = [o for o in orbits if o not in done_orbits]

    orbits = [o for o in orbits if coverage[o] > 0]

    f = lambda o: os.path.exists(mex.data_directory + \
                    'marsis/ais_digitizations/%05d/%05d.dig' % (o/1000 * 1000, o))

    if not repeat:
        orbits = [o for o in orbits if f(o)]

    def cb(s):
        print(s)


    queue = mp.JoinableQueue()
    # f = open('output.txt','w')
    # f.write("Starting...\n\n")
    # f.close()

    print('-- Starting writer')
    writer = mp.Process(target=queue_writer)
    writer.start()

    runner = async_worker_review
    runner = async_worker_computer

    print('--- test --- ')
    print(runner(4264, debug=True, verbose=True))
    print('--- /test --- ')

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
    r = [pool.apply_async(runner, (o,)) for o in orbits]
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
