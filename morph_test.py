import numpy as np
import numpy.linalg
import numpy.random

import matplotlib.pylab as plt
import matplotlib as mpl
from scipy.interpolate import griddata
import datetime
import os
import sys
import struct
import stat
import copy as copy_module

import mex
from . import mex_time
import spiceypy

import celsius
# import scipy.ndimage.filters as filters
from scipy.signal import detrend
import scipy.ndimage.morphology as morphology

from . import ais

def main():
    plt.close("all")
    plt.figure(figsize=(20, 12))

    n_ig = 6
    igs = []
    count = 0
    while count < n_ig:
        orbit = np.random.randint(6000) + 2000

        file_name = ais.find_file(orbit)
        nsweeps = 0
        try:
            stats = os.stat(file_name)
            nsweeps = stats[stat.ST_SIZE] / 400
        except OSError as e:
            print(e)

        if nsweeps > (180 * 150):
            tmp = ais.read_ais(orbit)
            if len(tmp) > 150:
                ignumber = np.random.randint(len(tmp))
                igs.append(tmp[ignumber])
                count = count + 1
                print("ORBIT %d, Ionogram %d" % (orbit, ignumber))
                continue

        print("REJECTING")

    n = len(igs)
    igs = igs[0:n_ig]

    g = mpl.gridspec.GridSpec(6, 6, wspace=0.16, hspace=0.1,
        left=0.05, right=0.95, bottom=0.08, top=0.95)
    # plt.hot()
    vmin, vmax = -16, -13

    thresold = -15

    fp_structure = np.ones((10,1))
    td_structure = np.ones((1,4))

    for i, ig in enumerate(igs):
        ig.interpolate_frequencies()
        data = ig.data
        plt.subplot(g[i,0])
        plt.imshow(np.log10(data), interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
        if i == 0:
            plt.title("Data")

        plt.subplot(g[i,1])
        bdata = np.zeros(data.shape, dtype=bool)
        bdata[data > (10.**thresold)] = True
        plt.imshow(bdata, interpolation='nearest', aspect='auto')
        if i == 0:
            plt.title("Threshold @ %f" % thresold)

        plt.subplot(g[i,2])
        dfp = morphology.binary_opening(bdata, structure=fp_structure)
        plt.imshow(dfp, interpolation='nearest', aspect='auto')
        if i == 0:
            plt.title("FP-Lines: v-opened")

        # plt.subplot(g[i,3])
        # dtd = morphology.binary_opening(bdata, structure=td_structure)
        # plt.imshow(dtd, interpolation='nearest', aspect='auto')
        # if i == 0:
        #     plt.title("CYC LINES?")

        plt.subplot(g[i,3])
        dtd = np.logical_and(bdata, np.logical_not(dfp))
        plt.imshow(dtd, interpolation='nearest', aspect='auto')
        if i == 0:
            plt.title("Residual")

        plt.subplot(g[i,4])
        # dtd = np.logical_and(bdata, np.logical_not(dfp))
        # dtd = morphology.binary_dilation(dtd, structure=morphology.generate_binary_structure(2,1))
        # dtd = morphology.binary_fill_holes(dtd, structure=np.ones((1,3)))
        dtd = morphology.binary_opening(dtd, structure=np.ones((1,4)))

        structure = np.ones((2,1))
        structure[1,0] = 0

        dtd = morphology.binary_hit_or_miss(dtd, structure1=structure)


        plt.imshow(dtd, interpolation='nearest', aspect='auto')
        if i == 0:
            plt.title("Cyc. lines: residual h-opened")

        # plt.subplot(g[i,5])
        # dtd = np.logical_and(bdata, np.logical_not(np.logical_or(dtd, dfp)))
        # plt.imshow(dtd, interpolation='nearest', aspect='auto')
        # if i == 0:
        #     plt.title("RESIDUAL")

        plt.subplot(g[i,5])

        dtd = morphology.binary_closing(dtd)
        plt.imshow(dtd, interpolation='nearest', aspect='auto')
        if i == 0:
            plt.title("Ionosphere - topside edge")

        # plt.subplot(g[i,4])
        # d = np.logical_and(bdata, np.logical_not(dfp))
        # d = morphology.binary_erosion(d, structure=td_structure)
        # d = morphology.binary_opening(d, structure=td_structure)
        # plt.imshow(d, interpolation='nearest', aspect='auto')
        # if i == 0:
        #     plt.title("IONOSPHERE?")

        # plt.subplot(g[i,5])
        # # d = np.logical_and(bdata, np.logical_not(np.logical_or(dfp, d)))
        # # d = morphology.binary_closing(bdata,structure=morphology.generate_binary_structure(2,2), iterations=3)
        # d = morphology.binary_dilation(bdata, structure=np.ones((2,1)))
        # plt.imshow(d, interpolation='nearest', aspect='auto')
        # if i == 0:
        #     plt.title("RESIDUAL?")

    plt.show()


ionogram_list = []

def auto_test():

    plt.close("all")
    plt.figure(figsize=(20, 12))
    plt.rcParams['font.size'] = 6

    coverage = ais.get_ais_coverage()


    n_ig = 8
    global ionogram_list
    if len(ionogram_list) != n_ig:
        ionogram_list = []
        count = 0
        while count < n_ig:
            orbit = np.random.randint(6000) + 2000
            if coverage[orbit] < 151:
                continue

            file_name = ais.find_file(orbit)
            nsweeps = 0
            try:
                stats = os.stat(file_name)
                nsweeps = stats[stat.ST_SIZE] / 400
            except OSError as e:
                # print e
                pass

            if nsweeps > (180 * 150):
                tmp = ais.read_ais(orbit)
                if len(tmp) > 150:
                    ignumber = np.random.randint(len(tmp))
                    ionogram_list.append(tmp[ignumber])
                    count = count + 1
                    print("ORBIT %d, Ionogram %d" % (orbit, ignumber))
                    continue

            # print "REJECTING"

    rows = 2
    g = mpl.gridspec.GridSpec(3*rows, n_ig/rows, wspace=0.2, hspace=0.2,
        left=0.05, right=0.95, bottom=0.04, top=0.95)
    # plt.hot()
    vmin, vmax = -16, -13

    thresold = -15

    fp_structure = np.ones((10,1))
    td_structure = np.ones((1,4))

    for i, ig in enumerate(ionogram_list):
        print('===========')
        print(i)
        ig.interpolate_frequencies()

        d = [ais.IonogramDigitization()]
        ig.digitizations = d

        plt.subplot(g[i])
        ig.plot(colorbar=False, color='red', errors=False)

        ax = plt.subplot(g[i + 2*n_ig])
        # print ig.calculate_fp_local()
        print(ig.calculate_td_cyclotron( ax=ax))
        # print ig.calculate_ground_trace()
        # print ig.calculate_reflection()

        # ig.digitization.delete_fp_local()
        # ig.digitization.delete_ground()
        # ig.digitization.delete_trace()



        plt.subplot(g[i +n_ig])
        ig.plot(colorbar=False, color='red', errors=False, show_thresholded_data=True)
        plt.annotate(str(i), (0.9, 0.9), xycoords='axes fraction', color='white', fontsize=18)
    plt.show()






if __name__ == '__main__':
    auto_test()
