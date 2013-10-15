#! /usr/bin/env python
"""

Carry out PSF subtraction on LMIRcam GJ504 data using principal
component analysis/K-L.

"""

import numpy as np
from klip import *
import time as time
import pyfits
import sys
import os
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.colors

N_proc = 24
#diagnos_stride = 100
diagnos_stride = 24

dataset_label = 'gj504_longL_octcanon'
#dataset_label = 'gj504_longL_octcanonTR'
#dataset_label = 'gj504_longL_octcanonBL'
data_dir = os.path.expanduser('/disk1/zimmerman/GJ504/apr21_longL/reduc')
result_dir = os.path.expanduser('/disk1/zimmerman/GJ504/apr21_longL/klipsub_results')
#
# global PCA search zone config
#
R_inner = 10.
R_out = [20, 40, 80, 120, 170, 220, 260, 300]
#R_out = [240, 280]#, 80, 140]# 200, 240, 280]
mode_cut = [500]*8
#DPhi = [360., 360., 360., 360.]#, 180., 90., 90.]
DPhi = [360.]*8
Phi_0 = [0.]*8
fwhm = 4.
min_refgap_fac = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1., 1.]
#min_refgap_fac = [0.5]*7

test_mode = False
store_results = True
store_psf = False
store_archv = False
use_svd = True
coadd_full_overlap_only = True

klip_config, klip_data = global_klipsub(dataset_label, data_dir, result_dir, R_inner, R_out, mode_cut, DPhi, Phi_0,
                                        fwhm, min_refgap_fac, op_fr=None, N_proc=N_proc, diagnos_stride=diagnos_stride,
                                        test_mode=test_mode, use_svd=use_svd, coadd_full_overlap_only=coadd_full_overlap_only,
                                        store_results=store_results, store_psf=store_psf, store_archv=store_archv)

if store_archv:
    klipsub_archv_fname = "%s/%s_klipsub_archv.pkl" % (result_dir, result_label)
    if os.path.exists(klipsub_archv_fname):
        os.remove(klipsub_archv_fname)
    if store_archv:
       klipsub_archv = open(klipsub_archv_fname, 'wb') 
       pickle.dump((klip_config, klip_data), klipsub_archv, protocol=2)
       klipsub_archv.close()
       print "Wrote KLIP reduction (%.3f Mb) archive to %s" % (os.stat(klipsub_archv_fname).st_size/10.**6, klipsub_archv_fname)

