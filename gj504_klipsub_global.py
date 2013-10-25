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

#N_proc = 12 
diagnos_stride = 100
N_proc = 2

dataset_label = 'gj504_longL_octcanon'
#dataset_label = 'gj504_longL_octcanonTR'
#dataset_label = 'gj504_longL_octcanonBL'
data_dir = os.path.expanduser('/disk1/zimmerman/GJ504/apr21_longL/reduc')
result_dir = os.path.expanduser('/disk1/zimmerman/GJ504/apr21_longL/klipsub_results')
#
# global PCA search zone config
#
fwhm = 4.
R_inner = 8
#R_out = [20, 40, 80, 120, 170, 220, 260, 300]
R_out = [15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
mode_cut = [600]*len(R_out)
DPhi = [360.]*len(R_out)
Phi_0 = [0.]*len(R_out)
min_refgap_fac = [0.5]*len(R_out)
#min_refgap_fac = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1., 1.]

#R_inner = 220
#R_out = [260, 300]
#mode_cut = [500]*2
#DPhi = [360.]*2
#Phi_0 = [0.]*2
#fwhm = 4.
#min_refgap_fac = [0.5]*2

test_mode = True
store_results = False
store_psf = False
store_archv = False
use_svd = True
coadd_full_overlap_only = True

klip_config, klip_data, annular_rms, zonal_rms = klipsub(dataset_label, data_dir, result_dir, R_inner, R_out, mode_cut, DPhi, Phi_0,
                                                         fwhm, min_refgap_fac, op_fr=None, N_proc=N_proc, diagnos_stride=diagnos_stride,
                                                         fake_planet=None, synth_psf_img=None, test_mode=test_mode, use_svd=use_svd,
                                                         coadd_full_overlap_only=coadd_full_overlap_only, store_results=store_results,
                                                         store_psf=store_psf, store_archv=store_archv)

if store_archv:
    klipsub_archv_fname = "%s/%s_klipsub_archv.pkl" % (result_dir, result_label)
    if os.path.exists(klipsub_archv_fname):
        os.remove(klipsub_archv_fname)
    if store_archv:
       klipsub_archv = open(klipsub_archv_fname, 'wb') 
       pickle.dump((klip_config, klip_data), klipsub_archv, protocol=2)
       klipsub_archv.close()
       print "Wrote KLIP reduction (%.3f Mb) archive to %s" % (os.stat(klipsub_archv_fname).st_size/10.**6, klipsub_archv_fname)

