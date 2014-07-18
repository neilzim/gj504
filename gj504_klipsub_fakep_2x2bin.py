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

N_proc = 22
#diagnos_stride = 100
#N_proc = 6
diagnos_stride = 100

dataset_label = 'gj504_longL_octcanon_2x2bin'
#dataset_label = 'gj504_longL_octcanonTR'
#dataset_label = 'gj504_longL_octcanonBL'
data_dir = os.path.expanduser('/disk1/zimmerman/GJ504/apr21_longL/fakep')
result_dir = os.path.expanduser('/disk1/zimmerman/GJ504/apr21_longL/klipsub_results_jun')
#
# global PCA search zone config
#

fwhm = 2.
R_inner = 100
R_out = [130.]
mode_cut = [600]
DPhi = [360.]
Phi_0 = [0.]
min_refgap_fac = [0.5]

synth_psf_fname = '%s/gj504_longL_2x2bin_psf_model.fits' % data_dir
psf_hdulist = pyfits.open(synth_psf_fname)
psf_img = psf_hdulist[0].data
psf_hdulist.close()

test_mode = False
store_results = False
store_psf = False
store_archv = False
use_svd = True
coadd_full_overlap_only = True

N_trials = 6
N_fakep = 14
R_fakep = [117.86]*N_fakep
flux_fakep = [0.50]*N_fakep
pa_src = 326.6
pa_margin = 15.

for t in range(2,N_trials):
    result_label = 'gj504_longL_fakep_trial%02d_K%03d_refgap%02d' % (t+1, mode_cut[0], round(10*min_refgap_fac[0]*fwhm))
    PA_fakep = list() 
    while len(PA_fakep) < N_fakep:
        phi_try = np.random.random()*360.
        phi_ok = True
    
        phi_try_plus = phi_try + pa_margin
        phi_try_minus = phi_try - pa_margin
        if phi_try_minus < pa_src < phi_try_plus:
            continue
    
        for pa in PA_fakep:
            if phi_try_plus > 360:
                if (pa > phi_try_minus) or (pa < (phi_try_plus - 360.)):
                    phi_ok = False
                    break
            elif phi_try_minus < 0:
                if (pa > (phi_try_minus + 360)) or (pa < phi_try_plus):
                    phi_ok = False
                    break
            else:
                if phi_try_minus < pa < phi_try_plus:
                    phi_ok = False
                    break
        if phi_ok: 
            PA_fakep.append(phi_try)
    PA_fakep.sort()
    fakep_table = zip(R_fakep, PA_fakep, flux_fakep)
    print "trial %02d fake planet position angles:" % (t+1)
    print PA_fakep
    print
    
    klip_config, klip_data,\
    fakep_coadd_img, fakep_med_img = klip_subtract(dataset_label, data_dir, result_dir, R_inner, R_out, mode_cut, DPhi, Phi_0,
                                                   fwhm, min_refgap_fac, op_fr=None, N_proc=N_proc, diagnos_stride=diagnos_stride,
                                                   fake_planets=fakep_table, synth_psf_img=psf_img, test_mode=test_mode, use_svd=use_svd,
                                                   coadd_full_overlap_only=coadd_full_overlap_only, store_results=store_results,
                                                   store_psf=store_psf, result_label=result_label, store_archv=store_archv)
    if test_mode == False:
        fakep_coadd_img_fname = "%s/%s_fp_res_coadd.fits" % (result_dir, result_label)
        fakep_coadd_img_hdu = pyfits.PrimaryHDU(fakep_coadd_img.astype(np.float32))
        fakep_coadd_img_hdu.writeto(fakep_coadd_img_fname, clobber=True)
        print "Wrote average of derotated, KLIP-subtracted fake planet images (%.3f Mb) to %s" % (fakep_coadd_img.nbytes/10.**6, fakep_coadd_img_fname)
