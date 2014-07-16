#! /usr/bin/env python
"""

Compute contrast curve after PSF subtraction on LMIRcam GJ504 data using principal
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

N_proc = 20
#diagnos_stride = 100
diagnos_stride = -1 

#dataset_label = 'gj504_longL_octcanon_2x2bin'
dataset_label = 'gj504_longL_octcanon'
data_dir = os.path.expanduser('/disk1/zimmerman/GJ504/apr21_longL/reduc')
result_dir = os.path.expanduser('/disk1/zimmerman/GJ504/apr21_longL/klipsub_results')
log_fname = result_dir + '/gj504_contrast_curve_jun3_log.txt'
log_fobj = open(log_fname, 'w')

synthpsf_fname = '%s/gj504_longL_psf_model.fits' % data_dir
synthpsf_img = crop_and_rolloff_psf(synthpsf_fname, rolloff_rad=20.)
#synthpsf_fname = '%s/gj504_longL_2x2bin_psf_model.fits' % data_dir
#synthpsf_img = crop_and_rolloff_psf(synthpsf_fname, rolloff_rad=10.)

#test_res_img_fname = result_dir + '/gj504_longL_octcanon_globalklip_rad20-40-80-120-170-220-260-300_mode500-500-500-500-500-500-500-500_res_med.fits'
#test_res_img_fname = result_dir + '/gj504_longL_octcanon_globalklip_rad20-40-80-120-170-220-260-300_mode500-500-500-500-500-500-500-500_res_coadd.fits'
#test_res_img_fname = result_dir + '/gj504_longL_octcanon_2x2bin_globalklip_rad005-150_mode600-600_res_med.fits'
#test_res_img_fname = result_dir + '/gj504_longL_octcanon_2x2bin_globalklip_rad005-150_mode600-600_res_med.fits'
test_res_img_fname = result_dir + '/gj504_longL_octcanon_globalklip_rad008-300_mode600-600_res_coadd.fits'

pix_scale = 0.01076 #arc sec
fwhm = 4.
star_peak = 6297.63 * (0.523973 / 0.0582192)
#pix_scale = 2*0.01076 #arc sec
#fwhm = 2.
#star_peak = 6290.74 * (0.523973 / 0.0582192)

PA_arr = np.arange(0, 360, 30)
#PA_arr = np.arange(0, 360, 180)
#PA_arr = np.array([0, 90, 120])
N_PA = len(PA_arr)
#K_arr = np.array([0, 10, 20, 50, 100, 200, 400])
#K_arr = np.array([0, 10, 20, 50])
K_arr = np.array([20])
#R_arr = np.array([0.2, 0.25, 0.375, 0.5, 0.75, 1.0, 2.0, 3.0]) / pix_scale
R_arr = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]) / pix_scale
#R_arr = np.array([2.0]) / pix_scale
#R_arr = np.array([0.2]) / pix_scale
#refgap_fac = 1.0
refgap_fac = 1.0
N_rad = len(R_arr)
N_K = len(K_arr)

detlim_grid = np.zeros((N_rad, N_PA, N_K))
#
#Estimate good fake planet fluxes based on the rms in each zone
#
test_res_img_hdulist = pyfits.open(test_res_img_fname, 'readonly')
test_res_img = test_res_img_hdulist[0].data
test_res_img_hdulist.close()
rad_map = np.sqrt( get_radius_sqrd(test_res_img.shape) )
test_rms_arr = np.zeros(R_arr.shape)
for i, r_p in enumerate(R_arr):
    annulus_ind = np.nonzero( np.logical_and(rad_map > r_p - fwhm - 1, rad_map <= r_p + fwhm + 1) )
    test_rms_arr[i] = np.sqrt( nanmean( test_res_img[annulus_ind]**2 ) )
#flux_arr = test_rms_arr*15
#flux_arr = test_rms_arr*10000
flux_arr = np.array([150, 100, 80, 50, 35, 20, 8, 5, 3, 3, 2, 2])
#flux_arr = np.array([3])
#plt.plot(R_arr*pix_scale, test_rms_arr, 'x')
#plt.plot(R_arr/fwhm, test_rms_arr, 'x')
#plt.show()

test_mode = False
store_results = False
store_psf = False
store_archv = False
use_svd = True
coadd_full_overlap_only = True

for i, (R_fp, flux_fp) in enumerate(zip(R_arr, flux_arr)):
    for j, PA_fp in enumerate(PA_arr):
        for k, K in enumerate(K_arr):
            log_fobj.write("*********************************************************************************************************************************\n")
            log_fobj.write("Beginning fake planet detection limit test for R = %0.2f pixels = %0.3f\", pos ang = %0.2f deg, flux = %0.2f, K = %d\n" %  (R_fp, R_fp*pix_scale, PA_fp, flux_fp, K))
            log_fobj.write("**********************************************************************************************************************\n")
            R_inner = int(round(R_fp)) - 1.5*fwhm - 1
            R_out = [int(round(R_fp)) + 1.5*fwhm + 1]

            if j == 0:
                klip_config, klip_data, coadd_img, med_img, coadd_detlim, med_detlim = klip_subtract(dataset_label, data_dir, result_dir, R_inner, R_out, mode_cut=[K], DPhi=[360.], Phi_0=[0.],
                                                                                                     fwhm=fwhm, min_refgap_fac=[refgap_fac], op_fr=None, N_proc=N_proc, diagnos_stride=diagnos_stride,
                                                                                                     fake_planet=(R_fp, PA_fp, flux_fp), synth_psf_img=synthpsf_img, coadd_img=None, med_img=None,
                                                                                                     test_mode=test_mode, use_svd=True, coadd_full_overlap_only=True, store_results=store_results,
                                                                                                     store_psf=False, store_archv=False, log_fobj=log_fobj)
            else:
                klip_config, klip_data, coadd_img, med_img, coadd_detlim, med_detlim = klip_subtract(dataset_label, data_dir, result_dir, R_inner, R_out, mode_cut=[K], DPhi=[360.], Phi_0=[0.],
                                                                                                     fwhm=fwhm, min_refgap_fac=[refgap_fac], op_fr=None, N_proc=N_proc, diagnos_stride=diagnos_stride,
                                                                                                     fake_planet=(R_fp, PA_fp, flux_fp), synth_psf_img=synthpsf_img, coadd_img=coadd_img, med_img=med_img,
                                                                                                     test_mode=test_mode, use_svd=True, coadd_full_overlap_only=True, store_results=store_results,
                                                                                                     store_psf=False, store_archv=False, log_fobj=log_fobj)
            detlim_grid[i, j, k] = med_detlim

detlim_grid_bestK = np.min(detlim_grid, axis=2)

bestK_grid = np.zeros(detlim_grid_bestK.shape)
for i, R_fp in enumerate(R_arr):
    for j, PA_fp in enumerate(PA_arr):
        bestK_grid[i, j] = K_arr[ np.where(detlim_grid[i, j, :] == np.min(detlim_grid[i, j, :]))[0][0] ]

detlim_vsrad = np.median(detlim_grid_bestK, axis=1)
detlim_vsrad_deltamag = 2.5*np.log10(star_peak/detlim_vsrad)
log_fobj.write('detlim_grid:\n')
log_fobj.write( str(detlim_grid) )
log_fobj.write('\nbestK_grid:\n')
log_fobj.write( str(bestK_grid) )
log_fobj.write('\ndetlim vs rad (linear):\n')
log_fobj.write( str(detlim_vsrad) )
log_fobj.write('\ndetlim vs rad (delta mag):\n')
log_fobj.write( str(detlim_vsrad_deltamag) )
log_fobj.write('\n')

log_fobj.close()
