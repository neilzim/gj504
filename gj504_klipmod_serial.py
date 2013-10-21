#! /usr/bin/env python
"""

Implement KLIP point source forward model on LMIRcam kappa And ADI data.

"""
import numpy as np
import time as time
import pyfits
from scipy.ndimage.interpolation import *
from scipy.interpolate import *
from scipy.optimize import *
import multiprocessing
import sys
import os
import pdb
import cPickle as pickle
from klip import *

data_dir = '/disk1/zimmerman/GJ504/apr21_longL'
klipsub_result_dir = '%s/klipsub_results' % data_dir
klipmod_result_dir = '%s/klipmod_results' % data_dir
template_img_fname = '%s/gj504_longL_octcanon_srcklip_rad255_dphi90_mode000_res_coadd.fits' % klipsub_result_dir
#template_img_fname = '%s/gj504_longL_sepcanon_srcklip_rad260_dphi90_mode500_res_coadd.fits' % klipsub_result_dir
synthpsf_fname = '%s/reduc/gj504_longL_psf_model.fits' % data_dir
mode_cut = 500
N_proc = 20
synthpsf_rolloff = 20.

# October canonical reduction; bottom left nod position only
#print ''
#print 'Modeling PSF in KLIP residual from October reduction; bottom left nod position only'
#result_label = 'gj504_longL_octcanonBL_modecut%03d' % mode_cut
#klipsub_archv_fname =   "%s/gj504_longL_octcanonBL_srcklip_rad260_dphi50_mode010_klipsub_archv.pkl" % klipsub_result_dir
#psol_octcanonBL = klipmod(ampguess=0.7, posguess_rho=238., posguess_theta=-33.5, klipsub_archv_fname=klipsub_archv_fname,
#                          klipsub_result_dir=klipsub_result_dir, klipmod_result_dir=klipmod_result_dir,
#                          template_img_fname=template_img_fname, synthpsf_fname=synthpsf_fname,
#                          synthpsf_rolloff=synthpsf_rolloff, result_label=result_label, N_proc=N_proc,
#                          mode_cut=mode_cut, do_MLE=True)

# October canonical reduction; top right nod position only
#print ''
#print 'Modeling PSF in KLIP residual from October reduction; top right nod position only'
#result_label = 'gj504_longL_octcanonTR_modecut%03d' % mode_cut
#klipsub_archv_fname =   "%s/gj504_longL_octcanonTR_srcklip_rad260_dphi50_mode010_klipsub_archv.pkl" % klipsub_result_dir
#psol_octcanonTR = klipmod(ampguess=0.7, posguess_rho=235., posguess_theta=-33.5, klipsub_archv_fname=klipsub_archv_fname,
#                          klipsub_result_dir=klipsub_result_dir, klipmod_result_dir=klipmod_result_dir,
#                          template_img_fname=template_img_fname, synthpsf_fname=synthpsf_fname,
#                          synthpsf_rolloff=synthpsf_rolloff, result_label=result_label, N_proc=N_proc,
#                          mode_cut=mode_cut, do_MLE=True)

# October canonical reduction; full data set
print ''
print 'Modeling PSF in KLIP residual from October reduction; full data set'
result_label = 'gj504_longL_octcanon_modecut%03d_debug' % mode_cut
#klipsub_archv_fname = "%s/gj504_longL_octcanon_srcklip_rad260_dphi50_mode010_klipsub_archv.pkl" % klipsub_result_dir
#klipsub_archv_fname = "%s/gj504_longL_octcanon_srcklip_rad255_dphi50_mode000_klipsub_archv.pkl" % klipsub_result_dir
klipsub_archv_fname = "%s/gj504_longL_sepcanon_srcklip_rad260_dphi50_mode500_klipsub_archv.pkl" % klipsub_result_dir
psol_octcanon = klipmod(ampguess=0.7, posguess_rho=238., posguess_theta=-33., klipsub_archv_fname=klipsub_archv_fname,
                        klipsub_result_dir=klipsub_result_dir, klipmod_result_dir=klipmod_result_dir,
                        template_img_fname=template_img_fname, synthpsf_fname=synthpsf_fname,
                        synthpsf_rolloff=synthpsf_rolloff, result_label=result_label, N_proc=N_proc,
                        mode_cut=mode_cut, do_MLE=False)
