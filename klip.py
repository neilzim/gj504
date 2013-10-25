#! /usr/bin/env python
"""

Library of routines for PSF subtraction by principal component analysis

"""
import numpy as np
import time as time
import pyfits
import warnings
from scipy.ndimage.interpolation import *
from scipy.interpolate import *
from scipy.optimize import *
from scipy.stats import nanmean, nanmedian
from scipy.io.idl import readsav
from shutil import copyfile
import multiprocessing
import sys
import os
import pdb
import cPickle as pickle
import matplotlib.pyplot as plt

class Worker(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                self.task_queue.task_done()
                break
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return

class eval_adiklip_srcmodel_task(object):
    def __init__(self, fr_ind, fr_shape, zonemask_1d, Z, F, srcmodel, src_amp, src_abframe_xy):
        self.fr_ind = fr_ind
        self.fr_shape = fr_shape
        self.zonemask_1d = zonemask_1d
        self.Z = Z
        self.F = F
        self.srcmodel = srcmodel
        self.src_amp = src_amp
        self.src_abframe_xy = src_abframe_xy
    def __call__(self):
        fr_shape = self.fr_shape
        srcmodel_cent_xy = ((self.srcmodel.shape[1] - 1)/2., (self.srcmodel.shape[0] - 1)/2.)
        if self.src_abframe_xy[0] >= fr_shape[1] or self.src_abframe_xy[1] >= fr_shape[0] or min(self.src_abframe_xy) < 0:
            print 'bad dest:', self.src_abframe_xy
            return np.finfo(np.float).max 
        abframe_synthsrc_img = superpose_srcmodel(data_img = np.zeros(fr_shape), srcmodel_img = self.src_amp*self.srcmodel,
                                                  srcmodel_destxy = self.src_abframe_xy, srcmodel_centxy = srcmodel_cent_xy)
        Pmod = np.ravel(abframe_synthsrc_img)[ self.zonemask_1d ].copy()
        if self.Z != None:
            res_vec = self.F - Pmod + Pmod.dot(self.Z.T).dot(self.Z)
        else:
            res_vec = self.F - Pmod
        return (self.fr_ind, res_vec, np.sum(res_vec**2))
    def __str__(self):
        return 'frame %d' % (self.fr_ind+1)

class klipsub_task(object):
    #def __init__(self, fr_ind, data_cube, config_dict, result_dict, result_dir, diagnos_stride,
    #             store_psf=False, store_klbasis=False, use_svd=True):
    def __init__(self, fr_ind, data_cube, fr_shape, parang_seq, mode_cut, op_fr, op_rad,
                 op_az, ref_table, zonemask_1d, zonemask_2d, result_dict, result_dir,
                 diagnos_stride, store_psf=False, store_klbasis=False, use_svd=True):
         self.fr_ind = fr_ind
         self.data_cube = data_cube
         #self.config_dict = config_dict

         self.fr_shape = fr_shape
         self.parang_seq = parang_seq
         self.mode_cut = mode_cut
         self.op_fr = op_fr
         self.op_rad = op_rad
         self.op_az = op_az
         self.ref_table = ref_table
         self.zonemask_1d = zonemask_1d         
         self.zonemask_2d = zonemask_2d

         self.result_dict = result_dict
         self.result_dir = result_dir
         self.diagnos_stride = diagnos_stride
         self.store_psf = store_psf
         self.store_klbasis = store_klbasis
         self.use_svd = use_svd
    def __call__(self):
        #fr_shape = self.config_dict['fr_shape']
        #parang_seq = self.config_dict['parang_seq']
        #mode_cut = self.config_dict['mode_cut']
        #op_fr = self.config_dict['op_fr']
        #op_rad = self.config_dict['op_rad']
        #op_az = self.config_dict['op_az']
        #ref_table = self.config_dict['ref_table']
        #zonemask_table_1d = self.config_dict['zonemask_table_1d']
        #zonemask_table_2d = self.config_dict['zonemask_table_2d']
        fr_shape = self.fr_shape
        parang_seq = self.parang_seq
        mode_cut = self.mode_cut
        op_fr = self.op_fr
        op_rad = self.op_rad
        op_az = self.op_az
        ref_table = self.ref_table
        zonemask_1d = self.zonemask_1d
        zonemask_2d = self.zonemask_2d
        N_fr = len(parang_seq)
        N_op_fr = len(op_fr)

        fr_ind = self.fr_ind
        data_cube = self.data_cube
        result_dict = self.result_dict
        result_dir = self.result_dir
        diagnos_stride = self.diagnos_stride
        store_psf = self.store_psf
        store_klbasis = self.store_klbasis
        use_svd = self.use_svd

        klipsub_img = np.zeros(fr_shape)
        derot_klipsub_img = klipsub_img.copy()
        submask_img = klipsub_img.copy()
        if store_psf:
            klippsf_img = np.tile(np.nan, fr_shape)
        else:
            klippsf_img = None

        if diagnos_stride > 0 and fr_ind%diagnos_stride == 0:
            if max(mode_cut) > 0:
                klbasis_cube = np.zeros((max(mode_cut), fr_shape[0], fr_shape[1]))
            else:
                klbasis_cube = None
        for rad_ind in op_rad:
            for az_ind in op_az[rad_ind]:
                I = np.ravel(data_cube[fr_ind,:,:])[ zonemask_1d[rad_ind][az_ind] ].copy() 
                R = np.zeros((ref_table[rad_ind].shape[0], zonemask_1d[rad_ind][az_ind].shape[0]))
                #I = np.ravel(data_cube[fr_ind,:,:])[ zonemask_table_1d[fr_ind][rad_ind][az_ind] ].copy() 
                #R = np.zeros((ref_table[fr_ind][rad_ind].shape[0], zonemask_table_1d[fr_ind][rad_ind][az_ind].shape[0]))
                #for j, ref_fr_ind in enumerate(ref_table[fr_ind][rad_ind]):
                for j, ref_fr_ind in enumerate(ref_table[rad_ind]):
                    #R[j,:] = np.ravel(data_cube[ref_fr_ind,:,:])[ zonemask_table_1d[fr_ind][rad_ind][az_ind] ].copy()
                    R[j,:] = np.ravel(data_cube[ref_fr_ind,:,:])[ zonemask_1d[rad_ind][az_ind] ].copy()
                if mode_cut[rad_ind] > 0: # do PCA on reference PSF stack
                    if use_svd == False: # following Soummer et al. 2012
                        I_mean = R.mean(axis = 0)
                        I -= I_mean
                        R -= I_mean
                        Z, sv, N_modes = get_klip_basis(R = R, cutoff = mode_cut[rad_ind])
                    else:
                        I_mean = R.mean(axis = 0)
                        I -= I_mean
                        R -= I_mean
                        Z, sv, N_modes = get_pca_basis(R = R, cutoff = mode_cut[rad_ind])
                    F = I - I.dot(Z.T).dot(Z)
                    if store_psf:
                        #klippsf_zone_img = reconst_zone(I_mean + I - F, zonemask_table_2d[fr_ind][rad_ind][az_ind], fr_shape)
                        klippsf_zone_img = reconst_zone(I_mean + I - F, zonemask_2d[rad_ind][az_ind], fr_shape)
                else: # classical ADI: subtract mean refernce PSF
                    R_mean = R.mean(axis = 0)
                    F = I - R_mean
                    if store_psf:
                        #klippsf_zone_img = reconst_zone(R_mean, zonemask_table_2d[fr_ind][rad_ind][az_ind], fr_shape)
                        klippsf_zone_img = reconst_zone(R_mean, zonemask_2d[rad_ind][az_ind], fr_shape)
                if store_psf:
                    #klippsf_img[ zonemask_table_2d[fr_ind][rad_ind][az_ind] ] = klippsf_zone_img[ zonemask_table_2d[fr_ind][rad_ind][az_ind] ]
                    klippsf_img[ zonemask_2d[rad_ind][az_ind] ] = klippsf_zone_img[ zonemask_2d[rad_ind][az_ind] ]
                klipsub_zone_img = reconst_zone(F, zonemask_2d[rad_ind][az_ind], fr_shape)
                #klipsub_img[ zonemask_table_2d[fr_ind][rad_ind][az_ind] ] = klipsub_zone_img[ zonemask_table_2d[fr_ind][rad_ind][az_ind] ]
                #submask_img[ zonemask_table_2d[fr_ind][rad_ind][az_ind] ] = 1.
                klipsub_img[ zonemask_2d[rad_ind][az_ind] ] = klipsub_zone_img[ zonemask_2d[rad_ind][az_ind] ]
                submask_img[ zonemask_2d[rad_ind][az_ind] ] = 1.

                if result_dict != None:
                    result_dict[fr_ind][rad_ind][az_ind]['F'] = F.astype(np.float32)
                    if mode_cut[rad_ind] > 0:
                        result_dict[fr_ind][rad_ind][az_ind]['Z'] = Z.astype(np.float32)
                    #result_dict[fr_ind][rad_ind][az_ind]['I'] = I
                    #result_dict[fr_ind][rad_ind][az_ind]['I_mean'] = I_mean
                    #result_dict[fr_ind][rad_ind][az_ind]['sv'] = sv
                    #result_dict[fr_ind][rad_ind][az_ind]['I_proj'] = I_proj
                if diagnos_stride > 0 and fr_ind % diagnos_stride == 0 and mode_cut[rad_ind] > 0:
                    #klbasis_cube[:N_modes,:,:] += reconst_zone_cube(Z, zonemask_table_2d[fr_ind][rad_ind][az_ind],
                    #                                                cube_dim = (N_modes, fr_shape[0], fr_shape[1]))
                    klbasis_cube[:N_modes,:,:] += reconst_zone_cube(Z, zonemask_2d[rad_ind][az_ind],
                                                                    cube_dim = (N_modes, fr_shape[0], fr_shape[1]))
                    print "Frame %d, annulus %d/%d, sector %d/%d: RMS before/after sub: %0.2f / %0.2f" %\
                          (fr_ind+1, rad_ind+1, len(op_rad), az_ind+1, len(op_az[rad_ind]),\
                           np.sqrt(np.mean((I + I_mean)**2)), np.sqrt(np.mean(F**2)))
        # De-rotate the KLIP-subtracted image
        derot_klipsub_img = rotate(klipsub_img, -parang_seq[fr_ind], reshape=False)
        derot_submask_img = rotate(submask_img, -parang_seq[fr_ind], reshape=False)
        #derot_submask_hdu = pyfits.PrimaryHDU(derot_submask_img.astype(np.float32))
        #derot_submask_hdu.writeto("%s/submask_fr%03d.fits" % (result_dir, fr_ind), clobber=True)
        exc_ind = np.where(derot_submask_img < 0.9)
        derot_klipsub_img[exc_ind] = np.nan

        #derot_klipsub_img = rotate(klipsub_img, -parang_seq[fr_ind], reshape=False)
        if diagnos_stride > 0 and fr_ind % diagnos_stride == 0:
            print "***** Frame %d has been PSF-sub'd and derotated. *****" % (fr_ind+1)
            if store_klbasis == True and klbasis_cube:
                klbasis_cube_hdu = pyfits.PrimaryHDU(klbasis_cube.astype(np.float32))
                klbasis_cube_hdu.writeto("%s/klbasis_fr%03d.fits" % (result_dir, fr_ind), clobber=True)
        return (fr_ind, klipsub_img, derot_klipsub_img, klippsf_img, result_dict)
    def __str__(self):
        return 'frame %d' % (self.fr_ind+1)

def crop_and_rolloff_psf(psf_fname, rolloff_rad = None, cropmarg = 1):
    psf_hdulist = pyfits.open(psf_fname)
    psf_img = psf_hdulist[0].data
    psf_hdulist.close()
    psf_cent_xy = ((psf_img.shape[0] - 1.)/2., (psf_img.shape[1] - 1.)/2.)
    crop_psf_img = psf_img[cropmarg:-cropmarg, cropmarg:-cropmarg].copy()
    if rolloff_rad != None:
        crop_psf_cent_xy = ((crop_psf_img.shape[0] - 1.)/2., (crop_psf_img.shape[1] - 1.)/2.)
        Y, X = np.indices(crop_psf_img.shape)
        Rsqrd = (X - crop_psf_cent_xy[0])**2 + (Y - crop_psf_cent_xy[1])**2
        crop_psf_img *= np.exp( -(Rsqrd / rolloff_rad**2)**2 )
    return crop_psf_img

def add_fake_planets(data_cube, parang_seq, psf_img, R_p, PA_p, flux_p):
    fr_width = data_cube.shape[1]
    data_cent_xy = ((fr_width - 1)/2., (fr_width - 1)/2.)
    psf_width = psf_img.shape[0]
    psf_cent_xy = ((psf_width - 1)/2., (psf_width - 1)/2.)
    N_fr = len(parang_seq)
    psf_peak = np.max(psf_img)
    amp_p = flux_p/psf_peak
    fakep_data_cube = data_cube.copy()
    for R, PA, amp in zip(R_p, PA_p, amp_p):
        rot_PA_seq = np.deg2rad([PA - parang for parang in parang_seq])
        rot_xy_seq = [( data_cent_xy[0] + R*np.cos(np.pi/2 + rot_PA),
                        data_cent_xy[1] + R*np.sin(np.pi/2 + rot_PA) ) for rot_PA in rot_PA_seq]
        for i in range(N_fr):
            fakep_data_cube[i,:,:] = superpose_srcmodel(data_img=fakep_data_cube[i,:,:], srcmodel_img=amp*psf_img,
                                                        srcmodel_destxy = rot_xy_seq[i], srcmodel_centxy = psf_cent_xy)
    return fakep_data_cube

def get_residual_fake_planet_stats(config_dict, Phi_0, R_p, PA_p, flux_p, fwhm, planet_coadd_img, planet_med_img,
                                   true_coadd_img=None, true_med_img=None, xycent=None, log_fobj=sys.stdout):
    fr_shape = config_dict['fr_shape']
    parang_seq = config_dict['parang_seq']
    op_rad = config_dict['op_rad']
    op_az = config_dict['op_az']
    if xycent == None:
        xycent = ((fr_shape[0] - 1)/2., (fr_shape[1] - 1)/2.)
    R_inner = config_dict['R_inner']
    R_out = config_dict['R_out']
    rad_vec = np.sqrt(get_radius_sqrd(fr_shape, xycent)).ravel()

    Phi_0_derot = (Phi_0 + parang_seq[0]) % 360.
    coadd_annular_rms = list()
    med_annular_rms = list()
    log_fobj.write("RMS counts in KLIP results:\n")
    for rad_ind in op_rad:
        R2 = R_out[rad_ind]
        if rad_ind == 0:
            R1 = R_inner
        else:
            R1 = R_out[rad_ind-1]
        annular_mask_logic = np.vstack([np.less_equal(rad_vec, R2),\
                                        np.greater(rad_vec, R1),\
                                        np.isfinite(true_coadd_img.ravel())])
        annular_mask = np.nonzero( np.all(annular_mask_logic, axis=0) )[0]
        if true_coadd_img == None:
            coadd_annular_rms.append( np.sqrt( np.mean( np.ravel(planet_coadd_img.copy())[annular_mask]**2 ) ) )
            med_annular_rms.append( np.sqrt( np.mean( np.ravel(planet_med_img.copy())[annular_mask]**2 ) ) )
        else:
            coadd_annular_rms.append( np.sqrt( np.mean( np.ravel(true_coadd_img.copy())[annular_mask]**2 ) ) )
            med_annular_rms.append( np.sqrt( np.mean( np.ravel(true_med_img.copy())[annular_mask]**2 ) ) )
        log_fobj.write("\tannulus %d/%d: %.2f in KLIP sub'd, derotated, coadded annlus\n" % (rad_ind+1, len(op_rad), coadd_annular_rms[-1]))
        log_fobj.write("\tannulus %d/%d: %.2f in KLIP sub'd, derotated, median annlus\n" % (rad_ind+1, len(op_rad), med_annular_rms[-1]))
    for i, rad_ind in enumerate(op_rad):
        R2 = R_out[rad_ind]
        if rad_ind == 0:
            R1 = R_inner
        else:
            R1 = R_out[rad_ind-1]
        if R1 < R_p and R_p < R2:
            xy_p = (xycent[0] + R_p*np.cos(np.pi/2 + np.deg2rad(PA_p)), xycent[0] + R_p*np.sin(np.pi/2 + np.deg2rad(PA_p)))
            round_xy_p = (round(xy_p[0]), round(xy_p[1]))
            log_fobj.write("Measuring fake planet signal at location %d, %d in residual image\n" % (round_xy_p[0], round_xy_p[1]))
            coadd_planet_box = planet_coadd_img[round_xy_p[1] - fwhm:round_xy_p[1] + fwhm,
                                                round_xy_p[0] - fwhm:round_xy_p[0] + fwhm]
            coadd_planet_peak = np.nanmax(coadd_planet_box)
            coadd_planet_snr = coadd_planet_peak / coadd_annular_rms[i]
            coadd_planet_thrupt = coadd_planet_peak / flux_p
            coadd_min_detect = 5 * coadd_annular_rms[i] / coadd_planet_thrupt
            med_planet_box = planet_med_img[round_xy_p[1] - fwhm:round_xy_p[1] + fwhm,
                                            round_xy_p[0] - fwhm:round_xy_p[0] + fwhm]
            med_planet_peak = np.nanmax(med_planet_box)
            med_planet_snr = med_planet_peak / med_annular_rms[i]
            med_planet_thrupt = med_planet_peak / flux_p
            med_min_detect = 5 * med_annular_rms[i] / med_planet_thrupt
            
            log_fobj.write("Peak planet pixel value in final co-added planet image: %0.2f => SNR = %0.2f, throughput = %0.2f, min. detect. input flux = %0.2f\n" % \
                              (coadd_planet_peak, coadd_planet_snr, coadd_planet_thrupt, coadd_min_detect))
            log_fobj.write("Peak planet pixel value in median of de-rotated planet images: %0.2f => SNR = %0.2f, throughput = %0.2f, min. detect. input flux = %0.2f\n" % \
                              (med_planet_peak, med_planet_snr, med_planet_thrupt, med_min_detect))
    return coadd_annular_rms, coadd_min_detect, med_annular_rms, med_min_detect

def get_residual_stats(config_dict, Phi_0, coadd_img, med_img, xycent=None, log_fobj=sys.stdout):
    fr_shape = config_dict['fr_shape']
    parang_seq = config_dict['parang_seq']
    op_rad = config_dict['op_rad']
    op_az = config_dict['op_az']
    if xycent == None:
        xycent = ((fr_shape[0] - 1)/2., (fr_shape[0] - 1)/2.)
    rad_vec = np.sqrt(get_radius_sqrd(fr_shape, xycent)).ravel()
    R_inner = config_dict['R_inner']
    R_out = config_dict['R_out']
    N_az = config_dict['N_az']
    
    Phi_0_derot = (Phi_0 + parang_seq[0]) % 360.
    coadd_annular_rms = list()
    zonal_rms = [[None]*N_az[r] for r in range(len(R_out))]
    log_fobj.write("RMS counts in KLIP results:\n")
    for rad_ind in op_rad:
        R2 = R_out[rad_ind]
        if rad_ind == 0:
            R1 = R_inner
        else:
            R1 = R_out[rad_ind-1]
        annular_mask_logic = np.vstack([np.less_equal(rad_vec, R2),\
                                        np.greater(rad_vec, R1),\
                                        np.isfinite(coadd_img.ravel())])
        annular_mask = np.nonzero( np.all(annular_mask_logic, axis=0) )[0]
        coadd_annular_rms.append( np.sqrt( np.mean( np.ravel(coadd_img)[annular_mask]**2 ) ) )
        log_fobj.write("\tannulus %d/%d: %.3f in KLIP sub'd, derotated, coadded annlus\n" % (rad_ind+1, len(op_rad), coadd_annular_rms[-1]))
        if len(op_az[rad_ind]) > 1:
            Phi_beg = (Phi_0_derot - DPhi[rad_ind]/2.) % 360.
            Phi_end = [ (Phi_beg + i * DPhi[rad_ind]) % 360. for i in range(1, len(op_az[rad_ind])) ]
            Phi_end.append(Phi_beg)
            for az_ind in op_az[rad_ind]:
                Phi2 = Phi_end[az_ind]
                if az_ind == 0:
                    Phi1 = Phi_beg
                else:
                    Phi1 = Phi_end[az_ind-1]
                if Phi1 < Phi2:
                    mask_logic = np.vstack((np.less_equal(rad_vec, R2),\
                                            np.greater(rad_vec, R1),\
                                            np.less_equal(angle_vec, Phi2),\
                                            np.greater(angle_vec, Phi1)))
                else: # azimuthal region spans phi = 0
                    rad_mask_logic = np.vstack((np.less_equal(rad_vec, R2),\
                                                np.greater(rad_vec, R1)))
                    az_mask_logic = np.vstack((np.less_equal(angle_vec, Phi2),\
                                               np.greater(angle_vec, Phi1)))
                    mask_logic = np.vstack((np.any(az_mask_logic, axis=0),\
                                            np.all(rad_mask_logic, axis=0)))
                derot_zonemask = np.nonzero( np.all(mask_logic, axis = 0) )[0]
                zonal_rms[rad_ind][az_ind] = np.sqrt( np.mean( np.ravel(coadd_img)[derot_zonemask]**2 ) )
            delimiter = ', '
            log_fobj.write("\tby zone: %s\n" % delimiter.join(["%.3f" % zonal_rms[rad_ind][a] for a in op_az[rad_ind]]))
    log_fobj.write("Peak, min values in final co-added image: %0.3f, %0.3f\n" % (np.nanmax(coadd_img), np.nanmin(coadd_img)))
    log_fobj.write("Peak, min values in median of de-rotated images: %0.3f, %0.3f\n" % (np.nanmax(med_img), np.nanmin(med_img)))
    return coadd_annular_rms, zonal_rms

def superpose_srcmodel(data_img, srcmodel_img, srcmodel_destxy, srcmodel_centxy = None, rolloff_rad = None):
    assert(len(data_img.shape) == 2)
    assert(len(srcmodel_img.shape) == 2)
    assert( srcmodel_destxy[0] < data_img.shape[1] and srcmodel_destxy[1] < data_img.shape[0]\
            and min(srcmodel_destxy) >= 0 )
    if srcmodel_centxy == None:
        srcmodel_centxy = ((srcmodel_img.shape[0] - 1.)/2., (srcmodel_img.shape[1] - 1.)/2.)
    subpix_xyoffset = np.array( [(srcmodel_destxy[0] - srcmodel_centxy[0])%1.,\
                                 (srcmodel_destxy[1] - srcmodel_centxy[1])%1.] )
    if abs(round(srcmodel_centxy[0] + subpix_xyoffset[0]) - round(srcmodel_centxy[0])) == 1: # jump in pixel containing x center
        subpix_xyoffset[0] -= round(srcmodel_centxy[0] + subpix_xyoffset[0]) - round(srcmodel_centxy[0])
    if abs(round(srcmodel_centxy[1] + subpix_xyoffset[1]) - round(srcmodel_centxy[1])) == 1: # jump in pixel containing y center
        subpix_xyoffset[1] -= round(srcmodel_centxy[1] + subpix_xyoffset[1]) - round(srcmodel_centxy[1])
    #print "subpix_offset: ", subpix_xyoffset

    if rolloff_rad:
        Y, X = np.indices(srcmodel_img.shape)
        Rsqrd = (X - srcmodel_centxy[0])**2 + (Y - srcmodel_centxy[1])**2
        rolloff_arr = np.exp( -(Rsqrd / rolloff_rad**2)**2 )
        srcmodel_img *= rolloff_arr
    shifted_srcmodel_img = shift(input = srcmodel_img, shift = subpix_xyoffset[::-1], order=3)

    srcmodel_BLcorneryx = np.array( [round(srcmodel_destxy[1]) - round(srcmodel_centxy[1]),
                                     round(srcmodel_destxy[0]) - round(srcmodel_centxy[0])], dtype=np.int)
    srcmodel_TRcorneryx = srcmodel_BLcorneryx + np.array(srcmodel_img.shape)
    super_BLcorneryx = np.amax(np.vstack((srcmodel_BLcorneryx, np.zeros(2))), axis=0)
    super_TRcorneryx = np.amin(np.vstack((srcmodel_TRcorneryx, np.array(data_img.shape))), axis=0)
    BLcropyx = super_BLcorneryx - srcmodel_BLcorneryx
    TRcropyx = srcmodel_TRcorneryx - super_TRcorneryx
    super_img = data_img.copy()
    super_img[super_BLcorneryx[0]:super_TRcorneryx[0],\
              super_BLcorneryx[1]:super_TRcorneryx[1]] +=\
            shifted_srcmodel_img[BLcropyx[0]:srcmodel_img.shape[0]-TRcropyx[0],\
                                 BLcropyx[1]:srcmodel_img.shape[1]-TRcropyx[1]]
    return super_img 

def get_radius_sqrd(shape, c=None):
    if c is None:
        c = (0.5*float(shape[0] - 1),  0.5*float(shape[1] - 1))
    y, x = np.indices(shape)
    rsqrd = (x - c[0])**2 + (y - c[1])**2
    return rsqrd

def get_angle(s, c=None):
    if c is None:
        c = (0.5*float(s[0] - 1),  0.5*float(s[1] - 1))
    y, x = np.indices(s)
    theta = np.arctan2(y - c[1], x - c[0])
    # Change the angle range from [-pi, pi] to [0, 360]
    theta_360 = np.where(np.greater_equal(theta, 0), np.rad2deg(theta), np.rad2deg(theta + 2*np.pi))
    return theta_360

def reconst_zone_cube(data_mat, pix_table, cube_dim):
    reconstrd_cube = np.zeros(cube_dim)
    assert(data_mat.shape[0] >= cube_dim[0])
    for fr_ind in range(cube_dim[0]):
        for i, pix_val in enumerate(data_mat[fr_ind, :].flat):
            row = pix_table[0][i]
            col = pix_table[1][i]
            #if np.isreal(pix_val) == False:
            #    print "reconst_zone_cube: warning - complex valued pixel"
            reconstrd_cube[fr_ind, row, col] = pix_val
    return reconstrd_cube

def reconst_zone(data_vec, pix_table, img_dim):
    reconstrd_img = np.zeros(img_dim)
    for i, pix_val in enumerate(data_vec.flat):
        row = pix_table[0][i] 
        col = pix_table[1][i]
        reconstrd_img[row, col] = pix_val
    return reconstrd_img

def load_leech_adiseq(fname_root, N_fr, old_xycent, outer_search_rad):
    cropped_cube = np.zeros((N_fr, 2*outer_search_rad, 2*outer_search_rad))
    subpix_xyoffset = np.array( [0.5 - old_xycent[0]%1., 0.5 - old_xycent[1]%1.] )
    #print 'load_leech_adiseq: subpix_xyoffset = %0.2f, %0.2f' % (subpix_xyoffset[0], subpix_xyoffset[1])
    shifted_xycent = ( old_xycent[0] + subpix_xyoffset[0], old_xycent[1] + subpix_xyoffset[1] )

    for i in range(N_fr):
        img_fname = fname_root + '%05d.fits' % i
        img_hdulist = pyfits.open(img_fname, 'readonly')
        img = img_hdulist[0].data
        old_width = img.shape[1]
        shifted_img = shift(input = img, shift = subpix_xyoffset[::-1], order=3)    
        cropped_cube[i, :, :] = shifted_img[ round(shifted_xycent[1]) - outer_search_rad:round(shifted_xycent[1]) + outer_search_rad,
                                             round(shifted_xycent[0]) - outer_search_rad:round(shifted_xycent[0]) + outer_search_rad ].copy()
        img_hdulist.close()
    return cropped_cube

def load_adi_master_cube(datacube_fname, outer_search_rad, old_xycent=None, true_center=False):
    cube_hdulist = pyfits.open(datacube_fname, 'readonly')
    old_width = cube_hdulist[0].data.shape[1]
    if old_xycent is None:
        if true_center == True:
            old_xycent = ((old_width - 1)/2., (old_width - 1)/2.)
        else:
            old_xycent = (old_width/2, old_width/2)
    subpix_xyoffset = np.array( [0.5 - old_xycent[0]%1., 0.5 - old_xycent[1]%1.] )
    if subpix_xyoffset[0] > np.finfo(np.float64).eps or subpix_xyoffset[1] > np.finfo(np.float64).eps:
        #print 'load_adi_master_cube: subpix_xyoffset = %0.2f, %0.2f' % (subpix_xyoffset[0], subpix_xyoffset[1])
        shifted_xycent = ( old_xycent[0] + subpix_xyoffset[0], old_xycent[1] + subpix_xyoffset[1] )
        shifted_cube = shift(input = cube_hdulist[0].data, shift = [0, subpix_xyoffset[1], subpix_xyoffset[0]], order=3)
    else:
        #print 'load_adi_master_cube: No sub-pixel offset applied.'
        shifted_cube = cube_hdulist[0].data
        shifted_xycent = old_xycent
    cube_hdulist.close()
    cropped_cube = shifted_cube[:, round(shifted_xycent[1]) - outer_search_rad:round(shifted_xycent[1]) + outer_search_rad,
                                   round(shifted_xycent[0]) - outer_search_rad:round(shifted_xycent[0]) + outer_search_rad ].copy()
    return cropped_cube

def load_data_cube(datacube_fname, outer_search_rad):
    cube_hdu = pyfits.open(datacube_fname, 'readonly')
    old_width = cube_hdu[0].data.shape[1]
    crop_margin = (old_width - 2*outer_search_rad)/2
    print "Cropping %d-pixel wide frames down to %d pixels."%(old_width, old_width-2*crop_margin)
    cropped_cube = cube_hdu[0].data[:, crop_margin-1:-crop_margin-1, crop_margin-1:-crop_margin-1].copy()
    cube_hdu.close()
    return cropped_cube

def get_ref_and_pix_tables(xycent, fr_shape, N_fr, op_fr, mode_cut, N_rad, R_inner, R_out, DPhi, Phi_0, op_rad,
                           N_az, op_az, parang_seq, fwhm, min_refgap_fac, track_mode, diagnos_stride):
    #
    # Determine table of references for each frame, and form search zone pixel masks (1-D and 2-D formats).
    #
    if diagnos_stride > 0:
        print "Search zone scheme:"
        if track_mode:
            print "\tTrack mode ON"
        else:
            print "\tTrack mode OFF"
        print "\tR_inner:", R_inner, "; R_out:", R_out
        print "\tPhi_0, DPhi, N_az:", Phi_0, DPhi, N_az
        print "\tmode_cut:", mode_cut
    for rad_ind in op_rad:
        R2 = R_out[rad_ind]
        if rad_ind == 0:
            R1 = R_inner
        else:
            R1 = R_out[rad_ind-1]
        if track_mode:
            min_refang = DPhi[rad_ind]/2.
        else:
            min_refang = np.arctan(min_refgap_fac[rad_ind]*fwhm/((R1 + R2)/2))*180/np.pi
        if diagnos_stride > 0: print "\trad_ind = %d: min_refang = %0.2f deg" % (rad_ind, min_refang)
    if diagnos_stride > 0: print ""

    if xycent == None:
        xycent = ((fr_shape[0] - 1)/2., (fr_shape[0] - 1)/2.)
    rad_vec = np.sqrt(get_radius_sqrd(fr_shape, xycent)).ravel()
    angle_vec = get_angle(fr_shape, xycent).ravel()
    zonemask_table_1d = [[[None]*N_az[r] for r in range(N_rad)] for i in range(N_fr)]
    zonemask_table_2d = [[[None]*N_az[r] for r in range(N_rad)] for i in range(N_fr)]
    ref_table = [[list() for r in range(N_rad)] for i in range(N_fr)]

    for fr_ind in op_fr:
        for rad_ind in op_rad:
            R2 = R_out[rad_ind]
            zonemask_radlist_1d = list()
            zonemask_radlist_2d = list()
            if rad_ind == 0:
                R1 = R_inner
            else:
                R1 = R_out[rad_ind-1]
            if track_mode:
                Phi_beg = (Phi_0[rad_ind] - DPhi[rad_ind]/2. + parang_seq[0] - parang_seq[fr_ind]) % 360.
            else:
                Phi_beg = (Phi_0[rad_ind] - DPhi[rad_ind]/2.) % 360.
            Phi_end = [ (Phi_beg + i * DPhi[rad_ind]) % 360. for i in range(1, N_az[rad_ind]) ]
            Phi_end.append(Phi_beg)
            if track_mode:
                min_refang = DPhi[rad_ind]/2.
            else:
                min_refang = np.arctan(min_refgap_fac[rad_ind]*fwhm/((R1 + R2)/2))*180/np.pi
            ref_table[fr_ind][rad_ind] = np.where(np.greater_equal(np.abs(parang_seq - parang_seq[fr_ind]), min_refang))[0]
                
            for az_ind in op_az[rad_ind]:
                Phi2 = Phi_end[az_ind]
                if az_ind == 0:
                    Phi1 = Phi_beg
                else:
                    Phi1 = Phi_end[az_ind-1]
                if Phi1 < Phi2:
                    mask_logic = np.vstack((np.less_equal(rad_vec, R2),\
                                            np.greater(rad_vec, R1),\
                                            np.less_equal(angle_vec, Phi2),\
                                            np.greater(angle_vec, Phi1)))
                else: # azimuthal region spans phi = 0
                    rad_mask_logic = np.vstack((np.less_equal(rad_vec, R2),\
                                                np.greater(rad_vec, R1)))
                    az_mask_logic = np.vstack((np.less_equal(angle_vec, Phi2),\
                                               np.greater(angle_vec, Phi1)))
                    mask_logic = np.vstack((np.any(az_mask_logic, axis=0),\
                                            np.all(rad_mask_logic, axis=0)))
                zonemask_1d = np.nonzero( np.all(mask_logic, axis = 0) )[0]
                zonemask_2d = np.nonzero( np.all(mask_logic, axis = 0).reshape(fr_shape) )
                zonemask_table_1d[fr_ind][rad_ind][az_ind] = zonemask_1d
                zonemask_table_2d[fr_ind][rad_ind][az_ind] = zonemask_2d
                if fr_ind == op_fr[0] and diagnos_stride > 0:
                    print 'Search zone size for rad ind %d, az_ind %d is %d pixels' % (rad_ind, az_ind, zonemask_1d.shape[0])
                    if rad_ind == N_rad-1 and az_ind == N_az[rad_ind]-1:
                        print ""
                if zonemask_1d.shape[0] < len(ref_table[fr_ind][rad_ind]):
                    print "get_ref_table: warning - size of search zone for frame %d, rad_ind %d, az_ind %d is %d < %d, the # of ref frames for this annulus" %\
                          (fr_ind, rad_ind, az_ind, zonemask_1d.shape[0], len(ref_table[fr_ind][rad_ind]))
                    print "This has previously resulted in unexpected behavior, namely a reference covariance matrix that is not positive definite."
        for rad_ind in op_rad:
            if diagnos_stride > 0 and fr_ind%diagnos_stride == 0:
                print "Frame %d/%d, annulus %d/%d: %d valid reference frames." %\
                      (fr_ind+1, N_fr, rad_ind+1, N_rad, len(ref_table[fr_ind][rad_ind]))
            if len(ref_table[fr_ind][rad_ind]) < 1:
                print "Zero valid reference frames for fr_ind = %d, rad_ind = %d." % (fr_ind, rad_ind)
                print "The par ang of this frame is %0.2f deg; min_refang = %0.2f deg. Forced to exit." % (parang_seq[fr_ind], min_refang)
                sys.exit(-1)
    if diagnos_stride > 0: print ""
    for rad_ind in op_rad:
        num_ref = [len(ref_table[f][rad_ind]) for f in op_fr]
        if diagnos_stride > 0: print "annulus %d/%d: min, median, max number of ref frames = %d, %d, %d" %\
              ( rad_ind+1, N_rad, min(num_ref), np.median(num_ref), max(num_ref) )
    if diagnos_stride > 0: print ""
    return ref_table, zonemask_table_1d, zonemask_table_2d

def get_pca_basis(R, cutoff):
    U, sv, Vt = np.linalg.svd(R, full_matrices=False)
    N_modes = min([cutoff, Vt.shape[0]])
    return Vt[0:cutoff, :], sv, N_modes

def get_klip_basis(R, cutoff):
    w, V = np.linalg.eig(np.dot(R, np.transpose(R)))
    sort_ind = np.argsort(w)[::-1] #indices of eigenvals sorted in descending order
    sv = np.sqrt(w[sort_ind][w[sort_ind] > 0]).reshape(-1,1) #column of ranked singular values
    Z = np.dot(1./sv*np.transpose(V[:, sort_ind[w[sort_ind] > 0]]), R)
    N_modes = min([cutoff, Z.shape[0]])
    return Z[0:N_modes, :], sv, N_modes

def do_mp_klip_subtraction(N_proc, data_cube, config_dict, result_dict, result_dir, diagnos_stride=50, store_psf=False,
                           store_archv=False, store_klbasis=False, disable_sub=False, use_svd=True, log_fobj=sys.stdout):
    op_fr = config_dict['op_fr']
    fr_shape = config_dict['fr_shape']
    N_op_fr = len(op_fr)
    klipsub_cube = np.zeros((N_op_fr, fr_shape[0], fr_shape[1]))
    derot_klipsub_cube = klipsub_cube.copy()
    if store_psf:
        klippsf_cube = klipsub_cube.copy()
    else:
        klippsf_cube = None

    start_time = time.time()
    # Establish communication queues and start the 'workers'
    klipsub_tasks = multiprocessing.JoinableQueue()
    klipsub_results = multiprocessing.Queue()
    workers = [ Worker(klipsub_tasks, klipsub_results) for p in xrange(N_proc) ]
    for w in workers:
        w.start()
    # Enqueue the operand frames
    for fr_ind in op_fr:
        klipsub_tasks.put( klipsub_task(fr_ind, data_cube, config_dict['fr_shape'], config_dict['parang_seq'], config_dict['mode_cut'],
                                        config_dict['op_fr'], config_dict['op_rad'], config_dict['op_az'], config_dict['ref_table'][fr_ind],
                                        config_dict['zonemask_table_1d'][fr_ind], config_dict['zonemask_table_2d'][fr_ind], result_dict, result_dir,
                                        diagnos_stride, store_psf, store_klbasis, use_svd) )
        #klipsub_tasks.put( klipsub_task(fr_ind, data_cube, config_dict, result_dict, result_dir,
        #                                diagnos_stride, store_psf, store_klbasis, use_svd) )
    # Kill each worker
    for p in xrange(N_proc):
        klipsub_tasks.put(None)
    # Wait for all of the tasks to finish
    klipsub_tasks.join()
    # Organize results
    N_toget = N_op_fr
    while N_toget:
        result = klipsub_results.get()
        fr_ind = result[0]
        i = np.where(op_fr == fr_ind)[0][0]
        klipsub_cube[i,:,:] = result[1]
        derot_klipsub_cube[i,:,:] = result[2]
        if store_psf:
            klippsf_cube[i,:,:] = result[3]
        if store_archv:
            result_dict[fr_ind] = result[4][fr_ind]
        N_toget -= 1

    end_time = time.time()
    exec_time = end_time - start_time
    time_per_frame = exec_time/N_op_fr
    log_fobj.write("Took %dm%02ds to KLIP-subtract %d frames (%0.2f s per frame).\n" %\
          (int(exec_time/60.), exec_time - 60*int(exec_time/60.), N_op_fr, time_per_frame))
    if diagnos_stride > 0: log_fobj.write("\n")
    return klipsub_cube, klippsf_cube, derot_klipsub_cube

def mp_eval_adiklip_srcmodel(p, N_proc, op_fr, rad_ind, az_ind, mode_cut, adiklip_config, adiklip_data, srcmodel):
    if op_fr == None:
        op_fr = adiklip_config['op_fr']
    if rad_ind == None:
        rad_ind = 0
    if az_ind == None:
        az_ind = 0
    if mode_cut == None:
        mode_cut = adiklip_config['mode_cut'][rad_ind]
    else:
        assert mode_cut >= 0 and mode_cut <= adiklip_config['mode_cut'][rad_ind]

    fr_shape = adiklip_config['fr_shape']
    parang_seq = adiklip_config['parang_seq']
    amp = p[0]
    theta = np.arctan2(p[2], p[1])
    rho = np.sqrt(p[1]**2 + p[2]**2)
    abframe_theta_seq = [theta - np.deg2rad(parang) for parang in parang_seq[op_fr]]
    cent_xy = ((fr_shape[1] - 1)/2., (fr_shape[0] - 1)/2.)
    abframe_xy_seq = [( cent_xy[0] + rho*np.cos(t),
                        cent_xy[1] + rho*np.sin(t) ) for t in abframe_theta_seq]

    total_sumofsq_cost = 0
    # Establish communication queues and start the 'workers'
    eval_tasks = multiprocessing.JoinableQueue()
    eval_results = multiprocessing.Queue()
    workers = [ Worker(eval_tasks, eval_results) for j in xrange(N_proc) ]
    for w in workers:
        w.start()
    # Enqueue the operand frames
    for i, fr_ind in enumerate(op_fr):
        if mode_cut > 0:
            eval_tasks.put( eval_adiklip_srcmodel_task(fr_ind, fr_shape, adiklip_config['zonemask_table_1d'][fr_ind][rad_ind][az_ind],
                                                       adiklip_data[fr_ind][rad_ind][az_ind]['Z'][:mode_cut,:], adiklip_data[fr_ind][rad_ind][az_ind]['F'],
                                                       srcmodel, amp, abframe_xy_seq[i]) )
        else:
            eval_tasks.put( eval_adiklip_srcmodel_task(fr_ind, fr_shape, adiklip_config['zonemask_table_1d'][fr_ind][rad_ind][az_ind],
                                                       None, adiklip_data[fr_ind][rad_ind][az_ind]['F'], srcmodel, amp, abframe_xy_seq[i]) )
    #print 'placed all tasks on queue'
            
    # Kill each worker
    for j in xrange(N_proc):
        eval_tasks.put(None)
    # Wait for all of the tasks to finish
    eval_tasks.join()
    # Organize results
    N_toget = len(op_fr)
    while N_toget:
        result = eval_results.get()
        fr_ind = result[0]
        i = np.where(op_fr == fr_ind)[0][0]
        res_vec = result[1]
        cost = result[2]     
        total_sumofsq_cost += cost
        N_toget -= 1
    return total_sumofsq_cost

def lnprob_adiklip_srcmodel(p, N_proc, op_fr, adiklip_config, adiklip_data, srcmodel):
    img_shape = adiklip_config['fr_shape']
    if abs(p[1]) < (img_shape[1] - 1.)/2. and abs(p[2]) < (img_shape[0] - 1)/2.:
        cost = mp_eval_adiklip_srcmodel(p = p, N_proc = N_proc, op_fr = op_fr, adiklip_config = adiklip_config,
                                        adiklip_data = adiklip_data, srcmodel = srcmodel)
        lnprob = -cost/2.
    else:
        lnprob = np.finfo(np.float).min
    return lnprob

def eval_adiklip_srcmodel(p, op_fr, adiklip_config, adiklip_data, srcmodel, cost_queue=None, res_cube_fname=None):
    fr_shape = adiklip_config['fr_shape']
    parang_seq = adiklip_config['parang_seq']
    mode_cut = adiklip_config['mode_cut']
    track_mode = adiklip_config['track_mode']
    op_rad = adiklip_config['op_rad']
    op_az = adiklip_config['op_az']
    ref_table = adiklip_config['ref_table']
    zonemask_table_1d = adiklip_config['zonemask_table_1d']
    zonemask_table_2d = adiklip_config['zonemask_table_2d']
    N_op_fr = len(op_fr)
    N_fr = len(parang_seq)

    cent_xy = ((fr_shape[1] - 1)/2., (fr_shape[0] - 1)/2.)
    srcmodel_cent_xy = ((srcmodel.shape[1] - 1)/2., (srcmodel.shape[0] - 1)/2.)
    amp = p[0]
    deltax = p[1]
    deltay = p[2]
    theta = np.arctan2(deltay, deltax)
    rho = np.sqrt(deltax**2 + deltay**2)
    abframe_theta_seq = [theta - np.deg2rad(parang) for parang in parang_seq[op_fr]]
    abframe_xy_seq = [( cent_xy[0] + rho*np.cos(t),
                        cent_xy[1] + rho*np.sin(t) ) for t in abframe_theta_seq]
    abframe_synthsrc_cube = np.zeros((N_op_fr, fr_shape[0], fr_shape[1]))
    sumofsq_cost = 0.
    if res_cube_fname:
        if os.path.exists(res_cube_fname) == False:
            res_cube = np.zeros((N_fr, fr_shape[0], fr_shape[1]))
            res_cube_hdu = pyfits.PrimaryHDU(res_cube.astype(np.float32))
            res_cube_hdu.writeto(res_cube_fname)
        res_cube_hdulist = pyfits.open(res_cube_fname, mode='update')
        res_cube = res_cube_hdulist[0].data
    for op_fr_ind, fr_ind in enumerate(op_fr):
        if abframe_xy_seq[op_fr_ind][0] >= fr_shape[1] or abframe_xy_seq[op_fr_ind][1] >= fr_shape[0] or min(abframe_xy_seq[op_fr_ind]) < 0:
            print 'bad dest:', abframe_xy_seq[op_fr_ind]
            print 'p:', p[0:3]
            return np.finfo(np.float).max 
        abframe_synthsrc_cube[op_fr_ind,:,:] = superpose_srcmodel(data_img = np.zeros(fr_shape), srcmodel_img = amp*srcmodel,
                                                                  srcmodel_destxy = abframe_xy_seq[op_fr_ind], srcmodel_centxy = srcmodel_cent_xy)
        for rad_ind in op_rad:
            for az_ind in op_az[rad_ind]:
                Pmod = np.ravel(abframe_synthsrc_cube[op_fr_ind,:,:])[ zonemask_table_1d[fr_ind][rad_ind][az_ind] ].copy()
                #Pmod -= np.mean(Pmod)
                Z = klip_data[fr_ind][rad_ind][az_ind]['Z'][:,:]
                Projmat = np.dot(Z.T, Z)
                F = klip_data[fr_ind][rad_ind][az_ind]['F']
                Pmod_proj = np.dot(Pmod, Projmat)
                res_vec = F - Pmod + Pmod_proj
                sumofsq_cost += np.sum(res_vec**2)
                if res_cube_fname:
                    res_cube[fr_ind,:,:] += reconst_zone(res_vec, zonemask_table_2d[fr_ind][rad_ind][az_ind], fr_shape)
                    #print "rms(Pmod) = %.1f, rms(Pmod_proj) = %.1f" %\
                    #      (np.sqrt(np.sum(Pmod**2)), np.sqrt(np.sum(Pmod_proj**2)))
    if cost_queue:
        cost_queue.put(sumofsq_cost)
    if res_cube_fname:
        res_cube_hdulist.close()
    return sumofsq_cost

def klip_subtract(dataset_label, data_dir, result_dir, R_inner, R_out, mode_cut, DPhi, Phi_0,
                  fwhm, min_refgap_fac, op_fr=None, N_proc=1, diagnos_stride=50, fake_planet=None,
                  synth_psf_img=None, test_mode=False, use_svd=True, coadd_full_overlap_only=True,
                  store_results=True, store_psf=False, store_archv=False, store_klbasis=False,
                  track_mode=False, log_fobj=sys.stdout):
    #
    # Load the data
    #
    assert(os.path.exists(data_dir)), 'data_dir %s does not exist' % data_dir
    assert(os.path.exists(result_dir)), 'result_dir %s does not exist' % result_dir
    cube_fname = '%s/%s_cube.fits' % (data_dir, dataset_label)
    cropped_cube_fname = '%s/%s_cropped_cube.fits' % (data_dir, dataset_label)
    if os.path.exists(cropped_cube_fname):
        if diagnos_stride > 0:
            print "Loading existing centered, cropped data cube %s..." % cropped_cube_fname
        cube_hdulist = pyfits.open(cropped_cube_fname, 'readonly')
        data_cube = cube_hdulist[0].data
        cube_hdulist.close()
        if data_cube.shape[1] != 2*R_out[-1]:
            if diagnos_stride > 0:
                print "Loading, centering, and cropping master ADI data cube %s..." % cube_fname
            data_cube = load_adi_master_cube(cube_fname, R_out[-1], true_center=True)
            data_cube_hdu = pyfits.PrimaryHDU(data_cube.astype(np.float32))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                data_cube_hdu.writeto('%s/%s_cropped_cube.fits' % (data_dir, dataset_label), clobber=True)
    else:
        if diagnos_stride > 0:
            print "Loading, centering, and cropping master ADI data cube %s..." % cube_fname
        data_cube = load_adi_master_cube(cube_fname, R_out[-1], true_center=True)
        data_cube_hdu = pyfits.PrimaryHDU(data_cube.astype(np.float32))
        data_cube_hdu.writeto('%s/%s_cropped_cube.fits' % (data_dir, dataset_label))
    parang_fname = '%s/%s_parang.sav' % (data_dir, dataset_label)
    parang_seq = readsav(parang_fname).master_parang_arr
    N_fr = parang_seq.shape[0]
    fr_shape = data_cube.shape[1:]
    fr_width = fr_shape[1]
    N_parang = parang_seq.shape[0]
    assert(np.equal(N_fr, N_parang))
    if diagnos_stride > 0:
        print "The LMIRcam ADI sequence has been cropped to width %d pixels." % fr_width
        print "%d images with parallactic angle range %0.2f to %0.2f deg" % (N_fr, parang_seq[0], parang_seq[-1])
    #
    # Complete the geometric configuration.
    #
    if op_fr == None:
        if test_mode:
            if diagnos_stride > 0:
                op_fr = np.arange(0, N_fr, diagnos_stride)
            else:
                op_fr = np.arange(0, N_fr, 50)
        else:
            op_fr = np.arange(N_fr)
    N_rad = len(R_out)
    assert(len(mode_cut) == N_rad == len(DPhi) == len(Phi_0))
    N_az = [ int(np.ceil(360./DPhi[r])) for r in range(N_rad) ]
    op_rad = range(N_rad)
    op_az = [range(N_az[i]) for i in range(N_rad)]
    assert(len(op_rad) == len(op_az) == len(min_refgap_fac) == N_rad)
    #
    # Form a pixel mask for each search zone, and assemble the masks into two tables (1-D and 2-D formats).
    #
    ref_table, zonemask_table_1d, zonemask_table_2d = get_ref_and_pix_tables(xycent=None, fr_shape=fr_shape, N_fr=N_fr,
                                                                             op_fr=op_fr, mode_cut=mode_cut, N_rad=N_rad,
                                                                             R_inner=R_inner, R_out=R_out, DPhi=DPhi, Phi_0=Phi_0,
                                                                             op_rad=op_rad, N_az=N_az, op_az=op_az,
                                                                             parang_seq=parang_seq, fwhm=fwhm,
                                                                             min_refgap_fac=min_refgap_fac, track_mode=track_mode,
                                                                             diagnos_stride=diagnos_stride)
    # 
    # Perform zone-by-zone KLIP subtraction on each frame
    #
    klip_config = {'fr_shape':fr_shape, 'parang_seq':parang_seq, 'mode_cut':mode_cut,
                   'track_mode':track_mode, 'op_fr':op_fr, 'op_rad':op_rad, 'op_az':op_az,
                   'R_inner':R_inner, 'R_out':R_out, 'N_az':N_az,
                   'ref_table':ref_table, 'zonemask_table_1d':zonemask_table_1d,
                   'zonemask_table_2d':zonemask_table_2d}
    if store_archv:
        klip_data = [[[dict.fromkeys(['I', 'I_mean', 'Z', 'sv', 'F']) for a in range(N_az[r])] for r in range(N_rad)] for i in range(N_fr)]
    else:
        klip_data = None
    if diagnos_stride > 0:
        print "Using %d of the %d logical processors available" % (N_proc, multiprocessing.cpu_count())
    klipsub_cube, klippsf_cube, derot_klipsub_cube = do_mp_klip_subtraction(N_proc = N_proc, data_cube=data_cube, config_dict=klip_config,
                                                                            result_dict=klip_data, result_dir=result_dir, diagnos_stride=diagnos_stride,
                                                                            store_psf=store_psf, store_archv=store_archv, store_klbasis=store_klbasis,
                                                                            use_svd=use_svd, log_fobj=log_fobj)
    if fake_planet != None:
        R_fakep = fake_planet[0]
        PA_fakep = fake_planet[1]
        flux_fakep = fake_planet[2]
        fakep_data_cube = add_fake_planets(data_cube, parang_seq, psf_img=synth_psf_img, R_p=[R_fakep], PA_p=[PA_fakep], flux_p=[flux_fakep])
        fakep_klipsub_cube, fakep_klippsf_cube, fakep_derot_klipsub_cube = do_mp_klip_subtraction(N_proc = N_proc, data_cube=fakep_data_cube, config_dict=klip_config,
                                                                                                  result_dict=klip_data, result_dir=result_dir,
                                                                                                  diagnos_stride=diagnos_stride, store_psf=store_psf,
                                                                                                  store_archv=store_archv, store_klbasis=store_klbasis,
                                                                                                  use_svd=use_svd, log_fobj=log_fobj)
    #
    # Form mean and median of derotated residual images, and the mean and median of the PSF estimates.
    #
    coadd_img = nanmean(derot_klipsub_cube, axis=0)
    med_img = nanmedian(derot_klipsub_cube, axis=0)
    if store_psf:
        mean_klippsf_img = nanmean(klippsf_cube, axis=0)
        med_klippsf_img = nanmedian(klippsf_cube, axis=0)
    if coadd_full_overlap_only:
        sum_collapse_img = np.sum(derot_klipsub_cube, axis=0)
        exclude_ind = np.isnan(sum_collapse_img)
        coadd_img[exclude_ind] = np.nan
        med_img[exclude_ind] = np.nan
    coadd_rebin2x2_img = coadd_img.reshape(coadd_img.shape[0]/2, 2, coadd_img.shape[1]/2, 2).mean(1).mean(2)
    if fake_planet != None:
        fakep_coadd_img = nanmean(fakep_derot_klipsub_cube, axis=0)
        fakep_med_img = nanmedian(fakep_derot_klipsub_cube, axis=0)
        if coadd_full_overlap_only:
            fakep_coadd_img[exclude_ind] = np.nan
            fakep_med_img[exclude_ind] = np.nan
    #
    # Get statistics from co-added and median residual images
    #
    if fake_planet != None:
        #plt.figure(figsize=(12,6))
        #plt.subplot(1,3,1)
        #plt.imshow(coadd_img, origin='lower', interpolation='nearest')
        #plt.colorbar(orientation='vertical', shrink=0.5)
        #plt.subplot(1,3,2)
        #plt.imshow(fakep_coadd_img, origin='lower', interpolation='nearest')
        #plt.colorbar(orientation='vertical', shrink=0.5)
        #plt.subplot(1,3,3)
        #plt.imshow(fakep_coadd_img - coadd_img, origin='lower', interpolation='nearest')
        #plt.colorbar(orientation='vertical', shrink=0.5)
        #plt.show()
        coadd_rms, coadd_min_detect, med_rms, med_min_detect = get_residual_fake_planet_stats(config_dict=klip_config, Phi_0=Phi_0, R_p=R_fakep, PA_p=PA_fakep,
                                                                                              flux_p = flux_fakep, fwhm=int(round(fwhm)),
                                                                                              planet_coadd_img=fakep_coadd_img, planet_med_img=fakep_med_img,
                                                                                              true_coadd_img=coadd_img, true_med_img=med_img, log_fobj=log_fobj)
    else:
        annular_rms, zonal_rms = get_residual_stats(config_dict=klip_config, Phi_0=Phi_0, coadd_img=coadd_img, med_img=med_img)
    if store_results == True:
        #
        # Store the results
        #
        delimiter = '-'
        result_label = "%s_globalklip_rad%03d-%03d_mode%03d-%03d" % (dataset_label, R_inner, R_out[-1], mode_cut[0], mode_cut[-1])
        klipsub_cube_fname = "%s/%s_res_cube.fits" % (result_dir, result_label)
        klippsf_cube_fname = "%s/%s_psf_cube.fits" % (result_dir, result_label)
        derot_klipsub_cube_fname = "%s/%s_derot_res_cube.fits" % (result_dir, result_label)
        coadd_img_fname = "%s/%s_res_coadd.fits" % (result_dir, result_label)
        coadd_rebin2x2_img_fname = "%s/%s_res_coadd_rebin2x2.fits" % (result_dir, result_label)
        med_img_fname = "%s/%s_res_med.fits" % (result_dir, result_label)
        mean_klippsf_img_fname = "%s/%s_psf_mean.fits" % (result_dir, result_label)
        med_klippsf_img_fname = "%s/%s_psf_med.fits" % (result_dir, result_label)

        if store_psf:
            klippsf_cube_hdu = pyfits.PrimaryHDU(klippsf_cube.astype(np.float32))
            klippsf_cube_hdu.writeto(klippsf_cube_fname, clobber=True)
            print "\nWrote KLIP PSF estimate cube (%.3f Mb) to %s" % (klippsf_cube.nbytes/10.**6, klippsf_cube_fname)
            mean_klippsf_img_hdu = pyfits.PrimaryHDU(mean_klippsf_img.astype(np.float32))
            mean_klippsf_img_hdu.writeto(mean_klippsf_img_fname, clobber=True)
            print "Wrote average of KLIP PSF estimate cube (%.3f Mb) to %s" % (mean_klippsf_img.nbytes/10.**6, mean_klippsf_img_fname)
            med_klippsf_img_hdu = pyfits.PrimaryHDU(med_klippsf_img.astype(np.float32))
            med_klippsf_img_hdu.writeto(med_klippsf_img_fname, clobber=True)
            print "Wrote median of KLIP PSF estimate cube (%.3f Mb) to %s" % (med_klippsf_img.nbytes/10.**6, med_klippsf_img_fname)
        klipsub_cube_hdu = pyfits.PrimaryHDU(klipsub_cube.astype(np.float32))
        klipsub_cube_hdu.writeto(klipsub_cube_fname, clobber=True)
        print "Wrote KLIP-subtracted cube (%.3f Mb) to %s" % (klipsub_cube.nbytes/10.**6, klipsub_cube_fname)
        derot_klipsub_cube_hdu = pyfits.PrimaryHDU(derot_klipsub_cube.astype(np.float32))
        derot_klipsub_cube_hdu.writeto(derot_klipsub_cube_fname, clobber=True)
        print "Wrote derotated, KLIP-subtracted image cube (%.3f Mb) to %s" % (derot_klipsub_cube.nbytes/10.**6, derot_klipsub_cube_fname)
        coadd_img_hdu = pyfits.PrimaryHDU(coadd_img.astype(np.float32))
        coadd_img_hdu.writeto(coadd_img_fname, clobber=True)
        print "Wrote average of derotated, KLIP-subtracted images (%.3f Mb) to %s" % (coadd_img.nbytes/10.**6, coadd_img_fname)
        coadd_rebin2x2_img_hdu = pyfits.PrimaryHDU(coadd_rebin2x2_img.astype(np.float32))
        coadd_rebin2x2_img_hdu.writeto(coadd_rebin2x2_img_fname, clobber=True)
        print "Wrote 2x2-rebinned average of derotated, KLIP-subtracted images (%.3f Mb) to %s" % (coadd_rebin2x2_img.nbytes/10.**6, coadd_rebin2x2_img_fname)
        med_img_hdu = pyfits.PrimaryHDU(med_img.astype(np.float32))
        med_img_hdu.writeto(med_img_fname, clobber=True)
        print "Wrote median of derotated, KLIP-subtracted images (%.3f Mb) to %s" % (med_img.nbytes/10.**6, med_img_fname)
    if fake_planet != None:
        return klip_config, klip_data, coadd_min_detect, med_min_detect
    else:
        return klip_config, klip_data, annular_rms, zonal_rms

def klipmod(ampguess, posguess_rho, posguess_theta, klipsub_archv_fname, klipsub_result_dir, klipmod_result_dir,
            template_img_fname, synthpsf_fname, synthpsf_rolloff, result_label, N_proc, mode_cut=None, do_MLE=True):

    guess_srcmodel_fname = '%s/%s_guess_srcmodel_img.fits' % (klipmod_result_dir, result_label)
    guess_res_cube_fname = '%s/%s_guess_res_cube.fits' % (klipmod_result_dir, result_label)
    final_srcmodel_fname = '%s/%s_final_srcmodel_img.fits' % (klipmod_result_dir, result_label)
    final_res_cube_fname = '%s/%s_final_res_cube.fits' % (klipmod_result_dir, result_label)

    assert os.path.exists(klipsub_result_dir)
    assert os.path.exists(klipmod_result_dir)
    assert os.path.exists(klipsub_archv_fname)

    klipsub_res_img_hdulist = pyfits.open(template_img_fname, 'readonly')
    klipsub_res_img = klipsub_res_img_hdulist[0].data
    klipsub_res_img_hdulist.close()
    exclude_ind = np.isnan(klipsub_res_img)

    klipsub_archv = open(klipsub_archv_fname, 'rb')
    print 'Opened KLIP subtraction archive %s' % klipsub_archv_fname
    print 'Loading the configuration and data...'
    klip_config, klip_data = pickle.load(klipsub_archv)
    klipsub_archv.close()

    fr_shape = klip_config['fr_shape']
    parang_seq = klip_config['parang_seq']
    op_fr = klip_config['op_fr']
    #op_fr = np.arange(43,87)
    op_rad = klip_config['op_rad']
    op_az = klip_config['op_az']
    ref_table = klip_config['ref_table']
    zonemask_table_1d = klip_config['zonemask_table_1d']
    zonemask_table_2d = klip_config['zonemask_table_2d']
    N_op_fr = len(op_fr)
    #
    # Load PSF model, crop, and apply "roll-off" window to keep the edges smooth.
    #
    synthpsf_hdulist = pyfits.open(synthpsf_fname)
    synthpsf_img = synthpsf_hdulist[0].data
    synthpsf_hdulist.close()
    synthpsf_cent_xy = ((synthpsf_img.shape[0] - 1.)/2., (synthpsf_img.shape[1] - 1.)/2.)
    rolloff_rad = synthpsf_rolloff
    cropmarg = 1
    crop_synthpsf_img = synthpsf_img[cropmarg:-cropmarg, cropmarg:-cropmarg].copy()
    crop_synthpsf_cent_xy = ((crop_synthpsf_img.shape[0] - 1.)/2., (crop_synthpsf_img.shape[1] - 1.)/2.)
    Y, X = np.indices(crop_synthpsf_img.shape)
    Rsqrd = (X - crop_synthpsf_cent_xy[0])**2 + (Y - crop_synthpsf_cent_xy[1])**2
    crop_synthpsf_img *= np.exp( -(Rsqrd / rolloff_rad**2)**2 )
    #print synthpsf_img.shape, crop_synthpsf_img.shape
    print 'Sum of synthetic PSF before / after rolloff and cropping: %0.3f, %0.3f' % (synthpsf_img.sum(), crop_synthpsf_img.sum())

    cent_xy = ((fr_shape[1] - 1)/2., (fr_shape[0] - 1)/2.)
    posguess_xy = ( cent_xy[0] + posguess_rho*np.cos(np.deg2rad(posguess_theta + 90)),\
                    cent_xy[1] + posguess_rho*np.sin(np.deg2rad(posguess_theta + 90)) )
    posguess_deltaxy = (posguess_xy[0] - cent_xy[0], posguess_xy[1] - cent_xy[1])
    p0 = np.array([ampguess, posguess_deltaxy[0], posguess_deltaxy[1]])
    p_min = [ampguess*0.2, posguess_deltaxy[0] - 5., posguess_deltaxy[1] - 5.]
    p_max = [ampguess*5., posguess_deltaxy[0] + 5., posguess_deltaxy[1] + 5.]
    p_bounds = [(p_min[i], p_max[i]) for i in range(len(p_min))]
    print "p0:", p0

    start_time = time.time()
    guess_cost = mp_eval_adiklip_srcmodel(p = p0, N_proc = N_proc, op_fr = op_fr, rad_ind = None, az_ind = None, mode_cut = mode_cut,
                                          adiklip_config = klip_config, adiklip_data = klip_data, srcmodel = crop_synthpsf_img)
    end_time = time.time()
    exec_time = end_time - start_time
    print "Guess param cost func evaluation = %.1f. Took %dm%02ds to evaluate KLIP source model cost for %d ADI frames" %\
          (guess_cost, int(exec_time/60.), exec_time - 60*int(exec_time/60.), N_op_fr)
    if do_MLE:
        #
        # Optimize the flux and position of the source model to fit the KLIP subtraction residuals.
        #
        start_time = time.time()
        #p_sol, final_cost, info = fmin_l_bfgs_b(func = eval_adiklip_srcmodel, x0 = p0,
        #                                        args = (op_fr, klip_config, klip_data, synthpsf_img),
        #                                        approx_grad = True, bounds = p_bounds, factr=1e8, maxfun=100, disp=2)
        p_sol, final_cost, info = fmin_l_bfgs_b(func = mp_eval_adiklip_srcmodel, x0 = p0,
                                                args = (N_proc, op_fr, None, None, mode_cut, klip_config, klip_data, crop_synthpsf_img),
                                                approx_grad = True, bounds = p_bounds, factr=1e7, maxfun=100, disp=2)
        end_time = time.time()
        exec_time = end_time - start_time

        print "p_sol:", p_sol
        print "Took %dm%02ds to optimize KLIP source model for %d ADI frames" %\
              (int(exec_time/60.), exec_time - 60*int(exec_time/60.), N_op_fr)

        #eval_adiklip_srcmodel(p = p_sol, op_fr = op_fr, adiklip_config = klip_config, adiklip_data = klip_data,\
        #                      srcmodel = crop_synthpsf_img, res_cube_fname = final_res_cube_fname)

        #
        # Write the final MLE model of the source to an image
        #
        mle_sol_amp = p_sol[0]
        mle_sol_pos_xy = ( cent_xy[0] + p_sol[1], cent_xy[1] + p_sol[2] )
        final_srcmodel_img = superpose_srcmodel(data_img = np.zeros(fr_shape), srcmodel_img = mle_sol_amp*crop_synthpsf_img,
                                                srcmodel_destxy = mle_sol_pos_xy, srcmodel_centxy = crop_synthpsf_cent_xy)
        final_srcmodel_img[exclude_ind] = np.nan
        final_srcmodel_hdu = pyfits.PrimaryHDU(final_srcmodel_img.astype(np.float32))
        final_srcmodel_hdu.writeto(final_srcmodel_fname, clobber=True)
        print "Wrote the final source model to %s" % final_srcmodel_fname
        return p_sol
    else:
        return p0
