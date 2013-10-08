#! /usr/bin/env python
"""

Carry out PSF subtraction on LMIRcam ADI data set using principal
component analysis/K-L.

"""

import numpy as np
import time as time
import pyfits
from scipy.ndimage.interpolation import *
from scipy.interpolate import *
from scipy.ndimage.filters import *
from scipy.io.idl import readsav
from scipy.stats import nanmean, nanmedian
import multiprocessing
import sys
import os
import pdb
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.colors

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
                # Poison pill means shutdown
                #print '%s: Exiting' % proc_name
                self.task_queue.task_done()
                break
            #print '%s: doing KLIP subtraction on %s' % (proc_name, next_task)
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return

class klipsub_task(object):
    def __init__(self, fr_ind, data_cube, config_dict, result_dict, result_dir, diagnos_stride,
                 store_klbasis=False, disable_sub=False, use_svd=True):
         self.fr_ind = fr_ind
         self.data_cube = data_cube
         self.config_dict = config_dict
         self.result_dict = result_dict
         self.result_dir = result_dir
         self.diagnos_stride = diagnos_stride
         self.store_klbasis = store_klbasis
         self.disable_sub = disable_sub
         self.use_svd = use_svd
    def __call__(self):
        fr_shape = self.config_dict['fr_shape']
        parang_seq = self.config_dict['parang_seq']
        N_fr = len(parang_seq)
        mode_cut = self.config_dict['mode_cut']
        track_mode = self.config_dict['track_mode']
        op_fr = self.config_dict['op_fr']
        N_op_fr = len(op_fr)
        op_rad = self.config_dict['op_rad']
        op_az = self.config_dict['op_az']
        ref_table = self.config_dict['ref_table']
        zonemask_table_1d = self.config_dict['zonemask_table_1d']
        zonemask_table_2d = self.config_dict['zonemask_table_2d']

        fr_ind = self.fr_ind
        data_cube = self.data_cube
        result_dict = self.result_dict
        result_dir = self.result_dir
        diagnos_stride = self.diagnos_stride
        store_klbasis = self.store_klbasis
        disable_sub = self.disable_sub
        use_svd = self.use_svd

        klippsf_img = np.tile(np.nan, fr_shape)
        klipsub_img = np.zeros(fr_shape)
        derot_klipsub_img = klipsub_img.copy()

        if fr_ind%diagnos_stride == 0:
            if max(mode_cut) > 0:
                klbasis_cube = np.zeros((max(mode_cut), fr_shape[0], fr_shape[1]))
            else:
                klbasis_cube = None
        for rad_ind in op_rad:
            for az_ind in op_az[rad_ind]:
                I = np.ravel(data_cube[fr_ind,:,:])[ zonemask_table_1d[fr_ind][rad_ind][az_ind] ].copy() 
                R = np.zeros((ref_table[fr_ind][rad_ind].shape[0], zonemask_table_1d[fr_ind][rad_ind][az_ind].shape[0]))
                for j, ref_fr_ind in enumerate(ref_table[fr_ind][rad_ind]):
                    R[j,:] = np.ravel(data_cube[ref_fr_ind,:,:])[ zonemask_table_1d[fr_ind][rad_ind][az_ind] ].copy()
                if mode_cut[rad_ind] > 0: # do PCA on reference PSF stack
                    if use_svd == False: # following Soummer et al. 2012
                        I_mean = R.mean(axis = 0)
                        I -= R.mean(axis = 0)
                        R -= R.mean(axis = 0)
                        Z, sv, N_modes = get_klip_basis(R = R, cutoff = mode_cut[rad_ind])
                    else:
                        I_mean = R.mean(axis = 0)
                        I -= R.mean(axis = 0)
                        R -= R.mean(axis = 0)
                        Z, sv, N_modes = get_pca_basis(R = R, cutoff = mode_cut[rad_ind])
                    #if fr_ind % diagnos_stride == 0:
                    #    print "Frame %d/%d, annulus %d/%d, sector %d/%d:" %\
                    #          (fr_ind+1, N_fr, rad_ind+1, N_rad, az_ind+1, N_az[rad_ind])
                    #    print "\tForming PSF estimate..."
                    Projmat = np.dot(Z.T, Z)
                    I_proj = np.dot(I, Projmat)
                    if disable_sub:
                        F = I + I_mean
                    else:
                        F = I - I_proj
                    klippsf_zone_img = reconst_zone(I_proj + I_mean, zonemask_table_2d[fr_ind][rad_ind][az_ind], fr_shape)
                else: # classical ADI: subtract mean refernce PSF
                    R_mean = R.mean(axis = 0)
                    F = I - R_mean
                    klippsf_zone_img = reconst_zone(R_mean, zonemask_table_2d[fr_ind][rad_ind][az_ind], fr_shape)
                klippsf_img[ zonemask_table_2d[fr_ind][rad_ind][az_ind] ] = klippsf_zone_img[ zonemask_table_2d[fr_ind][rad_ind][az_ind] ]
                klipsub_zone_img = reconst_zone(F, zonemask_table_2d[fr_ind][rad_ind][az_ind], fr_shape)
                klipsub_img[ zonemask_table_2d[fr_ind][rad_ind][az_ind] ] = klipsub_zone_img[ zonemask_table_2d[fr_ind][rad_ind][az_ind] ]

                if store_archv and mode_cut[rad_ind] > 0:
                    result_dict[fr_ind][rad_ind][az_ind]['Z'] = Z.astype(np.float32)
                    result_dict[fr_ind][rad_ind][az_ind]['F'] = F.astype(np.float32)
                    #result_dict[fr_ind][rad_ind][az_ind]['I'] = I
                    #result_dict[fr_ind][rad_ind][az_ind]['I_mean'] = I_mean
                    #result_dict[fr_ind][rad_ind][az_ind]['sv'] = sv
                    #result_dict[fr_ind][rad_ind][az_ind]['Projmat'] = Projmat
                    #result_dict[fr_ind][rad_ind][az_ind]['I_proj'] = I_proj
                if fr_ind % diagnos_stride == 0 and mode_cut[rad_ind] > 0:
                    klbasis_cube[:N_modes,:,:] += reconst_zone_cube(Z, zonemask_table_2d[fr_ind][rad_ind][az_ind],
                                                                    cube_dim = (N_modes, fr_shape[0], fr_shape[1]))
                    print "Frame %d, annulus %d/%d, sector %d/%d: RMS before/after sub: %0.2f / %0.2f" %\
                          (fr_ind+1, rad_ind+1, len(op_rad), az_ind+1, len(op_az[rad_ind]),\
                           np.sqrt(np.mean((I + I_mean)**2)), np.sqrt(np.mean(F**2)))
        # De-rotate the KLIP-subtracted image
        submask_img = klipsub_img.copy()
        submask_img[ zonemask_table_2d[fr_ind][rad_ind][az_ind] ] = 1.
        derot_klipsub_img = rotate(klipsub_img, -parang_seq[fr_ind], reshape=False)
        derot_submask_img = rotate(submask_img, -parang_seq[fr_ind], reshape=False)
        #derot_submask_hdu = pyfits.PrimaryHDU(derot_submask_img.astype(np.float32))
        #derot_submask_hdu.writeto("%s/submask_fr%03d.fits" % (result_dir, fr_ind), clobber=True)
        exc_ind = np.where(derot_submask_img < 0.9)
        derot_klipsub_img[exc_ind] = np.nan

        #derot_klipsub_img = rotate(klipsub_img, -parang_seq[fr_ind], reshape=False)
        if fr_ind % diagnos_stride == 0:
            print "***** Frame %d has been PSF-sub'd and derotated. *****" % (fr_ind+1)
            if store_klbasis == True and klbasis_cube:
                klbasis_cube_hdu = pyfits.PrimaryHDU(klbasis_cube.astype(np.float32))
                klbasis_cube_hdu.writeto("%s/klbasis_fr%03d.fits" % (result_dir, fr_ind), clobber=True)
        return (fr_ind, klipsub_img, klippsf_img, derot_klipsub_img, result_dict)
    def __str__(self):
        return 'frame %d' % (self.fr_ind+1)

def get_radius_sqrd(s, c=None):
    if c is None:
        c = (0.5*float(s[0] - 1),  0.5*float(s[1] - 1))
    y, x = np.indices(s)
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
        #if np.isreal(pix_val) == False:
        #    print "reconst_zone_cube: warning - complex valued pixel"
        reconstrd_img[row, col] = pix_val
    return reconstrd_img

def load_leech_adiseq(fname_root, N_fr, old_xycent, outer_search_rad):
    cropped_cube = np.zeros((N_fr, 2*outer_search_rad, 2*outer_search_rad))
    subpix_xyoffset = np.array( [0.5 - old_xycent[0]%1., 0.5 - old_xycent[1]%1.] )
    print 'load_leech_adiseq: subpix_xyoffset = %0.2f, %0.2f' % (subpix_xyoffset[0], subpix_xyoffset[1])
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
        print 'load_adi_master_cube: subpix_xyoffset = %0.2f, %0.2f' % (subpix_xyoffset[0], subpix_xyoffset[1])
        shifted_xycent = ( old_xycent[0] + subpix_xyoffset[0], old_xycent[1] + subpix_xyoffset[1] )
        shifted_cube = shift(input = cube_hdulist[0].data, shift = [0, subpix_xyoffset[1], subpix_xyoffset[0]], order=3)
    else:
        print 'load_adi_master_cube: No sub-pixel offset applied.'
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

def get_ref_and_pix_tables(xycent, fr_shape, N_fr, op_fr, N_rad, R_inner, R_out, op_rad,
                           N_az, op_az, parang_seq, fwhm, min_refgap_fac, track_mode, diagnos_stride):
    #
    # Determine table of references for each frame, and form search zone pixel masks (1-D and 2-D formats).
    #
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
        print "\trad_ind = %d: min_refang = %0.2f deg" % (rad_ind, min_refang)
    print ""

    if xycent == None:
        xycent = ((fr_width - 1)/2., (fr_width - 1)/2.)
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
            if fr_ind%diagnos_stride == 0:
                print "\tFrame %d/%d, annulus %d/%d: %d valid reference frames." %\
                      (fr_ind+1, N_fr, rad_ind+1, N_rad, len(ref_table[fr_ind][rad_ind]))
            if len(ref_table[fr_ind][rad_ind]) < 1:
                print "Zero valid reference frames for fr_ind = %d, rad_ind = %d." % (fr_ind, rad_ind)
                print "The par ang of this frame is %0.2f deg; min_refang = %0.2f deg. Forced to exit." % (parang_seq[fr_ind], min_refang)
                sys.exit(-1)
                
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
                if zonemask_1d.shape[0] < len(ref_table[fr_ind][rad_ind]):
                    print "get_ref_table: warning - size of search zone for frame %d, rad_ind %d, az_ind %d is %d < %d, the # of ref frames for this annulus" %\
                          (fr_ind, rad_ind, az_ind, zonemask_1d.shape[0], len(ref_table[fr_ind][rad_ind]))
                    print "This has previously resulted in unexpected behavior, namely a reference covariance matrix that is not positive definite."
    for rad_ind in op_rad:
        num_ref = [len(ref_table[f][rad_ind]) for f in op_fr]
        print "annulus %d/%d: min, median, max number of ref frames = %d, %d, %d" %\
              ( rad_ind+1, N_rad, min(num_ref), np.median(num_ref), max(num_ref) )
    print ""
    return ref_table, zonemask_table_1d, zonemask_table_2d

def do_mp_klip_subtraction(N_proc, data_cube, config_dict, result_dict, result_dir, diagnos_stride=40, store_klbasis=False,
                           disable_sub=False, use_svd=True, mean_sub=True):
    op_fr = config_dict['op_fr']
    fr_shape = config_dict['fr_shape']
    N_op_fr = len(op_fr)
    klipsub_cube = np.zeros((N_op_fr, fr_shape[0], fr_shape[1]))
    klippsf_cube = klipsub_cube.copy()
    derot_klipsub_cube = klipsub_cube.copy()

    start_time = time.time()
    # Establish communication queues and start the 'workers'
    klipsub_tasks = multiprocessing.JoinableQueue()
    klipsub_results = multiprocessing.Queue()
    workers = [ Worker(klipsub_tasks, klipsub_results) for p in xrange(N_proc) ]
    for w in workers:
        w.start()
    # Enqueue the operand frames
    for fr_ind in op_fr:
        klipsub_tasks.put(klipsub_task(fr_ind, data_cube, config_dict, result_dict, result_dir,
                                       diagnos_stride, store_klbasis, disable_sub, use_svd))
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
        klippsf_cube[i,:,:] = result[2]
        derot_klipsub_cube[i,:,:] = result[3]
        result_dict[fr_ind] = result[4][fr_ind]
        N_toget -= 1

    end_time = time.time()
    exec_time = end_time - start_time
    time_per_frame = exec_time/N_op_fr
    print "Took %dm%02ds to KLIP-subtract %d frames (%0.2f s per frame).\n" %\
          (int(exec_time/60.), exec_time - 60*int(exec_time/60.), N_op_fr, time_per_frame)
    return klipsub_cube, klippsf_cube, derot_klipsub_cube

def do_klip_subtraction(data_cube, config_dict, result_dict, result_dir, diagnos_stride=40, store_klbasis=False,
                        disable_sub=False, use_svd=True, mean_sub=True, proc_ind=None, result_queue=None):
#def do_klip_subtraction(data_cube, config_dict, result_dict, result_dir, diagnos_stride=40, store_klbasis=False,
#                        disable_sub=False, use_svd=True, mean_sub=True, proc_ind=None, conn=None):
    fr_shape = config_dict['fr_shape']
    parang_seq = config_dict['parang_seq']
    N_fr = len(parang_seq)
    mode_cut = config_dict['mode_cut']
    track_mode = config_dict['track_mode']
    op_fr = config_dict['op_fr']
    N_op_fr = len(op_fr)
    op_rad = config_dict['op_rad']
    op_az = config_dict['op_az']
    ref_table = config_dict['ref_table']
    zonemask_table_1d = config_dict['zonemask_table_1d']
    zonemask_table_2d = config_dict['zonemask_table_2d']

    klipsub_cube = np.zeros((N_op_fr, fr_shape[0], fr_shape[1]))
    klippsf_cube = klipsub_cube.copy()
    derot_klipsub_cube = klipsub_cube.copy()

    #if conn == None:
    if result_queue == None:
        start_time = time.time()
    else:
        print current_process().name, 'began'
    for i, fr_ind in enumerate(op_fr):
        # Loop over operand frames
        if fr_ind%diagnos_stride == 0:
            klbasis_cube = np.zeros((max(mode_cut), fr_shape[0], fr_shape[1]))
        for rad_ind in op_rad:
            for az_ind in op_az[rad_ind]:
                I = np.ravel(data_cube[fr_ind,:,:])[ zonemask_table_1d[fr_ind][rad_ind][az_ind] ].copy() 
                R = np.zeros((ref_table[fr_ind][rad_ind].shape[0], zonemask_table_1d[fr_ind][rad_ind][az_ind].shape[0]))
                for j, ref_fr_ind in enumerate(ref_table[fr_ind][rad_ind]):
                    R[j,:] = np.ravel(data_cube[ref_fr_ind,:,:])[ zonemask_table_1d[fr_ind][rad_ind][az_ind] ].copy()
                if use_svd == False: # following Soummer et al. 2012
                    if mean_sub == True:
                        #I_mean = np.mean(I)
                        #I -= I_mean
                        #R -= R.mean(axis=1).reshape(-1, 1)
                        I_mean = R.mean(axis = 0)
                        I -= R.mean(axis = 0)
                        R -= R.mean(axis = 0)
                    Z, sv, N_modes = get_klip_basis(R = R, cutoff = mode_cut[rad_ind])
                else:
                    if mean_sub == True:
                        I_mean = R.mean(axis = 0)
                        I -= R.mean(axis = 0)
                        R -= R.mean(axis = 0)
                        #I_mean = I.mean()
                        #I -= I_mean
                        #R -= R.mean(axis=1).reshape(-1, 1)
                    Z, sv, N_modes = get_pca_basis(R = R, cutoff = mode_cut[rad_ind])
                #if fr_ind % diagnos_stride == 0:
                #    print "Frame %d/%d, annulus %d/%d, sector %d/%d:" %\
                #          (fr_ind+1, N_fr, rad_ind+1, N_rad, az_ind+1, N_az[rad_ind])
                #    print "\tForming PSF estimate..."
                Projmat = np.dot(Z.T, Z)
                I_proj = np.dot(I, Projmat)
                if disable_sub:
                    if mean_sub:
                        F = I + I_mean
                    else:
                        F = I
                else:
                    F = I - I_proj
                klipsub_cube[i,:,:] += reconst_zone(F, zonemask_table_2d[fr_ind][rad_ind][az_ind], fr_shape)
                if mean_sub:
                    klippsf_cube[i,:,:] += reconst_zone(I_proj + I_mean, zonemask_table_2d[fr_ind][rad_ind][az_ind], fr_shape)
                else:
                    klippsf_cube[i,:,:] += reconst_zone(I_proj, zonemask_table_2d[fr_ind][rad_ind][az_ind], fr_shape)
                if store_archv:
                    result_dict[fr_ind][rad_ind][az_ind]['I'] = I
                    result_dict[fr_ind][rad_ind][az_ind]['I_mean'] = I_mean
                    result_dict[fr_ind][rad_ind][az_ind]['Z'] = Z
                    result_dict[fr_ind][rad_ind][az_ind]['sv'] = sv
                    #result_dict[fr_ind][rad_ind][az_ind]['Projmat'] = Projmat
                    result_dict[fr_ind][rad_ind][az_ind]['I_proj'] = I_proj
                    result_dict[fr_ind][rad_ind][az_ind]['F'] = F
                if fr_ind % diagnos_stride == 0:
                    klbasis_cube[:N_modes,:,:] += reconst_zone_cube(Z, zonemask_table_2d[fr_ind][rad_ind][az_ind],
                                                                    cube_dim = (N_modes, fr_shape[0], fr_shape[1]))
                    if mean_sub == False:
                        I_mean = 0
                    print "Frame %d, annulus %d/%d, sector %d/%d: RMS before/after sub: %0.2f / %0.2f" %\
                          (fr_ind+1, rad_ind+1, len(op_rad), az_ind+1, len(op_az[rad_ind]),\
                           np.sqrt(np.mean((I + I_mean)**2)), np.sqrt(np.mean(F**2)))
        # De-rotate the KLIP-subtracted image
        derot_klipsub_img = rotate(klipsub_cube[i,:,:], -parang_seq[fr_ind], reshape=False)
        derot_klipsub_cube[i,:,:] = derot_klipsub_img
        if fr_ind % diagnos_stride == 0:
            print "***** Frame %d has been PSF-sub'd and derotated. *****" % (fr_ind+1)
            if store_klbasis == True:
                klbasis_cube_hdu = pyfits.PrimaryHDU(klbasis_cube.astype(np.float32))
                klbasis_cube_hdu.writeto("%s/klbasis_fr%03d.fits" % (result_dir, fr_ind), clobber=True)
    #if conn == None:
    if result_queue == None:
        end_time = time.time()
        exec_time = end_time - start_time
        time_per_frame = exec_time/N_op_fr
        print "Took %dm%02ds to KLIP-subtract %d frames (%0.2f s per frame).\n" %\
              (int(exec_time/60.), exec_time - 60*int(exec_time/60.), N_op_fr, time_per_frame)
    else:
        result_queue.put([proc_ind, klipsub_cube, klippsf_cube, derot_klipsub_cube, result_dict])
    #    conn.send([proc_ind, klipsub_cube, klippsf_cube, derot_klipsub_cube, result_dict])
    #    #conn.close()
        print current_process().name, 'ended'
    return klipsub_cube, klippsf_cube, derot_klipsub_cube

def get_pca_basis(R, cutoff):
    U, sv, Vt = np.linalg.svd(R, full_matrices=False)
    N_modes = min([cutoff, Vt.shape[0]])
    return Vt[0:cutoff, :], sv, N_modes

def get_klip_basis(R, cutoff):
    #np.linalg.cholesky(np.dot(R, np.transpose(R)))
    w, V = np.linalg.eig(np.dot(R, np.transpose(R)))
    sort_ind = np.argsort(w)[::-1] #indices of eigenvals sorted in descending order
    sv = np.sqrt(w[sort_ind]).reshape(-1,1) #column of ranked singular values
    Z = np.dot(1./sv*np.transpose(V[:, sort_ind]), R)
    #for i in range(w.shape[0]):
    #    if w[i] < 0:
    #        print "negative eigenval: w[%d] = %g" % (i, w[i])
    #        #pdb.set_trace()
    N_modes = min([cutoff, Z.shape[0]])
    return Z[0:N_modes, :], sv, N_modes

def get_residual_stats(config_dict, Phi_0, coadd_img, med_img, xycent=None):
    if xycent == None:
        xycent = ((fr_width - 1)/2., (fr_width - 1)/2.)
    fr_shape = config_dict['fr_shape']
    parang_seq = config_dict['parang_seq']
    op_rad = config_dict['op_rad']
    op_az = config_dict['op_az']
    rad_vec = np.sqrt(get_radius_sqrd(fr_shape, xycent)).ravel()
    
    Phi_0_derot = (Phi_0 + parang_seq[0]) % 360.
    coadd_annular_rms = list()
    zonal_rms = [[None]*N_az[r] for r in range(N_rad)]
    print "RMS counts in KLIP results:"
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
        print "\tannulus %d/%d: %.3f in KLIP sub'd, derotated, coadded annlus" % (rad_ind+1, len(op_rad), coadd_annular_rms[-1])
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
            print "\tby zone: %s" % delimiter.join(["%.3f" % zonal_rms[rad_ind][a] for a in op_az[rad_ind]])
    print "Peak, min values in final co-added image: %0.3f, %0.3f" % (np.nanmax(coadd_img), np.nanmin(coadd_img))
    print "Peak, min values in median of de-rotated images: %0.3f, %0.3f" % (np.nanmax(med_img), np.nanmin(med_img))
    return coadd_annular_rms, zonal_rms

if __name__ == "__main__":
    #
    # Set PCA parameters
    #
    use_svd = True
    coadd_full_overlap_only = True
    mean_sub = True
    track_mode = False
    #
    # Set additional program parameters
    #
    store_results = True
    store_archv = True
    diagnos_stride = 50
    N_proc = 10
    #
    # point PCA search zone config
    #
    mode_cut = [10]
    #mode_cut = [10]
    R_inner = 220.
    R_out = [260.]
    #R_inner = 110.
    #R_out = [130.]
    DPhi = [50.]
    #DPhi = [50.]
    #R_out = [130.]
    #DPhi = [90.]
    Phi_0 = [53.]
    #
    # global PCA search zone config
    #
    #track_mode = False
    #mode_cut = [1]*1
    #R_inner = 200.
    #R_out = [249.]
    #DPhi = [360.]*1
    #Phi_0 = [0.]*1

    N_rad = len(R_out)
    #fwhm = 2.
    fwhm = 4.
    min_refgap_fac = [2.0]
    assert(len(mode_cut) == N_rad == len(DPhi) == len(Phi_0))
    N_az = [ int(np.ceil(360./DPhi[r])) for r in range(N_rad) ]
    #
    # Load data
    #
    #dataset_label = 'gj504_longL_URnods'
    #dataset_label = 'gj504_longL_sepcanon_rebin2x2'
    dataset_label = 'gj504_longL_sepcanon'
    #dataset_label = 'gj504_longL_nfrcomb50'
    data_dir = os.path.expanduser('/disk1/zimmerman/GJ504/apr21_longL/reduc')
    result_dir = os.path.expanduser('/disk1/zimmerman/GJ504/apr21_longL/klipsub_results')
    assert(os.path.exists(data_dir)), 'data_dir %s does not exist' % data_dir
    assert(os.path.exists(result_dir)), 'result_dir %s does not exist' % result_dir
    cube_fname = '%s/%s_cube.fits' % (data_dir, dataset_label)
    cropped_cube_fname = '%s/%s_cropped_cube.fits' % (data_dir, dataset_label)
    if os.path.exists(cropped_cube_fname):
        print "Loading existing centered, cropped data cube %s..." % cropped_cube_fname
        cube_hdulist = pyfits.open(cropped_cube_fname, 'readonly')
        data_cube = cube_hdulist[0].data
        cube_hdulist.close()
        assert(data_cube.shape[1] == 2*R_out[-1])
    else:
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
    print "The LMIRcam ADI sequence has been cropped to width %d pixels." % fr_width
    print "%d images with parallactic angle range %0.2f to %0.2f deg" % (N_fr, parang_seq[0], parang_seq[-1])
    op_fr = np.arange(N_fr)
    #op_fr = np.arange(0, N_fr, diagnos_stride)
    op_rad = range(N_rad)
    #op_az = [range(N_az[i]) for i in range(N_rad)]
    op_az = [[0]]
    #op_az = [[0, 6]]
    assert(len(op_rad) == len(op_az) == N_rad)
    #
    # Form a pixel mask for each search zone, and assemble the masks into two tables (1-D and 2-D formats).
    #
    ref_table, zonemask_table_1d, zonemask_table_2d = get_ref_and_pix_tables(xycent=None, fr_shape=fr_shape, N_fr=N_fr,
                                                                             op_fr=op_fr, N_rad=N_rad, R_inner=R_inner, R_out=R_out,
                                                                             op_rad=op_rad, N_az=N_az, op_az=op_az,
                                                                             parang_seq=parang_seq, fwhm=fwhm,
                                                                             min_refgap_fac=min_refgap_fac, track_mode=track_mode,
                                                                             diagnos_stride=diagnos_stride)
    # 
    # Perform zone-by-zone KLIP subtraction on each frame
    #
    klip_config = {'fr_shape':fr_shape, 'parang_seq':parang_seq, 'mode_cut':mode_cut,
                   'track_mode':track_mode, 'op_fr':op_fr, 'op_rad':op_rad, 'op_az':op_az,
                   'ref_table':ref_table, 'zonemask_table_1d':zonemask_table_1d,
                   'zonemask_table_2d':zonemask_table_2d}
    klip_data = [[[dict.fromkeys(['I', 'I_mean', 'Z', 'sv', 'Projmat', 'I_proj', 'F']) for a in range(N_az[r])] for r in range(N_rad)] for i in range(N_fr)]
    print "Using %d of the %d logical processors available" % (N_proc, multiprocessing.cpu_count())
    klipsub_cube, klippsf_cube, derot_klipsub_cube = do_mp_klip_subtraction(N_proc = N_proc, data_cube=data_cube, config_dict=klip_config,
                                                                            result_dict=klip_data, result_dir=result_dir, diagnos_stride=diagnos_stride,
                                                                            store_klbasis=False, use_svd=use_svd, mean_sub=mean_sub)
    #klipsub_cube, klippsf_cube, derot_klipsub_cube = do_klip_subtraction(data_cube=data_cube, config_dict=klip_config,
    #                                                                     result_dict=klip_data, result_dir=result_dir,
    #                                                                     diagnos_stride=diagnos_stride, store_klbasis=False, use_svd=use_svd, mean_sub=mean_sub)
    #
    # Form mean and median of derotated residual images, and the mean and median of the PSF estimates.
    #
    coadd_img = nanmean(derot_klipsub_cube, axis=0)
    med_img = nanmedian(derot_klipsub_cube, axis=0)
    mean_klippsf_img = nanmean(klippsf_cube, axis=0)
    med_klippsf_img = nanmedian(klippsf_cube, axis=0)
    if coadd_full_overlap_only:
        sum_collapse_img = np.sum(derot_klipsub_cube, axis=0)
        exclude_ind = np.isnan(sum_collapse_img)
        coadd_img[exclude_ind] = np.nan
        med_img[exclude_ind] = np.nan
    coadd_rebin2x2_img = coadd_img.reshape(coadd_img.shape[0]/2, 2, coadd_img.shape[1]/2, 2).mean(1).mean(2)
    #
    # Get statistics from co-added and median residual images
    #
    annular_rms, zonal_rms = get_residual_stats(config_dict=klip_config, Phi_0=Phi_0,
                                                coadd_img=coadd_img, med_img=med_img)
    if store_results == True:
        #
        # Store the results
        #
        delimiter = '-'
        result_label = "%s_srcklip_rad%s_dphi%s_mode%s" % (dataset_label, delimiter.join(["%02d" % r for r in R_out]), delimiter.join(["%02d" % dp for dp in DPhi]), delimiter.join(["%03d" % m for m in mode_cut]))
        klipsub_cube_fname = "%s/%s_res_cube.fits" % (result_dir, result_label)
        klippsf_cube_fname = "%s/%s_psf_cube.fits" % (result_dir, result_label)
        derot_klipsub_cube_fname = "%s/%s_derot_res_cube.fits" % (result_dir, result_label)
        coadd_img_fname = "%s/%s_res_coadd.fits" % (result_dir, result_label)
        coadd_rebin2x2_img_fname = "%s/%s_res_coadd_rebin2x2.fits" % (result_dir, result_label)
        med_img_fname = "%s/%s_res_med.fits" % (result_dir, result_label)
        mean_klippsf_img_fname = "%s/%s_psf_mean.fits" % (result_dir, result_label)
        med_klippsf_img_fname = "%s/%s_psf_med.fits" % (result_dir, result_label)
        klipsub_archv_fname = "%s/%s_klipsub_archv.pkl" % (result_dir, result_label)

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
        if os.path.exists(klipsub_archv_fname):
            os.remove(klipsub_archv_fname)
        if store_archv:
            klipsub_archv = open(klipsub_archv_fname, 'wb') 
            pickle.dump((klip_config, klip_data), klipsub_archv, protocol=2)
            klipsub_archv.close()
            print "Wrote KLIP reduction (%.3f Mb) archive to %s" % (os.stat(klipsub_archv_fname).st_size/10.**6, klipsub_archv_fname)
