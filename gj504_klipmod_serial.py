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
import matplotlib.pyplot as plt
import matplotlib

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
    def __init__(self, fr_ind, adiklip_config, zonemask_1d, Z, F, srcmodel, src_amp, src_abframe_xy):
        self.fr_ind = fr_ind
        self.adiklip_config = adiklip_config
        self.zonemask_1d = zonemask_1d
        self.Z = Z
        self.F = F
        self.srcmodel = srcmodel
        self.src_amp = src_amp
        self.src_abframe_xy = src_abframe_xy
    def __call__(self):
        fr_shape = self.adiklip_config['fr_shape']
        srcmodel_cent_xy = ((self.srcmodel.shape[1] - 1)/2., (self.srcmodel.shape[0] - 1)/2.)
        if self.src_abframe_xy[0] >= fr_shape[1] or self.src_abframe_xy[1] >= fr_shape[0] or min(self.src_abframe_xy) < 0:
            print 'bad dest:', self.src_abframe_xy
            return np.finfo(np.float).max 
        abframe_synthsrc_img = superpose_srcmodel(data_img = np.zeros(fr_shape), srcmodel_img = self.src_amp*self.srcmodel,
                                                  srcmodel_destxy = self.src_abframe_xy, srcmodel_centxy = srcmodel_cent_xy)
        Pmod = np.ravel(abframe_synthsrc_img)[ self.zonemask_1d ].copy()
        if self.Z:
            Projmat = np.dot(self.Z.T, self.Z)
            Pmod_proj = np.dot(Pmod, Projmat)
            res_vec = self.F - Pmod + Pmod_proj
        else:
            res_vec = self.F - Pmod
        return (self.fr_ind, res_vec, np.sum(res_vec**2))
    def __str__(self):
        return 'frame %d' % (self.fr_ind+1)

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

def reconst_zone(data_vec, pix_table, img_dim):
    reconstrd_img = np.zeros(img_dim)
    for i, pix_val in enumerate(data_vec.flat):
        row = pix_table[0][i] 
        col = pix_table[1][i]
        reconstrd_img[row, col] = pix_val
    return reconstrd_img

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
            eval_tasks.put( eval_adiklip_srcmodel_task(fr_ind, adiklip_config, adiklip_config['zonemask_table_1d'][fr_ind][rad_ind][az_ind],
                                                       adiklip_data[fr_ind][rad_ind][az_ind]['Z'][:mode_cut,:], adiklip_data[fr_ind][rad_ind][az_ind]['F'],
                                                       srcmodel, amp, abframe_xy_seq[i]) )
        else:
            eval_tasks.put( eval_adiklip_srcmodel_task(fr_ind, adiklip_config, adiklip_config['zonemask_table_1d'][fr_ind][rad_ind][az_ind],
                                                       None, adiklip_data[fr_ind][rad_ind][az_ind]['F'], srcmodel, amp, abframe_xy_seq[i]) )
            
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

#    mp_op_fr = list()
#    for i in range(N_proc):
#        chunk_size = round(float(N_op_fr) / N_proc)
#        fr_ind_beg = i * chunk_size
#        if i != N_proc - 1:
#            fr_ind_end = fr_ind_beg + chunk_size
#        else:
#            fr_ind_end = N_op_fr
#        mp_op_fr.append(op_fr[fr_ind_beg:fr_ind_end])
#
#    for i in range(N_proc):
#        task = Process(target = eval_adiklip_srcmodel, args = (p, mp_op_fr[i], adiklip_config,
#                                                               adiklip_data, srcmodel, cost_queue))
#        task.start()
#
#    for i in range(N_proc):
#        total_sumofsq_cost += cost_queue.get()
#
#    return total_sumofsq_cost

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
     

if __name__ == "__main__":
    data_dir = '/disk1/zimmerman/GJ504/apr21_longL'
    klipsub_result_dir = '%s/klipsub_results' % data_dir
    klipmod_result_dir = '%s/klipmod_results' % data_dir
    template_img_fname = '%s/gj504_longL_sepcanon_srcklip_rad260_dphi90_mode500_res_coadd.fits' % klipsub_result_dir
    synthpsf_fname = '%s/reduc/psf_model.fits' % data_dir
    mode_cut = 0
    N_proc = 1
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
    result_label = 'gj504_longL_octcanon_modecut%03d' % mode_cut
    #klipsub_archv_fname =   "%s/gj504_longL_octcanon_srcklip_rad260_dphi50_mode010_klipsub_archv.pkl" % klipsub_result_dir
    klipsub_archv_fname =   "%s/gj504_longL_octcanon_srcklip_rad255_dphi50_mode000_klipsub_archv.pkl" % klipsub_result_dir
    psol_octcanon = klipmod(ampguess=0.7, posguess_rho=238., posguess_theta=-33., klipsub_archv_fname=klipsub_archv_fname,
                            klipsub_result_dir=klipsub_result_dir, klipmod_result_dir=klipmod_result_dir,
                            template_img_fname=template_img_fname, synthpsf_fname=synthpsf_fname,
                            synthpsf_rolloff=synthpsf_rolloff, result_label=result_label, N_proc=N_proc,
                            mode_cut=mode_cut, do_MLE=True)
