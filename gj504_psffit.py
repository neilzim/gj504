#! /usr/bin/env python
"""

Implement KLIP point source forward model on LMIRcam kappa And ADI data.

"""
import numpy as np
import time as time
import pyfits
from scipy.ndimage.interpolation import *
from scipy.interpolate import *
import multiprocessing
import sys
import os
import matplotlib.pyplot as plt
import matplotlib

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

def lnprob_simple_psf_model(p, src_img, psf_img, fit_ind, exc_ind):
    #if p[0] > 0 and p[0] < 2 and abs(p[1]) < (src_img.shape[0] - 1.)/2. and abs(p[2]) < (src_img.shape[0] - 1)/2.:
    #    cost = eval_simple_psf_model(p = p, src_img = src_img, psf_img = psf_img, fit_ind = fit_ind, exc_ind = exc_ind)
    #    lnprob = -cost/2.

    # The walkers go crazy without strong priors on position.
    if p[0] > 0 and 120. < p[1] < 140. and 185. < p[2] < 205.: 
        cost = eval_simple_psf_model(p = p, src_img = src_img, psf_img = psf_img, fit_ind = fit_ind, exc_ind = exc_ind)
        lnprob = -cost/2.
    else:
        lnprob = np.finfo(np.float).min
    return lnprob

def eval_simple_psf_model(p, src_img, psf_img, fit_ind, exc_ind, res_img_fname=None, model_img_fname=None):
    amp = p[0]
    src_cent_xy = ((src_img.shape[0] - 1)/2., (src_img.shape[0] - 1)/2.)
    psf_cent_xy = ((psf_img.shape[0] - 1)/2., (psf_img.shape[0] - 1)/2.)
    pos_xy = (src_cent_xy[0] + p[1], src_cent_xy[1] + p[2])
    model_img = superpose_srcmodel(data_img = np.zeros(src_img.shape), srcmodel_img = amp*psf_img,
                                   srcmodel_destxy = pos_xy, srcmodel_centxy = psf_cent_xy)
    res_img = np.zeros(src_img.shape)
    res_img[fit_ind] = model_img[fit_ind] - src_img[fit_ind]

    if res_img_fname:
        res_img[exc_ind] = np.nan
        res_img_hdu = pyfits.PrimaryHDU(res_img.astype(np.float32)) 
        res_img_hdu.writeto(res_img_fname, clobber=True)
        print 'Wrote residual image to %s' % res_img_fname
    if model_img_fname:
        model_img[exc_ind] = np.nan
        model_img_hdu = pyfits.PrimaryHDU(model_img.astype(np.float32)) 
        model_img_hdu.writeto(model_img_fname, clobber=True)
        print 'Wrote model image to %s' % model_img_fname

    return np.sum(res_img[fit_ind]**2)

def psffit(ampguess, posguess_rho, posguess_theta, src_img_fname, psffit_result_dir,
           template_img_fname, synthpsf_fname, synthpsf_rolloff, result_label,
           do_MLE=True, do_MCMC=False):

    guess_srcmodel_fname = '%s/%s_psffit_guess_src_img.fits' % (psffit_result_dir, result_label)
    guess_resimg_fname = '%s/%s_psffit_guess_res_img.fits' % (psffit_result_dir, result_label)
    final_srcmodel_fname = '%s/%s_psffit_final_src_img.fits' % (psffit_result_dir, result_label)
    final_resimg_fname = '%s/%s_psffit_final_res_img.fits' % (psffit_result_dir, result_label)

    #template_img_hdulist = pyfits.open(template_img_fname, 'readonly')
    #template_img = template_img_hdulist[0].data
    #template_img_hdulist.close()
    #exclude_ind = np.isnan(template_img)

    src_img_hdulist = pyfits.open(src_img_fname, 'readonly')
    src_img = src_img_hdulist[0].data
    src_img_hdulist.close()
    fit_ind = np.isfinite(src_img)
    exc_ind = np.isnan(src_img)

    fr_shape = src_img.shape
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
    print 'Sum of synthetic PSF before / after rolloff and cropping: %0.3f, %0.3f' % (synthpsf_img.sum(), crop_synthpsf_img.sum())

    cent_xy = ((fr_shape[1] - 1)/2., (fr_shape[0] - 1)/2.)
    posguess_xy = ( cent_xy[0] + posguess_rho*np.cos(np.deg2rad(posguess_theta + 90)),\
                    cent_xy[1] + posguess_rho*np.sin(np.deg2rad(posguess_theta + 90)) )
    posguess_deltaxy = (posguess_xy[0] - cent_xy[0], posguess_xy[1] - cent_xy[1])
    p0 = np.array([ampguess, posguess_deltaxy[0], posguess_deltaxy[1]])
    p_min = [ampguess*0.2, posguess_deltaxy[0] - 10., posguess_deltaxy[1] - 10.]
    p_max = [ampguess*5., posguess_deltaxy[0] + 10., posguess_deltaxy[1] + 10.]
    p_bounds = [(p_min[i], p_max[i]) for i in range(len(p_min))]
    print "p0:", p0

    guess_cost = eval_simple_psf_model(p = p0, src_img = src_img, psf_img = crop_synthpsf_img, fit_ind = fit_ind,
                                       exc_ind = exc_ind, model_img_fname = guess_srcmodel_fname)
    print "Cost of guess model = ", guess_cost

    if do_MLE:
        from scipy.optimize import fmin_l_bfgs_b
        start_time = time.time()
        p_sol, final_cost, info = fmin_l_bfgs_b(func = eval_simple_psf_model, x0 = p0,
                                                args = (src_img, crop_synthpsf_img, fit_ind, exc_ind),
                                                approx_grad = True, bounds = p_bounds, factr=1e7, maxfun=100, disp=2)
        end_time = time.time()
        exec_time = end_time - start_time
        print "p_sol:", p_sol
        print "Took %dm%02ds to optimize the source model" % (int(exec_time/60.), exec_time - 60*int(exec_time/60.))
        
        final_cost = eval_simple_psf_model(p = p_sol, src_img = src_img, psf_img = crop_synthpsf_img, fit_ind = fit_ind,
                                           exc_ind = exc_ind, res_img_fname = final_resimg_fname, model_img_fname = final_srcmodel_fname)
    else:
        p_sol = p0
    if do_MCMC:
        #
        # Use MCMC to sample the posterior probability distribution of flux and position
        #
        import emcee
        import shelve
        print ""
        print "Starting the MCMC calculation..."
        start_time = time.time()
        N_dim = 3
        #N_iter = 5000
        #N_burn = 1000
        #N_iter = 3000
        #N_burn = 1000
        #N_walkers = 50
        N_burn = 200
        N_iter = 200
        N_walkers = 500
        p_init = np.transpose( np.vstack(( p_sol[0] + 0.2 * (0.5 - np.random.rand(N_walkers)),
                                           p_sol[1] + 1 * (0.5 - np.random.rand(N_walkers)),
                                           p_sol[2] + 1 * (0.5 - np.random.rand(N_walkers)) )) )
#                                           p_sol[1] + 2 * (0.5 - np.random.rand(N_walkers)),
#                                           p_sol[2] + 2 * (0.5 - np.random.rand(N_walkers)) )) )
#                                           p_sol[1] + 0.1 * (0.5 - np.random.rand(N_walkers)),
#                                           p_sol[2] + 0.1 * (0.5 - np.random.rand(N_walkers)) )) )
        sampler = emcee.EnsembleSampler(N_walkers, N_dim, lnprob_simple_psf_model,
                                        args=[src_img, crop_synthpsf_img, fit_ind, exc_ind], threads=1)
        pos, prob, state = sampler.run_mcmc(p_init, N_burn)
        sampler.reset()
        sampler.run_mcmc(pos, N_iter)
        end_time = time.time()
        exec_time = end_time - start_time
        print "Took %02dh%02dm%02ds to run MCMC over the flux and position parameter space" %\
              (int(exec_time/60./60.), int((exec_time - 60*60*int(exec_time/60./60.))/60.), exec_time - 60*int(exec_time/60.))
        #T_struct = time.localtime()
        #mcmc_archv_fname = "%s/%s_MCMC_archive_Nwalk%03d_Niter%03d.shelve" %\
        #                   (psffit_result_dir, result_label, N_walkers, N_iter)
        #mcmc_archv_fname = "%s/%s_MCMC_archive_Nwalk%03d_Niter%03d_%04d-%02d-%02d_%02dh-%02dm.shelve" %\
        #                   (psffit_result_dir, result_label, N_walkers, N_iter,
        #                    T_struct.tm_year, T_struct.tm_mon, T_struct.tm_mday,
        #                    T_struct.tm_hour, T_struct.tm_min)
        #mcmc_archv = shelve.open(mcmc_archv_fname)
        #mcmc_archv['p_init'] = p_init
        #mcmc_archv['N_dim'] = N_dim
        #mcmc_archv['N_iter'] = N_iter
        #mcmc_archv['N_burn'] = N_burn
        #mcmc_archv['N_walkers'] = N_walkers
        #mcmc_archv['sampler'] = sampler
        #mcmc_archv.close()
        #print "Stored the MCMC results in %s" % mcmc_archv_fname

        #plt.hist(sampler.flatchain[:,0], bins=100, range=[0,1], normed=True)
        #plt.show()
        #plt.hist(sampler.flatchain[:,1])
        #plt.show()
        #plt.hist(sampler.flatchain[:,2])
        #plt.show()
        return p_sol, sampler
    else:
        return p_sol

if __name__ == "__main__":
    data_dir = '/disk1/zimmerman/GJ504/apr21_longL'
    klipsub_result_dir = '%s/klipsub_results' % data_dir
    klipmod_result_dir = '%s/klipmod_results' % data_dir
    template_img_fname = '%s/gj504_longL_sepcanon_srcklip_rad260_dphi90_mode500_res_coadd.fits' % klipsub_result_dir
    #synthpsf_fname = '%s/reduc/psf_model.fits' % data_dir
    synthpsf_fname = '%s/reduc/gj504_longL_psf_model.fits' % data_dir
    mode_cut = 0
    N_proc = 10
    synthpsf_rolloff = 20.

    # October canonical reduction; bottom left nod position only
    #print ''

    # October canonical reduction; top right nod position only
    #print ''

    # October canonical reduction; full data set
    print ''
    print 'Fitting PSF in KLIP residual from October reduction'
    result_label = 'gj504_longL_octcanon'
    res_img_fname = "%s/gj504_longL_octcanon_srcklip_rad255_dphi90_mode000_res_coadd.fits" % klipsub_result_dir 
    psol_octcanon = psffit(ampguess=0.7, posguess_rho=238., posguess_theta=-33., src_img_fname=res_img_fname,
                                             psffit_result_dir=klipmod_result_dir, template_img_fname=template_img_fname,
                                             synthpsf_fname=synthpsf_fname, synthpsf_rolloff=synthpsf_rolloff, result_label=result_label,
                                             do_MLE=True, do_MCMC=False)
#    amp_flatchain = sampler_octcanon.flatchain[:,0]
