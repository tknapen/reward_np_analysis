import unittest
import glob
import os.path as op

def fit_glm_nuisances_single_file(
                        in_file, 
                        slice_regressor_list = [], 
                        vol_regressors = '', 
                        num_components = 20, 
                        method = 'ICA', 
                        mapper = 'unpredictable', 
                        dm_upscale_factor = 10,
                        tsv_behavior_file = '',
                        out_folder = ''
                        ):
    """Performs a per-slice GLM on nifti-file in_file, 
    with per-slice regressors from slice_regressor_list of nifti files,
    and per-TR regressors from vol_regressors text file.
    the 'mapper' event definitions are taken from the events tsv file.
    Uses standard HRF and its time-derivative.
    Assumes slices to be the last spatial dimension of nifti files,
    and time to be the last.

    Parameters
    ----------
    in_file : str
        Absolute path to nifti-file.
    slice_regressor_list : list
        list of absolute paths to per-slice regressor nifti files
    vol_regressor_list : str
        absolute path to per-TR regressor text file

    Returns
    -------
    res_file : str
        Absolute path to nifti-file containing residuals after regression.
    rsq_file : str
        Absolute path to nifti-file containing rsq of regression.
    beta_file : str
        Absolute path to nifti-file containing betas from regression.

    """

    import nibabel as nib
    import numpy as np
    import numpy.linalg as LA
    import os
    from sklearn import decomposition
    from scipy.signal import savgol_filter, fftconvolve
    from scipy.stats import t
    from spynoza.nodes.utils import get_scaninfo
    from hrf_estimation.hrf import spmt, dspmt, ddspmt

    func_nii = nib.load(in_file)
    TR, dims, dyns, voxsize, affine = get_scaninfo(in_file)

    kernels = [eval(func + '(np.linspace(0,25, dm_upscale_factor*25/TR, endpoint = False))') for func in ['spmt', 'dspmt', 'ddspmt']]

    # import data and convert nans to numbers
    func_data = np.nan_to_num(func_nii.get_data())

    all_slice_reg = np.zeros((len(slice_regressor_list),dims[-2],dims[-1]))
    # fill the regressor array from files
    for i in range(len(slice_regressor_list)):
        all_slice_reg[i] = nib.load(slice_regressor_list[i]).get_data().squeeze()

    if vol_regressors != '':
        all_TR_reg = np.loadtxt(vol_regressors)
        if all_TR_reg.shape[-1] != all_slice_reg.shape[-1]: # check for the right format
            all_TR_reg = all_TR_reg.T

    if mapper == 'unpredictable':
        # design implemented in TRs
        reg_center = np.r_[np.zeros(4), np.tile(np.concatenate((np.ones(8), np.zeros(24))), 8)]
        reg_surround = np.roll(reg_center, 16)

        # blow up design matrix by dm_upscale_factor x TR, assumes dm_upscale_factor x TR is integer...
        reg_center = np.repeat(reg_center, int(dm_upscale_factor))
        reg_surround = np.repeat(reg_surround, int(dm_upscale_factor))

        # convolve and go back to TR-based time-resolution
        reg_center_cs = np.array([fftconvolve(reg_center, kernel)[::dm_upscale_factor] for kernel in kernels])
        reg_surround_cs = np.array([fftconvolve(reg_surround, kernel)[::dm_upscale_factor] for kernel in kernels])

        reg_center_cs = (reg_center_cs.T/reg_center_cs.std(axis = -1)).T
        reg_surround_cs = (reg_surround_cs.T/reg_surround_cs.std(axis = -1)).T

        # take the relevant timepoints, not the overhang
        reg_center = reg_center_cs[:,:dyns]
        reg_surround = reg_surround_cs[:,:dyns]

        visual_dm = np.vstack((np.ones((1,dyns)), reg_center, reg_surround))
        visual_dm_shape = visual_dm.shape[0]

    else:
        print('Design %s not yet implemented...'%mapper)
        # so, just the intercept is the 'visual_dm'
        visual_dm = np.ones((1,dyns))
        visual_dm_shape = visual_dm.shape[0]

    # data containers
    residual_data = np.zeros_like(func_data)
    rsq_data = np.zeros(list(dims[:-1]))
    nr_regressors = visual_dm_shape # there's always an intercept, and the visual design is also first.
    # the number of physio nuisance regressors depends on whether they exist, 
    # and on how many components should be selected at PCA/ICA time
    if num_components == 0:             
        nr_regressors += len(slice_regressor_list)
        # if there are moco regressors (per volume), these regressors should be added to the number of regressors,
        # but only if the num_components is 0.
        if vol_regressors != '':
            nr_regressors += all_TR_reg.shape[0]
    else:
        nr_regressors += num_components    

    beta_data = np.zeros(list(dims[:-1]) + [nr_regressors])
    T_data = np.zeros(list(dims[:-1]) + [nr_regressors])
    p_data = np.zeros(list(dims[:-1]) + [nr_regressors])

    # loop over slices
    for x in range(dims[-2]):
        slice_data = func_data[:,:,x,:].reshape((-1,dims[-1]))
        # demean data
        slice_data = np.nan_to_num((slice_data.T/slice_data.mean(axis = 1)).T)
        # fix the design matrix
        nuisance_regressors = all_slice_reg[:,x,:]
        if vol_regressors != '':
            nuisance_regressors = np.vstack((nuisance_regressors, all_TR_reg))
        nuisance_regressors = np.nan_to_num(nuisance_regressors)

        if num_components != 0:
            if method == 'PCA':
                pca = decomposition.PCA(n_components = num_components, whiten = True)
                nuisance_regressors = pca.fit_transform(nuisance_regressors.T).T
            elif method == 'ICA':
                ica = decomposition.FastICA(n_components = num_components, whiten = True)
                nuisance_regressors = ica.fit_transform(nuisance_regressors.T).T

        # normalize regressors
        nuisance_regressors = (nuisance_regressors.T/nuisance_regressors.std(axis = -1)).T

        # create actual design matrix
        all_regressors = np.vstack((visual_dm, nuisance_regressors))

        # fit
        betas, sse, rank, svs = LA.lstsq(all_regressors.T, slice_data.T)

        # predicted data, rsq and residuals
        prediction = np.dot(betas.T, all_regressors)
        rsq = 1.0 - np.sum((prediction - slice_data)**2, axis = -1) / np.sum(slice_data.squeeze()**2, axis = -1)
        residuals = slice_data - prediction

        # and do stats
        design_matrix_rank = np.linalg.matrix_rank(all_regressors)
        df = residuals.shape[-1] - design_matrix_rank

        contrasts = np.matrix(np.eye(all_regressors.shape[0]))
        contrasts_in_dm = [np.array(contrast * np.linalg.pinv(np.dot(all_regressors, all_regressors.T)) * contrast.T).squeeze() for contrast in contrasts]

        standard_errors = [np.sqrt((sse/df) * contrast_in_dm) for contrast_in_dm in contrasts_in_dm]
        T_stats = np.array([np.squeeze(np.array(np.dot(contrast, betas) / standard_error)) for contrast, standard_error in zip(contrasts, standard_errors)])

        p_vals = -np.log10(np.array([np.squeeze([t.cdf(-np.abs(ts), df) for ts in T_stat]) for T_stat in T_stats]))

        # reshape and save
        residual_data[:,:,x,:] = residuals.T.reshape((dims[0], dims[1], dims[-1]))
        rsq_data[:,:,x] = rsq.reshape((dims[0], dims[1]))
        beta_data[:,:,x,:] = betas.T.reshape((dims[0], dims[1],all_regressors.shape[0]))
        p_data[:,:,x,:] = p_vals.T.reshape((dims[0], dims[1],all_regressors.shape[0]))
        T_data[:,:,x,:] = T_stats.T.reshape((dims[0], dims[1],all_regressors.shape[0]))

        print("slice %d finished nuisance GLM for %s"%(x, in_file))

    # save files
    residual_img = nib.Nifti1Image(np.nan_to_num(residual_data), affine)
    res_file = os.path.abspath(os.path.join(out_folder, os.path.split(in_file)[1][:-7] + '_res.nii.gz'))
    nib.save(residual_img, res_file)
    
    rsq_img = nib.Nifti1Image(np.nan_to_num(rsq_data), affine)
    rsq_file = os.path.abspath(os.path.join(out_folder, os.path.split(in_file)[1][:-7] + '_rsq.nii.gz'))
    nib.save(rsq_img, rsq_file)

    beta_img = nib.Nifti1Image(np.nan_to_num(beta_data), affine)
    beta_file = os.path.abspath(os.path.join(out_folder, os.path.split(in_file)[1][:-7] + '_betas.nii.gz'))
    nib.save(beta_img, beta_file)

    T_img = nib.Nifti1Image(np.nan_to_num(T_data), affine)
    T_file = os.path.abspath(os.path.join(out_folder, os.path.split(in_file)[1][:-7] + '_T.nii.gz'))
    nib.save(T_img, T_file)

    p_img = nib.Nifti1Image(np.nan_to_num(p_data), affine)
    p_file = os.path.abspath(os.path.join(out_folder, os.path.split(in_file)[1][:-7] + '_logp.nii.gz'))
    nib.save(p_img, p_file)

    # return paths
    return res_file, rsq_file, beta_file, T_file, p_file


def fit_FIR_nuisances_all_files(
                        example_func,
                        in_files, 
                        slice_regressor_lists = [], 
                        vol_regressor_list = '', 
                        behavior_file_list = '', 
                        fir_frequency = 2,
                        fir_interval = [-3.0,15.0],
                        out_folder = ''
                        ):
    """Performs a per-slice FIR deconvolution on nifti-files in_files, 
    with per-slice regressors from slice_regressor_list of nifti files,
    and per-TR regressors from vol_regressors text file.
    Uses standard HRF and its time-derivative.
    Assumes slices to be the last spatial dimension of nifti files,
    and time to be the last.

    Parameters
    ----------
    in_file : str
        Absolute path to nifti-file.
    slice_regressor_list : list
        list of absolute paths to per-slice regressor nifti files
    vol_regressor_list : str
        absolute path to per-TR regressor text file

    Returns
    -------
    res_file : str
        Absolute path to nifti-file containing residuals after regression.
    rsq_file : str
        Absolute path to nifti-file containing rsq of regression.
    beta_file : str
        Absolute path to nifti-file containing betas from regression.

    """

    import nibabel as nib
    import numpy as np
    import numpy.linalg as LA
    import scipy as sp
    import os
    import pandas as pd
    from spynoza.nodes.utils import get_scaninfo
    from fir import FIRDeconvolution

    func_niis = [nib.load(in_file) for in_file in in_files]
    TR, dims, dyns, voxsize, affine = get_scaninfo(in_files[0])

    mask = nib.load(example_func).get_data() > 0 

    # import data and convert nans to numbers
    func_data = np.nan_to_num(np.concatenate([func_nii.get_data().T for func_nii in func_niis]).T)
    
    if slice_regressor_lists != []:
        nr_physio_regressors = len(slice_regressor_lists[0])
        all_slice_reg = np.zeros((nr_physio_regressors,dims[-2],func_data.shape[-1]))
        for x in range(len(slice_regressor_lists)):
            # fill the regressor array from files
            for i in range(len(slice_regressor_lists[x])):
                all_slice_reg[i,:,x*dyns:(x+1)*dyns] = nib.load(slice_regressor_lists[x][i]).get_data().squeeze()

    if vol_regressor_list != []:
        all_vol_regs = []
        nr_vol_regressors = np.loadtxt(vol_regressor_list[0]).shape[-1]
        all_vol_reg = np.zeros((nr_vol_regressors, func_data.shape[-1]))
        for x in range(len(vol_regressor_list)):
            all_vol_reg[:,x*dyns:(x+1)*dyns] = np.loadtxt(vol_regressor_list[x]).T

    event_names = ['visual_reward', 'fixation_reward', 'visual_no_reward', 'fixation_no_reward']
    # get the behavior and format event times and gains for FIR
    all_behav_event_times = {en:[] 
                for en in event_names}

    for x in range(len(behavior_file_list)):
        this_run_events = pd.read_csv(behavior_file_list[x], delimiter='\t')

        reward_events = (this_run_events.reward_time != 0)
        stim_events = (this_run_events.contrast != 0)

        visual_reward_times = np.array(this_run_events.stim_onset_time[reward_events & stim_events]) + x * dyns * TR
        fixation_reward_times = np.array(this_run_events.stim_onset_time[reward_events & -stim_events]) + x * dyns * TR
        visual_no_reward_times = np.array(this_run_events.stim_onset_time[-reward_events & stim_events]) + x * dyns * TR
        fixation_no_reward_times = np.array(this_run_events.stim_onset_time[-reward_events & -stim_events]) + x * dyns * TR

        # these are the times to be saved.
        all_behav_event_times['visual_reward'].append(visual_reward_times)
        all_behav_event_times['fixation_reward'].append(fixation_reward_times)
        all_behav_event_times['visual_no_reward'].append(visual_no_reward_times)
        all_behav_event_times['fixation_no_reward'].append(fixation_no_reward_times)

    # set the times up as a 1-D array
    all_behav_event_times['visual_reward'] = np.concatenate(all_behav_event_times['visual_reward'])
    all_behav_event_times['fixation_reward'] = np.concatenate(all_behav_event_times['fixation_reward'])
    all_behav_event_times['visual_no_reward'] = np.concatenate(all_behav_event_times['visual_no_reward'])
    all_behav_event_times['fixation_no_reward'] = np.concatenate(all_behav_event_times['fixation_no_reward'])

    # data containers
    rsq_data = np.zeros(list(dims[:-1]))
  
    visual_fir_timecourses = np.zeros(list(dims[:-1]) + [int((fir_interval[1] - fir_interval[0]) * fir_frequency)])
    reward_fir_timecourses = np.zeros(list(dims[:-1]) + [int((fir_interval[1] - fir_interval[0]) * fir_frequency)])
    interaction_fir_timecourses = np.zeros(list(dims[:-1]) + [int((fir_interval[1] - fir_interval[0]) * fir_frequency)])

    fir_timecourses = {en:np.zeros(list(dims[:-1]) + [int((fir_interval[1] - fir_interval[0]) * fir_frequency)]) 
                for en in event_names}

    # loop over slices
    for x in range(dims[-2]):
        slice_data = func_data[:,:,x,:]
        slice_data = slice_data[mask[:,:,x]].reshape((-1,func_data.shape[-1]))
                
        # fix the design matrix
        nuisance_regressors = all_slice_reg[:,x,:]
        if vol_regressor_list != '':
            nuisance_regressors = np.vstack((nuisance_regressors, all_vol_reg))
        nuisance_regressors = np.nan_to_num(nuisance_regressors)

        # normalize regressors
        nuisance_regressors = (nuisance_regressors.T/nuisance_regressors.std(axis = -1)).T

        fd = FIRDeconvolution(
            signal = slice_data, 
            events = [all_behav_event_times[en] for en in event_names], # dictate order
            event_names = event_names, 
            sample_frequency = 1.0/1.5,
            deconvolution_frequency = fir_frequency,
            deconvolution_interval = fir_interval
            )

        fd.resampled_signal = np.nan_to_num(fd.resampled_signal)
        # we then tell it to create its design matrix
        fd.create_design_matrix()

        # resample mocos and so forth
        all_nuisances = sp.signal.resample(nuisance_regressors, fd.resampled_signal_size, axis = -1)
        fd.add_continuous_regressors_to_design_matrix(all_nuisances)

        # fit
        fd.regress(method = 'lstsq')

        fd.calculate_rsq()

        for en in event_names:
            fir_timecourses[en][mask[:,:,x],x] = np.nan_to_num(fd.betas_for_cov(en).T)
         # reshape and save
        rsq_data[mask[:,:,x],x] = np.nan_to_num(fd.rsq) # .reshape((dims[0], dims[1]))

        print("slice %d finished nuisance FIR"%x)

    output_files = []
    # save files
    for en in event_names:
        event_fir_img = nib.Nifti1Image(np.nan_to_num(fir_timecourses[en]), affine)
        event_fir_file = os.path.abspath(os.path.join(out_folder, os.path.split(in_files[0])[1][:-7] + '_{0}.nii.gz'.format(en)))
        nib.save(event_fir_img, event_fir_file)
        output_files.append(event_fir_file)
    
    rsq_img = nib.Nifti1Image(np.nan_to_num(rsq_data), affine)
    rsq_file = os.path.abspath(os.path.join(out_folder, os.path.split(in_files[0])[1][:-7] + '_rsq.nii.gz'))
    nib.save(rsq_img, rsq_file)
    output_files.append(rsq_file)

    # return paths
    return tuple(output_files)



class FIRTestCase(unittest.TestCase):
    
    def setUp(self):
        # standard test path and subject
        test_path = '/home/shared/-2014/reward/new/'
        test_sj = 'sub-001'
        self.example_func = op.join(test_path, test_sj, 'reg', 'example_func.nii.gz')
        # input files
        self.in_files = glob.glob(op.join(test_path, test_sj, 'psc/*unpredictable_reward*.nii.gz'))
        self.in_files.sort()
        # physio nuisance files
        self.slice_regressor_lists = [glob.glob(op.join(test_path, test_sj, 'phys/evs/*unpredictable_reward_{0}_*.nii.gz').format(x)) 
                                for x in range(1,len(in_files)+1)]        
        for sl in self.slice_regressor_lists:
            sl.sort()
        # moco nuisance files
        self.vol_regressor_list = glob.glob(op.join(test_path, test_sj, 'mcf/ext_motion_pars/*unpredictable_reward*.par'))
        self.vol_regressor_list.sort()
        # behavior files
        self.behavior_file_list = glob.glob(op.join(test_path, test_sj, 'events/tsv/*unpredictable_reward*.tsv'))
        self.behavior_file_list.sort()

    def runTest(self):
        fit_FIR_nuisances_all_files(
            example_func = self.example_func
            in_files = self.in_files,
            slice_regressor_lists = self.slice_regressor_lists,
            vol_regressor_list = self.vol_regressor_list,
            behavior_file_list = self.behavior_file_list,
            fir_frequency = 4,
            fir_interval = [-3.0,15.0]
            )


if __name__ == '__main__':
    unittest.main()




