from __future__ import division, print_function

def fit_FIR_roi(experiment,
                h5_file,
                in_files,
                vol_regressor_list, 
                behavior_file_list, 
                eye_h5_file_list,
                mapper_file_list,
                mask_index,
                mask_threshold = 2.0,
                mask_direction = 'pos',
                fmri_data_type = 'psc',
                pupil_data_type = '_pupil_bp_zscore',
                fir_frequency = 2,
                fir_interval = [-3.0,15.0],
                roi_list = ['V1','V2','V3']
                ):

    import nibabel as nib
    import numpy as np
    import numpy.linalg as LA
    import scipy as sp
    from sklearn import decomposition
    import os
    import os.path as op
    import pandas as pd
    from spynoza.nodes.utils import get_scaninfo
    from fir import FIRDeconvolution
    from hedfpy import HDFEyeOperator
    import tempfile
    from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit, report_errors, fit_report
    # from utils.behavior import behavior_timing
    # from utils.utils import roi_data_from_hdf

    tr_key = 116.0

    TR, dims, dyns, voxsize, affine = get_scaninfo(in_files[1]) 
    header = nib.load(in_files[1]).header


    ################################################################################## 
    # behavior data generalizes across ROIs of course
    ##################################################################################
    event_names, all_behav_event_times, rename_dict, which_event_name_rewarded = behavior_timing(experiment, behavior_file_list, dyns, TR)

    mapper_alias = op.split(mapper_file_list[0])[-1][:-7] + '_T'

    ################################################################################## 
    # eye-related data generalizes across ROIs of course
    ##################################################################################

    aliases = ['eye/' + op.splitext(op.split(eye_h5_file)[-1])[0] for eye_h5_file in eye_h5_file_list]

    ho = HDFEyeOperator(h5_file)

    # blink_data, pupil_data = [], []

    # for x, alias in enumerate(aliases):
    #     ho.open_hdf_file('a')
    #     node = ho.h5f.get_node('/' + alias)
    #     trial_data = node.trials.read()
    #     blinks = pd.DataFrame(node.blinks_from_message_file.read())
    #     events = node.events.read()
    #     ho.h5f.close()

    #     # find first TR, and use it to define:
    #     # the duration of the datastream and the tr
    #     tr_timestamps = events['EL_timestamp'][(events['key'] == tr_key) & (events['up_down'] == 'Down')][:dyns]

    #     eye = ho.eye_during_trial(0, alias)
    #     sample_rate = ho.sample_rate_during_period([tr_timestamps[0], tr_timestamps[-1] + TR], alias)

    #     raw_data = ho.data_from_time_period([tr_timestamps[0], tr_timestamps[0] + (1000.0 * dyns * TR)], alias)
    #     selected_data = np.array(raw_data[eye + pupil_data_type])

    #     # take blinks
    #     run_blinks = blinks[(blinks.start_timestamp > tr_timestamps[0]) & (blinks.start_timestamp > (1000.0 * dyns * TR))]
    #     run_blinks.start_timestamp = (x * dyns * TR) + ((run_blinks.start_timestamp - tr_timestamps[0]) / 1000.0)
    #     run_blinks.end_timestamp = (x * dyns * TR) + ((run_blinks.end_timestamp - tr_timestamps[0]) / 1000.0)
    #     run_blinks.duration = run_blinks.duration / 1000.0
    #     blink_data.append(run_blinks)

    #     # bad recordings are characterized, among other things, by a very high amount of lost samples. 
    #     # if this lost sample rate crosses a threshold, we will throw the file out completely
    #     raw_pupil = np.array(raw_data[eye + '_pupil'])
    #     lost_signal_rate = (raw_pupil == 0).sum() / (dyns * TR * sample_rate)
    #     run_signal = np.zeros((sample_rate * dyns * TR))
    #     if lost_signal_rate > lost_signal_rate_threshold:
    #         print 'dropping this run, lost signal rate = %1.3f'%lost_signal_rate
    #     else:
    #         run_signal[:selected_data.shape[0]] = selected_data

    #     pupil_data.append(run_signal)

    # pupil_data = np.concatenate(pupil_data)
    # blink_data = pd.concat(blink_data)

    ################################################################################## 
    # whole-brain nuisance data generalizes across ROIs of course
    ##################################################################################

    if vol_regressor_list != []:
        all_vol_regs = []
        nr_vol_regressors = np.loadtxt(vol_regressor_list[0]).shape[-1]
        all_vol_reg = np.zeros((nr_vol_regressors, dyns * len(in_files)))
        for x in range(len(vol_regressor_list)):
            all_vol_reg[:,x*dyns:(x+1)*dyns] = np.loadtxt(vol_regressor_list[x]).T[...,:dyns]

    ################################################################################## 
    # per-roi data
    ##################################################################################


    out_figures = []
    for roi in roi_list:
        contrast_data = roi_data_from_hdf(data_types_wildcards = [mapper_alias], roi_name_wildcard = roi, hdf5_file = h5_file, folder_alias = 'GLM')
        time_course_data = [roi_data_from_hdf(data_types_wildcards = [os.path.split(in_f)[-1][:-7]], roi_name_wildcard = roi, hdf5_file = h5_file, folder_alias = fmri_data_type) for in_f in in_files]

        time_course_data = np.hstack(time_course_data)

        if threshold < 0:
            threshold = -threshold
            contrast_data = -contrast_data

        over_threshold = (contrast_data[:,1]>threshold)
        iceberg = contrast_data[over_threshold, 1]

        projected_time_course = np.dot(time_course_data[over_threshold].T, iceberg) / np.sum(iceberg)
        av_time_course = time_course_data[over_threshold].mean(axis = 0)

        both_time_courses = np.vstack([projected_time_course, av_time_course])

        fir_timecourses = {en:np.zeros([int((fir_interval[1] - fir_interval[0]) * fir_frequency)]) 
                for en in event_names}

        nuisance_regressors = np.nan_to_num(all_vol_reg)

        if num_components != 0:
            if method == 'PCA':
                pca = decomposition.PCA(n_components = num_components, whiten = True)
                nuisance_regressors = pca.fit_transform(nuisance_regressors.T).T
            elif method == 'ICA':
                ica = decomposition.FastICA(n_components = num_components, whiten = True)
                nuisance_regressors = ica.fit_transform(nuisance_regressors.T).T


        fd = FIRDeconvolution(
            signal = projected_time_course, 
            events = [all_behav_event_times[en] for en in event_names], # dictate order
            event_names = event_names, 
            sample_frequency = 1.0/TR,
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
            fir_timecourses[en] = np.squeeze(np.nan_to_num(fd.betas_for_cov(en).T))

        plot_fir_results_unpredictable(fd.deconvolution_interval_timepoints, event_names, fir_time_course_dict = fir_timecourses, zero_interval = [-0.75, -0.25], use_zero_interval = False, suffix = roi+'_roi')


        irf_timepoints = np.arange(0,30,1.0/fir_frequency)
        shape, loc, scale, gain = ( 6.91318195e+03,  -1.85261652e+02,   2.78354064e-02, -2.24467302e+00 )
        rew_irf = sp.stats.gamma.pdf(irf_timepoints, shape, loc = loc, scale = scale)*gain
        stim_irf = hrf.spmt(irf_timepoints)

        stim_events = np.concatenate([all_behav_event_times[en] for en in event_names if 'visual' in en])
        stim_regressor = np.zeros(fd.resampled_signal_size)
        stim_regressor[np.round(stim_events * fir_frequency).astype(int)] = 1.0
        stim_regressor_c = sp.signal.fftconvolve(stim_regressor, stim_irf)[:fd.resampled_signal_size]
        stim_regressor_c -= stim_regressor_c.mean()

        rew_events = np.concatenate([all_behav_event_times[en] for en in event_names])
        rew_events.sort()
        rew_regressor = np.zeros((rew_events.shape[0], fd.resampled_signal_size))
        rew_regressor_c = np.zeros((rew_events.shape[0], fd.resampled_signal_size))
        for i, re in enumerate(rew_events):
            rew_regressor[i, np.round(re * fir_frequency).astype(int)] = 1.0
            rew_regressor_c[i] = sp.signal.fftconvolve(rew_regressor[i], rew_irf, 'full')[:fd.resampled_signal_size]
        rew_regressor_c = (rew_regressor_c.T - rew_regressor_c.mean(axis = 1)).T


        # create actual design matrix
        all_regressors = np.vstack((stim_regressor_c, all_nuisances,np.ones(stim_regressor_c.shape)))

        # fit
        betas, sse, rank, svs = LA.lstsq(all_regressors.T, fd.resampled_signal.T)

        # predicted data, rsq and residuals
        prediction = np.dot(betas.T, all_regressors)
        rsq = 1.0 - np.sum((prediction - fd.resampled_signal)**2, axis = -1) / np.sum(fd.resampled_signal.squeeze()**2, axis = -1)
        residuals = np.squeeze(fd.resampled_signal - prediction)

        figure()
        plot(fd.resampled_signal.T)
        plot(residuals)
        plot(stim_regressor_c)
        plot(stim_regressor)

        # reward design matrix
        all_regressors = np.vstack((rew_regressor_c,np.ones(stim_regressor_c.shape)))

        # fit
        betas, sse, rank, svs = LA.lstsq(all_regressors.T, residuals)

        # predicted data, rsq and residuals
        prediction = np.dot(betas.T, all_regressors)
        rsq = 1.0 - np.sum((prediction - fd.resampled_signal)**2, axis = -1) / np.sum(fd.resampled_signal.squeeze()**2, axis = -1)
        residuals_2 = np.squeeze(residuals - prediction)
        figure()
        plot(residuals)
        plot(residuals_2)
        plot(rew_regressor.sum(axis = 0))
        plot(rew_regressor_c.sum(axis = 0))

        reward_betas = betas[:-1]
        rew_mins = np.argmin(rew_regressor_c, axis = -1).astype(int)
        reward_values = np.array([fd.resampled_signal[0,x] for x in rew_mins])


        timing = np.argsort(np.concatenate([all_behav_event_times[name] for symbol in ['visual_reward','visual_no_reward','fixation_reward','fixation_no_reward']]))
        trial_array = np.concatenate([[symbol for x in range(len(all_behav_event_times[name]))] for symbol in ['SR','SN','FR','FN']])[timing]

      

        phm = Model(PHmodel)


        figure()
        result = phm.fit(reward_betas, trial_array=trial_array, alpha=2, kappa=0.2, pos=0.75, which = 'V', fit_kws={'nan_policy': 'omit'})
        print result.fit_report()
        plot(reward_betas)
        plot(np.array(result.best_fit).T)

        figure()
        result = phm.fit(reward_betas, trial_array=trial_array, alpha=2, kappa=0.2, pos=0.75, which = 'PE', fit_kws={'nan_policy': 'omit'})
        print result.fit_report()
        plot(reward_betas)
        plot(np.array(result.best_fit).T)

        figure()
        result = phm.fit(reward_betas, trial_array=trial_array, alpha=2, kappa=0.2, pos=0.75, which = 'alphas', fit_kws={'nan_policy': 'omit'})
        print result.fit_report()
        plot(reward_betas)
        plot(np.array(result.best_fit).T)        



class FIRROITestCase(unittest.TestCase):
    
    def setUp(self):
        import glob
        import os.path as op
        from utils.plotting import *
        from utils.behavior import behavior_timing
        from utils.utils import roi_data_from_hdf
        import nibabel as nib
        import numpy as np
        import numpy.linalg as LA
        import scipy as sp
        from sklearn import decomposition
        import os
        import os.path as op
        import pandas as pd
        from spynoza.nodes.utils import get_scaninfo
        from fir import FIRDeconvolution
        from hedfpy import HDFEyeOperator
        import tempfile

        # standard test path and subject
        test_path = '/home/shared/-2014/reward/new/'
        test_sj = 'sub-00'
        h5_file= glob.glob(op.join(test_path, test_sj, 'h5/*roi.h5'))[0]
        fir_frequency = 10
        fir_interval = [-2.0,7.0]
        pupil_data_type = '_pupil_bp_zscore'
        fmri_data_type = 'psc'
        lost_signal_rate_threshold = 0.15
        method = 'fir'

        mask_threshold = 2.0
        mask_direction = 'pos'
        fmri_data_type = 'psc'
        pupil_data_type = '_pupil_bp_zscore'
        fir_frequency = 2
        fir_interval = [-3.0,15.0]
        roi_list = ['V1','V2','V3']
        roi = roi_list[0]

        # experiment = 'stream-up'
        # experiment_re = 'stream-up'
        experiment = 'unpredictable'
        experiment_re = 'unpredictable'
        mapper = 'unpredictable'
        #
        # experiment = 'predictable'
        # experiment_re = 'predictable'
        #
        # experiment = 'variable'
        # experiment_re = 'variable_*'
        #

        # input files
        fir_file_reward_list = glob.glob(op.join(test_path, test_sj, 'psc/*-%s_reward*.nii.gz'%experiment_re))
        fir_file_reward_list.sort()
        glm_file_mapper_list = glob.glob(op.join(test_path, test_sj, 'psc/*-%s_mapper*.nii.gz'%mapper))
        glm_file_mapper_list.sort()
        # behavior files
        behavior_file_list = glob.glob(op.join(test_path, test_sj, 'events/tsv/*-%s_reward*.tsv'%experiment_re))
        behavior_file_list.sort()
        # mcf files
        vol_regressor_list = glob.glob(op.join(test_path, test_sj, 'mcf/ext_motion_pars/*-%s_reward_*.par'%experiment_re))
        vol_regressor_list.sort()
        # eye files
        eye_h5_file_list = glob.glob(op.join(test_path, test_sj, 'eye/h5/*-%s_reward_*.h5'%experiment_re))
        eye_h5_file_list.sort()   
        # mapper files
        mapper_file_list = glm_file_mapper_list   

        in_files = fir_file_reward_list

        # from utils.behavior import behavior_timing
        # from utils.utils import roi_data_from_hdf

        tr_key = 116.0

        TR, dims, dyns, voxsize, affine = get_scaninfo(in_files[1]) 
        header = nib.load(in_files[1]).header

        ################################################################################## 
        # behavior data generalizes across ROIs of course
        ##################################################################################
        event_names, all_behav_event_times, rename_dict, which_event_name_rewarded = behavior_timing(experiment, behavior_file_list, dyns, TR)

        mapper_alias = op.split(mapper_file_list[0])[-1][:-7] + '_T'     

        contrast_data = roi_data_from_hdf(data_types_wildcards = [mapper_alias], roi_name_wildcard = roi, hdf5_file = h5_file, folder_alias = 'GLM')
        time_course_data = [roi_data_from_hdf(data_types_wildcards = [os.path.split(in_f)[-1][:-7]], roi_name_wildcard = roi, hdf5_file = h5_file, folder_alias = fmri_data_type) for in_f in in_files]

        time_course_data = np.hstack(time_course_data)

        if threshold < 0:
            threshold = -threshold
            contrast_data = -contrast_data

        over_threshold = (contrast_data[:,1]>threshold)
        iceberg = contrast_data[over_threshold, 1]

        projected_time_course = np.dot(time_course_data[over_threshold].T, iceberg) / np.sum(iceberg)
        av_time_course = time_course_data[over_threshold].mean(axis = 0)

        both_time_courses = np.vstack([projected_time_course, av_time_course])

        fir_timecourses = {en:np.zeros([int((fir_interval[1] - fir_interval[0]) * fir_frequency)]) 
                for en in event_names}

        nuisance_regressors = np.nan_to_num(all_vol_reg)

        if num_components != 0:
            if method == 'PCA':
                pca = decomposition.PCA(n_components = num_components, whiten = True)
                nuisance_regressors = pca.fit_transform(nuisance_regressors.T).T
            elif method == 'ICA':
                ica = decomposition.FastICA(n_components = num_components, whiten = True)
                nuisance_regressors = ica.fit_transform(nuisance_regressors.T).T


        fd = FIRDeconvolution(
            signal = projected_time_course, 
            events = [all_behav_event_times[en] for en in event_names], # dictate order
            event_names = event_names, 
            sample_frequency = 1.0/TR,
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
            fir_timecourses[en] = np.squeeze(np.nan_to_num(fd.betas_for_cov(en).T))

        plot_fir_results_unpredictable(fd.deconvolution_interval_timepoints, event_names, fir_time_course_dict = fir_timecourses, zero_interval = [-0.75, -0.25], use_zero_interval = False, suffix = roi+'_roi')


    def runTest(self):
        fit_FIR_nuisances_all_files(
            example_func = self.example_func,
            in_files = self.in_files,
            slice_regressor_lists = self.slice_regressor_lists,
            vol_regressor_list = self.vol_regressor_list,
            behavior_file_list = self.behavior_file_list,
            fir_frequency = 4,
            fir_interval = [-3.0,15.0]
            )
    

