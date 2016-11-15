def fit_FIR_pupil_files(
                        experiment, 
                        eye_h5_file_list = '', 
                        behavior_file_list = '', 
                        h5_file = '', 
                        in_files = '', 
                        fir_frequency = 20,
                        fir_interval = [-2.0,6.0],
                        data_type = '_pupil_bp_clean_zscore',
                        lost_signal_rate_threshold = 0.05
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
    import os.path as op
    import pandas as pd
    from spynoza.nodes.utils import get_scaninfo
    from fir import FIRDeconvolution
    from hedfpy import HDFEyeOperator
    import tempfile
    from behavior import behavior_timing


    # constants 
    tr_key = 116.0

    if len(eye_h5_file_list) == 0:
        print 'pupil FIR of experiment {} not performed for lack of input files'.format(experiment)
        # return

    # the first file is longer in one or two subjects, so we take the scaninfo of the second file
    TR, dims, dyns, voxsize, affine = get_scaninfo(in_files[1]) 

    # behavior
    event_names, all_behav_event_times = behavior_timing(experiment, behavior_file_list, dyns, TR)

    # pupil data
    aliases = ['eye/' + op.splitext(op.split(eye_h5_file)[-1])[0] for eye_h5_file in eye_h5_file_list]

    ho = HDFEyeOperator(h5_file)
    
    run_nr_samples = TR * dyns * fir_frequency
    pupil_data = np.zeros((run_nr_samples * len(aliases)))

    for x, alias in enumerate(aliases):
        ho.open_hdf_file('a')
        node = ho.h5f.get_node('/' + alias)
        trial_data = node.trials.read()
        events = node.events.read()
        ho.h5f.close()

        # find first TR, and use it to define:
        # the duration of the datastream and the tr
        tr_timestamps = events['EL_timestamp'][(events['key'] == tr_key) & (events['up_down'] == 'Down')][:dyns]

        eye = ho.eye_during_trial(0, alias)
        sample_rate = ho.sample_rate_during_period([tr_timestamps[0], tr_timestamps[-1] + TR], alias)

        raw_data = ho.data_from_time_period([tr_timestamps[0], tr_timestamps[0] + (1000.0 * (dyns+1))], alias)
        selected_data = np.array(raw_data[eye + data_type])
        resampled_data = sp.signal.resample(selected_data, run_nr_samples)

        # bad recordings are characterized, among other things, by a very high amount of lost samples. 
        # if this lost sample rate crosses a threshold, we will throw the file out completely
        raw_pupil = np.array(raw_data[eye + '_pupil'])
        lost_signal_rate = (raw_pupil == 0).sum() / (dyns * TR * sample_rate)
        if lost_signal_rate > lost_signal_rate_threshold:
            run_signal = np.ones(resampled_data.shape)
            run_signal = np.nan
        else:
            run_signal = resampled_data

        pupil_data[x*run_nr_samples:(x+1)*run_nr_samples] = run_signal

    fd = FIRDeconvolution(
                        signal = pupil_data, 
                        events = [all_behav_event_times[en] for en in event_names], # dictate order
                        event_names = event_names, 
                        sample_frequency = fir_frequency,
                        deconvolution_frequency = fir_frequency,
                        deconvolution_interval = fir_interval
    )

    fd.resampled_signal = np.nan_to_num(fd.resampled_signal)
    # we then tell it to create its design matrix
    fd.create_design_matrix()

    # fit
    fd.regress(method = 'lstsq')
    fd.calculate_rsq()
    fir_timecourses = {en: np.zeros(((fir_interval[1] - fir_interval[0])*fir_frequency)) for en in event_names }
    for en in event_names:
        fir_timecourses[en] = np.nan_to_num(fd.betas_for_cov(en).squeeze())

    for en in event_names:
        plot(np.linspace(fir_interval[0], fir_interval[1], fir_timecourses[en].shape[0]), fir_timecourses[en], label = en)
    legend()
     # reshape and save
    rsq_data = np.nan_to_num(fd.rsq) 





class FIRTestCase(unittest.TestCase):
    
    def setUp(self):
        import glob
        import os.path as op
        # standard test path and subject
        test_path = '/home/shared/-2014/reward/new/'
        test_sj = 'sub-002'
        experiment = 'unpredictable'
        # input files
        in_files = glob.glob(op.join(test_path, test_sj, 'psc/*-unpredictable_reward*.nii.gz'))
        in_files.sort()
        # behavior files
        behavior_file_list = glob.glob(op.join(test_path, test_sj, 'events/tsv/*-unpredictable_reward*.tsv'))
        behavior_file_list.sort()
        # behavior files
        eye_h5_file_list = glob.glob(op.join(test_path, test_sj, 'eye/h5/*-unpredictable_reward*.h5'))
        eye_h5_file_list.sort()
        # experiment = 'variable'
        # # input files
        # in_files = glob.glob(op.join(test_path, test_sj, 'psc/*-variable_*_reward*.nii.gz'))
        # in_files.sort()
        # # behavior files
        # behavior_file_list = glob.glob(op.join(test_path, test_sj, 'events/tsv/*-variable_*_reward*.tsv'))
        # behavior_file_list.sort()
        # # behavior files
        # eye_h5_file_list = glob.glob(op.join(test_path, test_sj, 'eye/h5/*-variable_*_reward*.h5'))
        # eye_h5_file_list.sort()        
        # h5 file
        h5_file= glob.glob(op.join(test_path, test_sj, 'h5/*roi.h5'))[0]
        fir_frequency = 10
        fir_interval = [-2.0,6.0]
        data_type = '_pupil_bp_clean_zscore'

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
    
