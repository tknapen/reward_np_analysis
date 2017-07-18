import unittest
import glob
import os.path as op


def fit_FIR_pupil_files(
                        experiment, 
                        eye_h5_file_list = '', 
                        behavior_file_list = '', 
                        h5_file = '', 
                        in_files = '', 
                        fir_frequency = 20,
                        fir_interval = [-2.0,6.0],
                        data_type = '_pupil_bp_zscore',
                        lost_signal_rate_threshold = 0.05,
                        method = 'fir'
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
    import bottleneck as bn
    # from behavior import behavior_timing

    # constants 
    tr_key = 116.0
    zero_interval = [-0.75, -0.25]

    if len(eye_h5_file_list) == 0:
        print 'pupil FIR of experiment {} not performed for lack of input files'.format(experiment)
        # return

    # the first file is longer in one or two subjects, so we take the scaninfo of the second file
    TR, dims, dyns, voxsize, affine = get_scaninfo(in_files[1]) 

    # behavior
    event_names, all_behav_event_times, rename_dict, which_event_name_rewarded = behavior_timing(experiment, behavior_file_list, dyns, TR)

    # pupil data
    aliases = ['eye/' + op.splitext(op.split(eye_h5_file)[-1])[0] for eye_h5_file in eye_h5_file_list]

    ho = HDFEyeOperator(h5_file)
    
    pupil_data = []    

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

        raw_data = ho.data_from_time_period([tr_timestamps[0], tr_timestamps[0] + (1000.0 * dyns * TR)], alias)
        selected_data = np.array(raw_data[eye + data_type])

        # bad recordings are characterized, among other things, by a very high amount of lost samples. 
        # if this lost sample rate crosses a threshold, we will throw the file out completely
        raw_pupil = np.array(raw_data[eye + '_pupil'])
        lost_signal_rate = (raw_pupil == 0).sum() / (dyns * TR * sample_rate)
        run_signal = np.zeros((sample_rate * dyns * TR))
        if lost_signal_rate > lost_signal_rate_threshold:
            print 'dropping this run, lost signal rate = %1.3f'%lost_signal_rate
        else:
            run_signal[:selected_data.shape[0]] = selected_data

        pupil_data.append(run_signal)
    pupil_data = np.concatenate(pupil_data)

    if method == 'fir':
        fd = FIRDeconvolution(
                            signal = pupil_data[::(sample_rate/fir_frequency)],             # downsampled when used as argument
                            events = [all_behav_event_times[en] for en in event_names],     # dictate order
                            event_names = event_names, 
                            sample_frequency = fir_frequency,
                            deconvolution_frequency = fir_frequency,
                            deconvolution_interval = fir_interval
        )

        # fd.resampled_signal = np.nan_to_num(fd.resampled_signal)
        fd.create_design_matrix()
        fd.regress(method = 'lstsq')
        fd.calculate_rsq()
        # convert to data that can be plotted
        fir_time_course_dict = {en: fd.betas_for_cov(en) for en in event_names}
        deconvolution_interval_timepoints = fd.deconvolution_interval_timepoints

    elif method == 'era':
        deconvolution_interval_timepoints = np.arange(fir_interval[0], fir_interval[1], 1.0/sample_rate)
        zero_interval_bools = (deconvolution_interval_timepoints > zero_interval[0]) & (deconvolution_interval_timepoints < zero_interval[1])
        sample_times = np.arange(0,pupil_data.shape[0] / sample_rate, 1.0/sample_rate)
        fir_time_course_dict = {}
        for en in event_names:
            event_times = all_behav_event_times[en]
            epoched_data = np.zeros((len(event_times), len(deconvolution_interval_timepoints)))
            for i, et in enumerate(event_times):
                if (et + fir_interval[0]) < sample_times[-1]:
                    epoch_start_index = np.arange(len(sample_times))[sample_times > (et + fir_interval[0])][0]
                    epoch_end_index = np.min([epoch_start_index+len(deconvolution_interval_timepoints), len(sample_times)])
                    epoched_data[i,0:epoch_end_index-epoch_start_index] = pupil_data[epoch_start_index:epoch_end_index] 
                                        
            epoched_data_zerod = (epoched_data.T - bn.nanmedian(epoched_data[:,zero_interval_bools], axis = 1)).T
            fir_time_course_dict.update({en: bn.nanmedian(epoched_data, axis = 0)}) 


    if experiment == 'unpredictable':
        out_figures = [plot_fir_results_unpredictable(deconvolution_interval_timepoints,
                                                    event_names, 
                                                    fir_time_course_dict, 
                                                    zero_interval = zero_interval, 
                                                    use_zero_interval = True, 
                                                    suffix = 'pupil'
                                                    ),
                        plot_fir_results_unpredictable(deconvolution_interval_timepoints,
                                                    event_names, 
                                                    fir_time_course_dict, 
                                                    zero_interval = zero_interval, 
                                                    use_zero_interval = False, 
                                                    suffix = 'pupil'
                                                    )]
    elif experiment == 'predictable':
        out_figures = [plot_fir_results_predictable(deconvolution_interval_timepoints,
                                                    rename_dict,
                                                    event_names, 
                                                    fir_time_course_dict, 
                                                    zero_interval = zero_interval, 
                                                    use_zero_interval = True, 
                                                    suffix = 'pupil'
                                                    ),
                        plot_fir_results_predictable(deconvolution_interval_timepoints,
                                                    rename_dict,
                                                    event_names, 
                                                    fir_time_course_dict, 
                                                    zero_interval = zero_interval, 
                                                    use_zero_interval = False, 
                                                    suffix = 'pupil'
                                                    )]
    elif experiment == 'variable':
        out_figures = [plot_fir_results_variable(deconvolution_interval_timepoints,
                                                    event_names, 
                                                    fir_time_course_dict, 
                                                    zero_interval = zero_interval, 
                                                    use_zero_interval = True, 
                                                    suffix = 'pupil'
                                                    ),
                        plot_fir_results_variable(deconvolution_interval_timepoints,
                                                    event_names, 
                                                    fir_time_course_dict, 
                                                    zero_interval = zero_interval, 
                                                    use_zero_interval = False, 
                                                    suffix = 'pupil'
                                                    )]


    # the derived data types are added in the plot functions, and overwritten such that the no-zero'd data are saved
    out_values = np.array([np.squeeze(fir_time_course_dict[key]) for key in fir_time_course_dict.iterkeys()]).T
    out_columns = [key for key in fir_time_course_dict.iterkeys()]

    out_df = pd.DataFrame(out_values, columns = out_columns, index = deconvolution_interval_timepoints)

    store = pd.HDFStore(h5_file)
    store['/pupil/'+experiment] = out_df
    store.close()

    return out_figures




class FIRTestCase(unittest.TestCase):
    
    def setUp(self):
        import glob
        import os.path as op
        from utils.plotting import *
        from utils.behavior import behavior_timing

        # standard test path and subject
        test_path = '/home/shared/-2014/reward/new/'
        test_sj = 'sub-004'
        h5_file= glob.glob(op.join(test_path, test_sj, 'h5/*roi.h5'))[0]
        fir_frequency = 10
        fir_interval = [-2.0,7.0]
        data_type = '_pupil_bp_zscore'
        lost_signal_rate_threshold = 0.15
        method = 'fir'



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



        experiment = 'predictable'
        # input files
        in_files = glob.glob(op.join(test_path, test_sj, 'psc/*-predictable_reward*.nii.gz'))
        in_files.sort()
        # behavior files
        behavior_file_list = glob.glob(op.join(test_path, test_sj, 'events/tsv/*-predictable_reward*.tsv'))
        behavior_file_list.sort()
        # behavior files
        eye_h5_file_list = glob.glob(op.join(test_path, test_sj, 'eye/h5/*-predictable_reward*.h5'))
        eye_h5_file_list.sort()


        experiment = 'variable'
        # # input files
        in_files = glob.glob(op.join(test_path, test_sj, 'psc/*-variable_*_reward*.nii.gz'))
        in_files.sort()
        # # behavior files
        behavior_file_list = glob.glob(op.join(test_path, test_sj, 'events/tsv/*-variable_*_reward*.tsv'))
        behavior_file_list.sort()
        # # behavior files
        eye_h5_file_list = glob.glob(op.join(test_path, test_sj, 'eye/h5/*-variable_*_reward*.h5'))
        eye_h5_file_list.sort()        
     

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
    
