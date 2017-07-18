import unittest
import glob
import os.path as op


def BOLD_FIR_files(     analysis_info,
                        experiment, 
                        fir_file_reward_list = '', 
                        glm_file_mapper_list = '', 
                        behavior_file_list = '',
                        mapper_contrast = 'stim', 
                        h5_file = '',
                        roi_list = ['V1','V2','V3']
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
    from utils import roi_data_from_hdf

    threshold = 1.67
    zero_interval = [-2.5,-0.5]

    # roi data
    mapper_alias = op.split(glm_file_mapper_list[0])[-1][:-7] + '_logp'

    # the first file is longer in one or two subjects, so we take the scaninfo of the second file
    TR, dims, dyns, voxsize, affine = get_scaninfo(fir_file_reward_list[1]) 

    # behavior
    event_names, all_behav_event_times, rename_dict, which_event_name_rewarded = behavior_timing(experiment, behavior_file_list, dyns, TR)
    fir_aliases = [op.split(fir_file_reward_list[0])[-1][:-7] + '_%s'%en for en in event_names]

    out_figures = []
    for roi in roi_list:
        contrast_data = roi_data_from_hdf(data_types_wildcards = [mapper_alias], roi_name_wildcard = roi, hdf5_file = h5_file, folder_alias = 'GLM')
        fir_data = [roi_data_from_hdf(data_types_wildcards = [fa], roi_name_wildcard = roi, hdf5_file = h5_file, folder_alias = 'GLM') for fa in fir_aliases]
        deconvolution_interval_timepoints = np.linspace(analysis_info['fir_interval'][0], analysis_info['fir_interval'][1], fir_data[0].shape[-1], endpoint = False)

        # for experiment 'unpredictable' and 'variable' the masking contrast is the initial mapper
        if experiment in ['unpredictable', 'variable', 'stream-up']:
            if mapper_contrast == 'stim':
                fir_time_course_dict = {en: fd[contrast_data[:,1]>threshold].mean(axis = 0) for en,fd in zip(event_names, fir_data)}
            elif mapper == 'surround':
                fir_time_course_dict = {en: fd[contrast_data[:,4]>threshold].mean(axis = 0) for en,fd in zip(event_names, fir_data)}
        elif experiment == 'predictable':
            rewarded_location, rewarded_orientation = which_event_name_rewarded.split('_')
            # check which stimulus was rewarded and implement correct contrast values
            if rewarded_location == 'left':
                rsci,nrsci = [1,4], [7,10]
            elif rewarded_location == 'right':
                rsci,nrsci = [7,10], [1,4]
            # create joined contrast values (not averaged just yet - T values should sort-of sum.)
            if mapper_contrast == 'rewarded_stim':
                contrast_data_joined = contrast_data[:,rsci[0]] + contrast_data[:,rsci[1]]
            elif mapper_contrast == 'nonrewarded_stim':
                contrast_data_joined = contrast_data[:,nrsci[0]] + contrast_data[:,nrsci[1]]
            else:
                print('unknown mask type for experiment predictable')
            # create ROIs with these contrasts
            fir_time_course_dict = {en: fd[contrast_data_joined>threshold].mean(axis = 0) for en,fd in zip(event_names, fir_data)}

        if experiment == 'unpredictable':
            out_figures.append(
                            plot_fir_results_unpredictable(deconvolution_interval_timepoints,
                                                        event_names, 
                                                        fir_time_course_dict, 
                                                        zero_interval = zero_interval, 
                                                        use_zero_interval = False, 
                                                        suffix = roi
                                                        ))
        elif experiment == 'predictable':
            out_figures.append(
                            plot_fir_results_predictable(deconvolution_interval_timepoints,
                                                        rename_dict,
                                                        event_names, 
                                                        fir_time_course_dict, 
                                                        zero_interval = zero_interval, 
                                                        use_zero_interval = False, 
                                                        suffix = roi
                                                        ))
        elif experiment == 'variable':
            out_figures.append(
                            plot_fir_results_variable(deconvolution_interval_timepoints,
                                                        event_names, 
                                                        fir_time_course_dict, 
                                                        zero_interval = zero_interval, 
                                                        use_zero_interval = False, 
                                                        suffix = roi
                                                        ))

        # the derived data types are added in the plot functions, and overwritten such that the no-zero'd data are saved
        out_values = np.array([np.squeeze(fir_time_course_dict[key]) for key in fir_time_course_dict.iterkeys()]).T
        out_columns = [key for key in fir_time_course_dict.iterkeys()]

        out_df = pd.DataFrame(out_values, columns = out_columns, index = deconvolution_interval_timepoints)

        store = pd.HDFStore(h5_file)
        store['/fir/'+experiment+'/'+roi+'_'+mapper_contrast] = out_df
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
        fir_file_reward_list = glob.glob(op.join(test_path, test_sj, 'psc/*-unpredictable_reward*.nii.gz'))
        fir_file_reward_list.sort()
        glm_file_mapper_list = glob.glob(op.join(test_path, test_sj, 'psc/*-unpredictable_mapper*.nii.gz'))
        glm_file_mapper_list.sort()
        # behavior files
        behavior_file_list = glob.glob(op.join(test_path, test_sj, 'events/tsv/*-unpredictable_reward*.tsv'))
        behavior_file_list.sort()

        experiment = 'predictable'
        # input files
        fir_file_reward_list = glob.glob(op.join(test_path, test_sj, 'psc/*-predictable_reward*.nii.gz'))
        fir_file_reward_list.sort()
        glm_file_mapper_list = glob.glob(op.join(test_path, test_sj, 'psc/*-predictable_mapper*.nii.gz'))
        glm_file_mapper_list.sort()
        # behavior files
        behavior_file_list = glob.glob(op.join(test_path, test_sj, 'events/tsv/*-predictable_reward*.tsv'))
        behavior_file_list.sort()

        experiment = 'variable'
        # # input files
        fir_file_reward_list = glob.glob(op.join(test_path, test_sj, 'psc/*-variable_*_reward*.nii.gz'))
        fir_file_reward_list.sort()
        glm_file_mapper_list = glob.glob(op.join(test_path, test_sj, 'psc/*-unpredictable_mapper*.nii.gz'))
        glm_file_mapper_list.sort()
        # behavior files
        behavior_file_list = glob.glob(op.join(test_path, test_sj, 'events/tsv/*-variable_*_reward*.tsv'))
        behavior_file_list.sort()
     
     

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
    
