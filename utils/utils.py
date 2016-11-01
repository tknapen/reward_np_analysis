from __future__ import division, print_function

def convert_edf_2_hdf5(edf_file, low_pass_pupil_f = 6.0, high_pass_pupil_f = 0.01):
    """converts the edf_file to hdf5 using hedfpy
    
    Requires hedfpy

    Parameters
    ----------
    edf_file : str
        absolute path to edf file.
    low_pass_pupil_f : float
        low pass cutoff frequency for band-pass filtering of the pupil signal
    high_pass_pupil_f : float
        high pass cutoff frequency for band-pass filtering of the pupil signal
    Returns
    -------
    hdf5_file : str
        absolute path to hdf5 file.
    """

    import os
    import os.path as op
    from hedfpy import HDFEyeOperator
    import tempfile

    tempdir = tempfile.mkdtemp()
    temp_edf = op.join(tempdir, op.split(edf_file)[-1])

    os.system('cp ' + edf_file + ' ' + temp_edf)

    hdf5_file = op.join(tempdir, op.split(op.splitext(edf_file)[0] + '.h5')[-1])
    alias = op.splitext(op.split(edf_file)[-1])[0]

    ho = HDFEyeOperator(hdf5_file)
    ho.add_edf_file(temp_edf)
    ho.edf_message_data_to_hdf(alias = alias)
    ho.edf_gaze_data_to_hdf(alias = alias, pupil_hp = high_pass_pupil_f, pupil_lp = low_pass_pupil_f)

    return hdf5_file

def convert_hdf_eye_to_tsv(hdf5_file, tsv_file = None, xy_intercepts = None, xy_slopes = None ):

    from hedfpy import HDFEyeOperator
    import tempfile
    import os

    tr_key = 116.0
    alias = op.splitext(op.split(hdf5_file)[-1])[0]

    ho = HDFEyeOperator(hdf5_file)
    ho.open_hdf_file('a')
    node = ho.h5f.get_node('/' + alias)
    trial_data = node.trials.read()
    events = node.events.read()

    if tsv_file == None:
        tempdir = tempfile.mkdtemp()
        temp_tsv = op.join(tempdir, alias + '.tsv')
    else:
        temp_tsv = tsv_file

    # find first TR, and use it to define:
    # the duration of the datastream and the tr
    tr_timestamps = events['EL_timestamp'][(events['key'] == tr_key) & (events['up_down'] == 'Down')]
    tr = np.round(np.mean(np.diff(tr_timestamps)))

    eye = ho.eye_during_trial(0, alias)
    sample_rate = ho.sample_rate_during_period([tr_timestamps[0], tr_timestamps[-1] + tr], alias)

    raw_data = ho.data_from_time_period([tr_timestamps[0], tr_timestamps[-1] + tr], alias)
    selected_data = np.array(raw_data[['time', eye + '_gaze_x', eye + '_gaze_y', eye + '_pupil']])
    
    # some post-processing:
    # time in seconds
    selected_data[:,0] = (selected_data[:,0] - selected_data[0,0]) / 1000.0,  # convert ms to s
    # linear gaze position scaling
    if (xy_intercepts != None) and (xy_slopes != None):
        for i in [1,2]:
            selected_data[:,i] = (selected_data[:,i] * xy_slopes[i-1]) + xy_intercepts[i-1]

    np.savetxt(temp_tsv, selected_data, fmt = '%3.3f', delimiter = '\t', header="time   X   Y   pupil")
    os.system('gzip ' + temp_tsv)

    return temp_tsv + '.gz'

def convert_unpredictable_trials_to_tsv(hdf5_file, tsv_file = None, reward_signal = 1.0 ):

    from hedfpy import HDFEyeOperator
    import tempfile
    import os
    import os.path as op
    import numpy as np
    import pandas as pd

    tr_key = 116.0

    alias = op.splitext(op.split(hdf5_file)[-1])[0]

    ho = HDFEyeOperator(hdf5_file)
    ho.open_hdf_file('a')
    node = ho.h5f.get_node('/' + alias)
    trial_phase_data = node.trial_phases.read()
    events = node.events.read()
    parameters = node.parameters.read()

    if tsv_file == None:
        tempdir = tempfile.mkdtemp()
        temp_tsv = op.join(tempdir, alias + '.tsv')
    else:
        temp_tsv = tsv_file

    # find first TR, and use it to define:
    # the duration of the datastream and the tr
    tr_timestamps = events['EL_timestamp'][(events['key'] == tr_key) & (events['up_down'] == 'Down')]
    tr = np.round(np.mean(np.diff(tr_timestamps)))
    sample_rate = ho.sample_rate_during_period([tr_timestamps[0], tr_timestamps[-1] + tr], alias)

    run_onset_EL_time = tr_timestamps[0]

    columns_df = ['sound', 'contrast', 'stim_eccentricity', 'orientation', 'mask_radius']
    tp_value = 2.0

    trial_phase_timestamps = np.array(trial_phase_data[trial_phase_data['trial_phase_index'] == tp_value]['trial_phase_EL_timestamp'])

    trial_parameters = pd.DataFrame(parameters[columns_df].T)

    # first, we re-code the parameters in terms of screen coordinates
    # convert stim position from proportion of screen (1920, 124 cm distance on BoldScreen32) to dva
    trial_parameters['stim_eccentricity'] = trial_parameters['stim_eccentricity'] * 11 # 11 degrees from fix to edge of screen
    trial_parameters['mask_radius'] = trial_parameters['mask_radius'] * 22 / 1920 # 11 degrees from fix to edge of screen

    # add the times at which stimuli and rewards were presented
    which_trials_reward = np.array(trial_parameters['sound'] == reward_signal)
    reward_times = np.zeros(len(trial_phase_timestamps))
    # the sound was 800 ms delayed wrt the stimulus onset
    reward_times[which_trials_reward] = \
                (800.0 + trial_phase_timestamps[which_trials_reward] - tr_timestamps[0] ) / 1000.0

    trial_parameters['reward_time'] = pd.Series(reward_times)
    trial_parameters['stim_onset_time'] = pd.Series((trial_phase_timestamps - tr_timestamps[0] ) / 1000.0)

    trial_parameters = trial_parameters.sort_values(by = 'stim_onset_time')

    trial_parameters.to_csv(temp_tsv, sep = '\t', float_format = '%3.2f', header = True, index = False)

    return temp_tsv

def convert_predictable_trials_to_tsv(hdf5_file, tsv_file = None ):

    from hedfpy import HDFEyeOperator
    import tempfile
    import os
    import os.path as op
    import numpy as np
    import pandas as pd
    import math

    tr_key = 116.0

    alias = op.splitext(op.split(hdf5_file)[-1])[0]

    ho = HDFEyeOperator(hdf5_file)
    ho.open_hdf_file('a')
    node = ho.h5f.get_node('/' + alias)
    trial_phase_data = node.trial_phases.read()
    events = node.events.read()
    parameters = node.parameters.read()

    if tsv_file == None:
        tempdir = tempfile.mkdtemp()
        temp_tsv = op.join(tempdir, alias + '.tsv')
    else:
        temp_tsv = tsv_file

    # find first TR, and use it to define:
    # the duration of the datastream and the tr
    tr_timestamps = events['EL_timestamp'][(events['key'] == tr_key) & (events['up_down'] == 'Down')]
    tr = np.round(np.mean(np.diff(tr_timestamps)))
    sample_rate = ho.sample_rate_during_period([tr_timestamps[0], tr_timestamps[-1] + tr], alias)

    run_onset_EL_time = tr_timestamps[0]

    columns_df = ['sound', 'contrast', 'x_position', 'orientation', 'mask_radius']
    tp_value = 2.0

    trial_phase_timestamps = np.array(trial_phase_data[trial_phase_data['trial_phase_index'] == tp_value]['trial_phase_EL_timestamp'])

    trial_parameters = pd.DataFrame(parameters[columns_df].T)

    # first, we re-code the parameters in terms of screen coordinates
    # convert stim position from proportion of screen (1920, 124 cm distance on BoldScreen32) to dva
    trial_parameters['x_position'] = trial_parameters['x_position'] * 11 # 11 degrees from fix to edge of screen
    trial_parameters['mask_radius'] = trial_parameters['mask_radius'] * 22 / 1920 # 11 degrees from fix to edge of screen

    # add the times at which stimuli and rewards were presented
    which_trials_reward = np.array(trial_parameters['sound'] == 1)
    reward_times = np.zeros(len(trial_phase_timestamps))
    # the sound was 800 ms delayed wrt the stimulus onset
    reward_times[which_trials_reward] = \
                (800.0 + trial_phase_timestamps[which_trials_reward] - tr_timestamps[0] ) / 1000.0

    trial_parameters['reward_time'] = pd.Series(reward_times)
    trial_parameters['stim_onset_time'] = pd.Series((trial_phase_timestamps - tr_timestamps[0] ) / 1000.0)

    trial_parameters = trial_parameters.sort_values(by = 'stim_onset_time')

    trial_parameters.to_csv(temp_tsv, sep = '\t', float_format = '%3.2f', header = True, index = False)

    return temp_tsv

def convert_variable_trials_to_tsv(hdf5_file, tsv_file = None, reward_signal = 1.0 ):

    from hedfpy import HDFEyeOperator
    import tempfile
    import os
    import os.path as op
    import numpy as np
    import pandas as pd
    import math

    tr_key = 116.0

    alias = op.splitext(op.split(hdf5_file)[-1])[0]

    ho = HDFEyeOperator(hdf5_file)
    ho.open_hdf_file('a')
    node = ho.h5f.get_node('/' + alias)
    trial_phase_data = node.trial_phases.read()
    events = node.events.read()
    parameters = node.parameters.read()

    if tsv_file == None:
        tempdir = tempfile.mkdtemp()
        temp_tsv = op.join(tempdir, alias + '.tsv')
    else:
        temp_tsv = tsv_file

    # find first TR, and use it to define:
    # the duration of the datastream and the tr
    tr_timestamps = events['EL_timestamp'][(events['key'] == tr_key) & (events['up_down'] == 'Down')]
    tr = np.round(np.mean(np.diff(tr_timestamps)))
    sample_rate = ho.sample_rate_during_period([tr_timestamps[0], tr_timestamps[-1] + tr], alias)

    run_onset_EL_time = tr_timestamps[0]

    columns_df = ['sound', 'contrast', 'stim_eccentricity', 'stim_orientation', 'mask_radius', 'reward_delay', 'which_stimulus_for_rgb_overlay']
    tp_value = 2.0

    trial_phase_timestamps = np.array(trial_phase_data[trial_phase_data['trial_phase_index'] == tp_value]['trial_phase_EL_timestamp'])

    trial_parameters = pd.DataFrame(parameters[columns_df].T)

    # first, we re-code the parameters in terms of screen coordinates
    # convert stim position from proportion of screen (1920, 124 cm distance on BoldScreen32) to dva
    trial_parameters['stim_eccentricity'] = trial_parameters['stim_eccentricity'] * 11 # 11 degrees from fix to edge of screen
    trial_parameters['mask_radius'] = trial_parameters['mask_radius'] * 22 / 1920 # 11 degrees from fix to edge of screen

    # add the times at which stimuli and rewards were presented
    which_trials_reward = np.array(trial_parameters['sound'] == reward_signal)
    reward_times = np.zeros(len(trial_phase_timestamps))
    # the sound was 800 ms delayed wrt the stimulus onset
    reward_times = trial_parameters['reward_delay'] + (trial_phase_timestamps - tr_timestamps[0] ) / 1000.0

    trial_parameters['reward_time'] = pd.Series(reward_times)
    trial_parameters['stim_onset_time'] = pd.Series((trial_phase_timestamps - tr_timestamps[0] ) / 1000.0)

    trial_parameters = trial_parameters.sort_values(by = 'stim_onset_time')

    trial_parameters.to_csv(temp_tsv, sep = '\t', float_format = '%3.2f', header = True, index = False)

    return temp_tsv



def mask_nii_2_hdf5(in_files, mask_files, hdf5_file, folder_alias):
    """masks data in in_files with masks in mask_files,
    to be stored in an hdf5 file

    Takes a list of 4D fMRI nifti-files and masks the
    data with all masks in the list of nifti-files mask_files.
    These files are assumed to represent the same space, i.e.
    that of the functional acquisitions. 
    These are saved in hdf5_file, in the folder folder_alias.

    Parameters
    ----------
    in_files : list
        list of absolute path to functional nifti-files.
        all nifti files are assumed to have the same ndim
    mask_files : list
        list of absolute path to mask nifti-files.
        mask_files are assumed to be 3D
    hdf5_file : str
    	absolute path to hdf5 file.
   	folder_alias : str
   		name of the to-be-created folder in the hdf5 file.

    Returns
    -------
    hdf5_file : str
        absolute path to hdf5 file.
    """

    import nibabel as nib
    import os.path as op
    import numpy as np
    import tables

    success = True

    mask_data = [nib.load(mf).get_data() for mf in mask_files]
    nifti_data = [nib.load(nf).get_data() for nf in in_files]

    mask_names = [op.split(mf)[-1].split('.nii.gz')[0] for mf in mask_files]
    nifti_names = [op.split(mf)[-1].split('.nii.gz')[0] for nf in in_files]

    data = nib.load(in_files[0])
    dims = data.shape
    n_dims = data.ndim

    h5file = tables.open_file(hdf5_file, mode = "w", title = hdf5_file)
    # get or make group for alias folder
    try:
        folder_alias_run_group = h5file.get_node("/", name = folder_alias, classname='Group')
    except NoSuchNodeError:
        print('Adding group ' + folder_alias + ' to this file')
        folder_alias_run_group = h5file.create_group("/", folder_alias, folder_alias)

    for (roi, roi_name) in zip(mask_data, mask_names):
        # get or make group for alias/roi
        try:
            run_group = h5file.get_node(where = "/" + folder_alias, name = roi_name, classname='Group')
        except NoSuchNodeError:
            print('Adding group ' + folder_alias + '_' + roi_name + ' to this file')
            run_group = h5file.create_group("/" + folder_alias, roi_name, folder_alias + '_' + roi_name)
        
        h5file.create_array(run_group, roi_name, roi, roi_name + ' mask file for reconstituting nii data from masked data')

        for (nii_d, nii_name) in zip(nifti_data, nifti_names):
            if n_dims == 3:
                these_roi_data = nii_d[roi]
            elif n_dims == 4:   # timeseries data, last dimension is time.
                these_roi_data = nii_d[roi,:]
            else:
                print("n_dims in data {nifti} do not fit with mask".format(nii_name))
                success = False

            h5file.create_array(run_group, nii_name, these_roi_data, roi_name + ' data from ' + nii_name)

    h5file.close()

    return hdf5_file

def roi_data_from_hdf(data_types_wildcards, roi_name_wildcard, hdf5_file, folder_alias):
    """takes data_type data from masks stored in hdf5_file

    Takes a list of 4D fMRI nifti-files and masks the
    data with all masks in the list of nifti-files mask_files.
    These files are assumed to represent the same space, i.e.
    that of the functional acquisitions. 
    These are saved in hdf5_file, in the folder folder_alias.

    Parameters
    ----------
    data_types_wildcards : list
        list of data types to be loaded.
        correspond to nifti_names in mask_2_hdf5
    roi_name_wildcard : str
        wildcard for masks. 
        corresponds to mask_name in mask_2_hdf5.
    hdf5_file : str
        absolute path to hdf5 file.
    folder_alias : str
        name of the folder in the hdf5 file from which data
        should be loaded.

    Returns
    -------
    output_data : list
        list of numpy arrays corresponding to data_types and roi_name_wildcards
    """
    import tables
    import itertools

    h5file = tables.open_file(hdf5_file, mode = "r")

    try:
        folder_alias_run_group = h5file.get_node(where = '/', name = folder_alias, classname='Group')
    except NoSuchNodeError:
        # import actual data
        print('No group ' + folder_alias + ' in this file')
        return None


    all_roi_names = h5file.list_nodes(where = '/' + folder_alias, classname = 'Group')
    roi_names = [rn for rn in all_roi_names if roi_name_wildcard in rn]
    if len(roi_names) == 0:
        print('No rois corresponding to ' + roi_wildcard + ' in group ' + folder_alias)
        return None
    
    data_arrays = []
    for rn in roi_names:
        try:
            roi_node = h5file.get_node(where = '/' + folder_alias, name = roi_name, classname='Group')
        except NoSuchNodeError:
            print('No data corresponding to ' + roi_name + ' in group ' + folder_alias)
            pass
        all_data_array_names = h5file.list_nodes(where = '/' + folder_alias + '/' + roi_name)
        data_array_names = [[dan for dan in all_data_array_names if dtwc in dan and dtwc + '_' not in dan] for dtwd in data_types_wildcards]
        data_array_names = list(itertools.chain(*data_array_names))
        
        if sort_data_types:
            data_array_names = sorted(data_array_names)

        if len(data_array_names) == 0:
            print('No data corresponding to ' + str(data_types_wildcards) + ' in group ' + folder_alias + '/' + rn)
            pass
        else:
            print('Taking data corresponding to ' + str(data_array_names) + ' from group ' + folder_alias + '/' + rn)
            data_arrays.append([])
            for dan in data_array_names:
                data_arrays[-1].append(eval('roi_node.' + dan + '.read()'))

    all_roi_data_np = np.hstack(all_roi_data).T

    return all_roi_data_np

