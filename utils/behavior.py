from __future__ import division, print_function

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
        temp_tsv = op.join(tempdir, alias.replace('eye', 'trials') + '.tsv')
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

def convert_streamup_trials_to_tsv(hdf5_file, tsv_file = None ):

    from hedfpy import HDFEyeOperator
    import tempfile
    import os
    import os.path as op
    import numpy as np
    import pandas as pd

    tr_key = 116.0
    reward_signal = 3.0

    alias = op.splitext(op.split(hdf5_file)[-1])[0]

    ho = HDFEyeOperator(hdf5_file)
    ho.open_hdf_file('a')
    node = ho.h5f.get_node('/' + alias)
    trial_phase_data = node.trial_phases.read()
    events = node.events.read()
    parameters = node.parameters.read()

    if tsv_file == None:
        tempdir = tempfile.mkdtemp()
        temp_tsv = op.join(tempdir, alias.replace('eye', 'trials') + '.tsv')
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
        temp_tsv = op.join(tempdir, alias.replace('eye', 'trials') + '.tsv')
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

def convert_variable_trials_to_tsv(hdf5_file, tsv_file = None ):

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
        temp_tsv = op.join(tempdir, alias.replace('eye', 'trials') + '.tsv')
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

    # the sound was 800 ms delayed wrt the stimulus onset
    reward_times = trial_parameters['reward_delay'] + (trial_phase_timestamps - tr_timestamps[0] ) / 1000.0

    # now, we need to re-code the probabilities. 
    # first, we detect which feedback sound was the high-reward one. 
    # we do this by looking at the 0-orientation trials, because
    # these were always no-stim reward trials: there were no no-stim no-reward trials
    reward_signal = np.array(trial_parameters['sound'])[np.array(np.array(trial_parameters['stim_orientation'] == 0.0))].mean()
    which_trials_reward = np.array(trial_parameters['sound'] == reward_signal)
    
    # now, we can find which orientations had which reward probability, since 
    # the reward probability is not theoretical, but practical, on every run
    orientations = np.sort(np.unique(np.array(trial_parameters['stim_orientation'])))
    reward_probabilities = [(np.array(trial_parameters['sound'])[np.array(trial_parameters['stim_orientation'] == ori)] == reward_signal).mean() 
                                for ori in orientations]

    # and we use this to fill in the per-trial value for reward probability
    reward_probs_per_trial = np.zeros(len(trial_phase_timestamps))
    for ori, rp in zip(orientations, reward_probabilities):
        reward_probs_per_trial[np.array(trial_parameters['stim_orientation'] == ori)] = rp
    trial_parameters['reward_probability'] = pd.Series(reward_probs_per_trial)
 
    trial_parameters['feedback_time'] = pd.Series(reward_times)
    trial_parameters['feedback_was_reward'] = pd.Series(np.array(which_trials_reward, dtype = np.int))

    trial_parameters['stim_onset_time'] = pd.Series((trial_phase_timestamps - tr_timestamps[0] ) / 1000.0)

    trial_parameters = trial_parameters.sort_values(by = 'stim_onset_time')

    trial_parameters.to_csv(temp_tsv, sep = '\t', float_format = '%3.2f', header = True, index = False)

    return temp_tsv