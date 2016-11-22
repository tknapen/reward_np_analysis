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
    which_trials_reward = (np.array(trial_parameters['sound'] + reward_signal, dtype = int) % 2).astype(bool)
    reward_times = np.zeros(len(trial_phase_timestamps))
    # the sound was 800 ms delayed wrt the stimulus onset
    reward_times[which_trials_reward] = \
                (800.0 + trial_phase_timestamps[which_trials_reward] - tr_timestamps[0] ) / 1000.0

    trial_parameters['reward_time'] = pd.Series(reward_times)
    trial_parameters['stim_onset_time'] = pd.Series((trial_phase_timestamps - tr_timestamps[0] ) / 1000.0)

    trial_parameters = trial_parameters.sort_values(by = 'stim_onset_time')

    trial_parameters.to_csv(temp_tsv, sep = '\t', float_format = '%3.2f', header = True, index = False)

    return temp_tsv

def convert_streamup_trials_to_tsv(hdf5_file, tsv_file = None, reward_signal = 1.0 ):

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
    which_trials_reward = (np.array(trial_parameters['sound']) % 2).astype(bool)
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

def behavior_timing(experiment, behavior_file_list, dyns, TR):
    import pandas as pd
    import numpy as np

    # always return these
    rename_dict = {}
    which_event_name_rewarded = ''
    # per-experiment implementation of event types and times:
    if (experiment == 'unpredictable') | (experiment == 'stream-up'):
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

    elif experiment == 'predictable':
        event_names = ['left_cw', 'left_ccw', 'right_cw', 'right_ccw', 'fixation_no_reward', 'fixation_reward']
        # get the behavior and format event times and gains for FIR
        all_behav_event_times = {en:[] 
                    for en in event_names}

        for x in range(len(behavior_file_list)):
            this_run_events = pd.read_csv(behavior_file_list[x], delimiter='\t')

            fixation_reward_events = (this_run_events.reward_time != 0) & (this_run_events.contrast == 0)
            fixation_no_reward_events = (this_run_events.reward_time == 0) & (this_run_events.contrast == 0)
            # stim position bools
            left_stim_events = this_run_events.x_position < 0
            right_stim_events = this_run_events.x_position > 0
            # stim orientation bools
            cw_stim_events = this_run_events.orientation > 0
            ccw_stim_events = this_run_events.orientation < 0

            # the four stimulus classes
            all_stim_events = [
                                np.array(this_run_events[left_stim_events & cw_stim_events].stim_onset_time) + x * dyns * TR,     # L_CW
                                np.array(this_run_events[left_stim_events & ccw_stim_events].stim_onset_time) + x * dyns * TR,    # L_CCW
                                np.array(this_run_events[right_stim_events & cw_stim_events].stim_onset_time) + x * dyns * TR,    # R_CW
                                np.array(this_run_events[right_stim_events & ccw_stim_events].stim_onset_time) + x * dyns * TR,   # R_CCW
                                np.array(this_run_events[fixation_no_reward_events].stim_onset_time) + x * dyns * TR,             # fixation_no_reward
                                np.array(this_run_events[fixation_reward_events].stim_onset_time) + x * dyns * TR,                # fixation_reward
                                ]

            for name, times in zip(event_names, all_stim_events):
                all_behav_event_times[name].append(times)

            # which stimulus is rewarded
            if x == 0: # only investigate this for the first run
                stim_rew_events = np.array(this_run_events[(this_run_events.reward_time != 0) & (this_run_events.contrast != 0)].stim_onset_time) + x * dyns * TR

                which_event_name_rewarded = [event_names[i] for i, ev in enumerate(all_stim_events[:4]) 
                                                if ev[0] == stim_rew_events[0]][0]
                rewarded_location, rewarded_orientation = which_event_name_rewarded.split('_')

                # secondary names for event types, depending on rewarded location and orientation
                for i in range(4):
                    condition_string = ''
                    if event_names[i].split('_')[0] == rewarded_location:
                        condition_string += 'rewarded_location'
                    else:
                        condition_string += 'nonrewarded_location'
                    condition_string += '-'
                    if event_names[i].split('_')[1] == rewarded_orientation:
                        condition_string += 'rewarded_orientation'
                    else:
                        condition_string += 'nonrewarded_orientation'
                    rename_dict.update({event_names[i]: condition_string})
                rename_dict.update({'fixation_no_reward': 'fixation_no_reward'})
                rename_dict.update({'fixation_reward': 'fixation_reward'})

        # set the times up as a 1-D array
        for name in event_names:
                all_behav_event_times[name] = np.concatenate(all_behav_event_times[name])

    elif experiment == 'variable':
        event_names = ['75S', '75r', '75p', '50S', '50r', '50p', '25S', '25r', '25p', 'fixation_reward']
        # get the behavior and format event times and gains for FIR
        all_behav_event_times = {en:[] 
                    for en in event_names}

        for x in range(len(behavior_file_list)):
            this_run_events = pd.read_csv(behavior_file_list[x], delimiter='\t')

            # which trials lacked a stimulus, with a reward tone
            fixation_reward_events = (this_run_events.feedback_time != 0) & (this_run_events.contrast == 0)
            # percentage bools for trials
            HR_stim_trials = this_run_events.reward_probability == 0.75
            MR_stim_trials = this_run_events.reward_probability == 0.5
            LR_stim_trials = this_run_events.reward_probability == 0.25
            # feedback bools for trials
            R_trials = this_run_events.feedback_was_reward == 1
            NR_trials = this_run_events.feedback_was_reward == 0

            # the times
            all_behav_event_times['fixation_reward'].append(np.array(this_run_events[fixation_reward_events].feedback_time) + x * dyns * TR)

            all_behav_event_times['75S'].append(np.array(this_run_events[HR_stim_trials].stim_onset_time) + x * dyns * TR)
            all_behav_event_times['75r'].append(np.array(this_run_events[HR_stim_trials & R_trials].feedback_time) + x * dyns * TR)
            all_behav_event_times['75p'].append(np.array(this_run_events[HR_stim_trials & NR_trials].feedback_time) + x * dyns * TR)

            all_behav_event_times['50S'].append(np.array(this_run_events[MR_stim_trials].stim_onset_time) + x * dyns * TR)
            all_behav_event_times['50r'].append(np.array(this_run_events[MR_stim_trials & R_trials].feedback_time) + x * dyns * TR)
            all_behav_event_times['50p'].append(np.array(this_run_events[MR_stim_trials & NR_trials].feedback_time) + x * dyns * TR)

            all_behav_event_times['25S'].append(np.array(this_run_events[LR_stim_trials].stim_onset_time) + x * dyns * TR)
            all_behav_event_times['25r'].append(np.array(this_run_events[LR_stim_trials & R_trials].feedback_time) + x * dyns * TR)
            all_behav_event_times['25p'].append(np.array(this_run_events[LR_stim_trials & NR_trials].feedback_time) + x * dyns * TR)

        for name in event_names:
                all_behav_event_times[name] = np.concatenate(all_behav_event_times[name])

    return event_names, all_behav_event_times, rename_dict, which_event_name_rewarded