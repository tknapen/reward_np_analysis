def plot_fir_results_unpredictable(deconvolution_interval_timepoints, event_names, fir_time_course_dict = {}, zero_interval = [-0.75, -0.25], use_zero_interval = True, suffix = ''):
    import numpy as np
    import matplotlib.pyplot as pl
    import tempfile
    import os.path as op
    import seaborn as sn
    import pandas as pd

    sn.set(style="ticks")
    low_alpha, high_alpha = 0.4, 1.0

    colors = [[en, c, a] for en, c, a in zip(
        event_names,
        ['g','b','g','b'],
        [high_alpha,high_alpha,low_alpha,low_alpha]
        )]

    # do we subtract a certain baseline level from the curves?
    if use_zero_interval:
        zero_interval_bool = (deconvolution_interval_timepoints > zero_interval[0]) & (deconvolution_interval_timepoints < zero_interval[1])
        used_fir_time_course_dict = {en:(fir_time_course_dict[en] - fir_time_course_dict[en][zero_interval_bool].mean()) for en in event_names}
    else:
        used_fir_time_course_dict = fir_time_course_dict
    
    # create reward time-course from earlier results
    # the 'all_reward' timecourse is the average response to reward regardless of stimulus presence.

    reward_fir_timecourse = 0.5 * (( used_fir_time_course_dict['visual_reward'] + used_fir_time_course_dict['fixation_reward'] ) - ( used_fir_time_course_dict['visual_no_reward'] + used_fir_time_course_dict['fixation_no_reward'] ))
    stim_fir_timecourse = 0.5 * (( used_fir_time_course_dict['visual_reward'] + used_fir_time_course_dict['visual_no_reward'] ) - ( used_fir_time_course_dict['fixation_reward'] + used_fir_time_course_dict['fixation_no_reward'] ))

    # add the added information
    fir_time_course_dict.update({'all_reward': reward_fir_timecourse, 'all_stim': stim_fir_timecourse})

    f, (ax1, ax2) = pl.subplots(1, 2, sharey=True, figsize = (11,4))
    # first subplot
    for en, c, a in colors:
        ax1.plot(deconvolution_interval_timepoints, used_fir_time_course_dict[en], c = c, alpha = a, label = en)
    ax1.axvline(x=0, linewidth=0.5)
    ax1.axhline(y=0, linewidth=0.5)
    ax1.legend()

    ax2.plot(deconvolution_interval_timepoints, reward_fir_timecourse, c = 'r', alpha = 1, label = 'reward')
    ax2.plot(deconvolution_interval_timepoints, stim_fir_timecourse, c = 'k', alpha = 1, label = 'visual')
    ax2.axvline(x=0, linewidth=0.5)
    ax2.axhline(y=0, linewidth=0.5)
    ax2.legend()

    sn.despine(ax=ax1, offset=10, trim=True)
    sn.despine(ax=ax2, offset=10, trim=True)

    temp_folder = tempfile.mkdtemp()
    out_figure = op.join(temp_folder, 'unpredictable_%i_%s.pdf'%(int(use_zero_interval), suffix))

    pl.savefig(out_figure)

    return out_figure

def plot_fir_results_predictable(deconvolution_interval_timepoints, rename_dict, event_names, fir_time_course_dict = {}, zero_interval = [-0.75, -0.25], use_zero_interval = True, suffix = ''):
    import numpy as np
    import matplotlib.pyplot as pl
    import tempfile
    import os.path as op
    import seaborn as sn

    sn.set(style="ticks")
    low_alpha, high_alpha = 0.4, 1.0

    # do we subtract a certain baseline level from the curves?
    if use_zero_interval:
        zero_interval_bool = (deconvolution_interval_timepoints > zero_interval[0]) & (deconvolution_interval_timepoints < zero_interval[1])
        used_fir_time_course_dict = {en:(fir_time_course_dict[en] - fir_time_course_dict[en][zero_interval_bool].mean()) for en in event_names}
    else:
        used_fir_time_course_dict = fir_time_course_dict
    

    colors = [[en, c, a] for en, c, a in zip(
        ['rewarded_location-rewarded_orientation', 
        'rewarded_location-nonrewarded_orientation', 
        'nonrewarded_location-rewarded_orientation', 
        'nonrewarded_location-nonrewarded_orientation',
        'fixation_reward',
        'fixation_no_reward'],
        ['r','r','g','g','k','k'],
        [high_alpha,low_alpha,high_alpha,low_alpha,high_alpha,low_alpha]
        )]

    rev_rename_dict = {value:key for key, value in rename_dict.iteritems()}

    # create reward time-course from earlier results
    # the 'all_reward' timecourse is the average response to reward regardless of stimulus presence.
    reward_stim_fir_timecourse = used_fir_time_course_dict[rev_rename_dict['rewarded_location-rewarded_orientation']] - \
                                used_fir_time_course_dict[rev_rename_dict['rewarded_location-nonrewarded_orientation']] 
    nonreward_stim_fir_timecourse = used_fir_time_course_dict[rev_rename_dict['nonrewarded_location-rewarded_orientation']] - \
                                used_fir_time_course_dict[rev_rename_dict['nonrewarded_location-nonrewarded_orientation']] 
    reward_fix_fir_timecourse = used_fir_time_course_dict[rev_rename_dict['fixation_reward']] - \
                                used_fir_time_course_dict[rev_rename_dict['fixation_no_reward']] 
    reward_fir_timecourse = 0.5 * (reward_stim_fir_timecourse + reward_fix_fir_timecourse)

    # add the added information
    fir_time_course_dict.update({   'all_reward': reward_fir_timecourse, 
                                    'reward_stim': reward_stim_fir_timecourse, 
                                    'nonreward_stim': nonreward_stim_fir_timecourse,
                                    'fix_reward': reward_fix_fir_timecourse
                                    })
   for en, c, a in colors[:4]:
        fir_time_course_dict.update({en: used_fir_time_course_dict[rev_rename_dict[en]]})
    
    f, (ax1, ax2) = pl.subplots(1, 2, sharey=True, figsize = (11,4))
    # first subplot
    for en, c, a in colors:
        ax1.plot(deconvolution_interval_timepoints, used_fir_time_course_dict[rev_rename_dict[en]], c = c, alpha = a, label = en)
    ax1.axvline(x=0, linewidth=0.5)
    ax1.axhline(y=0, linewidth=0.5)
    ax1.legend()

    ax2.plot(deconvolution_interval_timepoints, reward_stim_fir_timecourse, c = 'r', alpha = low_alpha, label = '+reward_stimulus')
    ax2.plot(deconvolution_interval_timepoints, reward_fix_fir_timecourse, c = 'k', alpha = low_alpha, label = '-reward_stimulus')
    ax2.plot(deconvolution_interval_timepoints, nonreward_stim_fir_timecourse, c = 'g', alpha = low_alpha, label = 'reward_fixation')

    ax2.plot(deconvolution_interval_timepoints, reward_fir_timecourse, c = 'k', alpha = high_alpha, label = 'mean_reward')

    ax2.axvline(x=0, linewidth=0.5)
    ax2.axhline(y=0, linewidth=0.5)
    ax2.legend()
    
    sn.despine(ax=ax1, offset=10, trim=True)
    sn.despine(ax=ax2, offset=10, trim=True)

    temp_folder = tempfile.mkdtemp()
    out_figure = op.join(temp_folder, 'predictable_%i_%s.pdf'%(int(use_zero_interval), suffix))

    pl.savefig(out_figure)

    return out_figure

def plot_fir_results_variable(deconvolution_interval_timepoints, event_names, fir_time_course_dict = {}, zero_interval = [-0.75, -0.25], use_zero_interval = True, suffix = ''):
    import numpy as np
    import matplotlib.pyplot as pl
    import tempfile
    import os.path as op
    import seaborn as sn

    sn.set(style="ticks")
    low_alpha, high_alpha = 0.4, 1.0

    # do we subtract a certain baseline level from the curves?
    if use_zero_interval:
        zero_interval_bool = (deconvolution_interval_timepoints > zero_interval[0]) & (deconvolution_interval_timepoints < zero_interval[1])
        used_fir_time_course_dict = {en:(fir_time_course_dict[en] - fir_time_course_dict[en][zero_interval_bool].mean()) for en in event_names}
    else:
        used_fir_time_course_dict = fir_time_course_dict

    colors = [[en, c, a, ls] for en, c, a, ls in zip(
        ['75S', '75r', '75p', '50S', '50r', '50p', '25S', '25r', '25p', 'fixation_reward'],
        ['g','g','g','b','b','b','r','r','r','k'],
        [high_alpha,high_alpha,low_alpha,high_alpha,high_alpha,low_alpha,high_alpha,high_alpha,low_alpha,high_alpha],
        ['-','-','--','-','-','--','-','-','--','-']
        )]

    diffs = [used_fir_time_course_dict[perc+'r'] - used_fir_time_course_dict[perc+'p'] for perc in ['75', '50', '25']]
    diffs = [np.squeeze(d) for d in diffs]
    diffs.extend([used_fir_time_course_dict['fixation_reward']])
    reward_colors = [[en, c, a, resp] for en, c, a, resp in zip(
            ['75', '50', '25','fix'],
            ['g','b','r','k'],
            [high_alpha, high_alpha, high_alpha, high_alpha],
            diffs
            )]
    
    f, (ax1, ax2, ax3) = pl.subplots(1, 3, sharey=True, figsize = (16,4))
    for en, c, a, ls in colors:
        if en[-1] == 'S':
            ax1.plot(deconvolution_interval_timepoints, used_fir_time_course_dict[en], c = c, alpha = a, label = en, ls = ls)
        elif (en[-1] == 'r') | (en[-1] == 'p'):
            ax2.plot(deconvolution_interval_timepoints, used_fir_time_course_dict[en], c = c, alpha = a, label = en, ls = ls)

    for en, c, a, resp in reward_colors:
        ax3.plot(deconvolution_interval_timepoints, resp, c = c, alpha = a, label = en)

    for ax in [ax1, ax2, ax3]:
        ax.axvline(x=0, linewidth=0.5)
        ax.axhline(y=0, linewidth=0.5)
        ax.legend()
        sn.despine(ax=ax, offset=10, trim=True)

    temp_folder = tempfile.mkdtemp()
    out_figure = op.join(temp_folder, 'variable_%i_%s.pdf'%(int(use_zero_interval), suffix))

    pl.savefig(out_figure)

    return out_figure