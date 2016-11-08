def which_files_have_physio(all_input_files, 
                                all_physio_files, 
                                input_extension = '_bold.nii.gz', 
                                physio_extension = '_physio.log'):
    '''selects nifti files that have corresponding physio log files.
    Parameters
    ----------
    all_input_files : list
        list of nifti, or other filenames
    all_physio_files : list
        list of physio filenames
    input_extension : string
        string that allows identification of nifti files, without
        deleting their filename correspondence with physio files.
        i.e. the input filename without this extension should be
        identical to the physio filename without its extension.
        This means that this string depends on the preceding
        preprocessing steps, adding a _suffix for each step.
    physio_extension : string
        string that allows identification of physio files, without
        deleting their filename correspondence with nifit files.

    Example
    -------
    >>> nii_physio_files = which_nii_files_have_physio(['bla1_bold.nii.gz', 'bla2_bold.nii.gz', 'bla3_bold.nii.gz'],
    >>>                                                 ['bla1_physio.log', 'bla2_physio.log'])
    
    Out: ['bla1_bold.nii.gz', 'bla2_bold.nii.gz']

    '''
    import os.path as op

    drs = [op.split(s)[0] for s in all_input_files]
    fns_decap = [op.split(s)[-1].split(input_extension)[0] for s in all_input_files]
    physio_fns_decap = [op.split(s)[-1].split(physio_extension)[0] for s in all_physio_files]

    files_with_physio_ids = [fns_decap.index(ph) for ph in physio_fns_decap if ph in fns_decap]
    files_with_physio = [all_input_files[i] for i in files_with_physio_ids]

    return files_with_physio

def convert_behavior(hdf5_files, reward_signal_unpredictable):
    '''convert all behavioral data from hdf5 files to .tsv files, 
    for BIDS format. Returns one tsv file for each hdf5 file
    
    Parameters
    ----------
    hdf5_files : list
        list of hdf5 filenames


    '''    

    from utils.behavior import convert_unpredictable_trials_to_tsv, \
                                convert_predictable_trials_to_tsv, \
                                convert_variable_trials_to_tsv, \
                                convert_streamup_trials_to_tsv


    tsv_files = []
    for h5 in hdf5_files:
        if 'stream-up' in h5:
            tsv_files.append(convert_streamup_trials_to_tsv(hdf5_file = h5))
        elif 'unpredictable' in h5:
            tsv_files.append(convert_unpredictable_trials_to_tsv(hdf5_file = h5, reward_signal = reward_signal_unpredictable))
        elif 'predictable' in h5:
            tsv_files.append(convert_predictable_trials_to_tsv(hdf5_file = h5))
        elif 'variable' in h5:
            tsv_files.append(convert_variable_trials_to_tsv(hdf5_file = h5))

    return tsv_files


def create_all_calcarine_reward_preprocessing_workflow(analysis_info, name='all_calcarine_reward'):
    import os.path as op
    import tempfile
    import nipype.pipeline as pe
    from nipype.interfaces import fsl
    from nipype.interfaces.utility import Function, Merge, IdentityInterface
    from spynoza.nodes.utils import get_scaninfo, dyns_min_1, topup_scan_params, apply_scan_params
    from nipype.interfaces.io import SelectFiles, DataSink


    # Importing of custom nodes from spynoza packages; assumes that spynoza is installed:
    # pip install git+https://github.com/spinoza-centre/spynoza.git@develop
    from spynoza.nodes.filtering import savgol_filter
    from spynoza.nodes.utils import get_scaninfo, pickfirst, percent_signal_change, average_over_runs, pickle_to_json, set_nifti_intercept_slope
    from spynoza.workflows.topup_unwarping import create_topup_workflow
    from spynoza.workflows.B0_unwarping import create_B0_workflow
    from spynoza.workflows.registration import create_registration_workflow
    from spynoza.workflows.retroicor import create_retroicor_workflow
    from spynoza.workflows.sub_workflows.masks import create_masks_from_surface_workflow
    from spynoza.nodes.fit_nuisances import fit_nuisances

    from motion_correction import create_motion_correction_workflow
    from utils.utils import convert_edf_2_hdf5, mask_nii_2_hdf5
    from utils.utils import convert_hdf_eye_to_tsv




    ########################################################################################
    # nodes
    ########################################################################################

    input_node = pe.Node(IdentityInterface(
                fields=['raw_directory', 
                    'output_directory', 
                    'FS_ID', 
                    'FS_subject_dir',
                    'sub_id', 
                    'sess_id',
                    'which_file_is_EPI_space',
                    'standard_file', 
                    'psc_func', 
                    'MB_factor',
                    'tr',
                    'slice_direction',
                    'phys_sample_rate',
                    'slice_timing',
                    'slice_order',
                    'nr_dummies',
                    'wfs',
                    'epi_factor',
                    'acceleration',
                    'te_diff',
                    'echo_time',
                    'phase_encoding_direction']), name='inputspec')

    # i/o node
    datasource_templates = dict(func='{sub_id}/{sess_id}/func/*bold.nii.gz',
                                physio='{sub_id}/{sess_id}/func/*.log',
                                events='{sub_id}/{sess_id}/func/*.pickle',
                                eye='{sub_id}/{sess_id}/func/*.edf',
                                anat='{sub_id}/{sess_id}/anat/*_inplaneT2.nii.gz',
                                reg='{sub_id}/{sess_id}/anat/*_inplaneT2.mat') # ,
    datasource = pe.Node(SelectFiles(datasource_templates, sort_filelist = True, raise_on_empty = False), 
        name = 'datasource')

    output_node = pe.Node(IdentityInterface(fields=([
            'temporal_filtered_files', 
            'percent_signal_change_files'])), name='outputspec')

    # node for temporal filtering
    sgfilter = pe.MapNode(Function(input_names=['in_file'],
                                    output_names=['out_file'],
                                    function=savgol_filter),
                      name='sgfilter', iterfield=['in_file'])

    # node for converting pickle files to json
    pj = pe.MapNode(Function(input_names=['in_file'],
                                    output_names=['out_file'],
                                    function=pickle_to_json),
                      name='pj', iterfield=['in_file'])

    # node for percent signal change
    psc = pe.MapNode(Function(input_names=['in_file', 'func'],
                                    output_names=['out_file'],
                                    function=percent_signal_change),
                      name='percent_signal_change', iterfield=['in_file'])

    # node to select the nii files that have physio information
    physio_for_niis = pe.Node(Function(input_names=['all_input_files', 'all_physio_files'],
                                output_names=['files_with_physio'],
                                function=which_files_have_physio),
                  name='physio_for_niis')

    physio_for_mocos =  pe.Node(Function(input_names=['all_input_files', 'all_physio_files', 'input_extension'],
                                output_names=['files_with_physio'],
                                function=which_files_have_physio),
                  name='physio_for_mocos')
    physio_for_mocos.inputs.input_extension = '_bold_brain_mcf.niiext_moco_pars.par'

    # node for nuisance regression
    fit_nuis = pe.MapNode(Function(input_names=['in_file', 'slice_regressor_list', 'vol_regressors'],
                                    output_names=['res_file', 'rsq_file', 'beta_file'],
                                    function=fit_nuisances),
                      name='fit_nuisances', iterfield=['in_file', 'slice_regressor_list', 'vol_regressors']) 

    edf_converter = pe.MapNode(Function(input_names=['edf_file'],
                                    output_names=['hdf5_file'],
                                    function=convert_edf_2_hdf5),
                      name='edf_converter', iterfield=['edf_file'])

    hdf_tsv_converter = pe.MapNode(Function(input_names=['hdf5_file'],
                                    output_names=['tsv_file'],
                                    function=convert_hdf_eye_to_tsv),
                      name='hdf_tsv_converter', iterfield=['hdf5_file'])

    behavior_tsv_converter = pe.Node(Function(input_names=['hdf5_files', 'reward_signal_unpredictable'],
                                    output_names=['tsv_files'],
                                    function=convert_behavior),
                      name='behavior_tsv_converter')
    behavior_tsv_converter.inputs.reward_signal_unpredictable = analysis_info['which_reward_sound_unpredictable']

    # node for datasinking
    datasink = pe.Node(DataSink(), name='sinker')
    datasink.inputs.parameterization = False

    ########################################################################################
    # workflow
    ########################################################################################

    # the actual top-level workflow
    all_calcarine_reward_workflow = pe.Workflow(name=name)

    all_calcarine_reward_workflow.connect(input_node, 'raw_directory', datasource, 'base_directory')
    all_calcarine_reward_workflow.connect(input_node, 'sub_id', datasource, 'sub_id')
    all_calcarine_reward_workflow.connect(input_node, 'sess_id', datasource, 'sess_id')

    # behavioral pickle to json
    all_calcarine_reward_workflow.connect(datasource, 'events', pj, 'in_file')
    all_calcarine_reward_workflow.connect(datasource, 'eye', edf_converter, 'edf_file')

    all_calcarine_reward_workflow.connect(edf_converter, 'hdf5_file', hdf_tsv_converter, 'hdf5_file')
    all_calcarine_reward_workflow.connect(edf_converter, 'hdf5_file', behavior_tsv_converter, 'hdf5_files')

    
    # motion correction, using T2 inplane anatomicals to prime 
    # the motion correction to the standard EPI space
    motion_proc = create_motion_correction_workflow(analysis_info, 'moco')
    all_calcarine_reward_workflow.connect(input_node, 'tr', motion_proc, 'inputspec.tr')
    all_calcarine_reward_workflow.connect(input_node, 'output_directory', motion_proc, 'inputspec.output_directory')
    all_calcarine_reward_workflow.connect(input_node, 'which_file_is_EPI_space', motion_proc, 'inputspec.which_file_is_EPI_space')
    all_calcarine_reward_workflow.connect(datasource, 'func', motion_proc, 'inputspec.in_files')
    all_calcarine_reward_workflow.connect(datasource, 'anat', motion_proc, 'inputspec.inplane_T2_files')
    all_calcarine_reward_workflow.connect(datasource, 'reg', motion_proc, 'inputspec.T2_files_reg_matrices')

    # registration
    reg = create_registration_workflow(analysis_info, name = 'reg')
    all_calcarine_reward_workflow.connect(input_node, 'output_directory', reg, 'inputspec.output_directory')
    all_calcarine_reward_workflow.connect(motion_proc, 'outputspec.EPI_space_file', reg, 'inputspec.EPI_space_file')
    all_calcarine_reward_workflow.connect(input_node, 'FS_ID', reg, 'inputspec.freesurfer_subject_ID')
    all_calcarine_reward_workflow.connect(input_node, 'FS_subject_dir', reg, 'inputspec.freesurfer_subject_dir')
    all_calcarine_reward_workflow.connect(input_node, 'standard_file', reg, 'inputspec.standard_file')
    # the T1_file entry could be empty sometimes, depending on the output of the
    # datasource. Check this.
    # all_calcarine_reward_workflow.connect(reg, 'outputspec.T1_file', reg, 'inputspec.T1_file')    

    # temporal filtering
    all_calcarine_reward_workflow.connect(motion_proc, 'outputspec.motion_corrected_files', sgfilter, 'in_file')

    # node for percent signal change
    all_calcarine_reward_workflow.connect(input_node, 'psc_func', psc, 'func')
    all_calcarine_reward_workflow.connect(sgfilter, 'out_file', psc, 'in_file')

    # connect filtering and psc results to output node 
    all_calcarine_reward_workflow.connect(sgfilter, 'out_file', output_node, 'temporal_filtered_files')
    all_calcarine_reward_workflow.connect(psc, 'out_file', output_node, 'percent_signal_change_files')

    # retroicor functionality
    retr = create_retroicor_workflow(name = 'retroicor', order_or_timing = analysis_info['retroicor_order_or_timing'])

    # select those nii files with physio
    all_calcarine_reward_workflow.connect(datasource, 'func', physio_for_niis, 'all_input_files')
    all_calcarine_reward_workflow.connect(datasource, 'physio', physio_for_niis, 'all_physio_files')
    all_calcarine_reward_workflow.connect(physio_for_niis, 'files_with_physio', retr, 'inputspec.in_files')

    all_calcarine_reward_workflow.connect(datasource, 'physio', retr, 'inputspec.phys_files')
    all_calcarine_reward_workflow.connect(input_node, 'nr_dummies', retr, 'inputspec.nr_dummies')
    all_calcarine_reward_workflow.connect(input_node, 'MB_factor', retr, 'inputspec.MB_factor')
    all_calcarine_reward_workflow.connect(input_node, 'tr', retr, 'inputspec.tr')
    all_calcarine_reward_workflow.connect(input_node, 'slice_direction', retr, 'inputspec.slice_direction')
    all_calcarine_reward_workflow.connect(input_node, 'slice_timing', retr, 'inputspec.slice_timing')
    all_calcarine_reward_workflow.connect(input_node, 'slice_order', retr, 'inputspec.slice_order')
    all_calcarine_reward_workflow.connect(input_node, 'phys_sample_rate', retr, 'inputspec.phys_sample_rate')

    # fit nuisances from retroicor
    # all_calcarine_reward_workflow.connect(retr, 'outputspec.evs', fit_nuis, 'slice_regressor_list')
    # select the relevant motion correction files, using selection function
    # all_calcarine_reward_workflow.connect(motion_proc, 'outputspec.extended_motion_correction_parameters', physio_for_mocos, 'all_input_files')
    # all_calcarine_reward_workflow.connect(datasource, 'physio', physio_for_mocos, 'all_physio_files')
    # all_calcarine_reward_workflow.connect(physio_for_mocos, 'files_with_physio', fit_nuis, 'vol_regressors')

    # all_calcarine_reward_workflow.connect(physio_for_niis, 'files_with_physio', fit_nuis, 'in_file')

    # surface-based label import in to EPI space
    masks_from_surface = create_masks_from_surface_workflow(name = 'masks_from_surface')
    masks_from_surface.inputs.inputspec.label_directory = 'retmap'
    masks_from_surface.inputs.inputspec.fill_thresh = 0.005
    masks_from_surface.inputs.inputspec.re = '*.label'
    
    all_calcarine_reward_workflow.connect(motion_proc, 'outputspec.EPI_space_file', masks_from_surface, 'inputspec.EPI_space_file')
    all_calcarine_reward_workflow.connect(input_node, 'output_directory', masks_from_surface, 'inputspec.output_directory')
    all_calcarine_reward_workflow.connect(input_node, 'FS_subject_dir', masks_from_surface, 'inputspec.freesurfer_subject_dir')
    all_calcarine_reward_workflow.connect(input_node, 'FS_ID', masks_from_surface, 'inputspec.freesurfer_subject_ID')
    all_calcarine_reward_workflow.connect(reg, 'rename_register.out_file', masks_from_surface, 'inputspec.reg_file')

    # ########################################################################################
    # # outputs via datasink
    # ########################################################################################

    all_calcarine_reward_workflow.connect(input_node, 'output_directory', datasink, 'base_directory')

    # # sink out events and eyelink files
    all_calcarine_reward_workflow.connect(pj, 'out_file', datasink, 'events')
    
    all_calcarine_reward_workflow.connect(sgfilter, 'out_file', datasink, 'tf')
    all_calcarine_reward_workflow.connect(psc, 'out_file', datasink, 'psc')

    all_calcarine_reward_workflow.connect(retr, 'outputspec.new_phys', datasink, 'phys.log')
    all_calcarine_reward_workflow.connect(retr, 'outputspec.fig_file', datasink, 'phys.figs')
    all_calcarine_reward_workflow.connect(retr, 'outputspec.evs', datasink, 'phys.evs')

    # all_calcarine_reward_workflow.connect(fit_nuis, 'res_file', datasink, 'phys.res')
    # all_calcarine_reward_workflow.connect(fit_nuis, 'rsq_file', datasink, 'phys.rsq')
    # all_calcarine_reward_workflow.connect(fit_nuis, 'beta_file', datasink, 'phys.betas')

    all_calcarine_reward_workflow.connect(masks_from_surface, 'outputspec.masks', datasink, 'masks')

    all_calcarine_reward_workflow.connect(datasource, 'eye', datasink, 'eye')
    all_calcarine_reward_workflow.connect(edf_converter, 'hdf5_file', datasink, 'eye.h5')
    all_calcarine_reward_workflow.connect(hdf_tsv_converter, 'tsv_file', datasink, 'eye.tsv')
    all_calcarine_reward_workflow.connect(behavior_tsv_converter, 'tsv_files', datasink, 'events.tsv')

    return all_calcarine_reward_workflow
