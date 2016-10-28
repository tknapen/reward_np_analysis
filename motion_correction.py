from spynoza.nodes import EPI_file_selector

def _extend_motion_parameters(moco_par_file, tr, sg_args = {'window_length': 120, 'deriv':0, 'polyorder':3, 'mode':'nearest'}):
    import os.path as op
    import numpy as np
    from sklearn import decomposition
    from scipy.signal import savgol_filter

    ext_out_file = moco_par_file[:-7] + 'ext_moco_pars.par'
    new_out_file = moco_par_file[:-7] + 'new_moco_pars.par'

    sg_args['window_length'] = int(sg_args['window_length'] / tr)
    # Window must be odd-shaped
    if sg_args['window_length'] % 2 == 0:
        sg_args['window_length'] += 1

    moco_pars = np.loadtxt(moco_par_file)
    moco_pars = moco_pars - savgol_filter(moco_pars, axis = 0, **sg_args)    

    dt_moco_pars = np.diff(np.vstack((np.ones((1,6)), moco_pars)), axis = 0)
    ddt_moco_pars = np.diff(np.vstack((np.ones((1,6)), dt_moco_pars)), axis = 0)

    ext_moco_pars = np.hstack((moco_pars, dt_moco_pars, ddt_moco_pars))

    # blow up using abs(), perform pca and take original number of 18 components
    amp = np.hstack((moco_pars, dt_moco_pars, ddt_moco_pars, dt_moco_pars**2, ddt_moco_pars**2))
    pca = decomposition.PCA(n_components = 18)
    pca.fit(amp)
    new_moco_pars = pca.transform(amp)

    np.savetxt(new_out_file, new_moco_pars, fmt='%f', delimiter='\t')
    np.savetxt(ext_out_file, ext_moco_pars, fmt='%f', delimiter='\t')

    return new_out_file, ext_out_file

def select_target_T2(T2_file_list, target_session):
    target_T2 = [T2 for T2 in T2_file_list if target_session in T2][0]

    return target_T2

def select_target_epi(epi_file_list, T2_file_list, target_session, which_file):
    from spynoza.nodes import EPI_file_selector

    target_T2 = [T2 for T2 in T2_file_list if target_session in T2][0]
    all_target_epis = [epi for epi in epi_file_list if target_session in epi]
    target_epi = EPI_file_selector(which_file, all_target_epis)

    print("XXXXX " + target_epi)

    return target_epi

def select_T2_for_epi(epi_file, T2_file_list):
    import os.path as op
    epi_filename = op.split(epi_file)[-1]
    T2_sessions = [op.split(T2)[-1].split('_t2w')[0] for T2 in T2_file_list]
    which_T2_file = [T2 for (T2f, T2) in zip(T2_sessions, T2_file_list) if T2f in epi_filename][0]

    return which_T2_file

def find_all_epis_for_inplane_anats(epi_file_list, inplane_anats, inplane_anat_suffix = '_t2w_brain.nii.gz'):
    '''selects epi nifti files that correspond to the session of each of inplane_anats.
    Parameters
    ----------
    epi_file_list : list
        list of nifti, or other filenames
    inplane_anats : list
        list of nifti filenames
    inplane_anat_suffix : string
        string that, when taken from the inplane_anat's filename, leaves the session's label.

    Returns:
    list of lists, with epi_file_list files distributed among len(inplane_anats) sublists.
    '''
    import os.path as op
    session_labels = [op.split(ipa)[-1].split(inplane_anat_suffix)[0] for ipa in inplane_anats]

    output_lists = []
    for sl in session_labels:
        output_lists.append([epi for epi in epi_file_list if sl in epi])

    return output_list

def create_motion_correction_workflow(analysis_info, name = 'moco'):
    """uses sub-workflows to perform different registration steps.
    Requires fsl and freesurfer tools
    Parameters
    ----------
    name : string
        name of workflow
    
    Example
    -------
    >>> motion_correction_workflow = create_motion_correction_workflow('motion_correction_workflow')
    >>> motion_correction_workflow.inputs.inputspec.output_directory = '/data/project/raw/BIDS/sj_1/'
    >>> motion_correction_workflow.inputs.inputspec.in_files = ['sub-001.nii.gz','sub-002.nii.gz']
    >>> motion_correction_workflow.inputs.inputspec.which_file_is_EPI_space = 'middle'
 
    Inputs::
          inputspec.output_directory : directory in which to sink the result files
          inputspec.in_files : list of functional files
          inputspec.which_file_is_EPI_space : determines which file is the 'standard EPI space'
    Outputs::
           outputspec.EPI_space_file : standard EPI space file, one timepoint
           outputspec.motion_corrected_files : motion corrected files
           outputspec.motion_correction_plots : motion correction plots
           outputspec.motion_correction_parameters : motion correction parameters
    """
    import os.path as op
    import nipype.pipeline as pe
    import nipype.interfaces.fsl as fsl
    import nipype.interfaces.utility as util
    import nipype.interfaces.io as nio
    from nipype.interfaces.utility import Function, IdentityInterface
    import nipype.interfaces.utility as niu
    

    ########################################################################################
    # nodes
    ########################################################################################

    input_node = pe.Node(IdentityInterface(fields=[
                'in_files', 
                'inplane_T2_files',
                'output_directory', 
                'which_file_is_EPI_space', 
                'sub_id', 
                'tr']), name='inputspec')
    output_node = pe.Node(IdentityInterface(fields=([
                'motion_corrected_files', 
                'EPI_space_file',
                'T2_space_file',
                'motion_correction_plots', 
                'motion_correction_parameters', 
                'extended_motion_correction_parameters', 
                'new_motion_correction_parameters'])), name='outputspec')

    EPI_file_selector_node = pe.Node(Function(input_names=['which_file', 'in_files'], output_names='raw_EPI_space_file',
                                       function=EPI_file_selector), name='EPI_file_selector_node')

    # motion_correct_EPI_space = pe.Node(interface=fsl.MCFLIRT(
    #                 save_mats = True, 
    #                 stats_imgs = True,
    #                 save_plots = True, 
    #                 save_rms = True,
    #                 cost = 'normmi', 
    #                 interpolation = 'sinc',
    #                 dof = 6, 
    #                 # ref_vol = 0
    #                 ), name='realign_space')

    # mean_bold = pe.Node(interface=fsl.maths.MeanImage(dimension='T'), name='mean_space')

    # new approach, which should aid in the joint motion correction of 
    # multiple sessions together, by pre-registering each run.
    # the strategy would be to, for each run, take the first TR
    # and FLIRT-align (6dof) it to the EPI_space file. 
    # then we can use this as an --infile argument to mcflirt.

    select_target_T2_node = pe.Node(Function(input_names=['T2_file_list', 'target_session'], output_names=['which_T2'],
                                    function=select_target_T2), name='select_target_T2_node')
    select_target_T2_node.inputs.target_session = analysis_info['target_session']

    # select_target_epi_node = pe.Node(Function(input_names=['epi_file_list', 'T2_file_list', 'target_session', 'which_file'], output_names=['target_epi'],
    #                                 function=select_target_epi), name='select_target_epi_node')
    # select_target_epi_node.inputs.target_session = analysis_info['target_session']

    select_T2_for_epi_node = pe.MapNode(Function(input_names=['epi_file', 'T2_file_list'], output_names=['which_T2_file'],
                                       function=select_T2_for_epi), name='select_T2_for_epi_node', iterfield = ['epi_file'])

    bet_T2_node = pe.MapNode(interface=
        fsl.BET(frac = analysis_info['T2_bet_f_value'], 
                vertical_gradient = analysis_info['T2_bet_g_value'], 
                functional=False, mask = True, padding = True), name='bet_T2', iterfield=['in_file'])

    bet_epi_node = pe.MapNode(interface=
        fsl.BET(frac = analysis_info['T2_bet_f_value'], 
                vertical_gradient = analysis_info['T2_bet_g_value'], 
                functional=True, mask = True), name='bet_epi', iterfield=['in_file'])

    motion_correct_all = pe.MapNode(interface=fsl.MCFLIRT(
                    save_mats = True, 
                    save_plots = True, 
                    cost = 'normmi', 
                    interpolation = 'sinc',
                    stats_imgs = True,
                    dof = 6
                    ), name='realign_all',
                                iterfield = ['in_file', 'ref_file'])

    plot_motion = pe.MapNode(interface=fsl.PlotMotionParams(in_source='fsl'),
                            name='plot_motion',
                            iterfield=['in_file'])

    extend_motion_pars = pe.MapNode(Function(input_names=['moco_par_file', 'tr'], output_names=['new_out_file', 'ext_out_file'],
                                       function=_extend_motion_parameters), name='extend_motion_pars', iterfield = ['moco_par_file'])

    # registration node is set up for rigid-body within-modality reg
    reg_flirt_N = pe.MapNode(fsl.FLIRT(cost_func='normcorr', output_type = 'NIFTI_GZ', dof = 7, interp = 'sinc'), 
                        name = 'reg_flirt_N', iterfield = ['in_file'])

    regapply_moco_node = pe.MapNode(interface=
        fsl.ApplyXfm(interp = 'spline'), name='regapply_moco_node', iterfield=['in_file', 'in_matrix_file'])

    resample_epis = pe.MapNode(fsl.maths.MathsCommand(args = ' -subsamp2offc '), name='resample_epis', iterfield = ['in_file'])
    resample_target_T2 = pe.Node(fsl.maths.MathsCommand(args = ' -subsamp2offc '), name='resample_target_T2')

    rename = pe.Node(niu.Rename(format_string='session_EPI_space',
                            keep_ext=True),
                    name='namer')

    rename_T2 = pe.Node(niu.Rename(format_string='session_T2_space',
                            keep_ext=True),
                    name='namer_T2')

    ########################################################################################
    # workflow
    ########################################################################################

    motion_correction_workflow = pe.Workflow(name=name)

    motion_correction_workflow.connect(input_node, 'in_files', bet_epi_node, 'in_file')
    motion_correction_workflow.connect(input_node, 'inplane_T2_files', bet_T2_node, 'in_file')

    # select example func data, and example T2 space
    # motion_correction_workflow.connect(input_node, 'which_file_is_EPI_space', select_target_epi_node, 'which_file')
    # motion_correction_workflow.connect(bet_epi_node, 'out_file', select_target_epi_node, 'epi_file_list')
    # motion_correction_workflow.connect(bet_T2_node, 'out_file', select_target_epi_node, 'T2_file_list')
    motion_correction_workflow.connect(bet_T2_node, 'out_file', select_target_T2_node, 'T2_file_list')

    # motion correct and average the standard EPI file
    # motion_correction_workflow.connect(select_target_epi_node, 'target_epi', motion_correct_EPI_space, 'in_file')
    # motion_correction_workflow.connect(motion_correct_EPI_space, 'out_file', mean_bold, 'in_file')

    # output node, for later saving
    # motion_correction_workflow.connect(mean_bold, 'out_file', output_node, 'EPI_space_file')
    motion_correction_workflow.connect(select_target_T2_node, 'which_T2', output_node, 'EPI_space_file')

    # find the relevant T2 files for each of the epi files
    motion_correction_workflow.connect(bet_epi_node, 'out_file', select_T2_for_epi_node, 'epi_file')
    motion_correction_workflow.connect(bet_T2_node, 'out_file', select_T2_for_epi_node, 'T2_file_list')

    # motion correction across runs
    # motion_correction_workflow.connect(prereg_flirt_N, 'out_matrix_file', motion_correct_all, 'init')
    motion_correction_workflow.connect(bet_epi_node, 'out_file', motion_correct_all, 'in_file')
    motion_correction_workflow.connect(select_T2_for_epi_node, 'which_T2_file', motion_correct_all, 'ref_file')
    # motion_correction_workflow.connect(mean_bold, 'out_file', motion_correct_all, 'ref_file')
    
    # the registration
    motion_correction_workflow.connect(select_T2_for_epi_node, 'which_T2_file', reg_flirt_N, 'in_file')
    motion_correction_workflow.connect(select_target_T2_node, 'which_T2', reg_flirt_N, 'reference')

    # output of motion correction of all files
    motion_correction_workflow.connect(motion_correct_all, 'par_file', output_node, 'motion_correction_parameters')
    motion_correction_workflow.connect(motion_correct_all, 'out_file', regapply_moco_node, 'in_file')

    motion_correction_workflow.connect(reg_flirt_N, 'out_matrix_file', regapply_moco_node, 'in_matrix_file')
    motion_correction_workflow.connect(select_target_T2_node, 'which_T2', regapply_moco_node, 'reference')


    motion_correction_workflow.connect(regapply_moco_node, 'out_file', resample_epis, 'in_file')
    motion_correction_workflow.connect(resample_epis, 'out_file', output_node, 'motion_corrected_files')

    motion_correction_workflow.connect(motion_correct_all, 'par_file', extend_motion_pars, 'moco_par_file')
    motion_correction_workflow.connect(input_node, 'tr', extend_motion_pars, 'tr')
    motion_correction_workflow.connect(extend_motion_pars, 'ext_out_file', output_node, 'extended_motion_correction_parameters')
    motion_correction_workflow.connect(extend_motion_pars, 'new_out_file', output_node, 'new_motion_correction_parameters')

    ########################################################################################
    # Plot the estimated motion parameters
    ########################################################################################

    plot_motion.iterables = ('plot_type', ['rotations', 'translations'])
    motion_correction_workflow.connect(motion_correct_all, 'par_file', plot_motion, 'in_file')
    motion_correction_workflow.connect(plot_motion, 'out_file', output_node, 'motion_correction_plots')

    ########################################################################################
    # outputs via datasink
    ########################################################################################
    datasink = pe.Node(nio.DataSink(), name='sinker')
    datasink.inputs.parameterization = False

    # first link the workflow's output_directory into the datasink.
    motion_correction_workflow.connect(input_node, 'output_directory', datasink, 'base_directory')
    motion_correction_workflow.connect(input_node, 'sub_id', datasink, 'container')

    motion_correction_workflow.connect(select_target_T2_node, 'which_T2', resample_target_T2, 'in_file')
    motion_correction_workflow.connect(resample_target_T2, 'out_file', rename, 'in_file')
    motion_correction_workflow.connect(rename, 'out_file', datasink, 'reg')

    motion_correction_workflow.connect(select_target_T2_node, 'which_T2', rename_T2, 'in_file')
    motion_correction_workflow.connect(rename_T2, 'out_file', datasink, 'reg.@T2')

    motion_correction_workflow.connect(regapply_moco_node, 'out_file', datasink, 'mcf.hr')
    motion_correction_workflow.connect(resample_epis, 'out_file', datasink, 'mcf')
    motion_correction_workflow.connect(motion_correct_all, 'par_file', datasink, 'mcf.motion_pars')
    motion_correction_workflow.connect(plot_motion, 'out_file', datasink, 'mcf.motion_plots')
    motion_correction_workflow.connect(extend_motion_pars, 'ext_out_file', datasink, 'mcf.ext_motion_pars')
    motion_correction_workflow.connect(extend_motion_pars, 'new_out_file', datasink, 'mcf.new_motion_pars')

    motion_correction_workflow.connect(bet_T2_node, 'out_file', datasink, 'mcf.T2s')
    motion_correction_workflow.connect(motion_correct_all, 'out_file', datasink, 'mcf.hr_per_session')
    motion_correction_workflow.connect(reg_flirt_N, 'out_file', datasink, 'mcf.T2_per_session')

    return motion_correction_workflow

