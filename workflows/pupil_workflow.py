def create_pupil_workflow(analysis_info, name = 'pupil'):
    import nipype.pipeline as pe
    from nipype.interfaces.utility import Function, Merge, IdentityInterface
    from nipype.interfaces.io import SelectFiles, DataSink

    from utils.pupil import fit_FIR_pupil_files

    imports = ['from utils.behavior import behavior_timing',
        'from utils.plotting import plot_fir_results_unpredictable',
        'from utils.plotting import plot_fir_results_predictable',
        'from utils.plotting import plot_fir_results_variable',
    ]

    input_node = pe.Node(IdentityInterface(
                            fields=['preprocessed_directory', 
                                    'sub_id'
                        ]), name='inputspec')

    # i/o node
    datasource_templates = dict(
                                all_roi_file='{sub_id}/h5/roi.h5',                                
                                # predictable reward experiment needs behavior files and moco but no physio
                                predictable_in_files='{sub_id}/psc/*-predictable_reward_*.nii.gz',
                                predictable_behavior_tsv_files='{sub_id}/events/tsv/*-predictable_reward_*.tsv',
                                predictable_eye_h5_files='{sub_id}/eye/h5/*-predictable_reward_*.h5',
                                # unpredictable reward experiment needs behavior files, moco and physio
                                unpredictable_in_files='{sub_id}/psc/*-unpredictable_reward_*.nii.gz',
                                unpredictable_behavior_tsv_files='{sub_id}/events/tsv/*-unpredictable_reward_*.tsv',
                                unpredictable_eye_h5_files='{sub_id}/eye/h5/*-unpredictable_reward_*.h5',
                                # variable reward experiment needs behavior files, moco and physio
                                variable_in_files='{sub_id}/psc/*-variable_*_reward_*.nii.gz',
                                variable_behavior_tsv_files='{sub_id}/events/tsv/*-variable_*_reward_*.tsv',
                                variable_eye_h5_files='{sub_id}/eye/h5/*-variable_*_reward_*.h5',
                                )
    datasource = pe.Node(SelectFiles(datasource_templates, sort_filelist = True, raise_on_empty = False), 
        name = 'datasource')

    predictable_pupil_FIR= pe.Node(Function(input_names=[
                                            'experiment',
                                            'eye_h5_file_list',
                                            'behavior_file_list',
                                            'h5_file',
                                            'in_files', 
                                            'fir_frequency',
                                            'fir_interval',
                                            'data_type',
                                            'lost_signal_rate_threshold'
                                            ],
                                output_names=['out_figures'],
                                function=fit_FIR_pupil_files,
                                imports=imports),
                                name='predictable_pupil_FIR')
    predictable_pupil_FIR.inputs.fir_frequency = analysis_info['pupil_fir_frequency']
    predictable_pupil_FIR.inputs.fir_interval = analysis_info['pupil_fir_interval']
    predictable_pupil_FIR.inputs.experiment = 'predictable'
    predictable_pupil_FIR.inputs.data_type = analysis_info['pupil_data_type'] 
    predictable_pupil_FIR.inputs.lost_signal_rate_threshold = analysis_info['pupil_lost_signal_rate_threshold'] 

    unpredictable_pupil_FIR= pe.Node(Function(input_names=[
                                            'experiment',
                                            'eye_h5_file_list',
                                            'behavior_file_list',
                                            'h5_file',
                                            'in_files', 
                                            'fir_frequency',
                                            'fir_interval',
                                            'data_type',
                                            'lost_signal_rate_threshold'
                                            ],
                                output_names=['out_figures'],
                                function=fit_FIR_pupil_files,
                                imports=imports),
                                name='unpredictable_pupil_FIR')
    unpredictable_pupil_FIR.inputs.fir_frequency = analysis_info['pupil_fir_frequency']
    unpredictable_pupil_FIR.inputs.fir_interval = analysis_info['pupil_fir_interval']
    unpredictable_pupil_FIR.inputs.experiment = 'unpredictable'
    unpredictable_pupil_FIR.inputs.data_type = analysis_info['pupil_data_type'] 
    unpredictable_pupil_FIR.inputs.lost_signal_rate_threshold = analysis_info['pupil_lost_signal_rate_threshold'] 

    variable_pupil_FIR= pe.Node(Function(input_names=[
                                            'experiment',
                                            'eye_h5_file_list',
                                            'behavior_file_list',
                                            'h5_file',
                                            'in_files', 
                                            'fir_frequency',
                                            'fir_interval',
                                            'data_type',
                                            'lost_signal_rate_threshold'
                                            ],
                                output_names=['out_figures'],
                                function=fit_FIR_pupil_files,
                                imports=imports),
                                name='variable_pupil_FIR')
    variable_pupil_FIR.inputs.fir_frequency = analysis_info['pupil_fir_frequency']
    variable_pupil_FIR.inputs.fir_interval = analysis_info['pupil_fir_interval']
    variable_pupil_FIR.inputs.experiment = 'variable'
    variable_pupil_FIR.inputs.data_type = analysis_info['pupil_data_type'] 
    variable_pupil_FIR.inputs.lost_signal_rate_threshold = analysis_info['pupil_lost_signal_rate_threshold'] 


    # the actual top-level workflow
    pupil_analysis_workflow = pe.Workflow(name=name)

    pupil_analysis_workflow.connect(input_node, 'preprocessed_directory', datasource, 'base_directory')
    pupil_analysis_workflow.connect(input_node, 'sub_id', datasource, 'sub_id')

    # variable reward pupil FIR
    pupil_analysis_workflow.connect(datasource, 'variable_eye_h5_files', variable_pupil_FIR, 'eye_h5_file_list')
    pupil_analysis_workflow.connect(datasource, 'variable_behavior_tsv_files', variable_pupil_FIR, 'behavior_file_list')
    pupil_analysis_workflow.connect(datasource, 'all_roi_file', variable_pupil_FIR, 'h5_file')
    pupil_analysis_workflow.connect(datasource, 'variable_in_files', variable_pupil_FIR, 'in_files')

    # predictable reward pupil FIR
    pupil_analysis_workflow.connect(datasource, 'predictable_eye_h5_files', predictable_pupil_FIR, 'eye_h5_file_list')
    pupil_analysis_workflow.connect(datasource, 'predictable_behavior_tsv_files', predictable_pupil_FIR, 'behavior_file_list')
    pupil_analysis_workflow.connect(datasource, 'all_roi_file', predictable_pupil_FIR, 'h5_file')
    pupil_analysis_workflow.connect(datasource, 'predictable_in_files', predictable_pupil_FIR, 'in_files')

    # unpredictable reward pupil FIR
    pupil_analysis_workflow.connect(datasource, 'unpredictable_eye_h5_files', unpredictable_pupil_FIR, 'eye_h5_file_list')
    pupil_analysis_workflow.connect(datasource, 'unpredictable_behavior_tsv_files', unpredictable_pupil_FIR, 'behavior_file_list')
    pupil_analysis_workflow.connect(datasource, 'all_roi_file', unpredictable_pupil_FIR, 'h5_file')
    pupil_analysis_workflow.connect(datasource, 'unpredictable_in_files', unpredictable_pupil_FIR, 'in_files')

    # datasink
    datasink = pe.Node(DataSink(), name='sinker')
    datasink.inputs.parameterization = False

    pupil_analysis_workflow.connect(input_node, 'preprocessed_directory', datasink, 'base_directory')
    pupil_analysis_workflow.connect(input_node, 'sub_id', datasink, 'container')

    pupil_analysis_workflow.connect(unpredictable_pupil_FIR, 'out_figures', datasink, 'pupil.@unpredictable_pupil_FIR')
    pupil_analysis_workflow.connect(predictable_pupil_FIR, 'out_figures', datasink, 'pupil.@predictable_pupil_FIR')
    pupil_analysis_workflow.connect(variable_pupil_FIR, 'out_figures', datasink, 'pupil.@variable_pupil_FIR')

    return pupil_analysis_workflow



