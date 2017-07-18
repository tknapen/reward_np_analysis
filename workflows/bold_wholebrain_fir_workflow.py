def create_bold_wholebrain_fir_workflow(analysis_info, name = 'wb_roi'):
    import nipype.pipeline as pe
    from nipype.interfaces.utility import Function, Merge, IdentityInterface
    from nipype.interfaces.io import SelectFiles, DataSink

    from utils.bold_wholebrain_fir import BOLD_FIR_files
    from spynoza.nodes.utils import pickfirst

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
                                # unpredictable reward experiment needs behavior files, moco and physio
                                unpredictable_in_files='{sub_id}/psc/*-unpredictable_reward_*.nii.gz',
                                unpredictable_behavior_tsv_files='{sub_id}/events/tsv/*-unpredictable_reward_*.tsv',
                                # variable reward experiment needs behavior files, moco and physio
                                variable_in_files='{sub_id}/psc/*-variable_*_reward_*.nii.gz',
                                variable_behavior_tsv_files='{sub_id}/events/tsv/*-variable_*_reward_*.tsv',
                                # mapper_files
                                unpredictable_glm_mapper_list='{sub_id}/psc/*-unpredictable_mapper*.nii.gz',
                                predictable_glm_mapper_list='{sub_id}/psc/*-predictable_mapper*.nii.gz',
                                )
    datasource = pe.Node(SelectFiles(datasource_templates, sort_filelist = True, raise_on_empty = False, force_lists = True), 
        name = 'datasource')

    predictable_FIR= pe.Node(Function(input_names=[
                                            'analysis_info',
                                            'experiment',
                                            'fir_file_reward_list',
                                            'glm_file_mapper_list',
                                            'behavior_file_list',
                                            'mapper_contrast',
                                            'h5_file',
                                            'roi_list'
                                            ],
                                output_names=['out_figures'],
                                function=BOLD_FIR_files,
                                imports=imports),
                                name='predictable_FIR')
    predictable_FIR.inputs.analysis_info = analysis_info
    predictable_FIR.inputs.experiment = 'predictable'
    predictable_FIR.inputs.mapper_contrast = 'rewarded_stim'
    predictable_FIR.inputs.roi_list = ['V1','V2','V3','V4','LO','V3AB','O']

    unpredictable_FIR= pe.Node(Function(input_names=[
                                            'analysis_info',
                                            'experiment',
                                            'fir_file_reward_list',
                                            'glm_file_mapper_list',
                                            'behavior_file_list',
                                            'mapper_contrast',
                                            'h5_file',
                                            'roi_list'
                                            ],
                                output_names=['out_figures'],
                                function=BOLD_FIR_files,
                                imports=imports),
                                name='unpredictable_FIR')
    unpredictable_FIR.inputs.analysis_info = analysis_info
    unpredictable_FIR.inputs.experiment = 'unpredictable'
    unpredictable_FIR.inputs.mapper_contrast = 'stim'
    unpredictable_FIR.inputs.roi_list = ['V1','V2','V3','V4','LO','V3AB','O']

    variable_FIR= pe.Node(Function(input_names=[
                                            'analysis_info',
                                            'experiment',
                                            'fir_file_reward_list',
                                            'glm_file_mapper_list',
                                            'behavior_file_list',
                                            'mapper_contrast',
                                            'h5_file',
                                            'roi_list'
                                            ],
                                output_names=['out_figures'],
                                function=BOLD_FIR_files,
                                imports=imports),
                                name='variable_FIR')
    variable_FIR.inputs.analysis_info = analysis_info
    variable_FIR.inputs.experiment = 'variable'
    variable_FIR.inputs.mapper_contrast = 'stim'
    variable_FIR.inputs.roi_list = ['V1','V2','V3','V4','LO','V3AB','O']


    # the actual top-level workflow
    wb_fir_roi_workflow = pe.Workflow(name=name)

    wb_fir_roi_workflow.connect(input_node, 'preprocessed_directory', datasource, 'base_directory')
    wb_fir_roi_workflow.connect(input_node, 'sub_id', datasource, 'sub_id')

    # variable reward pupil FIR
    wb_fir_roi_workflow.connect(datasource, 'variable_behavior_tsv_files', variable_FIR, 'behavior_file_list')
    wb_fir_roi_workflow.connect(datasource, ('all_roi_file', pickfirst), variable_FIR, 'h5_file')
    wb_fir_roi_workflow.connect(datasource, 'variable_in_files', variable_FIR, 'fir_file_reward_list')
    wb_fir_roi_workflow.connect(datasource, 'unpredictable_glm_mapper_list', variable_FIR, 'glm_file_mapper_list')

    # predictable reward pupil FIR
    wb_fir_roi_workflow.connect(datasource, 'predictable_behavior_tsv_files', predictable_FIR, 'behavior_file_list')
    wb_fir_roi_workflow.connect(datasource, ('all_roi_file', pickfirst), predictable_FIR, 'h5_file')
    wb_fir_roi_workflow.connect(datasource, 'predictable_in_files', predictable_FIR, 'fir_file_reward_list')
    wb_fir_roi_workflow.connect(datasource, 'predictable_glm_mapper_list', predictable_FIR, 'glm_file_mapper_list')

    # unpredictable reward pupil FIR
    wb_fir_roi_workflow.connect(datasource, 'unpredictable_behavior_tsv_files', unpredictable_FIR, 'behavior_file_list')
    wb_fir_roi_workflow.connect(datasource, ('all_roi_file', pickfirst), unpredictable_FIR, 'h5_file')
    wb_fir_roi_workflow.connect(datasource, 'unpredictable_in_files', unpredictable_FIR, 'fir_file_reward_list')
    wb_fir_roi_workflow.connect(datasource, 'unpredictable_glm_mapper_list', unpredictable_FIR, 'glm_file_mapper_list')

    # datasink
    datasink = pe.Node(DataSink(), name='sinker')
    datasink.inputs.parameterization = False

    wb_fir_roi_workflow.connect(input_node, 'preprocessed_directory', datasink, 'base_directory')
    wb_fir_roi_workflow.connect(input_node, 'sub_id', datasink, 'container')

    wb_fir_roi_workflow.connect(unpredictable_FIR, 'out_figures', datasink, 'fir.@unpredictable_FIR')
    wb_fir_roi_workflow.connect(predictable_FIR, 'out_figures', datasink, 'fir.@predictable_FIR')
    wb_fir_roi_workflow.connect(variable_FIR, 'out_figures', datasink, 'fir.@variable_FIR')

    return wb_fir_roi_workflow



