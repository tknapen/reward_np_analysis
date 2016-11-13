from __future__ import division

def sublists_for_phys(slice_regressor_list, in_files):
    # no need to assume a sorted list
    slice_regressor_list.sort()
    nr_phys_regressors_per_file = len(slice_regressor_list) / len(in_files)
    slice_regressor_lists = []
    if round(nr_phys_regressors_per_file) == nr_phys_regressors_per_file:
        for x in range(len(in_files)):
            slice_regressor_lists.append(slice_regressor_list[x*nr_phys_regressors_per_file:(x+1)*nr_phys_regressors_per_file])
    else:
        print('Unequal number of physiology regressors for retroicor. Check \n%s'%str(slice_regressor_list))

    return slice_regressor_lists

def create_whole_brain_GLM_workflow(analysis_info, name = 'GLM'):
    import nipype.pipeline as pe
    from nipype.interfaces.utility import Function, Merge, IdentityInterface
    from nipype.interfaces.io import SelectFiles, DataSink

    from utils.GLM import fit_glm_nuisances_single_file, fit_FIR_nuisances_all_files


    input_node = pe.Node(IdentityInterface(
                            fields=['preprocessed_directory', 
                                    'sub_id'
                        ]), name='inputspec')

    # i/o node
    datasource_templates = dict(
                                example_func='{sub_id}/reg/example_func.nii.gz',
                                # predictable experiment has no physiology
                                predictable_mapper_in_file='{sub_id}/psc/*-predictable_mapper_1_*.nii.gz',
                                predictable_mapper_tsv_file='{sub_id}/events/tsv/*-predictable_mapper_1_*.tsv',
                                predictable_mapper_mcf_par='{sub_id}/mcf/ext_motion_pars/*-predictable_mapper_1_*.par',
                                # predictable reward experiment needs behavior files and moco but no physio
                                predictable_in_files='{sub_id}/psc/*-predictable_reward_*.nii.gz',
                                predictable_behavior_tsv_file='{sub_id}/events/tsv/*-predictable_reward_*.tsv',
                                predictable_mcf_pars='{sub_id}/mcf/ext_motion_pars/*-predictable_reward_*.par',
                                # unpredictable experiment has physiology but no behavior because: block design
                                unpredictable_mapper_in_file='{sub_id}/psc/*-unpredictable_mapper_1_*.nii.gz',
                                unpredictable_mapper_physio_files='{sub_id}/phys/evs/*-unpredictable_mapper_1_*.nii.gz',
                                unpredictable_mapper_mcf_par='{sub_id}/mcf/ext_motion_pars/*-unpredictable_mapper_1_*.par',
                                # unpredictable reward experiment needs behavior files, moco and physio
                                unpredictable_in_files='{sub_id}/psc/*-unpredictable_reward_*.nii.gz',
                                unpredictable_behavior_tsv_file='{sub_id}/events/tsv/*-unpredictable_reward_*.tsv',
                                unpredictable_physio_files='{sub_id}/phys/evs/*-unpredictable_reward_*.nii.gz',
                                unpredictable_mcf_pars='{sub_id}/mcf/ext_motion_pars/*-unpredictable_reward_*.par',
                                # variable reward experiment needs behavior files, moco and physio
                                variable_in_files='{sub_id}/psc/*-variable_*_reward_*.nii.gz',
                                variable_behavior_tsv_file='{sub_id}/events/tsv/*-variable_*_reward_*.tsv',
                                variable_physio_files='{sub_id}/phys/evs/*-variable_*_reward_*.nii.gz',
                                variable_mcf_pars='{sub_id}/mcf/ext_motion_pars/*-variable_*_reward_*.par'
                                )
    datasource = pe.Node(SelectFiles(datasource_templates, sort_filelist = True, raise_on_empty = False), 
        name = 'datasource')

    unpredictable_split_phys_list = pe.Node(Function(input_names=['slice_regressor_list', 'in_files'],
                                output_names=['slice_regressor_lists'],
                                function=sublists_for_phys),
                                name='unpredictable_split_phys_list')

    variable_split_phys_list = pe.Node(Function(input_names=['slice_regressor_list', 'in_files'],
                                output_names=['slice_regressor_lists'],
                                function=sublists_for_phys),
                                name='variable_split_phys_list')

    unpredictable_GLM = pe.Node(Function(input_names=['in_file', 
                                            'slice_regressor_list', 
                                            'vol_regressors', 
                                            'num_components', 
                                            'method', 
                                            'mapper', 
                                            'dm_upscale_factor',
                                            'tsv_behavior_file'],
                                output_names=['out_files'],
                                function=fit_glm_nuisances_single_file),
                                name='unpredictable_GLM')
    unpredictable_GLM.inputs.mapper = 'unpredictable'
    unpredictable_GLM.inputs.num_components = 24
    unpredictable_GLM.inputs.method = 'PCA'
    unpredictable_GLM.inputs.dm_upscale_factor = 10

    predictable_GLM = pe.Node(Function(input_names=['in_file', 
                                            'slice_regressor_list', 
                                            'vol_regressors', 
                                            'num_components', 
                                            'method', 
                                            'mapper', 
                                            'dm_upscale_factor',
                                            'tsv_behavior_file'],
                                output_names=['out_files'],
                                function=fit_glm_nuisances_single_file),
                                name='predictable_GLM')
    predictable_GLM.inputs.mapper = 'predictable'
    predictable_GLM.inputs.num_components = 0   # no physio, just motion correction nuisances
    predictable_GLM.inputs.method = 'PCA'
    predictable_GLM.inputs.dm_upscale_factor = 10

    unpredictable_FIR= pe.Node(Function(input_names=[
                                            'experiment',
                                            'example_func',
                                            'in_files', 
                                            'slice_regressor_lists', 
                                            'vol_regressor_list', 
                                            'behavior_file_list',
                                            'fir_frequency',
                                            'fir_interval'
                                            ],
                                output_names=['out_files'],
                                function=fit_FIR_nuisances_all_files),
                                name='unpredictable_FIR')
    unpredictable_FIR.inputs.fir_frequency = analysis_info['fir_frequency']
    unpredictable_FIR.inputs.fir_interval = analysis_info['fir_interval']
    unpredictable_FIR.inputs.experiment = 'unpredictable'

    predictable_FIR= pe.Node(Function(input_names=[
                                            'experiment',
                                            'example_func',
                                            'in_files', 
                                            'slice_regressor_lists', 
                                            'vol_regressor_list', 
                                            'behavior_file_list',
                                            'fir_frequency',
                                            'fir_interval'
                                            ],
                                output_names=['out_files'],
                                function=fit_FIR_nuisances_all_files),
                                name='predictable_FIR')
    predictable_FIR.inputs.fir_frequency = analysis_info['fir_frequency']
    predictable_FIR.inputs.fir_interval = analysis_info['fir_interval']
    predictable_FIR.inputs.experiment = 'predictable'
    predictable_FIR.inputs.slice_regressor_lists = [[]] # no physio regressors

    variable_FIR= pe.Node(Function(input_names=[
                                            'experiment',
                                            'example_func',
                                            'in_files', 
                                            'slice_regressor_lists', 
                                            'vol_regressor_list', 
                                            'behavior_file_list',
                                            'fir_frequency',
                                            'fir_interval'
                                            ],
                                output_names=['out_files'],
                                function=fit_FIR_nuisances_all_files),
                                name='variable_FIR')
    variable_FIR.inputs.fir_frequency = analysis_info['fir_frequency']
    variable_FIR.inputs.fir_interval = analysis_info['fir_interval']
    variable_FIR.inputs.experiment = 'variable'

    # the actual top-level workflow
    whole_brain_analysis_workflow = pe.Workflow(name=name)

    whole_brain_analysis_workflow.connect(input_node, 'preprocessed_directory', datasource, 'base_directory')
    whole_brain_analysis_workflow.connect(input_node, 'sub_id', datasource, 'sub_id')

    # predictable mapper GLM
    whole_brain_analysis_workflow.connect(datasource, 'predictable_mapper_in_file', predictable_GLM, 'in_file')
    whole_brain_analysis_workflow.connect(datasource, 'predictable_mapper_mcf_par', predictable_GLM, 'vol_regressors')
    whole_brain_analysis_workflow.connect(datasource, 'predictable_mapper_tsv_file', predictable_GLM, 'tsv_behavior_file')

    # predictable reward FIR
    whole_brain_analysis_workflow.connect(datasource, 'predictable_in_files', predictable_FIR, 'in_files')
    whole_brain_analysis_workflow.connect(datasource, 'predictable_mcf_pars', predictable_FIR, 'vol_regressor_list')
    whole_brain_analysis_workflow.connect(datasource, 'predictable_behavior_tsv_file', predictable_FIR, 'behavior_file_list')
    whole_brain_analysis_workflow.connect(datasource, 'example_func', predictable_FIR, 'example_func')

    # unpredictable mapper GLM
    whole_brain_analysis_workflow.connect(datasource, 'unpredictable_mapper_in_file', unpredictable_GLM, 'in_file')
    whole_brain_analysis_workflow.connect(datasource, 'unpredictable_mapper_mcf_par', unpredictable_GLM, 'vol_regressors')
    whole_brain_analysis_workflow.connect(datasource, 'unpredictable_mapper_physio_files', unpredictable_GLM, 'slice_regressor_list')

    # unpredictable reward FIR; first split the 1D slice regressor list to 2D
    whole_brain_analysis_workflow.connect(datasource, 'unpredictable_physio_files', unpredictable_split_phys_list, 'slice_regressor_list')
    whole_brain_analysis_workflow.connect(datasource, 'unpredictable_in_files', unpredictable_split_phys_list, 'in_files')
    whole_brain_analysis_workflow.connect(unpredictable_split_phys_list, 'slice_regressor_lists', unpredictable_FIR, 'slice_regressor_lists')

    whole_brain_analysis_workflow.connect(datasource, 'unpredictable_in_files', unpredictable_FIR, 'in_files')
    whole_brain_analysis_workflow.connect(datasource, 'unpredictable_mcf_pars', unpredictable_FIR, 'vol_regressor_list')
    whole_brain_analysis_workflow.connect(datasource, 'unpredictable_behavior_tsv_file', unpredictable_FIR, 'behavior_file_list')
    whole_brain_analysis_workflow.connect(datasource, 'example_func', unpredictable_FIR, 'example_func')

    # variable reward FIR; first split the 1D slice regressor list to 2D
    whole_brain_analysis_workflow.connect(datasource, 'variable_physio_files', variable_split_phys_list, 'slice_regressor_list')
    whole_brain_analysis_workflow.connect(datasource, 'variable_in_files', variable_split_phys_list, 'in_files')
    whole_brain_analysis_workflow.connect(variable_split_phys_list, 'slice_regressor_lists', variable_FIR, 'slice_regressor_lists')

    whole_brain_analysis_workflow.connect(datasource, 'variable_in_files', variable_FIR, 'in_files')
    whole_brain_analysis_workflow.connect(datasource, 'variable_mcf_pars', variable_FIR, 'vol_regressor_list')
    whole_brain_analysis_workflow.connect(datasource, 'variable_behavior_tsv_file', variable_FIR, 'behavior_file_list')
    whole_brain_analysis_workflow.connect(datasource, 'example_func', variable_FIR, 'example_func')

    # datasink
    datasink = pe.Node(DataSink(), name='sinker')
    datasink.inputs.parameterization = False

    whole_brain_analysis_workflow.connect(input_node, 'preprocessed_directory', datasink, 'base_directory')
    whole_brain_analysis_workflow.connect(input_node, 'sub_id', datasink, 'container')

    whole_brain_analysis_workflow.connect(predictable_GLM, 'out_files', datasink, 'GLM.@predictable_GLM')
    whole_brain_analysis_workflow.connect(predictable_FIR, 'out_files', datasink, 'GLM.@predictable_FIR')
    whole_brain_analysis_workflow.connect(unpredictable_GLM, 'out_files', datasink, 'GLM.@unpredictable_GLM')
    whole_brain_analysis_workflow.connect(unpredictable_FIR, 'out_files', datasink, 'GLM.@unpredictable_FIR')
    whole_brain_analysis_workflow.connect(variable_FIR, 'out_files', datasink, 'GLM.@variable_FIR')


    return whole_brain_analysis_workflow

