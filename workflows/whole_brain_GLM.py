def sublists_for_phys(slice_regressor_list, in_files):
    from __future__ import division

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
                                unpredictable_mcf_pars='{sub_id}/mcf/ext_motion_pars/*-unpredictable_reward_*.par'
                                )
    datasource = pe.Node(SelectFiles(datasource_templates, sort_filelist = True, raise_on_empty = False), 
        name = 'datasource')

    split_phys_list = pe.Node(Function(input_names=['slice_regressor_list', 'in_files'],
                                output_names=['slice_regressor_lists'],
                                function=sublists_for_phys),
                                name='split_phys_list')

    unpredictable_GLM = pe.Node(Function(input_names=['in_file', 
                                            'slice_regressor_list', 
                                            'vol_regressors', 
                                            'num_components', 
                                            'method', 
                                            'mapper', 
                                            'dm_upscale_factor',
                                            'tsv_behavior_file'],
                                output_names=['output_files'],
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
                                output_names=['output_files'],
                                function=fit_glm_nuisances_single_file),
                                name='predictable_GLM')
    predictable_GLM.inputs.mapper = 'predictable'
    predictable_GLM.inputs.num_components = 0   # no physio, just motion correction nuisances
    predictable_GLM.inputs.method = 'PCA'
    predictable_GLM.inputs.dm_upscale_factor = 10

    unpredictable_FIR= pe.Node(Function(input_names=[
                                            'example_func'
                                            'in_files', 
                                            'slice_regressor_list', 
                                            'vol_regressor_list', 
                                            'behavior_file_list',
                                            'fir_frequency',
                                            'fir_interval'
                                            ],
                                output_names=['output_files'],
                                function=fit_FIR_nuisances_all_files),
                                name='unpredictable_FIR')
    unpredictable_FIR.inputs.fir_frequency = analysis_info['fir_frequency']
    unpredictable_FIR.inputs.fir_interval = analysis_info['fir_interval']


    # the actual top-level workflow
    whole_brain_analysis_workflow = pe.Workflow(name=name)

    whole_brain_analysis_workflow.connect(input_node, 'preprocessed_directory', datasource, 'base_directory')
    whole_brain_analysis_workflow.connect(input_node, 'sub_id', datasource, 'sub_id')

    # predictable mapper GLM
    whole_brain_analysis_workflow.connect(datasource, 'predictable_mapper_in_file', predictable_GLM, 'in_file')
    whole_brain_analysis_workflow.connect(datasource, 'predictable_mapper_mcf_par', predictable_GLM, 'vol_regressors')
    whole_brain_analysis_workflow.connect(datasource, 'predictable_mapper_tsv_file', predictable_GLM, 'tsv_behavior_file')

    # unpredictable mapper GLM
    whole_brain_analysis_workflow.connect(datasource, 'unpredictable_mapper_in_file', unpredictable_GLM, 'in_file')
    whole_brain_analysis_workflow.connect(datasource, 'unpredictable_mapper_mcf_par', unpredictable_GLM, 'vol_regressors')
    whole_brain_analysis_workflow.connect(datasource, 'unpredictable_mapper_physio_files', unpredictable_GLM, 'slice_regressor_list')

    # unpredictable reward FIR; first split the 1D slice regressor list to 2D
    whole_brain_analysis_workflow.connect(datasource, 'unpredictable_physio_files', split_phys_list, 'slice_regressor_list')
    whole_brain_analysis_workflow.connect(datasource, 'unpredictable_in_files', split_phys_list, 'in_files')
    whole_brain_analysis_workflow.connect(split_phys_list, 'slice_regressor_lists', unpredictable_FIR, 'slice_regressor_list')

    whole_brain_analysis_workflow.connect(datasource, 'unpredictable_in_files', unpredictable_FIR, 'in_files')
    whole_brain_analysis_workflow.connect(datasource, 'unpredictable_mcf_pars', unpredictable_FIR, 'vol_regressors')
    whole_brain_analysis_workflow.connect(datasource, 'unpredictable_behavior_tsv_file', unpredictable_FIR, 'behavior_file_list')

    # datasink
    datasink = pe.Node(nio.DataSink(), name='sinker')
    datasink.inputs.parameterization = False

    whole_brain_analysis_workflow.connect(input_node, 'preprocessed_directory', datasink, 'base_directory')
    whole_brain_analysis_workflow.connect(input_node, 'output_directory', datasink, 'container')

    whole_brain_analysis_workflow.connect(predictable_GLM, 'out_files', datasink, 'GLM.@predictable_GLM')
    whole_brain_analysis_workflow.connect(unpredictable_GLM, 'out_files', datasink, 'GLM.@unpredictable_GLM')
    whole_brain_analysis_workflow.connect(unpredictable_FIR, 'out_files', datasink, 'GLM.@unpredictable_FIR')


    return whole_brain_analysis_workflow

