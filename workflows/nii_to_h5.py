
def create_all_calcarine_reward_2_h5_workflow(analysis_info, name='all_calcarine_reward_nii_2_h5'):
    import os.path as op
    import tempfile
    import nipype.pipeline as pe
    from nipype.interfaces import fsl
    from nipype.interfaces.utility import Function, Merge, IdentityInterface
    from spynoza.nodes.utils import get_scaninfo, dyns_min_1, topup_scan_params, apply_scan_params
    from nipype.interfaces.io import SelectFiles, DataSink

    # Importing of custom nodes from spynoza packages; assumes that spynoza is installed:
    # pip install git+https://github.com/spinoza-centre/spynoza.git@develop
    from utils.utils import mask_nii_2_hdf5, combine_eye_hdfs_to_nii_hdf

    input_node = pe.Node(IdentityInterface(
                fields=['sub_id', 'preprocessed_data_dir']), name='inputspec')

    # i/o node
    datasource_templates = dict(mcf='{sub_id}/mcf/*.nii.gz',
                                psc='{sub_id}/psc/*.nii.gz',
                                tf='{sub_id}/tf/*.nii.gz',
                                eye='{sub_id}/eye/h5/*.h5',
                                rois='{sub_id}/roi/*_vol.nii.gz'
                                ) 
    datasource = pe.Node(SelectFiles(datasource_templates, sort_filelist = True, raise_on_empty = False), 
        name = 'datasource')

    hdf5_psc_masker = pe.Node(Function(input_names = ['in_files', 'mask_files', 'hdf5_file', 'folder_alias'], output_names = ['hdf5_file'],
                                    function = mask_nii_2_hdf5), 
                                    name = 'hdf5_psc_masker')
    hdf5_psc_masker.inputs.folder_alias = 'psc'
    hdf5_psc_masker.inputs.hdf5_file = op.join(tempfile.mkdtemp(), 'roi.h5')

    hdf5_tf_masker = pe.Node(Function(input_names = ['in_files', 'mask_files', 'hdf5_file', 'folder_alias'], output_names = ['hdf5_file'],
                                    function = mask_nii_2_hdf5), 
                                    name = 'hdf5_tf_masker')
    hdf5_tf_masker.inputs.folder_alias = 'tf'
    hdf5_psc_masker.inputs.hdf5_file = op.join(tempfile.mkdtemp(), 'roi.h5')

    hdf5_mcf_masker = pe.Node(Function(input_names = ['in_files', 'mask_files', 'hdf5_file', 'folder_alias'], output_names = ['hdf5_file'],
                                    function = mask_nii_2_hdf5), 
                                    name = 'hdf5_mcf_masker')
    hdf5_mcf_masker.inputs.folder_alias = 'mcf'

    eye_hdfs_to_nii_masker = pe.Node(Function(input_names = ['nii_hdf5_file', 'eye_hdf_filelist', 'new_alias'], output_names = ['nii_hdf5_file'],
                                    function = combine_eye_hdfs_to_nii_hdf), 
                                    name = 'eye_hdfs_to_nii_masker')
    eye_hdfs_to_nii_masker.inputs.new_alias = 'eye'

 
    # node for datasinking
    datasink = pe.Node(DataSink(), name='sinker')
    datasink.inputs.parameterization = False

    all_calcarine_reward_nii_2_h5_workflow = pe.Workflow(name=name)

    all_calcarine_reward_nii_2_h5_workflow.connect(input_node, 'preprocessed_data_dir', datasink, 'base_directory')
    all_calcarine_reward_nii_2_h5_workflow.connect(input_node, 'sub_id', datasink, 'container')

    all_calcarine_reward_nii_2_h5_workflow.connect(input_node, 'preprocessed_data_dir', datasource, 'base_directory')
    all_calcarine_reward_nii_2_h5_workflow.connect(input_node, 'sub_id', datasource, 'sub_id')

    all_calcarine_reward_nii_2_h5_workflow.connect(datasource, 'psc', hdf5_psc_masker, 'in_files')
    all_calcarine_reward_nii_2_h5_workflow.connect(datasource, 'rois', hdf5_psc_masker, 'mask_files')

    # the hdf5_file is created by the psc node, and then passed from masker to masker on into the datasink.
    all_calcarine_reward_nii_2_h5_workflow.connect(hdf5_psc_masker, 'hdf5_file', hdf5_tf_masker, 'hdf5_file')
    all_calcarine_reward_nii_2_h5_workflow.connect(datasource, 'tf', hdf5_tf_masker, 'in_files')
    all_calcarine_reward_nii_2_h5_workflow.connect(datasource, 'rois', hdf5_tf_masker, 'mask_files')

    all_calcarine_reward_nii_2_h5_workflow.connect(hdf5_tf_masker, 'hdf5_file', hdf5_mcf_masker, 'hdf5_file')
    all_calcarine_reward_nii_2_h5_workflow.connect(datasource, 'mcf', hdf5_mcf_masker, 'in_files')
    all_calcarine_reward_nii_2_h5_workflow.connect(datasource, 'rois', hdf5_mcf_masker, 'mask_files')

    all_calcarine_reward_nii_2_h5_workflow.connect(hdf5_mcf_masker, 'hdf5_file', eye_hdfs_to_nii_masker, 'nii_hdf5_file')
    all_calcarine_reward_nii_2_h5_workflow.connect(datasource, 'eye', eye_hdfs_to_nii_masker, 'eye_hdf_filelist')

    all_calcarine_reward_nii_2_h5_workflow.connect(eye_hdfs_to_nii_masker, 'nii_hdf5_file', datasink, 'h5')

    return all_calcarine_reward_nii_2_h5_workflow