


def create_all_calcarine_reward_h5_workflow(analysis_info, name='all_calcarine_reward_h5'):

    input_node = pe.Node(IdentityInterface(
                fields=[
                    'FS_ID', 
                    'FS_subject_dir',
                    'sub_id', 
                    'psc_func', 
                    'MB_factor',
                    'tr']), name='inputspec')

    preprocessed_data_dir = '/home/shared/-2014/reward/new/'


    # i/o node
    datasource_templates = dict(mcf='{sub_id}/mcf/*.nii.gz',
                                psc='{sub_id}/psc/*.nii.gz',
                                tf='{sub_id}/tf/*.nii.gz',
                                events='{sub_id}/events/*.json',
                                eye='{sub_id}/eye/*.edf',
                                labels='{sub_id}/labels/*/*.nii.gz',
                                ex_epi='{sub_id}/reg/example_func.nii.gz',
                                reg_dat='{sub_id}/reg/register.dat') # ,
    datasource = pe.Node(SelectFiles(datasource_templates, sort_filelist = True, raise_on_empty = False), 
        name = 'datasource')

    # output_node = pe.Node(IdentityInterface(fields=([
    #         'temporal_filtered_files', 
    #         'percent_signal_change_files'])), name='outputspec')


    # node for edf conversion
    imports = [
        'import os.path as op',
        'from hedfpy.HDFEyeOperator import HDFEyeOperator'
        ]

    edf_converter = pe.MapNode(Function(input_names = ['edf_file'], output_names = ['hdf5_file'],
                                    function = 'convert_edf_2_hdf5', imports=imports), 
                                    name = 'edf_converter', iterfield = ['edf_file'])

    # node for masking and hdf5 conversion
    imports = [
        'import nibabel as nib',
        'import os.path as op',
        'import numpy as np',
        'import tables'
    ]    
    hdf5_masker = pe.MapNode(Function(input_names = ['in_files', 'mask_files', 'hdf5_file', 'folder_alias'], output_names = ['hdf5_file'],
                                    function = 'mask_nii_2_hdf5', imports=imports), 
                                    name = 'hdf5_masker', iterfield = ['mask_files'])
    hdf5_masker.folder_alias = 'psc'
    hdf5_masker.hdf5_file = op.join(tempfile.mkdtemp(), 'roi.h5')

    # node for datasinking
    datasink = pe.Node(DataSink(), name='sinker')
    datasink.inputs.parameterization = False


    



