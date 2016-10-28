# from nipype import config
# config.enable_debug_mode()
# Importing necessary packages
import os
import os.path as op
import glob
import json
import nipype
from nipype import config, logging
import matplotlib.pyplot as plt
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
import nibabel as nib
from IPython.display import Image
from nipype.interfaces.utility import Function, Merge, IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from IPython.display import Image

from IPython import embed as shell


from preprocessing_pipeline import create_all_calcarine_reward_workflow

# we will create a workflow from a BIDS formatted input, at first for the specific use case 
# of a 7T PRF experiment's preprocessing. 

# a project directory that we assume has already been created. 
raw_data_dir = '/home/raw_data/-2014/reward/human_reward/data/'
preprocessed_data_dir = '/home/shared/-2014/reward/new/'
FS_subject_dir = os.path.join(raw_data_dir, 'FS_SJID')

# for now, testing on a single subject, with appropriate FS ID, this will have to be masked.
for si in range(1,2): # 
    sub_id, FS_ID = 'sub-00%i'%si, 'sub-00%i'%si
    sess_id = 'ses-*'

    # now we set up the folders and logging there.
    opd = op.join(preprocessed_data_dir, sub_id)
    try:
        os.makedirs(op.join(opd, 'log'))
    except OSError:
        pass

    config.update_config({  'logging': {
                                        'log_directory': op.join(opd, 'log'),
                                        'log_to_file': True,
                                        'workflow_level': 'INFO',
                                        'interface_level': 'INFO'
                                      },
                            'execution': {
                                        'stop_on_first_crash': True
                                        }
                        })
    logging.update_logging(config)

    # load the sequence parameters from json file
    with open(os.path.join(raw_data_dir, 'acquisition_parameters.json')) as f:
        json_s = f.read()
        acquisition_parameters = json.loads(json_s)

    # load the analysis parameters from json file
    with open(os.path.join(raw_data_dir, 'analysis_parameters.json')) as f:
        json_s = f.read()
        analysis_info = json.loads(json_s)

    # load the analysis/experimental parameters for this subject from json file
    with open(os.path.join(raw_data_dir, sub_id ,'experimental_parameters.json')) as f:
        json_s = f.read()
        experimental_parameters = json.loads(json_s)
    analysis_info.update(experimental_parameters)

    if not op.isdir(preprocessed_data_dir):
        try:
            os.makedirs(preprocessed_data_dir)
        except OSError:
            pass

    # the actual workflow
    all_calcarine_reward_workflow = create_all_calcarine_reward_workflow(analysis_info, name = 'all_calcarine_reward')

    # standard in/output variables
    all_calcarine_reward_workflow.inputs.inputspec.raw_directory = raw_data_dir
    all_calcarine_reward_workflow.inputs.inputspec.sub_id = sub_id
    all_calcarine_reward_workflow.inputs.inputspec.sess_id = sess_id
    all_calcarine_reward_workflow.inputs.inputspec.output_directory = opd

    all_calcarine_reward_workflow.inputs.inputspec.psc_func = analysis_info['psc_func']

    # to what file do we motion correct?
    all_calcarine_reward_workflow.inputs.inputspec.which_file_is_EPI_space = analysis_info['which_file_is_EPI_space']

    # registration details
    all_calcarine_reward_workflow.inputs.inputspec.FS_ID = FS_ID
    all_calcarine_reward_workflow.inputs.inputspec.FS_subject_dir = FS_subject_dir
    all_calcarine_reward_workflow.inputs.inputspec.standard_file = op.join(os.environ['FSL_DIR'], 'data/standard/MNI152_T1_1mm_brain.nii.gz')

    # all the input variables for retroicor functionality
    # the key 'retroicor_order_or_timing' determines whether slice timing
    # or order is used for regressor creation
    all_calcarine_reward_workflow.inputs.inputspec.MB_factor = acquisition_parameters['MultiBandFactor']
    all_calcarine_reward_workflow.inputs.inputspec.nr_dummies = acquisition_parameters['NumberDummyScans']
    all_calcarine_reward_workflow.inputs.inputspec.tr = acquisition_parameters['RepetitionTime']
    all_calcarine_reward_workflow.inputs.inputspec.slice_direction = acquisition_parameters['SliceDirection']
    all_calcarine_reward_workflow.inputs.inputspec.phys_sample_rate = acquisition_parameters['PhysiologySampleRate']
    all_calcarine_reward_workflow.inputs.inputspec.slice_timing = acquisition_parameters['SliceTiming']
    all_calcarine_reward_workflow.inputs.inputspec.slice_order = acquisition_parameters['SliceOrder']
    all_calcarine_reward_workflow.inputs.inputspec.acceleration = acquisition_parameters['SenseFactor']
    all_calcarine_reward_workflow.inputs.inputspec.epi_factor = acquisition_parameters['EpiFactor']
    all_calcarine_reward_workflow.inputs.inputspec.wfs = acquisition_parameters['WaterFatShift']
    all_calcarine_reward_workflow.inputs.inputspec.te_diff = acquisition_parameters['EchoTimeDifference']

    # write out the graph and run
    all_calcarine_reward_workflow.write_graph(opd + '.png')
    all_calcarine_reward_workflow.run('MultiProc', plugin_args={'n_procs': 32})
