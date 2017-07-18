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
from nipype.utils.filemanip import copyfile

import nibabel as nib
from IPython.display import Image
from nipype.interfaces.utility import Function, Merge, IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from IPython.display import Image

from IPython import embed as shell

from workflows.pupil_workflow import create_pupil_workflow
from workflows.bold_wholebrain_fir_workflow import create_bold_wholebrain_fir_workflow

# we will create a workflow from a BIDS formatted input, at first for the specific use case 
# of a 7T PRF experiment's preprocessing. 

# a project directory that we assume has already been created. 
raw_data_dir = '/home/raw_data/-2014/reward/human_reward/data/'
preprocessed_data_dir = '/home/shared/-2014/reward/new/'
FS_subject_dir = os.path.join(raw_data_dir, 'FS_SJID')

# booleans that determine whether given stages of the
# analysis are run
pupil = True
wb_fir = True

for si in range(1,7): # 
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
                                        'stop_on_first_crash': False
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

    if not op.isdir(os.path.join(preprocessed_data_dir, sub_id)):
        try:
            os.makedirs(os.path.join(preprocessed_data_dir, sub_id))
        except OSError:
            pass

    # copy json files to preprocessed data folder
    # this allows these parameters to be updated and synced across subjects by changing only the raw data files.
    copyfile(os.path.join(raw_data_dir, 'acquisition_parameters.json'), os.path.join(preprocessed_data_dir, 'acquisition_parameters.json'), copy = True)
    copyfile(os.path.join(raw_data_dir, 'analysis_parameters.json'), os.path.join(preprocessed_data_dir, 'analysis_parameters.json'), copy = True)
    copyfile(os.path.join(raw_data_dir, sub_id ,'experimental_parameters.json'), os.path.join(preprocessed_data_dir, sub_id ,'experimental_parameters.json'), copy = True)

    if pupil:
        pwf = create_pupil_workflow(analysis_info,'pupil')
        pwf.inputs.inputspec.sub_id = sub_id
        pwf.inputs.inputspec.preprocessed_directory = preprocessed_data_dir

        pwf.write_graph(opd + '_pupil.svg', format='svg', graph2use='colored', simple_form=False)

        pwf.run('MultiProc', plugin_args={'n_procs': 6})

    if wb_fir:
        wbfwf = create_bold_wholebrain_fir_workflow(analysis_info,'wb_fir')
        wbfwf.inputs.inputspec.sub_id = sub_id
        wbfwf.inputs.inputspec.preprocessed_directory = preprocessed_data_dir

        wbfwf.write_graph(opd + '_wb_fir.svg', format='svg', graph2use='colored', simple_form=False)

        wbfwf.run('MultiProc', plugin_args={'n_procs': 6})
