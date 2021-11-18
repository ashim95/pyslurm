import os, sys, pathlib
import yaml
from copy import deepcopy
import time
from datetime import datetime
import json
from filelock import FileLock

from classes import Slurm

LD_LIBRARY_PATHS = {
    'default': '/usr/local/cuda/lib64',
    'cu100': '',
    'cu110': '/usr/local/cuda-11/lib64',
    'cu113': '/usr/local/cuda-11.3/lib64',
    'cu114': '/usr/local/cuda-11.4/lib64',
    'cu900': '',
}

PARTITION_TO_ACCOUNT = {
    'kingspeak-gpu-guest': 'owner-gpu-guest',
    'soc-gpu-kp': 'soc-gpu-kp',
    'kingspeak-gpu': 'kingspeak-gpu',
    'notchpeak-gpu-guest': 'owner-gpu-guest',
    'notchpeak-gpu': 'notchpeak-gpu',
}

PARTITION_TO_TIME={
    'kingspeak-gpu-guest': '3-00:00:00',
    'soc-gpu-kp': '14-00:00:00',
    'kingspeak-gpu': '3-00:00:00',
    'notchpeak-gpu-guest': '3-00:00:00',
    'notchpeak-gpu': '3-00:00:00',
}

GPU_TYPES=['v100', 'p100', 'titanv', 'a100', 'a40', '3090', '2080ti', '1080ti', 't4', 'p40']

GPU_MEMS = {
    'v100': 16,
    'p100': 16,
    'titanv': 12,
    'a100': 40,
    '3090': 24,
    '2080ti': 11,
    '1080ti': 11,
    't4': 16,
    'p40': 24,

}

PARTITION_PRIORITY = {
    'notchpeak-gpu': 10,
    'soc-gpu-kp': 10,
    'kingspeak-gpu': 10,
    'notchpeak-gpu-guest': 5,
    'kingspeak-gpu-guest': 5,
}

GPU_PRIORITY= {
    'a40': 10,
    'a100': 10,
    '3090': 10,
    'v100': 8,
    'titanv': 6,
    'p100': 4,
    't4': 4,
    '2080ti': 3,
    '1080ti': 2,
    'p40': 1,
}

def add_arguments(parser):

    parser.add_argument(
        "--cli", 
        choices=['pause', 'resume', 'joblist', 'submit'],
        type=str,
        required=True,
        help='the cli command of pyslurm to run',
    )

    parser.add_argument(
        "--joblist_file",
        default=None,
        required=False,        
    )

    parser.add_argument(
        "--job_key",
        default='',
        required=False,        
    )

    parser.add_argument(
        "--command",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--env",
        type=str,
        help='Python pip environment to use, = path, so that we can run: source <path>/bin/activate in slurm script'
    )

    parser.add_argument(
        "--gpu_type",
        type=str,
        default='any',
        help='GPU types that we are allowed to use, comma separated'
    )

    parser.add_argument(
        "--min_gpu_mem",
        type=int,
        default=-1,
        help='Minimum memory on the GPU required to run this program'
    )

    parser.add_argument(
        '--hyperparams',
        type=str,
        help = 'the file to read hyperparams from'
    )

    parser.add_argument(
        '--priority',
        type=int,
        default=-1,
        help='priority to assign to the job'
    )

    parser.add_argument(
        '--num_gpus',
        type=int,
        default=1,
        help='number of GPUS required'
    )

    parser.add_argument(
        '--work_dir',
        type=str,
        help = 'the workdir for these set of jobs'
    )

    parser.add_argument(
        '--cuda_version',
        type=str,
        default='default',
        help = 'the workdir for these set of jobs'
    )

    return parser

def set_arguments(args):

    args.gpu_type = args.gpu_type.lower()
    args.cli = args.cli.lower()

    if args.gpu_type == 'any':
        args.gpu_type = GPU_TYPES
    elif ',' in args.gpu_type:
        args.gpu_type = [typ.strip() for typ in args.gpu_type.split(',')]
    else:
        pass

    return args

def read_command(command_file):

    commands = []
    with open(command_file, 'r') as fp:
        for line in fp:
            if len(line.strip()) > 0:
                commands.append(line.strip())

    return commands

def parse_hyperparams(filename):
    with open(filename, 'r') as fp:

        data = yaml.load(fp, Loader=yaml.FullLoader)

        return data

def put_hyperparams_in_command(command, hyperparams):

    # command: list of commands read from the file (say: ['nvidia-smi', 'python -c "import torch; torch.cuda.is_available()"'])

    if len(hyperparams) == 0:
        return [command]

    new_commands = []

    size = len(hyperparams[list(hyperparams.keys())[0]])

    for i in range(size):
        new_command = deepcopy(command)
        values = {}
        for key, val in hyperparams.items():
            values[key] = val[i]
        
        for key, val in values.items():
            for n in range(len(new_command)):
                if '<' + str(key) + '>' in new_command[n]:
                    new_command[n] = new_command[n].replace('<' + str(key) + '>', str(val))
        new_commands.append(new_command)

    return new_commands

def unique_job_id():

    return int(time.time() * 100000 // 1)

def create_slurm_object(partition, host, gpu, num_gpus, job):

    gpu = [gpu]
    num_gpus = [num_gpus]
    
    if job.job_name is None:
        job.job_name = job.job_key + '_' + str(job.job_id) + '_' + str(job.job_number)


    year, week_num, day = datetime.today().isocalendar()
    slurm_folder = 'slurms/' + 'year_' + str(year) + '_week_' + str(week_num) + '_day_' + str(day)
    output_folder = slurm_folder + '/outputs/'

    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True) 

    slurm_file = slurm_folder + '/' + job.job_name
    output_file = output_folder + '/' + job.job_name

    return Slurm(
        name=job.job_name,
        slurm_file=slurm_file,
        gpu=gpu,
        command=job.job_command,
        num_gpus=num_gpus,
        account=PARTITION_TO_ACCOUNT[partition],
        partition=partition,
        output_file=output_file,
        error_file=output_file,
        nodes=1,
        n_tasks=1,
        time=PARTITION_TO_TIME[partition],
        cpu_mem=32,
        env=job.job_env,
        work_dir=job.work_dir,
        library_path=LD_LIBRARY_PATHS[job.cuda_version],
        expected_host=host,
    )

def make_json_serializable(list_of_obj):
    return [obj.__to_json__() for obj in list_of_obj]

def write_json_with_filelock(filename, objects):

    lockpath = filename + '.lock'
    
    if os.path.isfile(lockpath):
        # sleep 
        time.sleep(10)
        return write_json_with_filelock(filename, objects)
    else:
        with FileLock(lockpath):
            # time.sleep(20)
            print('Lock acquired for file ', lockpath)
            with open(filename, 'w') as fp:
                json.dump(make_json_serializable(objects), fp, indent=2)
        os.remove(lockpath)

def read_json_with_filelock(filename):
    lockpath = filename + '.lock'

    if os.path.isfile(lockpath):
        # sleep 
        time.sleep(10)
        return read_json_with_filelock(filename)
    else:
        with FileLock(lockpath):
            # time.sleep(20)
            print('Lock acquired for file ', lockpath)
            with open(filename, 'r') as fp:
                res = json.load(fp)
        os.remove(lockpath)
    
    return res

def create_slurm_file(slurm):

    if slurm.slurm_file is None:
        raise ValueError("Please provide a valid name for slurm file, None provided")
    
    fp = open(slurm.slurm_file, 'w')

    sbatch = '#SBATCH --'

    fp.write('#!/bin/bash\n#\n')

    gpu_str = ""
    for i in range(len(slurm.gpu)):
        gpu_str += "gpu:" + slurm.gpu[i] + ":" + str(slurm.num_gpus[i]) + ','
    
    if gpu_str[-1] == ',':
        gpu_str = gpu_str[:-1]
    fp.write(sbatch + 'gres=' + gpu_str + '\n')

    fp.write(sbatch + 'partition=' + slurm.partition + '\n')

    fp.write(sbatch + 'account=' + slurm.account + '\n')

    fp.write(sbatch + 'mail-user=' + slurm.user_mail + '\n')

    fp.write(sbatch + 'mail-type=' + slurm.mail_type + '\n')

    fp.write(sbatch + 'nodes=' + str(slurm.nodes) + '\n')

    fp.write(sbatch + 'ntasks=' + str(slurm.n_tasks) + '\n')

    fp.write(sbatch + 'time=' + str(slurm.time) + '\n')
    fp.write(sbatch + 'mem=' + str(slurm.cpu_mem) + 'G' + '\n')

    fp.write(sbatch + 'job-name=' + str(slurm.name) + '\n')
    fp.write(sbatch + 'output=' + str(slurm.output_file) + '\n')
    fp.write(sbatch + 'error=' + str(slurm.error_file) + '\n')

    fp.write('WORK_DIR=' + str(os.path.abspath(slurm.work_dir)) + '\n')

    fp.write('export LD_LIBRARY_PATH=' + str(slurm.library_path) + '\n')

    fp.write('echo "Work Dir : $WORK_DIR"\n')
    fp.write('cd $WORK_DIR \n')
    fp.write('\n\n')

    fp.write('#Activate Environment\n')
    fp.write('source ' + slurm.env + '/bin/activate\n')

    fp.write('nvidia-smi\n')
    fp.write('python -c "import torch; torch.cuda.is_available()"\n')

    for command in slurm.command:
        fp.write(command + '\n\n')

    fp.close()