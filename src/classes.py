import os, sys
import pprint
from copy import deepcopy


class Job:

    """
    Job class that contains details of the jobs in different queues.

    job_key: identifies all the jobs with the same job identifier, say: 'fairseq_tt_embedding'
    """
    def __init__(self, job_id=None, slurm_job_id = None, job_name=None, job_key=None, job_command=None, job_cpu_mem=(-1, -1), 
                job_gpu_mem=(-1, -1), job_cpu_tasks=(-1,-1), job_gpu_types=[], partitions=[], 
                from_dict=None, job_priority=-1, job_env=None, job_num_gpus=1, job_number=-1, work_dir=None, cuda_version='default'):

        # job_number: index of the job in the batch of joblist

        self.job_id = job_id
        self.slurm_job_id = slurm_job_id
        self.job_name = job_name
        self.job_key = job_key
        self.job_command = job_command # list of commands
        self.job_cpu_mem = job_cpu_mem
        self.job_gpu_mem = job_gpu_mem
        self.job_cpu_tasks = job_cpu_tasks
        self.job_gpu_types = job_gpu_types
        self.job_num_gpus = job_num_gpus
        self.partitions = partitions
        self.history = None
        self.work_dir = work_dir
        self.job_priority = job_priority
        self.job_env = job_env
        self.job_number = job_number
        self.cuda_version = cuda_version

        if from_dict is not None:
            self.__set_dict_params__(from_dict)

    def __set_dict_params__(self, dict):
        # self.__dict__ = deepcopy(dict)
        for key, val in dict.items():
            setattr(self, key, val)

    def __repr__(self) -> str:

        return pprint.pformat(self.__dict__, indent=2)
    
    def __to_json__(self):
        return self.__dict__

    def __set_params__(self):
        pass
    
    def __set_partitions__(self):
        """
        Set the partition and account based on the type of GPU and the min GPU mem required.
        """
        pass

class Slurm:

    def __init__(self, name=None, slurm_file=None, gpu=[], num_gpus=[], account=None, partition=None, user_mail='ashim.gupta.cs@gmail.com', mail_type='ALL', 
                output_file=None, error_file=None, nodes=1, n_tasks=1, time=None, cpu_mem=None, 
                env=None, work_dir=None, library_path='default', from_dict=None, command=[], expected_host=None):

        
        self.name = name
        self.slurm_file = slurm_file
        self.gpu= gpu
        self.command = command
        self.num_gpus = num_gpus
        self.account = account
        self.partition = partition
        self.user_mail = user_mail
        self.mail_type = mail_type
        self.output_file = output_file
        self.error_file = error_file
        self.nodes = nodes
        self.n_tasks = n_tasks
        self.time = time
        self.cpu_mem = cpu_mem
        self.env = env
        self.work_dir = work_dir
        self.library_path = library_path
        self.expected_host = expected_host
        self.submitted = False
        self.slurm_job_id = None
        self.status = None
        if from_dict is not None:
            self.__set_dict_params__(from_dict)

    def __set_dict_params__(self, dict):
        # self.__dict__ = deepcopy(dict)
        for key, val in dict.items():
            setattr(self, key, val)

    def __repr__(self) -> str:

        return pprint.pformat(self.__dict__, indent=2)
    
    def __to_json__(self):
        return self.__dict__