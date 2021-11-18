import os, sys
import argparse
from pprint import pprint
import json

from classes import Job
from utils import set_arguments, add_arguments, read_command, parse_hyperparams, put_hyperparams_in_command, unique_job_id


"""
Has following supported functionalities:

1. pause: Don't send new jobs to sbatch/slurm

2. cancel: Cancel certain jobs from slurm queue and store in the a separate file

3. requeue: Cancel certain jobs from slurm queue and add it to the job queue of pyslurm

4. create_joblist: Create a bunch of slurm jobs (init Job class) and save to a new file

5. submit: Submit a list of jobs from a file and move to the top of pyslurm queue

6. remove: 

7. details: get details on a job id (either a slurm job id, or a job key (an id), or pyslurm job id)

8. run: Submit the slurm job file to the sbatch queue

9. resume: Start sending new jobs to sbatch/slurm

10. search: search if resources are available

"""

def create_joblist(args):

    jobs = []

    for i in range(len(args.parsed_command)):

        job = Job(
            job_id=unique_job_id(),
            job_key=args.job_key,
            job_gpu_types=args.gpu_type,
            job_command=args.parsed_command[i],
            job_gpu_mem=(args.min_gpu_mem, -1),
            job_priority=args.priority,
            job_env=args.env,
            job_num_gpus=args.num_gpus,
            job_number=i,
            work_dir=args.work_dir,
            cuda_version=args.cuda_version,
        )

        jobs.append(job.__to_json__())
    
    # pprint(jobs)

    with open(args.joblist_file, 'w') as fp:
        json.dump(jobs, fp, indent=2)

def main():

    parser = argparse.ArgumentParser()

    parser = add_arguments(parser)

    args = parser.parse_args()

    args = set_arguments(args)

    if args.cli == 'joblist':
        args.command = read_command(args.command)

        args.hyperparams = parse_hyperparams(args.hyperparams)

        args.parsed_command = put_hyperparams_in_command(args.command, args.hyperparams)

        create_joblist(args)
    
    elif args.cli == 'submit':

        if args.joblist_file is None:
            raise ValueError("Please provide suitable filename containing jobs to submit")
        
        with open(args.joblist_file, 'r') as fp:
            joblist = [Job(from_dict=j) for j in json.load(fp)]
        
        pprint(joblist)

    elif args.cli == 'resume':
        pass

    elif args.cli == 'pause':
        pass

    pprint(args)


if __name__=="__main__":

    main()