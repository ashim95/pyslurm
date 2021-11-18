import os, sys, time, json
from filelock import FileLock

from utils import GPU_TYPES, GPU_MEMS, PARTITION_PRIORITY, GPU_PRIORITY, create_slurm_file, create_slurm_object, make_json_serializable, read_json_with_filelock, unique_job_id, write_json_with_filelock
from classes import Job, Slurm
from subprocess import Popen, PIPE, STDOUT, run

def read_slurm_jobs(partition):

    filename = 'db/' + partition + '.json'

    if not os.path.isfile(filename):
        return None

    data = read_json_with_filelock(filename)

    slurms = []
    for d in data:
        slurms.append(Slurm(from_dict=d))
    
    return slurms

def remove_jobs_file(partition):

    filename = 'db/' + partition + '.json'

    if not os.path.isfile(filename):
        return None

    os.remove(filename)

def run_sbatch(slurm):

    filename = slurm.slurm_file

    p = Popen('sbatch ' + str(filename), stdout = PIPE, shell=True)
    slurm.submitted = True
    status = p.stdout.readline().strip().decode("utf-8")
    slurm.status =status
    slurm_job_id= status.split(' ')[-1]
    slurm.slurm_job_id = slurm_job_id

    return slurm_job_id


def submit(slurm):

    # submit the slurm job, creating the slurm file, and running sbatch
    create_slurm_file(slurm)

    slurm_job_id = run_sbatch(slurm)
    # then logging the output in a log file

    return slurm

def submit_jobs(partition):

    slurms = read_slurm_jobs(partition)

    new_slurms = []

    if slurms is None:
        print('No slurm jobs found for partition ', partition)
        return None
    for slurm in slurms:
        new_slurm = submit(slurm)
        new_slurms.append(new_slurm)
    return new_slurms

def save_details_of_submitted_jobs(partition, slurms):

    filename = 'db/submitted/' + str(unique_job_id()) + '_' + partition + '.json'

    write_json_with_filelock(filename, slurms)


if __name__=="__main__":

    partitions_ = sys.argv[1]

    if ',' in partitions_:
        partitions = [part.strip().lower() for part in partitions_.split(',')]
    else:
        partitions = [partitions_]

    for part in partitions:
        new_slurms = submit_jobs(part)
        if new_slurms is None:
            continue
        remove_jobs_file(part)
        save_details_of_submitted_jobs(part, new_slurms)


