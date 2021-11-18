import os, sys, time, json
from filelock import FileLock

from utils import GPU_TYPES, GPU_MEMS, PARTITION_PRIORITY, GPU_PRIORITY, create_slurm_object, write_json_with_filelock, make_json_serializable
from classes import Job

def read_resources(partition):

    filename = 'db/resources/' + partition + '.json'
    lockpath = filename + '.lock'

    if os.path.isfile(lockpath):
        # sleep 
        time.sleep(10)
        return read_resources(partition)
    else:
        with FileLock(lockpath):
            # time.sleep(20)
            print('Lock acquired for file ', lockpath)
            with open(filename, 'r') as fp:
                res = json.load(fp)
        os.remove(lockpath)
    
    return res

def read_job_details(filename):

    with open(filename, 'r') as fp:
        return [Job(from_dict=j) for j in json.load(fp)]

def gpu_based_on_mem(mem):

    possible = set([key for key, val in GPU_MEMS.items() if val >= mem])

    return possible

def is_gpu_available(compatible, resources, num_gpus):

    all_possible = {}
    possible = []

    for part, res in resources.items():
        for host, status in res.items():
            if status['state'] != 'alloc' and status['unused']['state'] != 'alloc':
                for gpu, num in status['unused'].items():
                    
                    if gpu not in GPU_TYPES: # not a gpu key
                        continue

                    if gpu in compatible:
                        if part not in all_possible:
                            all_possible[part] = []
                        # if host not in all_possible[part]:
                        #     all_possible[part][host] = []

                        if status['unused'][gpu] >= num_gpus:
                            all_possible[part].append((gpu, num_gpus, host, part))
                            possible.append((gpu, num_gpus, host, part))
    
    # now all_possible contains all possible hosts/partitions that can have this job, except the number of GPUs

    if len(possible) == 0:
        return None

    # count_possible = {} # TODO: filter based on if total number is available on that host

    # for part, hosts in all_possible.items():
    #     for host, val in hosts.items():
    #         sum_host = sum([val[1]])
    #         if sum_host >= num_gpus:
    #             if part not in count_possible:
    #                 count_possible[part] = []
    #             gpus_to_use = {}
    #             count_possible[part].append((host, val, ))

    # if len(count_possible) == 0:
    #     return None
    # return count_possible
    return all_possible


def allocate_resource(possible, resources):

    priortiy_possible = []
    for part, val in possible.items():
        for v in val:
            priority = PARTITION_PRIORITY[part] * GPU_PRIORITY[v[0]]
            priortiy_possible.append((part, v[0], v[1], v[2], priority))
    
    priortiy_possible = sorted(priortiy_possible, key=lambda x: x[-1], reverse=True)

    selected = priortiy_possible[0]
    part, gpu, num, host, priority = selected
    print(part, gpu, num, host, priority)
    resources[part][host]['unused'][gpu] -= num
    if resources[part][host]['unused'][gpu] <= 0:
        del resources[part][host]['unused'][gpu]
    
    return resources, selected


def schedule_job(job, resources):

    # find compatible GPU types (GPU_TYPES + min gpu mem)
    compatible_gpus = set(job.job_gpu_types).intersection(gpu_based_on_mem(job.job_gpu_mem[0]))

    # see if GPU available
    possible = is_gpu_available(compatible_gpus, resources, job.job_num_gpus)

    # if yes, allocate, update resources and return
    if possible:
        resources, selected = allocate_resource(possible, resources)
        return resources, selected
    else:
        return resources, None

def get_schedule(partitions, jobs_file):

    new_joblist = []

    resources = {}
    for part in partitions:
        resources[part] = read_resources(part)
    
    partition_to_job_dict = {}
    # jobs to be scheduled in each partition
 
    # read job details and sort by priority
    joblist = read_job_details(jobs_file)
    joblist = sorted(joblist, key=lambda x: x.job_priority, reverse=True)

    # schedule based on available resources and job details

    for job in joblist:
        resources, slurm_job = schedule_job(job, resources)
        if slurm_job is not None:
            part, gpu, num, host, priority = slurm_job
            slurm = create_slurm_object(part, host, gpu, num, job)
            if part not in partition_to_job_dict:
                partition_to_job_dict[part] = []
            partition_to_job_dict[part].append(slurm)
        else:
            new_joblist.append(job)
            continue
    return new_joblist, partition_to_job_dict

def write_joblist(filename, joblist):

    write_json_with_filelock(filename, joblist)

    # with open(filename, 'w') as fp:
    #     json.dump(make_json_serializable(joblist), fp, indent=2)

def write_partition_to_job_dict(partition_to_job_dict):

    for part, val in partition_to_job_dict.items():
        filename = 'db/' + part + '.json'
        write_json_with_filelock(filename, val)
        # with open(filename, 'w') as fp:
        #     json.dump(make_json_serializable(val), fp, indent=2)

if __name__=="__main__":

    partitions_ = sys.argv[1]

    if ',' in partitions_:
        partitions = [part.strip().lower() for part in partitions_.split(',')]
    else:
        partitions = [partitions_]

    jobs_file = sys.argv[2]
    new_joblist, partition_to_job_dict = get_schedule(partitions, jobs_file)

    # write partition specific jobs
    write_partition_to_job_dict(partition_to_job_dict)

    write_joblist(jobs_file, new_joblist)

    # also write remaining jobs in file jobs_file