from ast import parse
import os, sys, json
from pprint import pprint
from subprocess import Popen, PIPE, STDOUT, run
from utils import GPU_TYPES
import time
from filelock import FileLock

def parse_gpus(gpu):
    gpu = gpu.split(',')
    gpu_map = {}

    for g in gpu:
        typ = g.split(':')[1]
        num = int(g.split(':')[2].split('(')[0])
        gpu_map[typ] = num
    return gpu_map

def parse_joblist(jobs):
    
    used = {}
    for j in jobs:
        if 'gpu' in j:
            if ',' in j:
                gpu_used = parse_gpus(j)
                for key, val in gpu_used.items():
                    if key not in used:
                        used[key] = 0
                    used[key] += val
            else:
                gpu_split = j.split(':')
                if len(gpu_split) == 3:
                    typ = gpu_split[1]
                    num = int(gpu_split[2])
                else:
                    if gpu_split[1] in GPU_TYPES:
                        typ = gpu_split[1]
                        num = 1
                    else:
                        typ = 'gpu'
                        num = int(gpu_split[1])
                if typ not in used:
                    used[typ] = 0
                used[typ] +=num
        else:
            continue
    
    return used

def get_unused_resources(gpus, used):
    unused = {}

    total_available = sum([val for key, val in gpus.items()])

    total_used = sum([val for key, val in used.items()])

    if total_used == total_available:
        unused['state'] = 'alloc'
        return unused
    
    num_unused = total_available - total_used

    unused['state'] = 'mix'

    for key, val in gpus.items():
        if key not in used:
            unused[key] = val
        else:
            unused[key] = val - used[key]
    
    if 'gpu' in used:
        if len(gpus) == 1:
            key = list(gpus.keys())[0]
            unused[key] -= used['gpu']
        # else:
    # TODO: what gpu does gpu:1 get assigned to
    return unused



def parse_output(lines):

    columns = []

    resources = {}

    for line in lines:
        if line.startswith('GRES') or line.startswith('Print'):
            continue
        if line.startswith('Hostname'):
            columns = line.strip().split()

        if len(columns)!= 0 :
            if line.startswith('kp') or line.startswith('not'):
                vals = line.strip().split()
                host = vals[0]
                if host not in resources:
                    resources[host] = {}
                
                resources[host]['partition'] = vals[1]
                resources[host]['state'] = vals[2]
                resources[host]['gpus'] = parse_gpus(vals[8])
                resources[host]['used'] = parse_joblist(vals[9:])
                resources[host]['unused'] = get_unused_resources(resources[host]['gpus'], resources[host]['used'])
    # pprint(resources)
    
    return resources
    # print(columns)

def write_resources(partition, resources):

    filename = 'db/resources/' + partition + '.json'
    lockpath = filename + '.lock'
    if os.path.isfile(lockpath):
        # sleep 
        time.sleep(10)
        return write_resources(partition, resources)
    else:
        with FileLock(lockpath):
            # time.sleep(20)
            print('Lock acquired for file ', lockpath)
            with open(filename, 'w') as fp:
                json.dump(resources, fp, indent=2)
        os.remove(lockpath)

def get_status(partitions=[]):

    for part in partitions:
        outs = []
        p = Popen('pestat -G -p ' + str(part), stdout = PIPE, shell=True)
        while True:
            line = p.stdout.readline()
            if not line:
               break
            else:
                outs.append(line.decode("utf-8").strip())
        
        # for o in outs:
        #     print(o.split('\t'))
        
        resources = parse_output(outs)
        write_resources(part, resources)
        # pprint(outs)



if __name__=="__main__":

    partitions_ = sys.argv[1]

    if ',' in partitions_:
        partitions = [part.strip().lower() for part in partitions_.split(',')]
    else:
        partitions = [partitions_]

    get_status(partitions)