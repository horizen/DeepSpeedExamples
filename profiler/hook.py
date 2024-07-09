import os
import sys
import subprocess
import json
import urllib3
from pathlib import Path
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
from qcloud_cos.cos_exception import CosClientError, CosServiceError


def _dump_relevant_env_vars():
    relevant_env_vars = [
        "PYTHONPATH",
        "NVSHMEM_NVTX",
        "GPU_NUM_PER_NODE",
        "NODE_NUM",
        "INDEX",
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_PORT",
        "MASTER_ADDR",
        "CUDA_VISIBLE_DEVICES",
        "GLOO_SOCKET_IFNAME",
        "GLOO_DEVICE_TRANSPORT",
        "NCCL_SOCKET_IFNAME",
        "TORCH_NCCL_BLOCKING_WAIT",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "NCCL_IB_DISABLE",
        "NCCL_P2P_DISABLE",
        "NCCL_P2P_LEVEL",
        "NCCL_SHM_DISABLE",
        "NCCL_SOCKET_NTHREADS",
        "NCCL_NSOCKS_PERTHREAD",
        "NCCL_BUFFSIZE",
        "NCCL_NTHREADS",
        "NCCL_RINGS",
        "NCCL_MAX_NCHANNELS",
        "NCCL_MIN_NCHANNELS",
        "NCCL_CHECKS_DISABLE",
        "NCCL_CHECK_POINTERS",
        "NCCL_LAUNCH_MODE",
        "NCCL_IB_HCA",
        "NCCL_IB_TIMEOUT",
        "NCCL_IB_RETRY_CNT",
        "NCCL_IB_GID_INDEX",
        "NCCL_IB_SL",
        "NCCL_IB_TC",
        "NCCL_IB_AR_THRESHOLD",
        "NCCL_IB_CUDA_SUPPORT",
        "NCCL_NET_GDR_LEVEL",
        "NCCL_NET_GDR_READ",
        "NCCL_SINGLE_RING_THRESHOLD",
        "NCCL_LL_THRESHOLD",
        "NCCL_TREE_THRESHOLD",
        "NCCL_ALGO",
        "NCCL_PROTO",
        "NCCL_IGNORE_CPU_AFFINITY",
        "NCCL_DEBUG_FILE",
        "NCCL_COLLNET_ENABLE",
        "NCCL_TOPO_FILE",
        "NCCL_TOPO_DUMP_FILE",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    ]
    formatted_output = ""
    for var in relevant_env_vars:
        value = os.environ[var] if var in os.environ else "N/A"
        formatted_output += f"env:{var}={value}\n"
    print(formatted_output)

# TODO
def _pre_check():
    # 校验rdma网卡是否正常

    # 校验nccl相关环境变量是否设置

    # 确认rdma库是否安装

    # 校验GPU状态是否正常

    # 检查nvlink是否开启

    # 检查gpu p2p是否正常

    # 检查gpu direct rdma是否开启

    # 检查是否使用tccl插件

    # 检查torchrun检查参数是否符合预期
    current_file_path = Path(__file__)
    absolute_file_path = current_file_path.resolve()
    current_dir = str(absolute_file_path.parent)

    process = subprocess.Popen([os.path.join(current_dir, "pre_check.sh")], shell=True, stdout=sys.stdout, stderr=sys.stderr, env=os.environ)
    code = process.wait()
    print('pre check exit with {}'.format(code))
    if code != 0:
        sys.exit(code)


def inject_hook():
    _pre_check()

    args = sys.argv[1:]
    entrypoint = None
    for idx, arg in enumerate(args):
        if not arg.startswith('-') and arg.endswith('.py'):
            entrypoint=arg
            sys.argv[idx+1] = '/tmp/hook.py'
            break
    entrypoint_dir = os.getcwd()
    if entrypoint is not None:
        # hook entrypoint
        if not entrypoint.startswith('/'):
            entrypoint = os.path.join(os.getcwd(), entrypoint)
        else:
            entrypoint_dir = os.path.dirname(entrypoint)
        print('inject hook for script:', entrypoint)
        hooked_entrypoint = '/tmp/hook.py' 
        with open(hooked_entrypoint, 'w') as hook_file:
            hook_file.write("from ti_profiler import pre_hook,post_hook\npre_hook()\n")
            with open(entrypoint, 'r') as file:
                for line in file:
                    hook_file.write(line)
            hook_file.write("post_hook()\n")

    current_file_path = Path(__file__)
    absolute_file_path = current_file_path.resolve()
    current_dir = str(absolute_file_path.parent)
    sys.path = [p for p in sys.path if p != current_dir]
    python_path = os.environ['PYTHONPATH']
    os.environ['PYTHONPATH'] = entrypoint_dir + ":" + python_path.replace(current_dir, os.path.join(current_dir, 'ti'))
    sys.path_importer_cache.clear()
    print('sys path', sys.path)
    print('sys argv', sys.argv)
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NVSHMEM_NVTX'] = 'common'

    _dump_relevant_env_vars()

def _get_temporary_secret_and_token():
    credentials = os.environ.get("PROFILER_COS_CREDENTIAL")
    return credentials.split(':')

def _upload_to_cos(file):
    cred = _get_temporary_secret_and_token()

    region = os.environ['REGION'] if 'REGION' in os.environ else "ap-shanghai"
    config = CosConfig(Region=region, SecretId=cred[1], SecretKey=cred[2], Token=cred[3], Scheme='https')
    client = CosS3Client(config)

    client.upload_file()
    for i in range(0, 10):
        try:
            response = client.upload_file(
                MAXThread = 10,
                Bucket=cred[0],
                Key=file[len('/tmp'):],
                LocalFilePath=file)
            return True
        except CosClientError or CosServiceError as e:
            if i < 9:
                print("upload error, try again")
            else:
                print("upload failed", e)
                return False


# return code
# 100: profiler sucess, but failed upload result to cos
# 101: profiler failed
def spawn_process():
    task_id = os.environ["TI_INSTANCE_ID"] if "TI_INSTANCE_ID" in os.environ else "tmp"
    rank = os.environ["INDEX"] if "INDEX" in os.environ else "0"
    save_dir = os.path.join('/tmp', task_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    profiler = os.environ["TI_PROFILER"] if "TI_PROFILER" in os.environ else "nsys"
    profiler_args = []
    if profiler == "nsys":
        profiler_result = os.path.join(save_dir, 'profile_{}'.format(rank))
        profiler_args = ['nsys', 'profile', '-t', 'cuda,nvtx,osrt', 
                '--capture-range=cudaProfilerApi', '-s', 'cpu', '--cudabacktrace=true', '--nic-metrics=true', 
                '-x', 'true', '-w', 'true', 
                '-o', profiler_result]
        profiler_result = profiler_result + ".nsys-rep"
    else:
        profiler_result = os.path.join(save_dir, "profile_{}.json".format(rank))

    os.environ["PROFILER_RESULT_FILE"] = profiler_result
    cmd = sys.argv[0]
    if profiler == "nsys" and (cmd.endswith("mpirun") or cmd.endswith("deepspeed") or cmd.endswith("horovodrun")):
        for idx, arg in enumerate(sys.argv):
            if not arg.startswith('-'):
                args = sys.argv[0:idx] + profiler_args + sys.argv[idx:]
                break
    else:
        args = profiler_args + sys.argv
    process = subprocess.Popen(args, stdout=sys.stdout, stderr=sys.stderr, env=os.environ)
    code = process.wait()
    print('{} exit with {}'.format(args, code))

    if os.path.exists(profiler_result):
        # upload to cos
        if not _upload_to_cos(profiler_result):
            sys.exit(100)
    else:
        print("seems no profiler data generated")
        sys.exit(101)


