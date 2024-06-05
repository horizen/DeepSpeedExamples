from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import numpy as np
import torch.distributed as dist
import os
import time

def synthetic_parser():
    parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
    parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

    parser.add_argument('--num-warmup', type=int, default=5,
                    help='number of warm-up batches that don\'t count towards benchmark')
    parser.add_argument('--num-iters', type=int, default=15,
                    help='number of benchmark iterations')

    return parser


def benchmark_step(model, optimizer, data, target, profiler):
    optimizer.zero_grad()
    if profiler:
        torch.cuda.nvtx.range_push("forward")
    output = model(data)
    if profiler:
        torch.cuda.nvtx.range_pop()
    loss = F.cross_entropy(output, target)
    if profiler:
        torch.cuda.nvtx.range_push("backward")
    loss.backward()
    if profiler:
        torch.cuda.nvtx.range_pop()
    if profiler:
        torch.cuda.nvtx.range_push("opt.step()")
    optimizer.step()
    if profiler:
        torch.cuda.nvtx.range_pop()


def log(s, nl=True):
    if dist.get_rank() != 0:
        return
    print(s, end='\n' if nl else '')


def main(args):
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    cudnn.benchmark = True

    # Set up standard model.
    model = getattr(models, args.model)()

    model.cuda()
    model = DDP(model)

    optimizer = optim.SGD(model.parameters(), lr=0.01 * world_size)

    # Set up fixed fake data
    data = torch.randn(args.batch_size, 3, 224, 224)
    target = torch.LongTensor(args.batch_size).random_() % 1000
    data, target = data.cuda(), target.cuda()

    log('Model: %s' % args.model)
    log('Batch size: %d' % args.batch_size)
    log('Number of GPU: %d' % world_size)

    # Benchmark
    log('Running benchmark...')
    img_secs = []
    for x in range(args.num_iters):
        #if x == args.num_warmup:
        #    torch.cuda.cudart().cudaProfilerStart()
        #if x >= args.num_warmup:
        #    torch.cuda.nvtx.range_push("rank{}-iteration{}".format(dist.get_rank(), x))
        start = time.time()
        benchmark_step(model, optimizer, data, target, False)
        end = time.time()
        img_sec = args.batch_size / (end-start)
        log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, "GPU"))
        img_secs.append(img_sec)
        #if x >= args.num_warmup:
        #    torch.cuda.nvtx.range_pop()
    #torch.cuda.cudart().cudaProfilerStop()
    # Results
    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    log('Img/sec per %s: %.1f +-%.1f' % ("GPU", img_sec_mean, img_sec_conf))
    log('Total img/sec on %d %s(s): %.1f +-%.1f' % (world_size, "GPU",  world_size * img_sec_mean, world_size * img_sec_conf))


if __name__ == "__main__":
    args = synthetic_parser().parse_args()

    dist.init_process_group("nccl")

    main(args)
    dist.destroy_process_group()