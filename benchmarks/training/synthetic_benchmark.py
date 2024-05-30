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

    parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
    parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
    parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')

    return parser


def benchmark_step(model, optimizer, data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()


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

    optimizer = optim.SGD(model.parameters(), lr=0.01 * world_size)

    # Set up fixed fake data
    data = torch.randn(args.batch_size, 3, 224, 224)
    target = torch.LongTensor(args.batch_size).random_() % 1000
    data, target = data.cuda(), target.cuda()

    log('Model: %s' % args.model)
    log('Batch size: %d' % args.batch_size)
    log('Number of GPU: %d' % world_size)

    # Warm-up
    log('Running warmup...')
    benchmark_step(model, optimizer, data, target)

    # Benchmark
    log('Running benchmark...')
    img_secs = []
    for x in range(args.num_iters):
        start = time.time()
        benchmark_step(model, optimizer, data, target)
        end = time.time()
        img_sec = args.batch_size * args.num_batches_per_iter / (end-start)
        log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, "GPU"))
        img_secs.append(img_sec)

    # Results
    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    log('Img/sec per %s: %.1f +-%.1f' % ("GPU", img_sec_mean, img_sec_conf))
    log('Total img/sec on %d %s(s): %.1f +-%.1f' % (world_size, "GPU",  world_size * img_sec_mean, world_size * img_sec_conf))


if __name__ == "__main__":
    args = synthetic_parser().parse_args()

    dist.init_process_group("nccl")

    main(args)
