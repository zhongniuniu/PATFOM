import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import pickle
import numpy as np
from datetime import datetime
import nibabel as nib
import scipy.io as io
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
import torchvision.utils as vutils

import cv2
import sys

sys.path.append('.../PATFOM')
from loss_functions.dice_loss import SoftDiceLoss
from loss_functions.metrics import dice_pytorch, SegmentationMetric

from dataset import generate_dataset, generate_test_loader
from evaluate import test_synapse, test_acdc
from models.model_proxy_SAM import PATSAM, ModelWithLoss
from models.segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=4, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--model_type', type=str, default="vit_b", help='path to splits file')
parser.add_argument('--src_dir', type=str, default=".../PATFOM/Seg_test", help='path to splits file')
parser.add_argument('--data_dir', type=str, default=".../PATFOM/Seg_test/imgs/", help='path to datafolder')
parser.add_argument("--img_size", type=int, default=1024)
parser.add_argument("--classes", type=int, default=8)
parser.add_argument("--do_contrast", default=False, action='store_true')
parser.add_argument("--slice_threshold", type=float, default=1)  # 0.05
parser.add_argument("--num_classes", type=int, default=2)  # need to change
parser.add_argument("--fold", type=int, default=5)
parser.add_argument("--tr_size", type=int, default=1)
parser.add_argument("--save_dir", type=str, default="./Test/")
parser.add_argument("--load_saved_model", action='store_true',
                    help='whether freeze encoder of the segmenter')
parser.add_argument("--saved_model_path", type=str, default=None)
parser.add_argument("--load_pseudo_label", default=False, action='store_true')
parser.add_argument("--dataset", type=str, default="ACDC")


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


#
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')  # map_location
    return checkpoint


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model

    model = PATSAM("PAT", ".../PATFOM/PATFOM_Pre.pt",
                     num_classes=args.num_classes,
                     is_finetune_image_encoder=False,
                     use_adaptation=False,
                     adaptation_type='LORA',
                     head_type='custom',
                     reduction=4,
                     upsample_times=2,
                     groups=4)

    color_map = {
        0: (0, 0, 0),
        1: (255, 0, 0),  # Body
        2: (0, 255, 0),  # Liver
        3: (0, 0, 255),  # Spine
        4: (255, 255, 0),  # Vessel
        5: (255, 0, 255),  # Kidneys
        6: (0, 255, 255)  # Spleen
    }

    checkpoint_path = '.../PATFOM/.../MOST.pth.tar'  #
    if os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint(checkpoint_path)
        if 'state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['state_dict'])
                print(f"Successfully loaded weights from {checkpoint_path}")
            except RuntimeError as e:
                print(f"Error loading state_dict: {e}")
        else:
            print(f"Warning: 'state_dict' not found in checkpoint {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # freeze weights in the image_encoder
    for name, param in model.named_parameters():
        if param.requires_grad and "image_encoder" in name:
            # param.requires_grad = False
            param.requires_grad = True
        else:
            param.requires_grad = True
        # param.requires_grad = True
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    train_loader, train_sampler, val_loader, val_sampler, test_loader, test_sampler = generate_dataset(args)
    # val_loader, val_sampler, test_loader, test_sampler = generate_dataset(args)

    now = datetime.now()
    # args.save_dir = "output_experiment/Sam_h_seg_distributed_tr" + str(args.tr_size) # + str(now)[:-7]
    args.save_dir = ".../PATFOM/output_experiment/" + args.save_dir
    print(args.save_dir)
    writer = SummaryWriter(os.path.join(args.save_dir, 'tensorboard' + str(gpu)))

    # filename = os.path.join(args.save_dir, 'checkpoint_b%d.pth.tar' % (args.batch_size))
    best_loss = 100

    for epoch in range(args.start_epoch, args.epochs):
        is_best = False
        if args.distributed:
            # train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step=epoch)

        print(train_loader)
        # train for one epoch
        # train(train_loader, model, optimizer, scheduler, epoch, args, writer)

        filename = os.path.join(args.save_dir, 'checkpoint_epoch%d.pth.tar' % (epoch))
        test(model, args, epoch)
        if args.dataset == 'synapse':
            test_synapse(args)
        elif args.dataset == 'ACDC' or args.dataset == 'acdc':
            test_acdc(args)

    # Add color
    input_dir = ".../PATFOM/output_experiment/Test/infer/Epoch 0/"
    output_dir = ".../PATFOM/output_experiment/Test/infer/Epoch 0/"
    os.makedirs(output_dir, exist_ok=True)
    for imgname in os.listdir(input_dir):
        if imgname.endswith(".png"):
            img_path = os.path.join(input_dir, imgname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            colored_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    try:
                        colored_img[i, j] = color_map[img[i, j]]
                    except KeyError:
                        print(f"Warning: Unknown pixel value {img[i, j]} in {img_path}")
                        colored_img[i, j] = (128, 128, 128)

            output_path = os.path.join(output_dir, imgname)
            cv2.imwrite(output_path, colored_img)
    print(f"************Have add color to the mask************")

    # Add image
    image_dir = ".../PATFOM/Seg_test/imgs/case_156/"
    mask_dir = output_dir
    image_filenames = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png"))]

    for resultname in image_filenames:
        image_path = os.path.join(image_dir, resultname)
        mask_path = os.path.join(mask_dir, resultname)

        if os.path.exists(mask_path):

            overlay = add_segmentation_to_image(image_path, mask_path, alpha=0.4)

            # cv2.imshow("Overlayed Image", overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imwrite(os.path.join(
                ".../PATFOM/output_experiment/Test/infer/Epoch 0_result/",
                f"{resultname}"), overlay)
        else:
            print(f"no {mask_path}")


def test(model, args, epoch):
    epoch_str = 'Epoch%2d' % (epoch)
    save_dir1 = ".../PATFOM/output_experiment/Test/infer/"
    # save_dir2 = ".../PATFOM/output_experiment/Test/label/"

    print('Test')
    join = os.path.join
    if not os.path.exists(join(args.save_dir, "infer")):
        os.mkdir(join(args.save_dir, "infer"))

    if not os.path.exists(
            join(".../PATFOM/output_experiment/Test/infer/", epoch_str)):
        os.mkdir(join(".../PATFOM/output_experiment/Test/infer/", epoch_str))

    split_dir = os.path.join(args.src_dir, "splits.pkl")
    with open(split_dir, "rb") as f:
        splits = pickle.load(f)
    test_keys = splits[args.fold]['train']
    test_keys = test_keys[0:args.tr_size]
    print(test_keys)

    model.eval()

    for key in test_keys:
        preds = []
        labels = []
        data_loader = generate_test_loader(key, args)
        with torch.no_grad():
            for i, tup in enumerate(data_loader):
                if args.gpu is not None:
                    img = tup[0].float().cuda(args.gpu, non_blocking=True)
                    label = tup[1].long().cuda(args.gpu, non_blocking=True)
                else:
                    img = tup[0]
                    label = tup[1]

                mask = model(img)
                mask_softmax = F.softmax(mask, dim=1)
                mask = torch.argmax(mask_softmax, dim=1)

                preds.append(mask.cpu().numpy())
                # labels.append(label.cpu().numpy())

            preds = np.concatenate(preds, axis=0)
            # labels = np.concatenate(labels, axis=0).squeeze()
            print(preds.shape)
            if "." in key:
                key = key.split(".")[0]

            # ni_pred = nib.Nifti1Image(preds.astype(np.int8), affine=np.eye(4))
            # nib.save(ni_pred, join(save_dir1, epoch_str, key + '.nii'))
            output_pred_paths = save_separate_images(preds, key + '_pred', join(save_dir1, epoch_str))

        print("************Finish texting img file:", key)


def add_segmentation_to_image(image_path, mask_path, alpha=0.5):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, image.shape[:2], interpolation=cv2.INTER_NEAREST)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)

    return overlay


def save_separate_images(data, filename, save_dir):
    data = data.astype(np.uint8)
    output_paths = []
    for i in range(data.shape[0]):
        slice = data[i]
        output_path = os.path.join(save_dir, f"{i + 1}.png")
        cv2.imwrite(output_path, slice)
        output_paths.append(output_path)
    return output_paths


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, filename)
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

    # python main_moco.py --data_dir ./data/mmwhs/ --do_contrast --dist-url 'tcp://localhost:10001'
    # --multiprocessing-distributed --world-size 1 --rank 0

