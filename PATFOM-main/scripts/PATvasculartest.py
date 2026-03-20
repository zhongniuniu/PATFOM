import argparse
import builtins
import math
from math import exp
import os
import random
import shutil
import time
import warnings
import pickle
import numpy as np
from datetime import datetime
import nibabel as nib
from PIL import Image
from torch.autograd import Variable
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

from models.model_proxy_SAM import PATSAM, ModelWithLoss
from models.segment_anything.utils.transforms import ResizeLongestSide

from dataset import generate_dataset, generate_test_loader
from evaluate import test_synapse, test_acdc
import pytorch_ssim
import torchvision.utils as vutils

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=2001, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
# parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
#                     help='learning rate schedule (when to drop lr by 10x)')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum of SGD solver')
# parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)',
#                     dest='weight_decay')
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
parser.add_argument('--gpu', default=0
                    , type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--model_type', type=str, default="vit_b", help='path to splits file')
parser.add_argument('--src_dir', type=str, default=None, help='path to splits file')
parser.add_argument('--data_dir', type=str, default=None, help='path to datafolder')
parser.add_argument("--img_size", type=int, default=1024)
parser.add_argument("--classes", type=int, default=8)
parser.add_argument("--do_contrast", default=False, action='store_true')
parser.add_argument("--slice_threshold", type=float, default=1)  # 0.05
parser.add_argument("--num_classes", type=int, default=1)
parser.add_argument("--fold", type=int, default=0)

parser.add_argument("--tr_size", type=int, default=1)
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--load_saved_model", action='store_true',
                    help='whether freeze encoder of the segmenter')
parser.add_argument("--saved_model_path", type=str, default=None)
parser.add_argument("--load_pseudo_label", default=False, action='store_true')
parser.add_argument("--dataset", type=str, default="synapse")

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            'You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.')

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

    model = torch.load('checkpoint path', map_location='cuda:1')

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

    freeze_image_encoder_name = "image_encoder"
    # freeze weights in the image_encoder
    for name, param in model.named_parameters():
        if param.requires_grad and freeze_image_encoder_name in name:
            param.requires_grad = True  # false
        else:
            # nn.init.normal_(param, mean=0, std=0.25)
            param.requires_grad = True
        # param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9)

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

    test(test_loader, model, args)


def sobel_edge_detection(image):
    """计算 Sobel 边缘"""
    # Sobel 滤波器
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)

    # 转换为灰度图计算梯度
    image_gray = image.mean(dim=1, keepdim=True)  # 转为单通道
    edge_x = F.conv2d(image_gray, sobel_x, padding=1)
    edge_y = F.conv2d(image_gray, sobel_y, padding=1)
    edges = torch.sqrt(edge_x**2 + edge_y**2)
    return edges



def gram(x):
    # get the batch size, channels, height, and width of the image
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w * h)
    G = f.bmm(f.transpose(1, 2)) / (ch * h * w)
    return G

def test(test_loader, model, args):
    # epoch_str = 'Epoch%2d' % (epoch)
    join = os.path.join
    path1 = join(args.save_dir, "infer")
    path2 = join(args.save_dir, "label")
    if not os.path.exists(path1):
        os.mkdir(path1)
    if not os.path.exists(path2):
        os.mkdir(path2)

    if not os.path.exists(join("/public2/zhongyutian/PATFOM-main/infer/")):
        os.mkdir(join("/public2/zhongyutian/PATFOM-main/infer/"))


    split_dir = os.path.join(args.src_dir, "splits.pkl")
    with open(split_dir, "rb") as f:
        splits = pickle.load(f)
    test_keys = splits[args.fold]['test']
    test_keys = test_keys[0:args.tr_size]
    print(test_keys)  # 'patient_014'

    model.eval()

    for key in test_keys:
        preds = []
        labels = []
        k = 1

        with torch.no_grad():
            for i, tup in enumerate(test_loader):
                if args.gpu is not None:

                    img = tup[0].float().cuda(args.gpu, non_blocking=True)
                    label = tup[1].float().cuda(args.gpu, non_blocking=True)
                else:
                    img = tup[0]
                    label = tup[1]

                # print(img.shape())
                img = img / torch.max(img)


                mask = model(img)

                save_dir1 = "/public2/zhongyutian/PATFOM-main/infer"
                # image1 = Image.fromarray(mask.astype('uint8'))
                # 构建图像保存路径和文件名
                image_name1 = f"image_{k}.png"
                image_path1 = f"{save_dir1}/{image_name1}"

                image_grid1 = vutils.make_grid(mask, nrow=1, padding=0)
                vutils.save_image(image_grid1, image_path1, nrow=1)

                k = k + 1

        print("finish saving file:", key)

def tensor_to_image(tensor, save_path):
    # 将张量转换为 PIL 图像
    image = tensor.numpy()
    image = image.astype(np.uint8)
    image = Image.fromarray(image)

    # 保存图像到指定路径
    image.save(save_path)

class StyleTransferLoss(nn.Module):
    def __init__(self, content_weight=0, style_weight=5000):
        super(StyleTransferLoss, self).__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target, style_image):
        # 计算内容损失
        content_loss = self.mse_loss(output, target)

        # 计算风格损失
        # 这里假设您有一个方法 `gram_matrix` 来计算风格图像的Gram矩阵
        style_gram = self.gram_matrix(style_image)

        # 假设输出图像也需要计算Gram矩阵
        output_gram = self.gram_matrix(output)

        style_loss = self.mse_loss(output_gram, style_gram)

        # 组合损失
        total_loss = (self.content_weight * content_loss) + (self.style_weight * style_loss)
        return total_loss

    def gram_matrix(self, input):
        # 将输入展平并计算 Gram 矩阵
        batch_size, channels, height, width = input.size()
        features = input.view(batch_size, channels, height * width)  # 使用 view 更高效
        G = torch.bmm(features, features.transpose(1, 2))  # 计算 Gram 矩阵
        return G.div(channels * height * width)  # 归一化

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

class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer=8):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:feature_layer+1].cuda()
        self.feature_extractor = nn.Sequential(*vgg).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, img1, img2):
        features_img1 = self.feature_extractor(img1)
        features_img2 = self.feature_extractor(img2)
        loss = nn.functional.mse_loss(features_img1, features_img2)
        return loss

def normalize_to_0_255(tensor):
    # 找到张量的最小值和最大值
    min_val = tensor.min()
    max_val = tensor.max()

    # 归一化到 [0, 1] 范围
    normalized_tensor = (tensor - min_val) / (max_val - min_val)

    # 缩放到 [0, 255] 范围
    scaled_tensor = 255 * normalized_tensor

    # 转换为整数类型
    scaled_tensor = scaled_tensor.to(torch.uint8)

    return scaled_tensor


class GradientLoss(nn.Module):
    def __init__(self, operator="Roberts", channel_mean=True):
        r"""
       :param operator: in ['Sobel', 'Prewitt','Roberts','Scharr']
       :param channel_mean: 是否在通道维度上计算均值
       """
        super(GradientLoss, self).__init__()
        assert operator in ['Sobel', 'Prewitt', 'Roberts', 'Scharr'], "Unsupported operator"
        self.channel_mean = channel_mean
        self.operators = {
            "Sobel": {
                'x': torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float),
                'y': torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float)
            },
            "Prewitt": {
                'x': torch.tensor([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]], dtype=torch.float),
                'y': torch.tensor([[[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]]], dtype=torch.float)
            },
            "Roberts": {
                'x': torch.tensor([[[[1, 0], [0, -1]]]], dtype=torch.float),
                'y': torch.tensor([[[[0, -1], [1, 0]]]], dtype=torch.float)
            },
            "Scharr": {
                'x': torch.tensor([[[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]]], dtype=torch.float),
                'y': torch.tensor([[[[-3, 10, -3], [0, 0, 0], [3, 10, 3]]]], dtype=torch.float)
            },
        }
        self.op_x = self.operators[operator]['x'].cuda()
        self.op_y = self.operators[operator]['y'].cuda()

    def gradients(self, img_tensor):
        op_x, op_y = self.op_x, self.op_y
        if self.channel_mean:
            img_tensor = img_tensor.mean(dim=1, keepdim=True)
            groups = 1
        else:
            groups = img_tensor.shape[1]
            op_x = op_x.repeat(groups, 1, 1, 1)
            op_y = op_y.repeat(groups, 1, 1, 1)
        grad_x = F.conv2d(img_tensor, op_x, groups=groups)
        grad_y = F.conv2d(img_tensor, op_y, groups=groups)
        return grad_x, grad_y

    def forward(self, img1, img2):
        grad_x1, grad_y1 = self.gradients(img1)
        grad_x2, grad_y2 = self.gradients(img2)
        diff_x = torch.abs(grad_x1 - grad_x2)
        diff_y = torch.abs(grad_y1 - grad_y2)
        total_loss = torch.mean(diff_x + diff_y)
        return total_loss


def save_image_3d(tensor, slice_idx, file_name):
    '''
    tensor: [bs, c, h, w, 1]
    '''
    image_num = len(slice_idx)
    tensor = tensor[0, slice_idx, ...].permute(0, 3, 1, 2).cpu().data  # [c, 1, h, w]
    image_grid = vutils.make_grid(tensor, nrow=image_num, padding=0)
    vutils.save_image(image_grid, file_name, nrow=1)


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

if __name__ == '__main__':
    main()

