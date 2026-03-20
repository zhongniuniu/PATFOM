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
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
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
parser.add_argument('--gpu', default=5, type=int,
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
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

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

    test(test_loader, model, args, epoch)

def test(test_loader, model, args, epoch):
    epoch_str = 'Epoch%2d' % (epoch)
    print('Test')
    join = os.path.join
    path1 = join(args.save_dir, "infer")
    path2 = join(args.save_dir, "label")
    path3 = join(args.save_dir, "mask")
    if not os.path.exists(path1):
        os.mkdir(path1)
    if not os.path.exists(path2):
        os.mkdir(path2)
    if not os.path.exists(path3):
        os.mkdir(path3)

    if not os.path.exists(join("/public/zhongyutian/PATFOM-main/infer/", epoch_str)):
        os.mkdir(join("/public/zhongyutian/PATFOM-main/infer/", epoch_str))
    if not os.path.exists(join("/public/zhongyutian/PATFOM-main/label/", epoch_str)):
        os.mkdir(join("/public/zhongyutian/PATFOM-main/label/", epoch_str))
    if not os.path.exists(join("/public/zhongyutian/PATFOM-main/mask/", epoch_str)):
        os.mkdir(join("/public/zhongyutian/PATFOM-main/mask/", epoch_str))

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

                img = img / torch.max(img)
                # label = label / torch.max(label)

                crop_size = 200

                # 获取原图像尺寸
                _, _, h, w = img.size()

                # 计算裁剪起始位置
                start_h = (h - crop_size) // 2
                start_w = (w - crop_size) // 2

                # 裁剪图像
                img = img[:, :, start_h:start_h + crop_size, start_w:start_w + crop_size]

                block_size = 10
                mask_ratio = 0.1  # 掩膜比例，20%

                save_dir0 = "/public/zhongyutian/PATFOM-main/mask"

                image_name0 = f"image_{k}.png"
                image_path0 = f"{save_dir0}/{epoch_str}/{image_name0}"

                # 应用随机多个平均掩膜
                img = apply_random_average_mask_pytorch(img.cpu(), block_size, mask_ratio, image_path0)
                img = img.cuda()

                mask = model(img)


                save_dir1 = "/public/zhongyutian/PATFOM-main/infer"
                # image1 = Image.fromarray(mask.astype('uint8'))
                # 构建图像保存路径和文件名
                image_name1 = f"image_{k}.png"
                image_path1 = f"{save_dir1}/{epoch_str}/{image_name1}"

                image_grid1 = vutils.make_grid(mask, nrow=1, padding=0)
                vutils.save_image(image_grid1, image_path1, nrow=1)

                save_dir2 = "/public/zhongyutian/PATFOM-main/label"
                # image1 = Image.fromarray(mask.astype('uint8'))
                # 构建图像保存路径和文件名
                image_name2 = f"image_{k}.png"
                image_path2 = f"{save_dir2}/{epoch_str}/{image_name2}"

                image_grid2 = vutils.make_grid(img, nrow=1, padding=0)
                vutils.save_image(image_grid2, image_path2, nrow=1)

                k = k + 1

        print("finish saving file:", key)


def tensor_to_image(tensor, save_path):
    # 将张量转换为 PIL 图像
    image = tensor.numpy()
    image = image.astype(np.uint8)
    image = Image.fromarray(image)

    # 保存图像到指定路径
    image.save(save_path)


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

# 感知损失（Perceptual Loss）
class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer=2):
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

# 对抗损失（Adversarial Loss）
class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.adversarial_loss = nn.BCELoss()

    def forward(self, prediction, is_real):
        if is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        loss = self.adversarial_loss(prediction, target)
        return loss

# 边缘感知损失（Edge-aware Loss）
class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super(EdgeAwareLoss, self).__init__()

    def forward(self, generated, target):
        generated_grad = self.compute_gradient(generated)
        target_grad = self.compute_gradient(target)
        return F.l1_loss(generated_grad, target_grad)

    def compute_gradient(self, x):
        grad_x = x[:, :, :-1, :-1] - x[:, :, 1:, :-1]
        grad_y = x[:, :, :-1, :-1] - x[:, :, :-1, 1:]
        return torch.sqrt(grad_x**2 + grad_y**2)

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

def apply_random_average_mask_pytorch(image, num_masks, mask_ratio, mask_save_path=None):
    """
    对图像进行固定比例随机分布的掩膜，其中掩膜的灰度值为该掩膜位置中图像的像素值的平均值。
    掩膜之间不会相互重叠，并返回以白色为背景的掩膜图像，并在提供保存路径时保存该掩膜图像。

    :param image: 输入图像，格式为torch.Size([1, 3, 1024, 1024])
    :param mask_ratio: 掩膜的比例（0到1之间）
    :param num_masks: 掩膜的数量
    :param mask_save_path: 保存掩膜图像的路径，可选
    :return: 掩膜处理后的图像，格式为torch.Size([1, 3, 1024, 1024])，以及掩膜图像
    """
    # 确保输入图像的格式为torch.Size([1, 3, 1024, 1024])
    assert image.dim() == 4 and image.size(0) == 1 and image.size(1) == 3

    # 获取图像的尺寸
    _, _, height, width = image.size()

    # 转换为numpy数组进行处理
    image_np = image.squeeze().numpy()

    # 创建一个全零的输出数组
    output_np = image_np

    # 创建一个全白的掩膜图像
    mask_image = np.ones((3, height, width), dtype=np.float32)

    # 确定掩膜的大小
    mask_height = int(height * mask_ratio)
    mask_width = int(width * mask_ratio)

    # 跟踪已被掩膜覆盖的区域
    covered_area = np.zeros((height, width), dtype=bool)

    for _ in range(num_masks):
        while True:
            # 随机选择掩膜的起始位置
            i = random.randint(0, height - mask_height)
            j = random.randint(0, width - mask_width)

            # 检查掩膜区域是否已被覆盖
            if not covered_area[i:i + mask_height, j:j + mask_width].any():
                break

        # 标记已覆盖区域
        covered_area[i:i + mask_height, j:j + mask_width] = True

        # 获取当前掩膜区域的子图像
        mask_region = image_np[:, i:i + mask_height, j:j + mask_width]

        # 计算子图像的平均值
        mean_value = np.mean(mask_region, axis=(1, 2), dtype=float)
        # mean_value = np.zeros(mask_region, axis=(1, 2), dtype=float)

        # 将平均值赋值给输出图像的相应区域
        # output_np[:, i:i + mask_height, j:j + mask_width] = mean_value[:, None, None]
        output_np[:, i:i + mask_height, j:j + mask_width] = 0

        # 将平均值赋值给掩膜图像的相应区域
        # mask_image[:, i:i + mask_height, j:j + mask_width] = mean_value[:, None, None]
        mask_image[:, i:i + mask_height, j:j + mask_width] = 0

    # 转换回PyTorch张量
    output_tensor = torch.from_numpy(output_np).unsqueeze(0).float()

    # 如果提供了保存路径，保存掩膜图像
    if mask_save_path:
        # 转换掩膜图像到0-255范围，并转换为PIL图像
        mask_image_pil = Image.fromarray((mask_image.transpose(1, 2, 0) * 255).astype(np.uint8))
        mask_image_pil.save(mask_save_path)

    return output_tensor


def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False):
    # Assumes that img1 and img2 have the shape torch.Size([1, 1, 200, 200])
    channel = img1.size(1)
    device = img1.device
    if window is None:
        window = create_window(window_size, channel)

    window = window.to(device)
    # print(f'img1 device: {img1.device}, img2 device: {img2.device}, window device: {window.device}')
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        return 1 - ssim(img1, img2, window_size=self.window_size, window=self.window, size_average=self.size_average)


if __name__ == '__main__':
    main()

    # python main_moco.py --data_dir ./data/mmwhs/ --do_contrast --dist-url 'tcp://localhost:10001'
    # --multiprocessing-distributed --world-size 1 --rank 0
