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
parser.add_argument('--epochs', default=161, type=int, metavar='N',
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
parser.add_argument('--gpu', default=1
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
parser.add_argument("--img_size", type=int, default=300)
parser.add_argument("--classes", type=int, default=8)
parser.add_argument("--do_contrast", default=False, action='store_true')
parser.add_argument("--slice_threshold", type=float, default=1)  # 0.05
parser.add_argument("--num_classes", type=int, default=255)
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

    model = PATSAM("PAT", "checkpoint",
                     num_classes=1,
                     is_finetune_image_encoder=True,
                     use_adaptation=False,
                     adaptation_type='LORA',
                     head_type='custom',
                     reduction=4, upsample_times=2, groups=4)

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

    now = datetime.now()
    # args.save_dir = "output_experiment/Sam_h_seg_distributed_tr" + str(args.tr_size) # + str(now)[:-7]
    args.save_dir = "output_experiment/" + args.save_dir
    print(args.save_dir)
    writer = SummaryWriter(os.path.join(args.save_dir, 'tensorboard' + str(gpu)))

    best_loss = 1

    for epoch in range(args.start_epoch, args.epochs):
        is_best = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step=epoch)
        loss_list = []
        # train for one epoch
        train(train_loader, model, optimizer, scheduler, epoch, args, writer, loss_list)
        loss = validate(val_loader, model, epoch, args, writer)

        filename = os.path.join(args.save_dir, 'checkpoint_e%d.pth.tar' % (epoch))
        filename_best = os.path.join(args.save_dir, 'checkpoint_best_e%d.pth.tar' % (epoch))

        if epoch % 10 == 0:
            torch.save(model, filename)


def train(train_loader, model, optimizer, scheduler, epoch, args, writer, loss_list):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    dice_loss = SoftDiceLoss(batch_dice=True, do_bg=False)
    ce_loss = torch.nn.CrossEntropyLoss()

    # switch to train mode
    model.train()

    end = time.time()
    for i, tup in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            img = tup[0].float().cuda(args.gpu, non_blocking=True)
            label = tup[1].float().cuda(args.gpu, non_blocking=True)
        else:
            img = tup[0].float()
            label = tup[1].long()

        img = img / torch.max(img)

        img = img.cuda(args.gpu, non_blocking=True)
        label = label.cuda(args.gpu, non_blocking=True)

        pred = model(img)
        pred = F.interpolate(pred, size=(1024, 1024), mode='bilinear', align_corners=False)

        # print(pred)
        MSELoss = nn.MSELoss()
        L1_LOSS = nn.L1Loss()
        TV_loss = TVLoss()
        perceptual_loss_fn = PerceptualLoss()
        loss_function = StyleTransferLoss().cuda(args.gpu)

        pred1 = pred.expand(-1, 3, -1, -1)
        label1 = label.expand(-1, 3, -1, -1)
        img1 = img.expand(-1, 3, -1, -1)

        pred1 = pred1.cuda(args.gpu, non_blocking=True)
        label1 = label1.cuda(args.gpu, non_blocking=True)
        img1 = img1.cuda(args.gpu, non_blocking=True)

        loss = L1_LOSS(pred, label) + perceptual_loss_fn(pred1, label1) 
        loss_list.append(loss.item())

        loss.requires_grad_(True)
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        writer.add_scalar('train_loss', loss, global_step=i + epoch * len(train_loader))

        if i % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss:.4f}'.format(epoch, i, len(train_loader), loss=loss.item()))

    print('Epoch: %2d Loss: %.4f' % (epoch, np.mean(loss_list)))
    if epoch >= 5:
        scheduler.step(loss)

class StyleTransferLoss(nn.Module):
    def __init__(self, content_weight=1, style_weight=25):
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
        # 将输入展平并计算Gram矩阵
        batch_size, channels, height, width = input.size()
        features = input.view(batch_size, channels, height * width)
        G = torch.bmm(features, features.transpose(1, 2))  # 计算Gram矩阵
        return G.div(channels * height * width)  # 归一化

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

def crop_center(image, label, crop_size=(220, 220)):
    """
    自动寻找图像中物体的中心点，并根据指定的尺寸裁剪。

    参数：
        image (torch.Tensor): 输入图像，规格为 torch.Size([1, 1, H, W])。
        crop_size (tuple): 裁剪尺寸，格式为 (height, width)。

    返回：
        torch.Tensor: 裁剪后的图像。
    """
    # 将图像从torch.Tensor转为numpy数组
    image_1 = image.squeeze().cpu().numpy()

    image_np = label.squeeze().cpu().numpy()

    # print(image_1.shape)
    # print(image_np.shape)

    # 计算图像的质心
    y_indices, x_indices = np.where(image_np > 0)  # 获取非零像素的位置
    if len(y_indices) == 0 or len(x_indices) == 0:
        # 如果图像为空或全零，返回原图
        print("Image appears empty; returning original.")
        return label

    # 计算质心
    center_x = int(np.mean(x_indices))
    center_y = int(np.mean(y_indices))

    # 裁剪尺寸
    crop_height, crop_width = crop_size
    half_crop_height = crop_height // 2
    half_crop_width = crop_width // 2

    # 计算裁剪区域
    start_x = max(center_x - half_crop_width, 0)
    end_x = min(center_x + half_crop_width, image_np.shape[1])
    start_y = max(center_y - half_crop_height, 0)
    end_y = min(center_y + half_crop_height, image_np.shape[0])

    # 对图像进行裁剪
    cropped_image_1 = image_1[:, start_y:end_y, start_x:end_x]

    cropped_image_np = image_np[start_y:end_y, start_x:end_x]

    # 如果裁剪区域小于指定大小，则填充边界
    # padded_cropped_image1 = np.zeros((3, crop_height, crop_width), dtype=image_1.dtype)

    padded_cropped_image = np.zeros((crop_height, crop_width), dtype=image_np.dtype)

    # padded_cropped_image1[2, :cropped_image_1.shape[0], :cropped_image_1.shape[1]] = cropped_image_1

    padded_cropped_image[:cropped_image_np.shape[0], :cropped_image_np.shape[1]] = cropped_image_np

    # 将裁剪后的图像转回torch.Tensor
    cropped_image_2 = torch.tensor(cropped_image_1).unsqueeze(0)

    cropped_image = torch.tensor(padded_cropped_image).unsqueeze(0).unsqueeze(0)

    return cropped_image_2, cropped_image

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

