import numpy as np
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.wrn import WideResNet
from skimage.filters import gaussian as gblur
from PIL import Image as PILImage

# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std, print_tnr95
    import utils.svhn_loader as svhn
    import utils.lsun_loader as lsun_loader
    from utils.maha_detector import MahaDetector

parser = argparse.ArgumentParser(description='Evaluates a Tiny ImageNet OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--use_modified', '-x', action='store_true', help='Use CVPR20 modified version of Mahalanibis detector.')
parser.add_argument('--method_name', '-m', type=str, default='wrn_baseline', help='Method name.')
# Loading details
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--load', '-l', type=str, default='./snapshots/baseline', help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
args = parser.parse_args()

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)

print(args.method_name)
# mean and standard deviation of channels of ImageNet images
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

train_data = dset.ImageFolder(
    root="/data/share/ood_datasets/tinyImageNet/tiny-imagenet-200/train",
    transform=test_transform)
test_data = dset.ImageFolder(
    root="/data/share/ood_datasets/tinyImageNet/tiny-imagenet-200/val",
    transform=test_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.test_bs, shuffle=True,
                                          num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)
num_classes = 200

# Create model
net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

# Restore model
assert args.load != ''
model_name = os.path.join(args.load, args.method_name + '.pt')
net.load_state_dict(torch.load(model_name))
print('Model restored! File:', model_name)

net.eval()

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()

# Train Mahalanobis detector
net_type = 'wrn'
md = MahaDetector(net, train_loader, std, modified_flag=args.use_modified)

# /////////////// Detection Prelims ///////////////

ood_num_examples = len(test_data)
print('OOD examples number {}'.format(ood_num_examples))
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
            break

        data = data.cuda()

        output = net(data)
        smax = to_np(F.softmax(output, dim=1))

        maha_scores = -md.scores(data)
        _score.append(maha_scores)

        if in_dist:
            preds = np.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            _right_score.append(maha_scores[right_indices])
            _wrong_score.append(maha_scores[wrong_indices])

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

num_right = len(right_score)
num_wrong = len(wrong_score)
print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))

# /////////////// End Detection Prelims ///////////////

print('\nUsing TinyImageNet as typical data')

# /////////////// Error Detection ///////////////

print('\n\nError Detection')
show_performance(wrong_score, right_score, method_name=args.method_name)
print_tnr95(wrong_score, right_score)

# /////////////// OOD Detection ///////////////
auroc_list, aupr_list, fpr_list = [], [], []


def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):

    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader)
        measures = get_measures(out_score, in_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])

    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    else:
        print_measures(auroc, aupr, fpr, args.method_name)
    print_tnr95(out_score, in_score)

# /////////////// Gaussian Noise ///////////////

dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
ood_data = torch.from_numpy(np.float32(np.clip(
    np.random.normal(size=(ood_num_examples * args.num_to_avg, 3, 64, 64), scale=0.5), -1, 1)))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nGaussian Noise (sigma = 0.5) Detection')
get_and_print_results(ood_loader)

# /////////////// Rademacher Noise ///////////////

dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
ood_data = torch.from_numpy(np.random.binomial(
    n=1, p=0.5, size=(ood_num_examples * args.num_to_avg, 3, 64, 64)).astype(np.float32)) * 2 - 1
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nRademacher Noise Detection')
get_and_print_results(ood_loader)

# /////////////// Blob ///////////////

ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(ood_num_examples * args.num_to_avg, 64, 64, 3)))
for i in range(ood_num_examples * args.num_to_avg):
    ood_data[i] = gblur(ood_data[i], sigma=2, multichannel=False)
    ood_data[i][ood_data[i] < 0.75] = 0.0

dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
ood_data = torch.from_numpy(ood_data.transpose((0, 3, 1, 2))) * 2 - 1
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nBlob Detection')
get_and_print_results(ood_loader)

# /////////////// Textures ///////////////

ood_data = dset.ImageFolder(root="/data/share/ood_datasets/dtd/images",
                            transform=trn.Compose([trn.Resize(64), trn.CenterCrop(64),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nTexture Detection')
get_and_print_results(ood_loader)

# /////////////// LSUN ///////////////

ood_data = lsun_loader.LSUN("/data/share/ood_datasets/lsun/data", classes='test',
                            transform=trn.Compose([trn.Resize(64), trn.CenterCrop(64),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nLSUN Detection')
get_and_print_results(ood_loader)

# /////////////// iSUN ///////////////

ood_data = dset.ImageFolder("/data/share/ood_datasets/iSUN",
                            transform=trn.Compose([trn.Resize(64), trn.CenterCrop(64),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\niSUN Detection')
get_and_print_results(ood_loader)

# /////////////// CIFAR Data ///////////////

ood_data = dset.CIFAR10('/data/share/ood_datasets/cifar', train=False,
                        transform=trn.Compose([trn.Resize(64), trn.CenterCrop(64),
                                               trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nCIFAR-10 Detection')
get_and_print_results(ood_loader)

# /////////////// Held-out ImageNet Classes ///////////////

# ood_data = dset.ImageFolder(
#     root="/share/data/vision-greg2/users/dan/datasets/TinyImageNet/out/val", transform=test_transform)
# ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
#                                          num_workers=args.prefetch, pin_memory=True)


# print('\n\nHeld-out ImageNet Classes Detection')
# get_and_print_results(ood_loader)

# /////////////// Mean Results ///////////////

print('\n\nMean Test Results')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)

# /////////////// OOD Detection of Validation Distributions ///////////////

if args.validate is False:
    exit()

auroc_list, aupr_list, fpr_list = [], [], []

# /////////////// Uniform Noise ///////////////

dummy_targets = torch.ones(ood_num_examples * args.num_to_avg)
ood_data = torch.from_numpy(
    np.random.uniform(size=(ood_num_examples * args.num_to_avg, 3, 64, 64),
                      low=-1.0, high=1.0).astype(np.float32))
ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True)

print('\n\nUniform[-1,1] Noise Detection')
get_and_print_results(ood_loader)


# /////////////// Arithmetic Mean of Images ///////////////


class AvgOfPair(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))

        return self.dataset[i][0] / 2. + self.dataset[random_idx][0] / 2., 0

    def __len__(self):
        return len(self.dataset)


ood_data = dset.ImageFolder(
    root="/share/data/vision-greg2/users/dan/datasets/TinyImageNet/out/val", transform=test_transform)
ood_loader = torch.utils.data.DataLoader(AvgOfPair(ood_data),
                                         batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

print('\n\nArithmetic Mean of Random Image Pair Detection')
get_and_print_results(ood_loader)


# /////////////// Geometric Mean of Images ///////////////


class GeomMeanOfPair(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))

        return trn.Normalize(mean, std)(torch.sqrt(self.dataset[i][0] * self.dataset[random_idx][0])), 0

    def __len__(self):
        return len(self.dataset)


ood_data = dset.ImageFolder(
    root="/share/data/vision-greg2/users/dan/datasets/TinyImageNet/out/val", transform=trn.ToTensor())
ood_loader = torch.utils.data.DataLoader(
    GeomMeanOfPair(ood_data), batch_size=args.test_bs, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

print('\n\nGeometric Mean of Random Image Pair Detection')
get_and_print_results(ood_loader)

# /////////////// Jigsaw Images ///////////////

ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=args.prefetch, pin_memory=True)

jigsaw = lambda x: torch.cat((
    torch.cat((torch.cat((x[:, 16:32, :32], x[:, :16, :32]), 1),
               x[:, 32:, :32]), 2),
    torch.cat((x[:, 32:, 32:],
               torch.cat((x[:, :32, 48:], x[:, :32, 32:48]), 2)), 2),
), 1)

ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), jigsaw, trn.Normalize(mean, std)])

print('\n\nJigsawed Images Detection')
get_and_print_results(ood_loader)

# /////////////// Speckled Images ///////////////

speckle = lambda x: torch.clamp(x + x * torch.randn_like(x), 0, 1)
ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), speckle, trn.Normalize(mean, std)])

print('\n\nSpeckle Noised Images Detection')
get_and_print_results(ood_loader)

# /////////////// Pixelated Images ///////////////

pixelate = lambda x: x.resize((int(64 * 0.2), int(64 * 0.2)), PILImage.BOX).resize((64, 64), PILImage.BOX)
ood_loader.dataset.transform = trn.Compose([pixelate, trn.ToTensor(), trn.Normalize(mean, std)])

print('\n\nPixelate Detection')
get_and_print_results(ood_loader)

# /////////////// RGB Ghosted/Shifted Images ///////////////

rgb_shift = lambda x: torch.cat((x[1:2].index_select(2, torch.LongTensor([i for i in range(64 - 1, -1, -1)])),
                                 x[2:, :, :], x[0:1, :, :]), 0)
ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), rgb_shift, trn.Normalize(mean, std)])

print('\n\nRGB Ghosted/Shifted Image Detection')
get_and_print_results(ood_loader)

# /////////////// Inverted Images ///////////////

# not done on all channels to make image ood with higher probability
invert = lambda x: torch.cat((x[0:1, :, :], 1 - x[1:2, :, ], 1 - x[2:, :, :],), 0)
ood_loader.dataset.transform = trn.Compose([trn.ToTensor(), invert, trn.Normalize(mean, std)])

print('\n\nInverted Image Detection')
get_and_print_results(ood_loader)

# /////////////// Mean Results ///////////////

print('\n\nMean Validation Results')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)
