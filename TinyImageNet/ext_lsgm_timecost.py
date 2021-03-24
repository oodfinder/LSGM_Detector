import numpy as np
import sys
import os
import pickle
import argparse
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.wrn import WideResNet
from skimage.filters import gaussian as gblur
from PIL import Image as PILImage
import time

# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std, print_tnr95
    import utils.svhn_loader as svhn
    import utils.lsun_loader as lsun_loader

parser = argparse.ArgumentParser(description='Evaluates a Tiny ImageNet OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--method_name', '-m', type=str, default='wrn_baseline', help='Method name.')
# Loading details
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--load', '-l', type=str, default='./snapshots/baseline', help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
args = parser.parse_args()

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.test_bs, shuffle=False,
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

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.cuda()

            # process data
            output, out_list = net.feature_list(data)
            probs_test = []
            for i, layer in enumerate(out_list):
                if layer.dim() == 4:
                    layer = F.avg_pool2d(layer, layer.size(2))
                    out_list[i] = layer.reshape(layer.shape[0], -1)
                    # get probabilities
                    probs_test.append(gmm_list[i]._estimate_weighted_log_prob(out_list[i].cpu()))
            
            # get scores
            from scipy.special import logsumexp
            scores = []
            for j in range(len(data)):
                m = probs_test[1][j].reshape(-1, 1) + probs_test[2][j].reshape(1, -1) 
                # m.shape == (k1, k2)
                m += bigram[1]

                # layer 2->3
                for i in range(3, n_layers - 1):
                    m = logsumexp(m, axis=0)
                    m = m[:, np.newaxis] + probs_test[i][j].reshape(1, -1)
                    m += bigram[i-1]  # layer i-1 -> i

                scores.append(logsumexp(m))

            smax = to_np(F.softmax(output, dim=1))
            lsgm_scores = -np.array(scores)
            _score.append(lsgm_scores)

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                _right_score.append(lsgm_scores[right_indices])
                _wrong_score.append(lsgm_scores[wrong_indices])

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
                return concat(_score)[:ood_num_examples].copy()


for n_components in [50, 100, 150, 200, 250]:
    print('n_components:', n_components)
    time0 = time.time()

    ### Generate features ###
    train_list = []
    net.eval()
    with torch.no_grad():
        # extract training data
        for itr, (input, target) in enumerate(train_loader):
            input, target = input.cuda(), target  # .cuda()
            y, out_list = net.feature_list(input)

            # process the data
            for i, layer in enumerate(out_list):
                # if itr == 0:
                #     print(tuple(layer.shape), '->', end=' ')
                if layer.dim() == 4:
                    layer = F.avg_pool2d(layer, layer.size(2))
                    out_list[i] = layer.reshape(layer.shape[0], -1)
                # if itr == 0:
                #     print(tuple(out_list[i].shape))

            # save data to list
            train_list.append([layer.cpu() for layer in out_list] + [y.cpu(), target])
            # if itr % 50 == 49:
            #     print((itr + 1) * args.test_bs)

    train_features = [np.concatenate(f) for f in zip(*train_list)]
    n_layers = len(out_list)


    ### train clustering ###
    probs_train = []
    labels_train = []
    gmm_list = []
    # for layers, train...
    for i, features in enumerate(train_features):
        if i == n_layers:  # last layer
            break

        x_train = train_features[i]

        # train kmeans
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag', max_iter=1000,
                                    init_params='kmeans', reg_covar=1e-6, random_state=seed)
        gmm.fit(x_train)
        gmm_list.append(gmm)

        labels_train.append(gmm.predict(x_train))

    path_train = np.vstack(labels_train).T

    bigram = []
    for i in range(0, n_layers - 1):
        count = np.zeros([n_components, n_components]) + 1e-8
        for path in path_train:
            u, v = path[i], path[i + 1]
            count[u][v] += 1

        for j in range(n_components):
            count[j] /= count[j].sum()
        count = np.log(count)

        bigram.append(count)

    print('train:', time.time()-time0)
    time0 = time.time()

    ### testing ###
    in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)
    print('test:', time.time()-time0)
