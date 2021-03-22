import argparse
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from torchvision import transforms
from torch.autograd import Variable


class MahaDetector:
    def __init__(self, model, train_loader, std, modified_flag=False):
        self.std = std
        self.model = model
        self.modified_flag = modified_flag
        # set information about feature extaction
        model.eval()
        temp_x, _ = iter(train_loader).next()
        out, temp_list = model.feature_list(temp_x.cuda())
        self.num_classes = out.shape[1]
        self.num_output = len(temp_list)
        feature_list = np.empty(self.num_output)
        count = 0
        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1

        self.sample_mean, self.precision = self.sample_estimator(model, self.num_classes, feature_list, train_loader)

        self.best_magnitude = 0
        if modified_flag:
            self.best_magnitude = self.find_best_magnitude(model, train_loader)
            print('best_magnitude:', self.best_magnitude)


    def scores(self, inputs):
        # print('get Mahalanobis scores')
    
        if self.modified_flag:
            Mahalanobis_all = []
            for i in range(self.num_output):
                Mahalanobis = self.get_Mahalanobis_score(self.model, inputs, self.num_classes, self.std, self.sample_mean, self.precision, i, self.best_magnitude)
                Mahalanobis = np.array(Mahalanobis)
                # Mahalanobis.shape == (10000, )
                if Mahalanobis.mean() > 0:
                    continue
                Mahalanobis_all.append(Mahalanobis)
            Mahalanobis_all = np.vstack(Mahalanobis_all).T
            # add all layer scores
            scores = Mahalanobis_all.sum(axis=1)

        else:
            scores = self.get_Mahalanobis_score(self.model, inputs, self.num_classes, self.std, 
                                                         self.sample_mean, self.precision, layer_index=self.num_output-1, magnitude=0)

        # print(Mahalanobis_all.shape)
    
        return np.asarray(scores)


    @staticmethod
    def sample_estimator(model, num_classes, feature_list, train_loader):
        """
        compute sample mean and precision (inverse of covariance)
        return: sample_class_mean: list of class mean
                precision: list of precisions
        """
        import sklearn.covariance
        
        model.eval()
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        correct, total = 0, 0
        num_output = len(feature_list)
        num_sample_per_class = np.empty(num_classes)
        num_sample_per_class.fill(0)
        list_features = []
        for i in range(num_output):
            temp_list = []
            for j in range(num_classes):
                temp_list.append(0)
            list_features.append(temp_list)
        
        for data, target in train_loader:
            total += data.size(0)
            data = data.cuda()
            # data = Variable(data, volatile=True)
            output, out_features = model.feature_list(data)
            
            # get hidden features
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)
                
            # compute the accuracy
            pred = output.data.max(1)[1]
            equal_flag = pred.eq(target.cuda()).cpu()
            correct += equal_flag.sum()
            
            # construct the sample matrix
            for i in range(data.size(0)):
                label = target[i]
                if num_sample_per_class[label] == 0:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] = out[i].view(1, -1)
                        out_count += 1
                else:
                    out_count = 0
                    for out in out_features:
                        list_features[out_count][label] \
                        = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                        out_count += 1                
                num_sample_per_class[label] += 1

            # if total >= 1000:
            #     print('num:', total)
            #     break
                
        sample_class_mean = []
        out_count = 0
        for num_feature in feature_list:
            temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
            for j in range(num_classes):
                temp_list[j] = torch.mean(list_features[out_count][j], 0)
            sample_class_mean.append(temp_list)
            out_count += 1
            
        precision = []
        for k in range(num_output):
            X = 0
            for i in range(num_classes):
                if i == 0:
                    X = list_features[k][i] - sample_class_mean[k][i]
                else:
                    X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

            # print('X:', X.size())
            # find inverse
            group_lasso.fit(X.cpu().numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float().cuda()
            precision.append(temp_precision)
            
        print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

        return sample_class_mean, precision


    @staticmethod
    def get_Mahalanobis_score(model, data, num_classes, std, sample_mean, precision, layer_index, magnitude):
        '''
        Compute the proposed Mahalanobis confidence score on input dataset
        return: Mahalanobis score from layer_index
        '''
        model.eval()
        Mahalanobis = []
        
        # if out_flag == True:
        #     temp_file_name = '%s/confidence_Ga%s_In.txt'%(outf, str(layer_index))
        # else:
        #     temp_file_name = '%s/confidence_Ga%s_Out.txt'%(outf, str(layer_index))
            
        # g = open(temp_file_name, 'w')
        
        # for data, target in test_loader:
        if True:
            # data = data.cuda()
            data = Variable(data, requires_grad = True)
            
            out_features = model.intermediate_forward(data, layer_index)
            out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
            out_features = torch.mean(out_features, 2)
            
            # compute Mahalanobis score
            gaussian_score = 0
            for i in range(num_classes):
                batch_sample_mean = sample_mean[layer_index][i]
                zero_f = out_features.data - batch_sample_mean
                term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
                if i == 0:
                    gaussian_score = term_gau.view(-1,1)
                else:
                    gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
            
            # Input_processing
            sample_pred = gaussian_score.max(1)[1]
            batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
            zero_f = out_features - Variable(batch_sample_mean)
            pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
            loss = torch.mean(-pure_gau)
            loss.backward()
            
            gradient =  torch.ge(data.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            # if net_type == 'densenet':
            #     gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
            #     gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
            #     gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
            # elif net_type == 'resnet':
            #     gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            #     gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            #     gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (std[0]))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (std[1]))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (std[2]))
            tempInputs = torch.add(data.data, -magnitude * gradient)
    
            noise_out_features = model.intermediate_forward(Variable(tempInputs), layer_index)
            noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
            noise_out_features = torch.mean(noise_out_features, 2)
            noise_gaussian_score = 0
            for i in range(num_classes):
                batch_sample_mean = sample_mean[layer_index][i]
                zero_f = noise_out_features.data - batch_sample_mean
                term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
                if i == 0:
                    noise_gaussian_score = term_gau.view(-1,1)
                else:
                    noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

            # with torch.no_grad():
            #     noise_gaussian_score = torch.sigmoid(noise_gaussian_score/100)*2 * F.softmax(model(data), dim=1) # p(x,y) = p(x) * p(y|x)

            noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
            Mahalanobis.extend(noise_gaussian_score.cpu().numpy())
            
        #     for i in range(data.size(0)):
        #         g.write("{}\n".format(noise_gaussian_score[i]))
        # g.close()

        return Mahalanobis


    def find_best_magnitude(self, model, train_loader):
        '''
        Find best magnitude with validation set. See Generalized ODIN in CVPR2020
        return: magnitude
        '''
        validation_size = 2000
        data_list = []
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= validation_size // len(data):
                break
            data = data.cuda()
            data_list.append(data)

        m_list = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]
        score_list = []
        for magnitude in m_list:
            Mahalanobis_all = []
            for i in range(self.num_output):
                tmp_list = []
                for data in data_list:
                    Mahalanobis = self.get_Mahalanobis_score(model, data, self.num_classes, self.std, self.sample_mean, self.precision, i, magnitude)
                    Mahalanobis = np.array(Mahalanobis)
                    tmp_list.append(Mahalanobis)
                # Mahalanobis_one.shape == (10000, )
                Mahalanobis_one = np.concatenate(tmp_list)
                # hack for overflow
                if Mahalanobis_one.mean() > 0:
                    continue
                Mahalanobis_all.append(Mahalanobis_one)

            Mahalanobis_all = np.vstack(Mahalanobis_all).T
            # add all layer scores
            score = Mahalanobis_all.sum(axis=1)
            # average over samples
            mean_score = score.mean()
            score_list.append(mean_score)
            print(f'{magnitude}: {mean_score}')
        best_index = int(np.array(score_list).argmax())
        best_magnitude = m_list[best_index]
        return best_magnitude
