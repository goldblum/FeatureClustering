# -*- coding: utf-8 -*-
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import sys
sys.path.append('..')
from models.R2D2_embedding import R2D2Embedding
from models.protonet_embedding import ProtoNetEmbedding
from models.ResNet12_embedding import resnet12
from models.WRN_embedding import wrn_28_10
from models.MAML_architecture import MAML_Embedding
from utils import set_gpu, Timer, count_accuracy, check_dir, log
import pdb

def feature_clustering(outputs):
    chunk = outputs.shape[0]//4
    assert chunk*4 == outputs.shape[0], "images per batch must be divisible by 4"
    im1_c1 = outputs[0:chunk]         # The first half of batch 1
    im2_c1 = outputs[chunk:2*chunk]   # The second half of batch 1
    im1_c2 = outputs[2*chunk:3*chunk] # First half of batch 2
    im2_c2 = outputs[3*chunk:4*chunk] # Second half of batch 2

    d1 = im1_c1-im1_c2  # Compute the difference between features vectors for different class labels using batch 1
    d2 = im2_c1-im2_c2  # Compute corresponding differences using batch 2
    error = (d1-d2).norm(dim=1)    # Compute the discrepancy between the differences on batch 1 and batch 2
    mean_norm = (d2.norm(dim=1)+d1.norm(dim=1))/2+1e-4  # Compute the average magnitude of the difference vectors
    # The pair loss is the error/mismatch between the two sets of differences, normalized by the size of the difference vectors
    feature_clustering = error/mean_norm #av_norm
    feature_clustering = feature_clustering*feature_clustering
    feature_clustering = feature_clustering.mean()
    return feature_clustering

def hyperplane_variation(outputs):
    chunk = outputs.shape[0]//4
    assert chunk*4 == outputs.shape[0], "images per batch must be divisible by 4"
    im1_c1 = outputs[0:chunk]         # The first half of batch 1
    im2_c1 = outputs[chunk:2*chunk]   # The second half of batch 1
    im1_c2 = outputs[2*chunk:3*chunk] # First half of batch 2
    im2_c2 = outputs[3*chunk:4*chunk] # Second half of batch 2

    # calculate the mean for each class
    m1 = (im1_c1 + im2_c1)/2
    m2 = (im1_c2 + im2_c2)/2
    m = (m1 + m2)/2
    sw1 = ((im1_c1 - m1).norm(dim=1))**2 + ((im2_c1 - m1).norm(dim=1))**2
    sw2 = ((im1_c2 - m2).norm(dim=1))**2 + ((im2_c2 - m2).norm(dim=1))**2
    sb = ((m1 - m).norm(dim=1))**2 + ((m2 - m).norm(dim=1))**2 + 1e-6
    loss = (sw1 + sw2)/sb
    loss = loss.mean()
    return loss

def label_converter(options, dataset):
    if options.dataset == 'miniImageNet':
        classes = range(64)
        label_dict = {}
        for label in classes:
            label_dict[label]=torch.Tensor([label])
    elif options.dataset == 'CIFAR_FS':
        # Get the labels
        label_dict = {}
        labels = torch.LongTensor([xy[1] for xy in dataset])
        idx = torch.LongTensor(range(len(labels)))
        # Extract the dataset indices for each class
        classes = []
        for c in range(100):
            if idx[labels==c].view(-1,1).shape[0] != 0:
                classes.append(c)
        counter = 0
        for c in classes:
            label_dict[c]=torch.Tensor([counter])
            counter += 1
    return classes, label_dict

class PairData(Dataset):
    def __init__(self, dataset, classes):
        self.dataset=dataset
        self.classes=classes
        self.num_classes, self.images_per_class, self.class_idx = self.get_info()
        self.class_map, self.data_map, self.counter = self.generate_maps()
    def generate_maps(self):
        class_map = torch.stack([torch.randperm(self.num_classes) for i in range(self.images_per_class//2)]) #300x64
        class_map = class_map.view(-1, 2) #9600x2
        data_map = torch.stack([torch.randperm(self.images_per_class) for i in range(self.num_classes)], dim=1)# 600x64
        counter = torch.zeros(self.num_classes, dtype=torch.long)
        return class_map, data_map, counter
    def shuffle_maps(self):
        self.class_map, self.data_map, self.counter = self.generate_maps()
    def get_info(self):
        labels = torch.LongTensor([xy[1] for xy in self.dataset])
        idx = torch.LongTensor(range(len(labels)))
        class_idx = []
        for c in classes:
            class_idx.append(idx[labels==c].view(-1,1))
        class_idx = torch.cat(class_idx,1) # 600 x 64
        images_per_class = class_idx.shape[0]
        num_classes = class_idx.shape[1]
        return num_classes, images_per_class, class_idx
    def __len__(self):
        return self.num_classes*self.images_per_class//4
    def __getitem__(self, idx):
        c1 = self.class_map[idx,0]
        c2 = self.class_map[idx,1]
        im1_c1 = self.dataset[self.class_idx[self.data_map[self.counter[c1], c1], c1]]
        im2_c1 = self.dataset[self.class_idx[self.data_map[self.counter[c1]+1, c1], c1]]
        im1_c2 = self.dataset[self.class_idx[self.data_map[self.counter[c2], c2], c2]]
        im2_c2 = self.dataset[self.class_idx[self.data_map[self.counter[c2]+1, c2], c2]]
        self.counter[c1]+=2
        self.counter[c2]+=2
        return (im1_c1[0], im2_c1[0], im1_c2[0], im2_c2[0],im1_c1[1], im2_c1[1], im1_c2[1], im2_c2[1])


def get_model(options):
    # Choose the embedding network & corresponding linear head
    if options.dataset == 'miniImageNet':
        if options.network == 'ProtoNet':
            network = ProtoNetEmbedding().cuda()
            cls_head = torch.nn.Linear(1600, 64).cuda()
        elif options.network == 'R2D2':
            network = R2D2Embedding().cuda()
            cls_head = torch.nn.Linear(51200, 64).cuda()
        elif options.network == 'ResNet':
            if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
                network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5).cuda()
                network = torch.nn.DataParallel(network)
            else:
                network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2).cuda()
            cls_head = torch.nn.Linear(16000, 64).cuda()
        elif options.network == 'WideResNet':
            if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
                network = wrn_28_10().cuda()
                network = torch.nn.DataParallel(network)
            else:
                network = wrn_28_10().cuda()
            cls_head = torch.nn.Linear(2560, 64).cuda()
        elif options.network == 'MAML':
            network = MAML_Embedding().cuda()
            cls_head = torch.nn.Linear(800,64).cuda()
        else:
            print ("Cannot recognize the network type")
            assert(False)
    elif options.dataset == 'CIFAR_FS':
        if options.network == 'ProtoNet':
            network = ProtoNetEmbedding().cuda()
            cls_head = torch.nn.Linear(256, 64).cuda()
        elif options.network == 'R2D2':
            network = R2D2Embedding().cuda()
            cls_head = torch.nn.Linear(8192, 64).cuda()
        elif options.network == 'ResNet':
            if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
                network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=5).cuda()
                network = torch.nn.DataParallel(network)
            else:
                network = resnet12(avg_pool=False, drop_rate=0.1, dropblock_size=2).cuda()
            cls_head = torch.nn.Linear(2560, 64).cuda()
        elif options.network == 'WideResNet':
            if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
                network = wrn_28_10().cuda()
                network = torch.nn.DataParallel(network)
            else:
                network = wrn_28_10().cuda()
            cls_head = torch.nn.Linear(640, 64).cuda()
        elif options.network == 'MAML':
            network = MAML_Embedding().cuda()
            cls_head = torch.nn.Linear(800,64).cuda()
        else:
            print ("Cannot recognize the network type")
            assert(False)
    return (network, cls_head)

def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet
        dataset_train = MiniImageNet(phase='train')
        #dataset_val = MiniImageNet(phase='val')
    elif options.dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet
        dataset_train = tieredImageNet(phase='train')
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS
        dataset_train = CIFAR_FS(phase='train')
        dataset_val = CIFAR_FS(phase='val')
    elif options.dataset == 'FC100':
        from data.FC100 import FC100
        dataset_train = FC100(phase='train')
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
        
    #return (dataset_train, dataset_val, DataLoader)
    return (dataset_train, DataLoader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=300,
                            help='number of training epochs')
    parser.add_argument('--num-warmup', type=int, default=10,
                            help='number of warm up training epochs')
    parser.add_argument('--save-epoch', type=int, default=150,
                            help='frequency of model saving')
    parser.add_argument('--save-path', default='../checkpoint/debug')
    parser.add_argument('--train_batch_size', type=int, default=64,
                            help='training batch size')
    parser.add_argument('--gpu', default='0, 1, 2, 3')
    parser.add_argument('--network', type=str, default='ResNet',
                            help='choose which embedding network to use. ProtoNet, R2D2, ResNet, MAML')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which dataset to use. Currently, only miniImageNet and CIFAR_FS are supported)
    parser.add_argument('--lr', type=float, default=0.01,
                            help='initial learning rate')
    parser.add_argument('--lr_schedule', nargs='+', default=[100, 150, 200, 250], type=int, help='when to decrease lr')
    parser.add_argument('--feature_clustering_coeff', type=float, default=1.0,
                            help='coefficient for feature clustering regularizer')
    parser.add_argument('--hyperplane_variation_coeff', type=float, default=0.0,
                            help='coefficient for hyperplane variation regularizer')

    opt = parser.parse_args()

    (dataset_train, data_loader) = get_dataset(opt)
    classes, label_dict = label_converter(opt, dataset_train)
    pair_dataset_train = PairData(dataset_train, classes)
    dloader_train = data_loader(
        dataset=pair_dataset_train,
        batch_size=opt.train_batch_size//4,
        num_workers=4,
        shuffle=True
    )

    set_gpu(opt.gpu)
    check_dir('../checkpoint/')
    check_dir(opt.save_path)
    
    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model(opt)


    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()}, 
                                     {'params': cls_head.parameters()}], lr=opt.lr, momentum=0.9, \
                                              weight_decay=5e-4, nesterov=True)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_schedule, gamma=0.1)

    criterion = torch.nn.CrossEntropyLoss()

    max_train_acc = 0.0

    timer = Timer()

    for epoch in range(1, opt.num_epoch + 1):
        embedding_net.train()
        cls_head.train()

        pair_dataset_train.shuffle_maps()
  
        if epoch <= opt.num_warmup:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.01

        # Fetch the current epoch's learning rate
        #epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']
            
        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
                            epoch, epoch_learning_rate))
        
        _, _ = [x.train() for x in (embedding_net, cls_head)]
        
        train_accuracies = []
        train_losses = []

        for i, (im1, im2, im3, im4, lab1, lab2, lab3, lab4) in enumerate(dloader_train):
            images = torch.cat([im1,im2,im3,im4], 0).cuda()
            labels = torch.cat([lab1,lab2,lab3,lab4], 0)
            for j in range(labels.shape[0]):
                labels[j]=label_dict[labels[j].item()]
            labels=labels.cuda()

            outputs = embedding_net(images)
            logit_query = cls_head(outputs)
            loss = criterion(logit_query, labels)

            if opt.feature_clustering_coeff > 0.0:
                loss = loss + opt.feature_clustering_coeff * feature_clustering(outputs)

            if opt.hyperplane_variation_coeff > 0.0:
                loss = loss + opt.hyperplane_variation_coeff * hyperplane_variation(outputs)

            acc = count_accuracy(logit_query, labels)
            
            train_accuracies.append(acc.item())
            train_losses.append(loss.item())

            if (i % 100 == 0):
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(
                            epoch, i, len(dloader_train), loss.item(), train_acc_avg, acc))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        train_acc_avg = np.mean(np.array(train_accuracies))
        train_loss_avg = np.mean(np.array(train_losses))

        if train_acc_avg > max_train_acc:
            max_train_acc = train_acc_avg
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()},\
                       os.path.join(opt.save_path, 'best_model.pth'))
            log(log_file_path, 'Train Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} % (Best)'\
                  .format(epoch, train_loss_avg, train_acc_avg))
        else:
            log(log_file_path, 'Train Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} %'\
                  .format(epoch, train_loss_avg, train_acc_avg))

        torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                   , os.path.join(opt.save_path, 'last_epoch.pth'))

        if epoch % opt.save_epoch == 0:
            torch.save({'embedding': embedding_net.state_dict(), 'head': cls_head.state_dict()}\
                       , os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch)))

        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))
