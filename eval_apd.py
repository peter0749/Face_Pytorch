#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import pickle
import os
import json
import torch.utils.data
from backbone import mobilefacenet, resnet, arcfacenet, cbam
from dataset.apd import APD
import torchvision.transforms as transforms
from torch.nn import DataParallel
import argparse

def evaluation(positive_pair, negative_pair, feature_path='./result/cur_epoch_result.pkl', threshold=0.0):
    with open(feature_path, 'rb') as fp:
        result = pickle.load(fp)
    basename2feature = result['basename2feature']
    features = result['features']

    # normalize features
    mu = np.mean(features, axis=0, keepdims=True)
    features = features - mu
    features = features / np.linalg.norm(features, axis=1, ord=2, keepdims=True)

    scores = []
    labels = []
    with open(positive_pair, 'r') as fp:
        '''
        Example:
        xxx\t90\t34
        '''
        for line in fp:
            id_, im1, im2 = line.split('\t')
            id_ = id_.strip()
            im1 = int(im1)
            im2 = int(im2)
            id1 = '%s_%d'%(id_,im1)
            id2 = '%s_%d'%(id_,im2)
            feat1 = features[basename2feature[id1]]
            feat2 = features[basename2feature[id2]]
            score = (np.sum(feat1 * feat2) + 1.0) / 2.0
            scores.append(score)
            labels.append(1)
    with open(negative_pair, 'r') as fp:
        '''
        Example:
        aaa\t90\tbbb\t34
        '''
        for line in fp:
            id1, im1, id2, im2 = line.split('\t')
            id1 = id1.strip()
            id2 = id2.strip()
            im1 = int(im1)
            im2 = int(im2)
            id1 = '%s_%d'%(id1,im1)
            id2 = '%s_%d'%(id2,im2)
            feat1 = features[basename2feature[id1]]
            feat2 = features[basename2feature[id2]]
            score = (np.sum(feat1 * feat2) + 1.0) / 2.0
            scores.append(score)
            labels.append(0)
    fpr, tpr, _ = roc_curve(labels, scores)
    score = auc(fpr, tpr)
    return score, fpr, tpr


def loadModel(data_root, backbone_net, gpus='0', resume=None):

    if backbone_net == 'MobileFace':
        net = mobilefacenet.MobileFaceNet()
    elif backbone_net == 'CBAM_50':
        net = cbam.CBAMResNet(50, feature_dim=args.feature_dim, mode='ir')
    elif backbone_net == 'CBAM_50_SE':
        net = cbam.CBAMResNet(50, feature_dim=args.feature_dim, mode='ir_se')
    elif backbone_net == 'CBAM_100':
        net = cbam.CBAMResNet(100, feature_dim=args.feature_dim, mode='ir')
    elif backbone_net == 'CBAM_100_SE':
        net = cbam.CBAMResNet(100, feature_dim=args.feature_dim, mode='ir_se')
    else:
        print(backbone_net, ' is not available!')

    # gpu init
    multi_gpus = False
    if len(gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if (not resume is None) and os.path.exists(resume):
        net.load_state_dict(torch.load(resume)['net_state_dict'])

    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    apd_dataset = APD(data_root, transform=transform)
    apd_loader = torch.utils.data.DataLoader(apd_dataset, batch_size=128,
                                             shuffle=False, num_workers=2, drop_last=False)

    return net.eval(), device, apd_dataset, apd_loader

def getFeatureFromTorch(feature_save_path, net, device, data_set, data_loader, verbose=False):
    features = None
    basename2feature = {}
    if verbose:
        print("Generating feature vectors...")
        pbar = tqdm(total=len(data_loader))
    for img, img_flip, basenames in data_loader:
        img = img.to(device)
        img_flip = img_flip.to(device)
        with torch.no_grad():
            res = [net(img).data.cpu().numpy(), net(img_flip).data.cpu().numpy()]
        feature = np.concatenate((res[0], res[1]), 1) # batch_size, 1024
        L = 0 if features is None else len(features)
        for n, basename in enumerate(basenames):
            assert not basename in basename2feature
            basename2feature[basename] = L + n
        features = feature if features is None else np.append(features, feature, axis=0)
        if verbose:
            pbar.update(1)
    if verbose:
        pbar.close()
        print('Finishing generating feature vectors...')

    result = {'basename2feature': basename2feature, 'features': features}
    with open(feature_save_path, 'wb') as fp:
        pickle.dump(result, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--root', type=str, required=True, help='The path of apd data')
    parser.add_argument('--positive_pair', type=str, required=True, help='The path of apd positive pair')
    parser.add_argument('--negative_pair', type=str, required=True, help='The path of apd negative pair')
    parser.add_argument('--backbone_net', type=str, default='CBAM_100_SE', help='MobileFace, CBAM_50, CBAM_50_SE, CBAM_100, CBAM_100_SE')
    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension')
    parser.add_argument('--resume', type=str, default='./model/SERES100_SERES100_IR_20190528_132635/Iter_342000_net.ckpt',
                        help='The path pf save model')
    parser.add_argument('--feature_save_path', type=str, default='./result/cur_epoch_apd_result.pkl',
                        help='The path of the extract features save, must be .mat file')
    parser.add_argument('--gpus', type=str, default='', help='gpu list')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    net, device, apd_dataset, apd_loader = loadModel(args.root, args.backbone_net, args.gpus, args.resume)
    getFeatureFromTorch(args.feature_save_path, net, device, apd_dataset, apd_loader, verbose=True)
    score, fpr, tpr = evaluation(args.positive_pair, args.negative_pair, args.feature_save_path)
    print('AUC_ROC    {:.4f}'.format(score * 100))

    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                         lw=lw, label='ROC curve (area = %0.2f)' % score)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve (APD)')
        plt.legend(loc="lower right")
        plt.show()

