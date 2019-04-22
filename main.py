#Written by Saurav Rai
#Implementation of SE-CNN
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pdb
import os
from torch.utils.data import DataLoader 

from sklearn import metrics
import numpy as np

from light_cnn import LightCNN_4Layers, LightCNN_9Layers
from aifrmodel import aifrNet
from arcface import Arcface
from LFWDataset import LFWDataset
from eval_metrics import evaluate, myevaluate
from tqdm import tqdm
from PIL import *

from torch.autograd import Function
from agedata import AgeFaceDataset


def main():

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")


    parser = argparse.ArgumentParser(description='PyTorch AIFR')
    parser.add_argument('--batch_size', type=int, default = 64 , metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default = 2, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--iters', type=int, default = 2000, metavar='N',
                        help='number of iterations to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    #parser.add_argument('--no-cuda', action='store_true', default=False,
    #                    help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--pretrained', default = True, type = bool,
                    metavar='N', help='use pretrained ligthcnn model:True / False no pretrainedmodel )')

    parser.add_argument('--basemodel', default='LightCNN-4', type=str, metavar='BaseModel',
                    help='model type:ContrastiveCNN-4 LightCNN-4 LightCNN-9, LightCNN-29, LightCNN-29v2')

    parser.add_argument('--feature_size', default = 128, type=int, metavar='N',
                    help='ifeature size is 128 for lightcnn model )')

    parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')

    parser.add_argument('--resume', default=False, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

    parser.add_argument('--start-epoch', default = 0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    #Training dataset on Morph

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
    parser.add_argument('--root_path', default='/home/titanx/DB/Morph_preprocess/Morph_aligned/Morph/', type=str, metavar='PATH',
                    help='path to root path of images (default: none)')

    parser.add_argument('--num_classes', default=1000, type=int,
                    metavar='N', help='number of classes (default: 10000)')


    args = parser.parse_args()

    
    if args.basemodel == 'LightCNN-4':
        basemodel = LightCNN_4Layers(num_classes=args.num_classes)
        print('4 layer Lightcnn model')
    elif args.basemodel == 'LightCNN-9':
        basemodel = LightCNN_9Layers(num_classes=args.num_classes)
        print('9 layer Lightcnn model')
    else:
        print('Model not found so existing.')
        assert(False)
    


    basemodel = nn.DataParallel(basemodel).to(device)
 
    params1 = []

    for name, param in basemodel.named_parameters():

        if 'fc2' not in name:
            param.requires_grad = False
        else:
            params1.append(param)


    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.pretrained is True:
        #pre_trained_dict = torch.load('./LightCNN_9Layers_checkpoint.pth.tar', map_location = lambda storage, loc: storage) #by DG
        pre_trained_dict = torch.load('./LightenedCNN_4_torch.pth', map_location = lambda storage, loc: storage) #by DG

        #pre_trained_dict = pre_trained_dict['state_dict'] #THIS LINE IS USED EXCEPT FOR LIGHTCNN 4 MODEL
        model_dict = basemodel.state_dict()
        
        # THIS ONE IS FOR CHANGING THE NAME IN THE MODEL:
        # IF WE ARE USING CUDA THEN WE NEED TO MENTION ( module ) LIKE HOW WE HAVE DONE BELOW         
        '''THIS FOLLOWING LINES ARE USED ONLY FOR LIGHTCNN 4 MODEL'''

        pre_trained_dict['module.features.0.filter.weight'] = pre_trained_dict.pop('0.weight')
        pre_trained_dict['module.features.0.filter.bias'] = pre_trained_dict.pop('0.bias')
        pre_trained_dict['module.features.2.filter.weight'] = pre_trained_dict.pop('2.weight')
        pre_trained_dict['module.features.2.filter.bias'] = pre_trained_dict.pop('2.bias')
        pre_trained_dict['module.features.4.filter.weight'] = pre_trained_dict.pop('4.weight')
        pre_trained_dict['module.features.4.filter.bias'] = pre_trained_dict.pop('4.bias')
        pre_trained_dict['module.features.6.filter.weight'] = pre_trained_dict.pop('6.weight')
        pre_trained_dict['module.features.6.filter.bias'] = pre_trained_dict.pop('6.bias')
        pre_trained_dict['module.fc1.filter.weight'] = pre_trained_dict.pop('9.1.weight')
        pre_trained_dict['module.fc1.filter.bias'] = pre_trained_dict.pop('9.1.bias')
        pre_trained_dict['module.fc2.weight'] = pre_trained_dict.pop('12.1.weight')
        pre_trained_dict['module.fc2.bias'] = pre_trained_dict.pop('12.1.bias')
       
       # 1. filter out unnecessary keys  
        pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if ("fc2" not in k)}
       # 2. overwrite entries in the existing state dict
        model_dict.update(pre_trained_dict)
       # 3. load the new state dict  
        basemodel.load_state_dict(model_dict, strict = False)
    


    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_transform = transforms.Compose([transforms.Resize(144), transforms.RandomCrop(128), transforms.ToTensor()])#,    
                                           #transforms.Normalize(mean = [0.5224], std = [0.1989])])

    valid_transform = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])#,

    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.1, patience = 5) 

    train_loader = DataLoader(AgeFaceDataset(transform = train_transform, istrain = True, isvalid = False),
                    batch_size = args.batch_size, shuffle = True,
                    num_workers =args.workers, pin_memory = False)


    valid_loader = DataLoader(AgeFaceDataset(transform = valid_transform, istrain = False, isvalid = True),
                    batch_size = args.batch_size, shuffle = False,
                    num_workers =args.workers, pin_memory = False)

    test_loader = DataLoader(AgeFaceDataset(transform = valid_transform,  istrain = False, isvalid = False),
                    batch_size = args.batch_size, shuffle = False,
                    num_workers = args.workers, pin_memory = False)

    trainprobe_loader = DataLoader(AgeFaceDataset(transform = valid_transform, istrain = True, isvalid = True),
                    batch_size = args.batch_size, shuffle = False,
                    num_workers = args.workers, pin_memory = False)   #for train accuracy

    basemodel = basemodel.to(device)

    aifrmodel = aifrNet(channels = 686).to(device)  #channels is embedding from lightcnn base model
    arcface = Arcface(embedding_size= 128, classnum=10000).to(device)

    params =    list(aifrmodel.parameters())  #+ list(arcface.parameters())

    optimizer = optim.SGD(params , lr=args.lr, momentum=args.momentum)

    optimizer = optim.Adam(params , lr=args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['iterno']
            genmodel.load_state_dict(checkpoint['state_dict1'])
            basemodel.load_state_dict(checkpoint['state_dict2'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Test acc at checkpoint was:',checkpoint['testacc'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    criterion1   = nn.CrossEntropyLoss().to(device)         #for  Identification loss
    criterion2   = nn.CrossEntropyLoss().to(device)         #for  Identification loss
    criterion3   = nn.CrossEntropyLoss().to(device)         #for  Identification loss
    #criterion1   = nn.MSELoss().to(device)         #for Age regression loss

    print('Device being used is :' + str(device))

    for iterno in range(args.start_epoch , 200):

        accuracy = test(test_loader, basemodel, aifrmodel,device)
        print('test accuracy is :', accuracy)
        #adjust_learning_rate(optimizer, iterno)

        #print('args iters',args.iters)

        train(args, basemodel, aifrmodel, arcface, device, train_loader, optimizer, criterion1, criterion2, criterion3,iterno)
      
        #TEST ACCURACY
        accuracy = test(test_loader, basemodel, aifrmodel,device)
        print('test accuracy is :', accuracy)
        f = open('Morph_performance_with_resentlike_lightcnn4_senet_acc_3resblocks_correct','a')
        f.write('\n'+str(iterno)+'\t'+str( accuracy) );
        f.close()

  
def  train(args, basemodel, aifrmodel, arcface, device, train_loader, optimizer, criterion1,criterion2,criterion3, iteration):

    for batch_idx ,(data,agelabel, identitylabel) in enumerate(train_loader):
        data, agelabel, identitylabel = (data).to(device),  torch.from_numpy(np.asarray(agelabel)).to(device), torch.from_numpy(np.asarray(identitylabel)).to(device)
        
        #agelabel = agelabel.float().unsqueeze(1)

        optimizer.zero_grad()

        out, feature = basemodel(data)
        age, identity_feat, embeddings = aifrmodel(out)

        #thetas = arcface(embeddings,identitylabel) 
        #loss1 = criterion1(age, agelabel)  #mse  loss

        loss1 = criterion1(age, agelabel)  #cross entropy loss
        loss2 = criterion2(identity_feat ,identitylabel) #cross entropy loss
        loss3 = criterion3(embeddings,identitylabel) #cross entropy loss for identity using thetas 

        #print('Losses 1 & 2 are :',loss1.item(), loss2.item())
        #print('Losses 1 is :',loss1.item())
        #loss  = loss1 + loss2 + loss3
        loss  = loss3

        loss.backward()
        #pdb.set_trace() 
        '''
        nn.utils.clip_grad_value_( list(genmodel.parameters()) + list( reg_model.parameters()) + list(idreg_model.parameters()), clip_value = 5)
        '''
        optimizer.step()

        print('Train iter: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} {:.4f} {:.4f} {:.4f}'.format(
             iteration, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item(),loss1.item(),loss2.item(),loss3.item()))
        
        f = open('Morph_performance_with_resentlike__structure_lightcnn4_senet_loss_3resblocks_correct','a')
        f.write('\n'+str(iteration) + '\t' + str(loss.item()));
        f.close()

def test(test_loader,basemodel,aifrmodel,device):
    acc = 0
    target =[]
    galleryfeatures = []
    probefeatures = []
    basemodel.train(False)

    with torch.no_grad():
        for (x1,age_yng,x2,age_old,label) in test_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)

            #num = np.array(list(age_yng),dtype=int)
            label = torch.tensor(label, dtype = torch.long)
            label = label.to(device)

            '''CHANGES MADE BY SAURAV 20 AUG'''
            '''
            _,iden_feat1 = basemodel(x1)
            _,iden_feat2 = basemodel(x2)

            '''
            out1, feature1 = basemodel(x1)
            out2 , feature2 = basemodel(x2)

            age_feat, iden_feat1 , embedding_feat1 = aifrmodel(out1)
            age_feat, iden_feat2 , embedding_feat2 = aifrmodel(out2)
           
            #print('The len(iden_feat1',len(iden_feat1))
            #print('The target value length:',len(label))

            #for j in range(len(embedding_feat1)):
            for j in range(len(iden_feat1)):
                #galleryfeatures.append(embedding_feat1[j].cpu().numpy())
                galleryfeatures.append(iden_feat1[j].cpu().numpy())

                #probefeatures.append(embedding_feat2[j].cpu().numpy())
                probefeatures.append(iden_feat2[j].cpu().numpy())

                target.append(label[j].cpu().numpy())

        total = len(probefeatures)

        probe = np.array(probefeatures)
        gallery = np.array(galleryfeatures)
        #print('probe ',probe)
        #print('gallery',gallery)
         
        
        dist  = metrics.pairwise.cosine_similarity(probe, gallery)
        output = np.argmax(dist, axis = 1)
        
        target = np.array(target)
        correct =0
        for i in range(len(probefeatures)):
            #print('target[output[i]] , target[i]',target[output[i]],target[i])
            if(target[output[i]] == target[i]):
                correct +=1
        print('correct and total',correct,total)
        acc = correct * 100.0 / total

        return acc



def save_checkpoint(state, filename):
    torch.save(state, filename)



if __name__ == '__main__':
    main()

