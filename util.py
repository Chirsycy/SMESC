
import numpy as np
import torch
from torch.utils.data import DataLoader
from operator import truediv
import scipy.io as scio
import tifffile as tf
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score


def get_data_augement(a):
    left=a.shape[0]
    right=a.shape[1]
    band=a.shape[2]
    c=np.zeros((3*left,3*right,band))
    c[:left,:right]=a
    c[left:left*2,:right]=a
    c[left*2:left*3,:right]=a

    c[:left,right:right*2]=a
    c[left:left*2,right:right*2]=a
    c[left*2:left*3,right:right*2]=a

    c[:left,right*2:right*3]=a
    c[left:left*2,right*2:right*3]=a
    c[left*2:left*3,right*2:right*3]=a


    return c



class HSI_dataset(torch.utils.data.Dataset):

    def __init__(self,path_data,hou_zhui='mat',path_label=None):
        print(path_data)

        if 'train' in path_data:
            key_data='patches_train'
            key_label = 'labels_train'

        if 'val' in path_data:
            key_data = 'patches_val'
            key_label = 'labels_val'

        if 'test' in path_data:
            key_data = 'patches_test'
            key_label = 'labels_test'

        if 'pretrain' in  path_data:
            key_data = 'patches_pretrain'
            key_label = 'labels_pretrain'


        if 'mat'==hou_zhui:
            patches = scio.loadmat(path_data)[key_data]
            self.train_data = torch.from_numpy(patches).float()
            self.len = patches.shape[0]
            self.labels=None
            labels_numpy = scio.loadmat(path_data)[key_label]
            self.labels = torch.from_numpy(labels_numpy).reshape(-1).long()

        if 'tif' ==hou_zhui:
            input_mat = tf.imread(path_data).transpose([1,2,0])
            self.patches_index=scio.loadmat(path_label)['patches_test']
            labels_numpy = scio.loadmat(path_label)['labels_test'].reshape(-1)
            self.HEIGHT=input_mat.shape[0]
            self.WIDTH=input_mat.shape[1]
            input_mat = get_data_augement(input_mat).transpose([2,0,1])

            self.train_data = torch.from_numpy(input_mat).float()
            self.len = len(labels_numpy)
            self.labels = torch.from_numpy(labels_numpy).long()

        self.houzhui=hou_zhui





    def __getitem__(self, index):

          if self.houzhui=='tif':
              i,j=self.patches_index[index]
              a= self.train_data[:,i-7+self.HEIGHT:i+8+self.HEIGHT,j-7+self.WIDTH:j+8+self.WIDTH]
              b=self.labels[index]
              return   a,b

          else:
                if self.labels==None:
                    return self.train_data[index]
                else:
                    return self.train_data[index], self.labels[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len




def reports(test_loader,model):
    model.eval()
    count=0
    for batch, (data, target) in enumerate(test_loader):

        data = data.cuda()
        pred= model(data)
        outputs = pred
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        target_numpy=target.detach().cpu().numpy()
        if count == 0:
            y_pred_test = outputs
            y_test=target_numpy
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, target_numpy))
    print(y_test.shape)
    print(y_pred_test.shape)

    classification = classification_report(y_test, y_pred_test)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, confusion, list(np.round( np.array( list(each_acc)+[oa, aa, kappa]  ) * 100, 2))








def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc






def train_and_acc(trainloader, model, criterion, optimizer):
    with torch.autograd.set_detect_anomaly(True):
        model.train()
        accs   = np.ones((len(trainloader))) * -1000.0
        losses = np.ones((len(trainloader))) * -1000.0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses[batch_idx] = loss.item()
            accs[batch_idx] = accuracy(outputs.data, targets.data)[0].item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return (np.average(losses), np.average(accs))


import torch.nn.functional as F






def train_and_acc_spa_spe_fre(trainloader, model, criterion, optimizer,loss_contrast):
    with torch.autograd.set_detect_anomaly(True):
        model.train()
        accs   = np.ones((len(trainloader))) * -1000.0
        losses = np.ones((len(trainloader))) * -1000.0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            outputs,spa_proj,fre_proj = model(inputs,True)
            loss_cre = criterion(outputs, targets)
            loss_con=loss_contrast(spa_proj,fre_proj)
            loss=loss_con+loss_cre
            losses[batch_idx] = loss.item()
            accs[batch_idx] = accuracy(outputs.data, targets.data)[0].item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return (np.average(losses), np.average(accs))




def test_and_acc(testloader, model, criterion):
    model.eval()
    accs   = np.ones((len(testloader))) * -1000.0
    losses = np.ones((len(testloader))) * -1000.0
    with torch.no_grad():
        for batch_idx, (inputs,targets) in enumerate(testloader):

            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
            outputs = model(inputs)
            losses[batch_idx] = criterion(outputs, targets).item()
            accs[batch_idx] = accuracy(outputs.data, targets.data, topk=(1,))[0].item()
    return (np.average(losses), np.average(accs))




def get_test_data(path_test,batch_size_test = 16,label_path=None,hou_zhui='mat', drop_last=False):

            test_dataset = HSI_dataset(path_test,hou_zhui=hou_zhui,path_label=label_path)
            test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, drop_last=drop_last, num_workers=0)
            return test_loader


def get_pretrain_data(path_pretrain, batch_size_pretrain=16):
    pretrain_dataset = HSI_dataset(path_data=path_pretrain)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size_pretrain, shuffle=True, drop_last=True, num_workers=0)

    return pretrain_loader



def get_train_and_val_data(path_train,path_val,batch_size_train=16,batch_size_val=16):


        train_dataset = HSI_dataset(path_data=path_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, drop_last=True, num_workers=0)

        val_dataset = HSI_dataset(path_data=path_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, drop_last=True, num_workers=0)

        return train_loader,val_loader






def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


