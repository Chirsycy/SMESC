from experiment.util import *
from model import *
import pandas as pd

def get_class():

    if dataset == 'houston':
        if model_name == 'SCCRNet':
            model = SMESC(start_channel=144).cuda()
    if dataset == 'indian':
        if model_name == 'SCCRNet':
            model = SMESC(start_channel=200).cuda()

    return model


def  test():
    model = get_class()
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    test_loader = get_test_data(path_test=path_test,batch_size_test=64)
    classification, confusion, results = reports(test_loader,model)
    print('==============' + model_name + '==============')
    print('==============' + model_name + '==============')
    print('==============' + model_name + '==============')
    print(classification)
    print(dataset, results)
    print('==============' + model_name + '==============')
    print('==============' + model_name + '==============')
    print('==============' + model_name + '==============')
    return results

def train():
    model = get_class()
    torch.backends.cudnn.benchmark = True
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    best_acc=-1
    for total in range(3):
        for path_train,path_val in train_sequence:
            train_loader, val_loader= get_train_and_val_data(path_train=path_train,path_val=path_val)
            for epoch in range(0, 100):
                adjust_learning_rate(optimizer, epoch, lr)
                train_loss, train_acc = train_and_acc(train_loader, model, loss_criterion, optimizer)
                test_loss, test_acc = test_and_acc(val_loader, model, loss_criterion)
                print('第',total+1,'轮次','训练的尺度是'+path_train,"第", epoch+1,'epoch', "训练损失是", train_loss, "训练精度是", train_acc, end=',')
                print("验证集损失是", test_loss, "验证集精度是", test_acc)
                print('==============' + model_name + '==============')
                print('==============')
                print('==============')
                if test_acc > best_acc:
                    torch.save(model.state_dict(), model_path)
                    best_acc = test_acc
            model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)


if __name__ == '__main__':

    DEVICE=torch.device('cuda:0')

    for model_name in ['SMESC']:

        for dataset  in [ 'houston','indian']:#
            path_test='../'+dataset+'_test_15.mat'
            model_path ='./'+dataset+'_'+model_name+'.pth'
            train_sequence = []
            for size in [5,7,9,11,13,15]:
                train_sequence.append(('../'+dataset+'_train_size_'+str(size)+'.mat','../'+dataset+'_val_'+str(size)+'.mat'))
            data_excel = {}
            for cishu in range(5):
                lr = 0.0001
                train()
                result=test()
                data_excel[str(cishu)] = result
            data_excel =pd.DataFrame(data_excel)
            data_excel.to_excel(dataset + '数据集' + model_name + '模型.xlsx', index=False)






