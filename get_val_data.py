from random import shuffle
import numpy as np
import scipy.io as io




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

for dataset in ['indian','houston','ksc']:#,'botswana'
        for size in [5,7,9,11,13,15]:
            percent=0.05
            name = "./" + dataset + "_val_" + str(size) + ".mat"
            if dataset=='indian':
                input_mat = io.loadmat("./Indian_pines_corrected.mat")['indian_pines_corrected']
                target_mat = io.loadmat("./Indian_pines_gt.mat")['indian_pines_gt']
                class_num=16
            if dataset == 'houston':
                input_mat = io.loadmat("./Houston.mat")['Houston']
                target_mat = io.loadmat("./Houston_gt.mat")['Houston_gt']
                class_num = 15
            if dataset == 'ksc':
                input_mat = io.loadmat("./KSC.mat")['KSC']
                target_mat = io.loadmat("./KSC_gt.mat")['KSC_gt']
                class_num = 13
            if dataset == 'botswana':
                input_mat = io.loadmat("./Botswana.mat")['Botswana']
                target_mat = io.loadmat("./Botswana_gt.mat")['Botswana_gt']
                class_num = 14
            input_mat_augement = get_data_augement(input_mat)
            HEIGHT = input_mat.shape[0]
            WIDTH = input_mat.shape[1]
            BAND = input_mat.shape[2]
            t=size//2
            patches=[[] for i  in range(class_num)]
            for i in range(HEIGHT):
                for j in range(WIDTH):
                     centre_target=target_mat[i,j]
                     if centre_target!=0:
                         patches[centre_target-1].append((i,j))
            patches_train=[]
            labels_train=[]
            counter=0
            for data in patches:
                shuffle(data)
                num = 0
                train_num=max(1,int(len(data)*percent))
                for item in data:
                    i,j=item
                    centre_mat = input_mat_augement[i - t + HEIGHT:i + t + 1 + HEIGHT, j - t + WIDTH:j + t + 1 + WIDTH].copy()
                    if num<train_num:
                        patches_train.append(centre_mat.copy())
                        labels_train.append(counter)
                        num+=1
                    else:
                        break
                counter += 1


            patches_train=np.array(patches_train,dtype=float).transpose(0,3,1,2)
            print(patches_train.shape)
            dict={}
            dict['patches_val']=patches_train
            dict['labels_val']=labels_train
            io.savemat(name, dict)

