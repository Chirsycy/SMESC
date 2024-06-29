
from random import shuffle


import numpy as np
import scipy.io as io
from sklearn.decomposition import PCA


# from tqdm import tqdm
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


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


for dataset in ['indian', 'houston', 'ksc', 'botswana']:
    for size in [5, 7, 9, 11, 13, 15]:
            name= "./"+dataset+"_test_" + str(size) + ".mat"
            if dataset == 'indian':
                input_mat = io.loadmat("../dataset/Indian_pines_corrected.mat")['indian_pines_corrected']
                target_mat = io.loadmat("../dataset/Indian_pines_gt.mat")['indian_pines_gt']
                class_num = 16
            if dataset == 'houston':
                input_mat = io.loadmat("../dataset/Houston.mat")['Houston']
                target_mat = io.loadmat("../dataset/Houston_gt.mat")['Houston_gt']
                class_num = 15
            if dataset == 'ksc':
                input_mat = io.loadmat("../dataset/KSC.mat")['KSC']
                target_mat = io.loadmat("../dataset/KSC_gt.mat")['KSC_gt']
                class_num = 13
            if dataset == 'botswana':
                input_mat = io.loadmat("../dataset/Botswana.mat")['Botswana']
                target_mat = io.loadmat("../dataset/Botswana_gt.mat")['Botswana_gt']
                class_num = 14
            input_mat_augement = get_data_augement(input_mat)
            HEIGHT = input_mat.shape[0]
            WIDTH = input_mat.shape[1]
            BAND = input_mat.shape[2]
            t=size//2
            patches=[[] for i  in range(class_num)]
            patches_test,labels_test,index_map=[],[],[]
            for i in range(HEIGHT):
                for j in range(WIDTH):
                     centre_target=target_mat[i,j]
                     if centre_target!=0:
                         centre_mat=input_mat_augement[i-t+HEIGHT:i+t+1+HEIGHT,j-t+WIDTH:j+t+1+WIDTH].copy()
                         patches[centre_target-1].append(centre_mat.copy())
            counter=0
            for i, data in enumerate(patches):
                shuffle(data)
                patches_test += data
                labels_test += [counter] * len(data)
                counter += 1
            counter=0
            patches_test =np.transpose(patches_test,[0,3,1,2])
            index_map=np.array(index_map,dtype=int)
            print(index_map.shape)
            print(patches_test.shape)
            dict={}
            dict['patches_test']=patches_test
            dict['labels_test']=labels_test

            io.savemat(name, dict)

#rot data
# for dataset in ['indian', 'houston', 'ksc']:#, 'botswana'
#     for size in [15]:
#             name= "./"+dataset+"_test_" + str(size) + ".mat"
#             if dataset == 'indian':
#                 input_mat = io.loadmat("./Indian_pines_corrected.mat")['indian_pines_corrected']
#                 target_mat = io.loadmat("./Indian_pines_gt.mat")['indian_pines_gt']
#                 class_num = 16
#             if dataset == 'houston':
#                 input_mat = io.loadmat("./Houston.mat")['Houston']
#                 target_mat = io.loadmat("./Houston_gt.mat")['Houston_gt']
#                 class_num = 15
#             if dataset == 'ksc':
#                 input_mat = io.loadmat("./KSC.mat")['KSC']
#                 target_mat = io.loadmat("./KSC_gt.mat")['KSC_gt']
#                 class_num = 13
#             if dataset == 'botswana':
#                 input_mat = io.loadmat("./Botswana.mat")['Botswana']
#                 target_mat = io.loadmat("./Botswana_gt.mat")['Botswana_gt']
#                 class_num = 14
#             input_mat_augement = get_data_augement(input_mat)
#             HEIGHT = input_mat.shape[0]
#             WIDTH = input_mat.shape[1]
#             BAND = input_mat.shape[2]
#             t=size//2
#             patches=[[] for i  in range(class_num)]
#             patches_test,labels_test,index_map=[],[],[]
#             for i in range(HEIGHT):
#                 for j in range(WIDTH):
#                      centre_target=target_mat[i,j]
#                      if centre_target!=0:
#                          centre_mat=input_mat_augement[i-t+HEIGHT:i+t+1+HEIGHT,j-t+WIDTH:j+t+1+WIDTH].copy()
#                          patches[centre_target-1].append(centre_mat.copy())
#             counter=0
#             for i, data in enumerate(patches):
#                 shuffle(data)
#                 patches_test += data
#                 labels_test += [counter] * len(data)
#                 counter += 1
#             counter=0
#             patches_test =np.transpose(patches_test,[0,3,1,2])
#             index_map=np.array(index_map,dtype=int)
#             print(index_map.shape)
#             print(patches_test.shape)
#             dict={}
#             dict['patches_test']=patches_test
#             dict['labels_test']=labels_test
#
#             io.savemat(name, dict)
