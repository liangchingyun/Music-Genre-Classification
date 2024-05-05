#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import cv2
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score

dd=os.listdir('TIN')
dd.remove('.DS_Store')
f1 = open('train.txt', 'w')
f2 = open('test.txt', 'w')
for i in range(len(dd)):
    d2 = os.listdir('TIN/%s/images/'%(dd[i]))
    for j in range(len(d2)-2):
        str1='TIN/%s/images/%s'%(dd[i], d2[j])
        f1.write("%s %d\n" % (str1, i))
    str1='TIN/%s/images/%s'%(dd[i], d2[-1])
    f2.write("%s %d\n" % (str1, i))

f1.close()
f2.close()

# Feature Extraction Methods
def Color_Histogram(gray):
    return cv2.calcHist([gray], [0], None, [255], [1, 256])

def SIFT(gray):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des

def HOG(gray):
    winSize = gray.shape
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    return hog.compute(gray).flatten()

#Classfication Models
def KNN(x, y, tx, ty):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x,y)
    pred = knn.predict(tx)
    print('\n\nKNN\n')
    print('F1 Score: ', f1_score(ty, pred, average='weighted'))
    print('KNN Accuracy: ', accuracy_score(ty, pred))

def SVM(x, y, tx, ty):
    clf = svm.SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    clf.fit(x, y)
    pred = clf.predict(tx)
    print('\n\nSVM\n')
    print('F1 Score: ', f1_score(ty, pred, average='weighted'))
    print('KNN Accuracy: ', accuracy_score(ty, pred))

def Random_Forest(x, y, tx, ty):
    clf = RandomForestClassifier()
    clf.fit(x, y)
    pred = clf.predict(tx)
    print('\n\nRandom Forest\n')
    print('F1 Score: ', f1_score(ty, pred, average='weighted'))
    print('KNN Accuracy: ', accuracy_score(ty, pred))
    

# Load Image
def load_img(f, method):
    t = 0
    f=open(f)
    lines=f.readlines()
    imgs, lab=[], []
    
    all_des = []  # for SIFT

    if method == SIFT:
        for i in range(len(lines)):
            fn, label = lines[i].split(" ")
            img = cv2.imread(fn)
            img_resized = cv2.resize(img, (32,32))
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            _, des = SIFT(img_gray)
            if des is not None:
                all_des.append(des)
            t += 1
            print('\r' + '[Progress]:|%s%s|%.2f%%;' % ('█' * int(t * 20 / len(lines)), ' ' * (20 - int(t * 20 / len(lines))), float(t / len(lines) * 100)), end='')
        # Build visual vocabulary
        des_array = np.vstack(all_des)
        kmeans = KMeans(n_clusters=100, random_state=42)
        kmeans.fit(des_array)
        
        t = 0

    for i in range(len(lines)):
        fn, label = lines[i].split(' ')
        
        img = cv2.imread(fn)
        img_resized = cv2.resize(img, (32,32)) #調整大小32*32
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) #轉為灰階
        
        '''===============================
        影像處理的技巧可以放這邊，來增強影像的品質
        
        ==============================='''
        
        if method == SIFT:
            kp, des = method(img_gray)
            if des is not None:
                kmeans.predict(des)
                im1 = np.bincount(kmeans.labels_, minlength=kmeans.n_clusters)

        else:
            im1 = method(img_gray)

        if im1 is not None:  
            vec = np.reshape(im1, [-1]) #展平為一維向量
            imgs.append(vec) 
            lab.append(int(label))

        t += 1
        print('\r' + '[Progress]:|%s%s|%.2f%%;' % ('█' * int(t * 20 / len(lines)), ' ' * (20 - int(t * 20 / len(lines))), float(t / len(lines) * 100)), end='')

    imgs= np.asarray(imgs, np.float32)
    lab= np.asarray(lab, np.int32)
    return imgs, lab 

start = time.time()

x, y = load_img('train.txt', SIFT)
tx, ty = load_img('test.txt', SIFT)

KNN(x, y, tx, ty)
#SVM(x, y, tx, ty)
#Random_Forest(x, y, tx, ty)

end = time.time()
print('Time: %f sec' %(end - start))
#======================================
#X就是資料，Y是Label，請設計不同分類器來得到最高效能
#必須要計算出分類的正確率
#======================================





