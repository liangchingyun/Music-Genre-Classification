#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
from numpy import linalg as LA
from FeatureExtraction import HOG, ColorHistogram, SIFT, build_vocabulary, bow_encoding
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import time


def prepare_data():
    # train test split percentage -> 80% train, 20% test
    train_test_split = 0.8

    dd = os.listdir("TIN")
    f1 = open("train.txt", "w")
    f2 = open("test.txt", "w")
    print(f"start data spliting, train_test_split = {train_test_split}")
    for i in range(len(dd)):
        d2 = os.listdir("TIN/%s/images/" % (dd[i]))
        # training data
        for j in range(int(len(d2) * train_test_split)):  # 80% train
            str1 = "TIN/%s/images/%s" % (dd[i], d2[j])
            f1.write("%s %d\n" % (str1, i))  # write to train.txt
        # testing data
        for j in range(int(len(d2) * train_test_split), len(d2)):
            str1 = "TIN/%s/images/%s" % (dd[i], d2[-1])
            f2.write("%s %d\n" % (str1, i))  # write to test.txt
        # end of training and testing data for class i

    f1.close()
    f2.close()

    print("data spliting done")


# downsample image to 32x32 for faster processing
def resize_img(img, size=(32, 32)):
    img = cv2.imread(img)
    img = cv2.resize(img, size)
    return img


def load_img(f, method="HOG"):
    f = open(f)
    lines = f.readlines()
    imgs, lab = [], []
    all_descriptors = []  # for SIFT

    if method == "SIFT":
        print("start SIFT")
        for i in range(len(lines)):
            fn, label = lines[i].split(" ")
            im1 = resize_img(fn)
            _, descriptors = SIFT(im1)
            if descriptors is not None:
                all_descriptors.append(descriptors)
        print("SIFT done")

        # Build visual vocabulary
        visual_words = build_vocabulary(all_descriptors)

    for i in range(len(lines)):
        fn, label = lines[i].split(" ")
        im1 = resize_img(fn)

        """===============================
        影像處理的技巧可以放這邊，來增強影像的品質
        
        ==============================="""
        # feature extraction from FeatureExtraction.py
        if method == "HOG":
            im1 = HOG(im1)
        elif method == "ColorHistogram":
            im1 = ColorHistogram(im1)
        elif method == "SIFT":
            keypoints, im1 = SIFT(im1)
            if im1 is not None:
                im1 = bow_encoding(im1, visual_words)

        if im1 is not None:
            # print(im1.shape)
            im1 = im1.flatten()
            imgs.append(im1)
            lab.append(int(label))

    imgs = np.asarray(imgs, np.float32)
    lab = np.asarray(lab, np.int32)
    return imgs, lab


# ======================================
# X就是資料，Y是Label，請設計不同分類器來得到最高效能
# 必須要計算出分類的正確率
# ======================================


def KNN(x, y):
    print("start KNN")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x, y)
    print("KNN training done")
    return model


def SVM(x, y):
    print("start SVM")
    model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    model.fit(x, y)
    print("SVM training done")
    return model


def RandomForest(x, y):
    print("start Random Forest")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x, y)
    print("Random Forest training done")
    return model


def train(x, y, model="SVM"):
    if model == "KNN":
        model = KNN(x, y)
    elif model == "SVM":
        model = SVM(x, y)
    elif model == "RandomForest":
        model = RandomForest(x, y)
    return model


def test(x, y, model):
    y_pred = model.predict(x)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")
    classification = classification_report(y, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Classification Report: {classification}")
    return accuracy, f1, classification


# main
if __name__ == "__main__":

    # data preparation
    # prepare_data()

    methods = ["HOG", "ColorHistogram", "SIFT"]
    models = ["KNN", "SVM", "RandomForest"]

    for method in methods:
        method_start = time.time()  # start method
        print(f"Method: {method}")
        # load data
        x, y = load_img("train.txt", method=method)
        tx, ty = load_img("test.txt", method=method)
        # print(x.shape)    # (number of images, number of features)
        # print(y.shape)    # (number of labels of images)
        # print(tx.shape)
        # print(ty.shape)

        method_end = time.time()  # end method
        method_time = method_end - method_start  # method time

        for model_name in models:
            model_start = time.time()  # start model
            print(f"Model: {model_name}")
            # training
            model = train(x, y, model=model_name)
            # testing
            accuracy, f1, classification = test(tx, ty, model)
            model_end = time.time()  # end model
            model_time = model_end - model_start  # model time

            # write to log
            with open("log.txt", "a") as f:
                f.write(f"Method: {method}, Model: {model_name}\n")
                f.write(f"Method Time: {method_time}\n")
                f.write(f"Model Time: {model_time}\n")
                f.write(f"Total Time: {method_time + model_time} seconds\n")
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"F1 Score: {f1}\n")
                f.write(f"Classification Report: {classification}\n")
                f.write("\n")

    # load data
    # x, y = load_img("train.txt", method="SIFT")
    # tx, ty = load_img("test.txt", method="ColorHistogram")
    # print(x.shape)    # (number of images, number of features)
    # print(y.shape)    # (number of labels of images)
    # print(tx.shape)
    # print(ty.shape)

    # training
    # model = train(tx, ty, model="RandomForest")
    # testing
    # test(tx, ty, model)
