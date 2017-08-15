#!/usr/bin/env python2
#put training and inferring together

import time

start = time.time()

import math
import cv2
import os
import shutil
import pickle
import pandas as pd
import random
import numpy as np
np.set_printoptions(precision=2)

import openface
from openface.data import iterImgs


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB



fileDir = os.path.dirname(os.path.realpath(__file__))
regDir = os.path.join(fileDir, "registeredPeople")
modelDir = os.path.join(fileDir, "..","models")
lfwDir = os.path.join(fileDir, "..", "..", "lfw")
dlibModelDir = os.path.join(modelDir, "dlib")
openfaceModelDir = os.path.join(modelDir, "openface")
dataDir = os.path.join(fileDir, "data")
fold1path = os.path.join(dataDir, "fold1")
fold2path = os.path.join(dataDir, "fold2")
fold3path = os.path.join(dataDir, "fold3")
fold4path = os.path.join(dataDir, "fold4")


align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
net = openface.TorchNeuralNet(os.path.join(
    openfaceModelDir,
    'nn4.small2.v1.t7'), imgDim=96, cuda=False)

def generate_datafolds(dirpath):
    for sec_dirname in os.listdir(dirpath):
        sec_dirpath = os.path.join(dirpath, sec_dirname)
        image_list = os.listdir(sec_dirpath)
        if len(image_list) >= 12:
            person_path1 = os.path.join(fold1path, sec_dirname)
            person_path2 = os.path.join(fold2path, sec_dirname)
            person_path3 = os.path.join(fold3path, sec_dirname)
            person_path4 = os.path.join(fold4path, sec_dirname)

            os.makedirs(person_path1)
            os.makedirs(person_path2)
            os.makedirs(person_path3)
            os.makedirs(person_path4)

            for filename in image_list[:3]:
                image_path = os.path.join(sec_dirpath, filename)
                shutil.copy(image_path, person_path1)
            for filename in image_list[3:6]:
                image_path = os.path.join(sec_dirpath, filename)
                shutil.copy(image_path, person_path2)
            for filename in image_list[6:9]:
                image_path = os.path.join(sec_dirpath, filename)
                shutil.copy(image_path, person_path3)
            for filename in image_list[9:12]:
                image_path = os.path.join(sec_dirpath, filename)
                shutil.copy(image_path, person_path4)

def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle) # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

#return samples and labels of images in a directory.
def alignAndforwardDir(imgPath, infer=False):

    imgs = list(iterImgs(imgPath))

    # Shuffle so multiple versions can be run at once.
    random.shuffle(imgs)

    landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE

    samples = []
    labels = []

    for imgObject in imgs:
        name = imgObject.cls
        rgb = imgObject.getRGB()
        #regPic = os.path.join(regDir, "{}.jpg".format(name))
        #if not os.path.exists(regDir):
        #    os.mkdir(regDir)
        #if os.path.exists(regPic) == None and rgb != None:
        #    cv2.imwrite(regPic, rgb)
        if rgb is None:
            print("  + Unable to load.")
            alignedFace1 = None
            alignedFace2 = None
        else:
            if not infer:
                rgb1 = rgb
                rgb2 = rotate_about_center(rgb, 10)
                rgb3 = rotate_about_center(rgb, 350)
                alignedFace1 = align.align(96, rgb1, landmarkIndices=landmarkIndices, skipMulti=True)
                alignedFace2 = align.align(96, rgb2, landmarkIndices=landmarkIndices, skipMulti=True)

                alignedFace3 = align.align(96, rgb3, landmarkIndices=landmarkIndices, skipMulti=True)

                if alignedFace1 is None:
                    print("  + Unable to align.")

                if alignedFace1 is not None:
                    imageclass = imgObject.cls
                    labels.append(imageclass)
                    samples.append(net.forward(alignedFace1))
                    print("finished 1st forwad passes.")


                if alignedFace2 is None:
                    print("  + Unable to align2.")

                if alignedFace2 is not None:
                    imageclass = imgObject.cls
                    labels.append(imageclass)
                    samples.append(net.forward(alignedFace2))
                    print("finished 2nd forwad passes.")


                if alignedFace3 is None:
                    print("  + Unable to align3.")

                if alignedFace3 is not None:
                    imageclass = imgObject.cls
                    labels.append(imageclass)
                    samples.append(net.forward(alignedFace3))
                    print("finished 3rd forwad passes.")
            else:
                alignedFace = align.align(96, rgb, landmarkIndices=landmarkIndices, skipMulti=True)
                if alignedFace is None:
                    print("  + Unable to align.")

                if alignedFace is not None:
                    imageclass = imgObject.cls
                    labels.append(imageclass)
                    samples.append(net.forward(alignedFace))
                    print("finished 1 forwad pass.")

    samples = np.array(samples)
    labels = np.array([labels])
    data = np.concatenate((labels.T, samples), axis=1)
    return data

#return a sample of a single image with single face detection.
def alignAndforwardSingle(rgb):

    landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE

    bbs = align.getAllFaceBoundingBoxes(rgb)

    rep = None
    for bb in bbs:
         start = time.time()
         alignedFace = align.align(96, rgb, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
         if alignedFace is None:
             raise Exception("Unable to align image")

         start = time.time()
         rep = net.forward(alignedFace)
         print("Neural network forward pass took {} seconds.".format(
                 time.time() - start))
    return rep

def saveReps(imgPath, foldname):
    data = alignAndforwardDir(imgPath)
    data = np.tile(data, (2, 1))
    fName = "{}/{}.npy".format(dataDir, foldname)
    np.save(fName, data)
    print "Saving representations finished."

def generate_reps():
    saveReps(fold1path, "fold1repforP")
    saveReps(fold2path, "fold2repforP")
    saveReps(fold3path, "fold3repforP")
    saveReps(fold4path, "fold4repforP")

#train a classifier using repname, saving in app/data/
#return training time
def train(repname, clf, clfname):
    start = time.time()
    print "start training..."

    rep_path = os.path.join(dataDir, repname)
    train_data = np.load(rep_path)

    labels = train_data[:, 0]
    samples = train_data[:, 1:].astype(np.float)

    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)

    if labelsNum.size > 1:
        clf.fit(samples, labelsNum)
    else:
        clf.fit(samples, labels)

    fName = "{}/{}.pkl".format(dataDir, clfname)
    duration = time.time() - start
    print "Training took {} seconds.".format(duration)
    print "Saving my classifier to '{}'".format(fName)
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)
    return duration

#return the name of the person
def infer(img, clfname, threshold):


    rep = alignAndforwardSingle(img)
    person_name = None
    if not os.path.exists(clfname):
        raise Exception("classifier doesn't exist!")

    with open(clfname, 'r') as f:
        (le, clf) = pickle.load(f)

    predictions = clf.predict_proba(rep).ravel()
    maxI = np.argmax(predictions)
    person_name = le.inverse_transform(maxI)
    confidence = predictions[maxI]
    if confidence < threshold:
        person_name = "unknown"
    else:
        person_name = person
    return person_name

#return the best accuracy of predicting in a directory(using 5 thresholds)
def inferDir(repname, clfname):
    start = time.time()
    print "start predicting..."

    clfpath = os.path.join(dataDir, "{}.pkl".format(clfname))
    if not os.path.exists(clfpath):
        raise Exception("classifier doesn't exist!")

    with open(clfpath, 'r') as f:
        (le, clf) = pickle.load(f)

    thresholds = [0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_accuracy = {}
    reppath = os.path.join(dataDir, repname)
    rep = np.load(reppath)
    test_labels = rep[:, 0]
    test_samples = rep[:, 1:].astype(np.float)


    for threshold in thresholds:
        predictions = clf.predict_proba(test_samples)
        maxP = np.max(predictions, axis=1)
        maxI = np.argmax(predictions, axis=1)
        person_name = le.inverse_transform(maxI)
        person_name[maxP < threshold] = "unknown"

        accur = np.sum(person_name == test_labels) / float(test_labels.size)
        threshold_accuracy[threshold] = accur
    best_threshold_pair = max(threshold_accuracy.items(), key=lambda x: x[1])
    duration = time.time() - start
    print "best threshold for clf is {}".format(best_threshold_pair[0])
    print "accuracy is {}".format(best_threshold_pair[1])
    print "predicting took {} seconds.".format(duration)
    return (best_threshold_pair[1], duration, best_threshold_pair[0])


if __name__ == "__main__":

    training_reps = ["fold1rep.npy", "fold2rep.npy", "fold3rep.npy", "fold4rep.npy"]
    predicting_reps = ["fold1repforP.npy",  "fold2repforP.npy",  "fold3repforP.npy",  "fold4repforP.npy"]


    linearsvm_clf = SVC(C=1, kernel='linear', probability=True)

    param_grid = [
        {'C': [1, 10, 100, 1000],
         'kernel': ['linear']},
        {'C': [1, 10, 100, 1000],
         'gamma': [0.001, 0.0001],
         'kernel': ['rbf']}
    ]
    grid_clf = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5)

    rbfsvm_clf = SVC(C=1, kernel='rbf', probability=True, gamma=2)

    tmp_path = os.path.join(dataDir, training_reps[0])
    tmp_data = np.load(tmp_path)
    tmp_le = LabelEncoder().fit(tmp_data[:, 0])
    nClasses = len(tmp_le.classes_)
    GMM_clf = GMM(n_components=nClasses)

    DT_clf = DecisionTreeClassifier(max_depth=20)

    LR_clf = linear_model.LogisticRegression()

    Gau_clf = GaussianNB()

    clfname_clf = {"linearsvm_clf":linearsvm_clf, "grid_clf":grid_clf, "rbfsvm_clf":rbfsvm_clf, "GMM_clf":GMM_clf, "DT_clf":DT_clf, "LR_clf":LR_clf, "Gau_clf":Gau_clf}

    clf_accuracy = {}
    clf_training_time = {}
    clf_predicting_time = {}
    clf_threshold = {}

    for clfname, clftmplate in  clfname_clf.items():
        clf_total_accur = 0
        clf_total_training_time = 0
        clf_total_predicting_time = 0
        for t_index in xrange(4):
            clf = clftmplate
            training_time = train(training_reps[t_index], clf, clfname)
            total_accur = 0
            total_time = 0
            clf_total_training_time += training_time
            for p_index in range(0, t_index)+range(t_index+1, 4):
                triple = inferDir(predicting_reps[p_index], clfname)
                total_accur += triple[0]
                total_time += triple[1]
                clf_threshold[clfname] = triple[2]
            clf_total_accur += total_accur / 3
            clf_total_predicting_time += total_time / 3
        clf_accuracy[clfname] = clf_total_accur / 4
        clf_training_time[clfname] = clf_total_training_time / 4
        clf_predicting_time[clfname] = clf_total_predicting_time / 4

    f = open("test_report", "w")
    f.write("This is a test report for facial recognition using different classifier\n")
    for clfname, dummy in clfname_clf.items():
        f.write("Average accuracy for {} is {}.".format(clfname, clf_accuracy[clfname]) + "\n")

    for clfname, dummy in clfname_clf.items():
        f.write("Average training time for {} is {}.".format(clfname, clf_training_time[clfname]) + "\n")

    for clfname, dummy in clfname_clf.items():
        f.write("Average predicting time for {} is {}.".format(clfname, clf_predicting_time[clfname]) + "\n")

    for clfname, dummy in clfname_clf.items():
        f.write("Best threshold for {} is {}.".format(clfname, clf_threshold[clfname]) + "\n")
