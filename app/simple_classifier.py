#!/usr/bin/env python2
#put training and inferring together

import time

start = time.time()

import argparse
import cv2
import os
import pickle
import pandas as pd
import random
import numpy as np
np.set_printoptions(precision=2)

import openface
from openface.data import iterImgs


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..','models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
net = openface.TorchNeuralNet(os.path.join(
    openfaceModelDir,
    'nn4.small2.v1.t7'), imgDim=96, cuda=False)


#return samples and labels of images in a directory.
def alignAndforwardMulti(rgbs, name):

    #imgs = list(iterImgs(fromDir))

    # Shuffle so multiple versions can be run at once.
    #random.shuffle(imgs)

    landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE

    samples = []
    labels = []

    for rgb in rgbs:
        alignedFace = align.align(96, rgb,
                             landmarkIndices=landmarkIndices,
                             skipMulti=True)
        if alignedFace is None:
            print("  + Unable to align.")

        if alignedFace is not None:
            labels.append(name)
            samples.append(net.forward(alignedFace))
    samples = np.array(samples)
    labels = np.array([labels])
    data = np.concatenate((labels.T, samples), axis=1)
    return data

#return a sample of a single image with multiple faces detection.
def alignAndforwardSingle(rgb):

    landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE

    bbs = align.getAllFaceBoundingBoxes(rgb)

    reps = []
    for bb in bbs:
         start = time.time()
         alignedFace = align.align(
             96,
             rgb,
             bb,
             landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
         if alignedFace is None:
             raise Exception("Unable to align image")

         start = time.time()
         rep = net.forward(alignedFace)
         print("Neural network forward pass took {} seconds.".format(
                 time.time() - start))
         reps.append((bb.center().x, bb.center().y, bb.width(), rep))
    sreps = sorted(reps, key=lambda x: (x[0], x[1]))
    return sreps

def train(rgbs, name):
    start = time.time()

    data = alignAndforwardMulti(rgbs, name)
    fName = "{}/prev_data.npy".format(fileDir)
    labels = data[:, 0]
    samples = data[:, 1:]
    if os.path.exists(fName):
        prev = np.load(fName)
        new_data = np.concatenate((prev, data), axis=0)
        np.save(fName, new_data)
        labels = new_data[:, 0]
        samples = new_data[:, 1:]
    else:
        np.save(fName, data)
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)

    clf = SVC(C=1, kernel='linear', probability=True)
    if labelsNum.size > 1:
        clf.fit(samples, labelsNum)
    else:
        clf.fit(samples, labels)

    fName = "{}/myclassifier.pkl".format(fileDir)
    print "Training took {} seconds.".format(time.time() - start)
    print "Saving my classifier to '{}'".format(fName)
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)

def infer(img):

    sreps = alignAndforwardSingle(img)
    persons = []

    if not os.path.exists("myclassifier.pkl"):
        for rep in sreps:
            px = rep[0]
            py = rep[1]
            wid = rep[2]
            persons.append(("unknown", px, py, wid))
        return persons

    with open("myclassifier.pkl", 'r') as f:
        (le, clf) = pickle.load(f)


    for rep in sreps:
        px = rep[0]
        py = rep[1]
        wid = rep[2]
        rep = rep[3].reshape(1, -1)
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]
        if confidence < 0.5:
            persons.append(("unknown", px, py, wid))
        else:
            persons.append((person, px, py, wid))
    return persons
