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



#return samples and labels of images in a directory.
def alignAndforwardDir(fromDir, align):

    imgs = list(iterImgs(fromDir))

    # Shuffle so multiple versions can be run at once.
    random.shuffle(imgs)

    landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE

    samples = []
    labels = []

    for imgObject in imgs:
        rgb = imgObject.getRGB()
        if rgb is None:
            if args.verbose:
                print("  + Unable to load.")
            alignedFace = None
        else:
            alignedFace = align.align(96, rgb,
                                 landmarkIndices=landmarkIndices,
                                 skipMulti=True)
            if alignedFace is None and args.verbose:
                print("  + Unable to align.")

            if alignedFace is not None:
                imageclass = imgObject.cls
                labels.append(imageclass)
                samples.append(net.forward(alignedFace))
    samples = np.array(samples)
    labels = np.array([labels])
    data = np.concatenate((labels.T, samples), axis=1)
    return data

#return a sample of a single image.
def alignAndforwardSingle(imgPath):

    start = time.time()

    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))

    landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE

    rgb = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if rgb is None:
        if args.verbose:
            print("  + Unable to load.")
        alignedFace = None
    else:
        alignedFace = align.align(96, rgb,
                             landmarkIndices=landmarkIndices,
                             skipMulti=True)
        if alignedFace is None and args.verbose:
            print("  + Unable to align.")

        if alignedFace is not None:
            rep = net.forward(alignedFace)
            print("Forwarding took {} seconds".format(time.time() - start))
            return rep

def train(data):
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
    start = time.time()
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))

    clf = SVC(C=1, kernel='linear', probability=True)
    clf.fit(samples, labelsNum)

    fName = "{}/myclassifier.pkl".format(fileDir)
    print("Training took {} seconds.".format(time.time() -start))
    print("Saving my classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)

def infer(args):
    with open(args.classifierModel, 'r') as f:
        (le, clf) = pickle.load(f)

    for img in args.imgs:
        print("\n=== {} ===".format(img))
        rep = alignAndforwardSingle(img)

        rep = rep.reshape(1, -1)

        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]
        if args.verbose:
            print("Prediction took {} seconds.".format(time.time() - start))

        print("Predict {} with {:.2f} confidence.".format(person, confidence))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')

    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    trainParser = subparsers.add_parser('train',
                                        help="Train a new classifier.")
    trainParser.add_argument('--ldaDim', type=int, default=-1)
    trainParser.add_argument(
        '--classifier',
        type=str,
        choices=[
            'LinearSvm',
            'GridSearchSvm',
            'GMM',
            'RadialSvm',
            'DecisionTree',
            'GaussianNB',
            'DBN'],
        help='The type of classifier to use.',
        default='LinearSvm')
    trainParser.add_argument(
        'imageDir',
        type=str,
        help="The training set.")

    inferParser = subparsers.add_parser(
        'infer', help='Predict who an image contains from a trained classifier.')
    inferParser.add_argument(
        'classifierModel',
        type=str,
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')
    inferParser.add_argument('imgs', type=str, nargs='+',
                             help="Input image.")

    args = parser.parse_args()
    if args.verbose:
        print("Argument parsing and import libraries took {} seconds.".format(time.time() - start))

    if args.mode == 'infer' and args.classifierModel.endswith(".t7"):
        raise Exception("""
Torch network model passed as the classification model,
which should be a Python pickle (.pkl)

See the documentation for the distinction between the Torch
network and classification models:

        http://cmusatyalab.github.io/openface/demo-3-classifier/
        http://cmusatyalab.github.io/openface/training-new-models/

Use `--networkModel` to set a non-standard Torch network model.""")
    start = time.time()

    align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
    net = openface.TorchNeuralNet(os.path.join(
        openfaceModelDir,
        'nn4.small2.v1.t7'), imgDim=96, cuda=False)

    if args.verbose:
        print("Loading the dlib and OpenFace models took {} seconds.".format(time.time() - start))
        start = time.time()

    if args.mode == 'train':
        data = alignAndforwardDir(args.imageDir, align)
        train(data)
    elif args.mode == 'infer':
        infer(args)
