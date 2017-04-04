#!/usr/bin/env python2
#put training and inferring together

import time

start = time.time()

import argparse
import cv2
import os
import pickle
import random
import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import openface
from openface.data import iterImgs


from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')



#return a dictionary of aligned faces.
def alignAndforward(fromDir, align):

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
                if args.verbose:
                    print("  + Writing aligned file to disk.")
                imageclass = imgObject.cls
                labels.append(imageclass)
                samples.append(net.forward(alignedFace))
    return samples, labels

def train(samples_l, labels_l):
    le = LabelEncoder().fit(labels_l)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))

    clf = SVC(C=1, kernel='linear', probability=True)
    clf.fit(samples_l, labelsNum)

    fName = "{}/myclassifier.pkl".format(fileDir)
    print("Saving my classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)

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
    inferParser.add_argument('--multi', help="Infer multiple faces in image",
                             action="store_true")

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
        samples, labels = alignAndforward(args.imageDir, align)
        train(samples, labels)
    elif args.mode == 'infer':
        infer(args, args.multi)
