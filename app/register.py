#!/usr/bin/env python2
import cv2
import numpy as np
import sys

import simple_classifier

import time
from PyQt4 import QtGui
from PyQt4 import QtCore


class OpenCVQImage(QtGui.QImage):

    def __init__(self, bgrImg):
        h, w = bgrImg.shape[:2]
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
        self._imgData = rgbImg.tostring()
        super(OpenCVQImage, self).__init__(self._imgData, w, h, \
            QtGui.QImage.Format_RGB888)


class CameraDevice(QtCore.QObject):

    _DEFAULT_FPS = 20

    newFrame = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, cameraId=0, parent=None):
        super(CameraDevice, self).__init__(parent)


        self._cameraDevice = cv2.VideoCapture(cameraId)
        self._cameraDevice.set(3, 1500)
        self._cameraDevice.set(4, 1300)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._queryFrame)
        self._timer.setInterval(1000/self._DEFAULT_FPS)

        self.paused = False

    @QtCore.pyqtSlot()
    def _queryFrame(self):
        ret, frame = self._cameraDevice.read()
        mirroredFrame = frame.copy()
        mirroredFrame = cv2.flip(frame, 1)
        self.newFrame.emit(mirroredFrame)

    @property
    def paused(self):
        return not self._timer.isActive()

    @paused.setter
    def paused(self, p):
        if p:
            self._timer.stop()
        else:
            self._timer.start()

    @property
    def frameSize(self):
        w = self._cameraDevice.get(3)
        h = self._cameraDevice.get(4)
        return int(w), int(h)


class CameraWidget(QtGui.QWidget):

    alFrame = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, cameraDevice, parent=None):
        super(CameraWidget, self).__init__(parent)

        self._frame = None
        self._count = 0

        self._cameraDevice = cameraDevice
        self._cameraDevice.newFrame.connect(self._onNewFrame)

        w, h = self._cameraDevice.frameSize
        self.setMinimumSize(w, h)
        self.setMaximumSize(w, h)

    @QtCore.pyqtSlot(np.ndarray)
    def _onNewFrame(self, frame):
        self._frame = frame.copy()
        self._count += 1
        if self._count == 12:
            self.alFrame.emit(self._frame)
            self._count = 0

        self.update()

    def changeEvent(self, e):
        if e.type() == QtCore.QEvent.EnabledChange:
            if self.isEnabled():
                self._cameraDevice.newFrame.connect(self._onNewFrame)
            else:
                self._cameraDevice.newFrame.disconnect(self._onNewFrame)

    def paintEvent(self, e):
        if self._frame is None:
            return
        painter = QtGui.QPainter(self)
        painter.drawImage(QtCore.QPoint(0, 0), OpenCVQImage(self._frame))

class wholeWidget(QtGui.QWidget):
    TRAINING_NUM = 12

    def __init__(self, camWidget):
        super(wholeWidget, self).__init__()
        self._camWidget = camWidget

        self._grid = QtGui.QGridLayout()
        self._grid.setSpacing(6)


        self._count = 0
        self._trainList = []
        self._trainPerson = None

        self._procDialog = None

        self._inputBtt = QtGui.QPushButton("From camera")
        self._inputBtt.clicked.connect(self._textDialog)
        self._fileBtt = QtGui.QPushButton("From file")
        self._fileBtt.clicked.connect(self._fileDialog)
        self._regBtt = QtGui.QPushButton("Start registration")
        self._regBtt.clicked.connect(self._train)

        self._grid.addWidget(self._inputBtt, 0, 0, 1, 2)
        self._grid.addWidget(self._fileBtt, 0, 2, 1, 2)
        self._grid.addWidget(self._regBtt, 0, 4, 1, 2)

        self._grid.addWidget(self._camWidget, 1, 0, 5, 6)

        self.setLayout(self._grid)

        self.setGeometry(300, 300, 350, 300)



    @QtCore.pyqtSlot()
    def _textDialog(self):
        text, ok = QtGui.QInputDialog.getText(self, "", "Enter your name:")

        if ok:
            self.receiveName(str(text))
            self._camWidget.alFrame.connect(self._sample)



    @QtCore.pyqtSlot(np.ndarray)
    def _sample(self, trainFrame):
        if not self._trainPerson:
            raise Exception("Haven't register persons")
        while self._count < self.TRAINING_NUM * 5:
            if not self._count % 5:
                self._trainList.append(trainFrame)
            self._count += 1
            return
        simple_classifier.saveReps(self._trainPerson, self._trainList)
        self._count = 0
        self._trainPerson = None
        self._camWidget.alFrame.disconnect()

    @QtCore.pyqtSlot()
    def _fileDialog(self):
        dirName = QtGui.QFileDialog.getExistingDirectory()
        dirName = str(dirName)
        simple_classifier.saveReps("", None, True, dirName)

    @QtCore.pyqtSlot()
    def _train(self):
        simple_classifier.train()


    def receiveName(self, name):
        self._trainPerson = name


    @QtCore.pyqtSlot()
    def _interrTrain(self):
        self._camWidget.alFrame.disconnect()
        self._count = 0
        self._trainPerson = None
        self._trainList = []
        self._procDialog = None

def main():

    app = QtGui.QApplication(sys.argv)

    cameraDevice = CameraDevice()

    cameraWidget1 = CameraWidget(cameraDevice)
    theWidget = wholeWidget(cameraWidget1)
    theWidget.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
