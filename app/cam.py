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
        self._persons = None

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

    @QtCore.pyqtSlot(list)
    def updateRecLoc(self, persons):
        self._persons = persons

    def changeEvent(self, e):
        if e.type() == QtCore.QEvent.EnabledChange:
            if self.isEnabled():
                self._cameraDevice.newFrame.connect(self._onNewFrame)
            else:
                self._cameraDevice.newFrame.disconnect(self._onNewFrame)

    def paintEvent(self, e):
        if self._frame is None:
            return
        if self._persons:
            for person in self._persons:
                name, px, py, wid = person
                hw = wid / 2
                cv2.rectangle(self._frame, (px-hw,py-hw), (px+hw,py+hw), (55,255,155), 5)
                tx = px - hw
                ty = py + hw + hw / 3
                cv2.putText(self._frame, name,  (tx,ty), 2, 2, (55,255,155),2)
        painter = QtGui.QPainter(self)
        painter.drawImage(QtCore.QPoint(0, 0), OpenCVQImage(self._frame))

class wholeWidget(QtGui.QWidget):
    fbInfo = QtCore.pyqtSignal(list)
    TRAINING_NUM = 10

    def __init__(self, camWidget):
        super(wholeWidget, self).__init__()
        self._camWidget = camWidget

        self._grid = QtGui.QGridLayout()
        self._grid.setSpacing(5)

        self._predictBtt = QtGui.QPushButton("predict")
        self._predictBtt.clicked.connect(self._predictFunc)

        self._trainBtt = QtGui.QPushButton("train")
        self._trainBtt.clicked.connect(self._trainFunc)
        self._count = 0
        self._trainList = []
        self._trainPerson = None

        self._procDialog = None

        self._grid.addWidget(self._predictBtt, 0, 0, 1, 1)
        self._grid.addWidget(self._trainBtt, 0, 1, 1, 1)

        self._grid.addWidget(self._camWidget, 1, 0, 5, 2)

        self.fbInfo.connect(self._camWidget.updateRecLoc)

        self.setLayout(self._grid)

        self.setGeometry(300, 300, 350, 300)

    @QtCore.pyqtSlot()
    def _predictFunc(self):
        try:
            self._camWidget.alFrame.disconnect()
        except Exception:
            pass
        self._camWidget.alFrame.connect(self._predict)

    @QtCore.pyqtSlot(np.ndarray)
    def _predict(self, predictFrame):
        rgb = cv2.cvtColor(predictFrame, cv2.COLOR_BGR2RGB)
        persons = simple_classifier.infer(rgb)
        self.fbInfo.emit(persons)

    @QtCore.pyqtSlot()
    def _trainFunc(self):
        try:
            self._camWidget.alFrame.disconnect()
        except Exception:
            pass
        self.fbInfo.emit([])
        self._camWidget.alFrame.connect(self._train)

        text, ok = QtGui.QInputDialog.getText(self, "", "Enter your name:")

        if ok:
            self.receiveName(str(text))



    @QtCore.pyqtSlot(np.ndarray)
    def _train(self, trainFrame):
        if not self._trainPerson:
            return
        if not self._procDialog:
            total_steps = self.TRAINING_NUM
            self._procDialog = QtGui.QProgressDialog("", "Cancel", 0, total_steps, self)
            self._procDialog.setLabelText("Training...")
            self._procDialog.canceled.connect(self._interrTrain)
            self._procDialog.setAutoClose(False)
            self._procDialog.setAutoReset(False)
            self._procDialog.show()
        rgb = cv2.cvtColor(trainFrame, cv2.COLOR_BGR2RGB)
        self._trainList.append(rgb)
        self._count += 1
        self._procDialog.setValue(self._count)
        if self._count == self.TRAINING_NUM:
            simple_classifier.train(self._trainList, self._trainPerson)
            self._procDialog.setLabelText("Training completed!")


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
