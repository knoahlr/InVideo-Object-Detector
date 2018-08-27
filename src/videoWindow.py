from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QWidget, QFormLayout, QMainWindow, QGroupBox, QVBoxLayout, \
QLabel, QTextEdit, QLineEdit, QPushButton

from errorWindow import ErrorWindow

import cv2
import tensorflow as tf
import numpy as np

import os, sys, time
from pathlib import Path
import tarfile
import urllib

sys.path.append('../')

from utils import label_map_util
from utils import visualization_utils as vis_util

ICON = r'..\articles\atom.png'


class Thread(QtCore.QThread):

    changePixmap = QtCore.pyqtSignal(QtGui.QImage, float)

    def __init__(self, image_processor, videoFilePath, modelTimeLineEdit):
        
        super().__init__()

        self.cap = cv2.VideoCapture(videoFilePath)
        self.rgbImage = None
        self.convertToQtFormat = None
        self.p = None
        self.frame = None
        self.frameProcessor = image_processor
        self.session = None


    def run(self):
        
        with tf.Session(graph=self.frameProcessor.detectionGraph) as self.session:

            while True:
                ret, self.frame = self.cap.read()
                # print("frame", self.frame)
                self.frame, time = self.frameProcessor.runDetection(self.frame, self.session)

                self.rgbImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                self.convertToQtFormat = QtGui.QImage(self.rgbImage.data, self.rgbImage.shape[1], self.rgbImage.shape[0], QtGui.QImage.Format_RGB888)
                self.p = self.convertToQtFormat.scaled(800, 640, QtCore.Qt.KeepAspectRatio)
                self.changePixmap.emit(self.p, time)
            
    def quit(self):

        self.cap.release()
        self.changePixmap.disconnect()
        self.blockSignals(True)
        super().quit()


class VideoWindow(QMainWindow):

    '''
    Window for loading in video and model for classification
    
    '''

    def __init__(self):

        super().__init__()
        '''Fields'''
        self.args = None
        self.th = None
        self.modelsDirPath = r"..\models"
        self.modelsDownloadPath = r"..\downloaded_models"
        self.downloadBase = 'http://download.tensorflow.org/models/object_detection/'
        self.modelInput = None
        self.image_processor = None
        self.error = None
        self.createModelsDir()
        

        self.specificationsInfo = "Note: Module has to be loaded first.\n"

        ''' Window Properties'''

        self.Icon = QtGui.QIcon(ICON)
        self.setMinimumSize(self.sizeHint())
        self.resize(1200, 800)
        self.setWindowTitle('Multi Model Video-Object Classifier')
        self.setWindowIcon(self.Icon)


        ''' Setting window layout and central widget '''
        self.centralwidget = QtWidgets.QWidget()
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setAlignment(QtCore.Qt.AlignCenter)

        '''Box Frames'''

        self.videoFrame = QGroupBox('Video')
        
        self.inputsFrame = QGroupBox('Specifications')
        self.statsFrame = QGroupBox('Statistics')

        ''' Frame Size Policy'''
        self.videoFrame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.videoFrame.setAlignment(QtCore.Qt.AlignTop)

        ''' Frame layout '''

        self.videoFrameLayout = QVBoxLayout(self.videoFrame)
        self.inputsFrameLayout = QFormLayout(self.inputsFrame)
        self.statsFrameLayout = QFormLayout(self.statsFrame)



        ''' input labels '''
        self.info = QLabel(self.specificationsInfo)
        self.infoFont = self.info.font()
        self.infoFont.setPointSize(10)
        self.infoFont.setItalic(True)
        self.infoFont.setFamily("Comic Sans MS")
        self.info.setFont(self.infoFont)


        self.videoFilePath = QLabel('Path to Video')
        self.modelImportButton = QPushButton('Load Module')
        self.modelImportButton.clicked.connect(self.handleLoadModule)
        

        self.videoLineEdit = QLineEdit(self.inputsFrame)
        self.modelLineEdit = QLineEdit(self.inputsFrame)
        self.modelLineEdit.setText('ssd_mobilenet_v1_coco_2018_01_28')
        self.videoLineEdit.setText(r"D:\OneDrive - Carleton University\Noah\Movies\Avatar - the last Airbender - Season 2 Complete - NXOR\S02E01 Avatar - The Last Airbender - Book 2 - Chapter 01 - The Avatar State.avi")

        ''' stat Fields '''
        self.modelTimeLabel = QLabel('Time')
        self.modelTimeLineEdit = QLineEdit(self.statsFrame)

        '''run button'''
        self.runButton = QPushButton('Run')
        self.runButton.clicked.connect(self.handleRun)
        self.runButton.setEnabled(False)


        ''' video Widget'''

        self.videoWidget = QtWidgets.QGraphicsView(self.videoFrame)
        self.videoFrameLayout.addWidget(self.videoWidget)

        ''' Add Labels and line edits to group boxes '''
        self.inputsFrameLayout.addWidget(self.info)
        self.inputsFrameLayout.setWidget(1, QFormLayout.LabelRole, self.videoFilePath)
        self.inputsFrameLayout.setWidget(1, QFormLayout.FieldRole, self.videoLineEdit)

        self.inputsFrameLayout.setWidget(2, QFormLayout.LabelRole, self.modelImportButton)
        self.inputsFrameLayout.setWidget(2, QFormLayout.FieldRole, self.modelLineEdit)
        self.inputsFrameLayout.addWidget(self.runButton)


        self.statsFrameLayout.setWidget(0, QFormLayout.LabelRole, self.modelTimeLabel)
        self.statsFrameLayout.setWidget(0, QFormLayout.FieldRole, self.modelTimeLineEdit)

        self.verticalLayout.addWidget(self.videoFrame)
        self.verticalLayout.addWidget(self.inputsFrame)
        self.verticalLayout.addWidget(self.statsFrame)

        self.setCentralWidget(self.centralwidget)


    def handleRun(self):

        if self.th:
            self.th.session.close()
            self.th.terminate()
            self.th.quit()
            self.th= None
            
        self.th =Thread(self.image_processor, self.videoLineEdit.text(), self.modelTimeLineEdit)
        self.th.changePixmap.connect(self.setImage)
        self.th.start()


    def handleLoadModule(self):
        
        
        self.modelInput = self.modelLineEdit.text()
        pathOrLink = self.modelInput

        _bool, fileDirName = self.pathOrLinkCheck(Path(pathOrLink))

        if _bool:

            self.loadModule(fileDirName)

        else:

            try:
                opener = urllib.request.URLopener()
                opener.retrieve(self.downloadBase+pathOrLink+".tar.gz", filename=self.modelsDownloadPath+os.sep+pathOrLink)

                ''' Extract File '''
                self.extractModels(self.modelsDownloadPath+os.sep+pathOrLink)

                self.loadModule(Path(pathOrLink))
            
            except urllib.error.URLError:

                self.modelLineEdit.clear()
                self.error = ErrorWindow("Path provided is not a directory, a tarFile or a link", self.Icon)
                self.error.show()
                

                
#ssd_mobilenet_v1_coco_11_06_2017
    def setImage(self, image, time):

        self.modelTimeLineEdit.setText(str(time))
        scene = QtWidgets.QGraphicsScene()
        pixmapItem = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(image))
        scene.addItem(pixmapItem)
        self.videoWidget.setScene(scene)

 

    def loadModule(self, fileDirName):

        '''
        - Looks for unpacked model, if found opens and loads inference graph into memory
        - Downloads model if link is provided, unpacks model and loads graph into memory
        '''

        # pathOrLink = Path(self.modelInput)

        # _bool, fileDirName = self.pathOrLinkCheck(pathOrLink)

        # if not _bool: self.modelLineEdit.clear()
        
        graphPath = Path(self.modelsDirPath + os.sep + str(fileDirName) + os.sep + 'frozen_inference_graph.pb')

        if 'frozen_inference_graph.pb' in os.path.basename(graphPath):
            self.image_processor = FrameProcessor(graphPath)

            self.runButton.setEnabled(True)

        

    def pathOrLinkCheck(self, path):


        if path.is_absolute():
            if not path.is_dir():
                self.error = ErrorWindow("Path provided is not a directory", self.Icon)
                self.error.show()
                return False, None
            else:
                if path.is_dir(): return True, path
                if tarfile.is_tarfile(path):self.extractModels(path)

                return True, self.modelsDirPath
        else:

            relativePath = path
            fullPath = Path(self.modelsDirPath + os.sep + str(path))

            if relativePath.is_dir(): return True, relativePath

            elif fullPath.is_dir(): return True, fullPath

            else:
                try:
                    if tarfile.is_tarfile(relativePath):

                        self.extractModels(relativePath)
                        return True, self.modelsDirPath

                except FileNotFoundError:

                    # self.error = ErrorWindow("Path provided is not a directory or a tarFile", self.Icon)
                    # self.error.show()

                    return False, None
 
                try:
                    if tarfile.is_tarfile(fullPath): 

                        self.extractModels(fullPath)
                        return True, self.modelsDirPath

                except FileNotFoundError:

                    return False, None

                    # self.error = ErrorWindow("Path provided is not a directory or a tarFile", self.Icon)
                    # self.error.show()

        
            # self.error = ErrorWindow("Path provided is not a directory or a tarFile", self.Icon)
            # self.error.show()

            return False, None
        

    def createModelsDir(self):

        if not os.path.exists(Path(self.modelsDownloadPath)): os.makedirs(Path(self.modelsDownloadPath))
        
        if not os.path.exists(Path(self.modelsDirPath)): os.makedirs(Path(self.modelsDirPath))


    def extractModels(self, path):

        if tarfile.is_tarfile(path):
            tar_file = tarfile.open(path)
            for file in tar_file.getmembers():
                file_name = os.path.basename(file.name)
                if 'frozen_inference_graph.pb' in file_name:
                    tar_file.extract(file, self.modelsDirPath)



class FrameProcessor():


    def __init__(self, graphPath):
        '''
        misc
        '''

        self.NUM_CLASSES = 90

        self.graphPath = graphPath
        self.detectionGraph = tf.Graph()
        self.od_graph_def = tf.GraphDef()

        '''Categories'''
        self.PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

        self.label_map = label_map_util.load_labelmap(r'C:\Users\Noah Workstation\Desktop\P_PR\repo\Video_Detector\data\mscoco_label_map.pbtxt')
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        self.createDetectionGraph()

    def createDetectionGraph(self):

        '''
        Loading frozen tensorflow model into memory
        '''
        with self.detectionGraph.as_default():
            with tf.gfile.GFile(str(self.graphPath), 'rb') as fid:
                serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')
    
    def runDetection(self, image, sess):

        

        image_expanded = np.expand_dims(image, axis=0)
        image_tensor = self.detectionGraph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detectionGraph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detectionGraph.get_tensor_by_name('detection_scores:0')
        classes = self.detectionGraph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detectionGraph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        startTime = time.time()
        (boxes, scores, classes, num_detections) = sess.run( [boxes, scores, classes, num_detections], feed_dict={image_tensor: image_expanded})
        endTime = time.time()
        #print(( '\n\nboxes\n\n', boxes, '\n\nscores\n\n', scores ,'\n\nclasses\n\n', classes, '\n\nnum_detections\n\n', num_detections))
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        return image, endTime - startTime

    def filter_boxes():

        pass








