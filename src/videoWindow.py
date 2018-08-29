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
import collections

sys.path.append(os.path.abspath(r"../"))

try:
    from utils import label_map_util
    from utils import visualization_utils as vis_util
except ModuleNotFoundError as e:
    print(str(e))

ICON = Path(r'..\articles\atom.png')

DATA = r"..\data"


class CheckableComboBox(QtWidgets.QComboBox):

    def __init__(self):

        super().__init__()
        self.view().pressed.connect(self.handleItemPressed)
        self.setModel(QtGui.QStandardItemModel(self))

    def initComboBox(self, categoriesIndex):
        
        elemIndex = 0

        for elem in categoriesIndex:

            self.addItem(categoriesIndex[elem]["name"])
            item = self.model().item(elemIndex, 0)
            item.setCheckState(QtCore.Qt.Unchecked)
            elemIndex += 1

    def handleItemPressed(self, index):

        item = self.model().itemFromIndex(index)

        if item.checkState() == QtCore.Qt.Checked:

            item.setCheckState(QtCore.Qt.Unchecked)
        else:
            item.setCheckState(QtCore.Qt.Checked)

class VideoWindow(QMainWindow):

    '''
    Window for loading in video and model for classification
    
    '''
    pixmapChanged = QtCore.pyqtSignal()

    def __init__(self):

        super().__init__()
        '''Fields'''
        self.args = None
        self.th = None
        self.modelsDirPath = r"..\models"
        self.modelsDownloadPath = r"..\downloaded_models"
        self.downloadBase = 'http://download.tensorflow.org/models/object_detection/'
        self.modelLabelMap = None
        self.modelInput = None
        self.image_processor = None
        self.error = None
        self.createModelsDir()
        

        self.specificationsInfo = "Note: Module has to be loaded first.\n"

        ''' Window Properties'''

        self.Icon = QtGui.QIcon(str(ICON))
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
        
        ''' Categories Specifications '''
        self.categoriesLabel = QLabel("Categories")
        self.categoriesComboBox = CheckableComboBox()

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

        self.inputsFrameLayout.setWidget(2, QFormLayout.LabelRole, self.categoriesLabel)
        self.inputsFrameLayout.setWidget(2, QFormLayout.FieldRole, self.categoriesComboBox)

        self.inputsFrameLayout.setWidget(3, QFormLayout.LabelRole, self.modelImportButton)
        self.inputsFrameLayout.setWidget(3, QFormLayout.FieldRole, self.modelLineEdit)
        self.inputsFrameLayout.addWidget(self.runButton)


        self.statsFrameLayout.setWidget(0, QFormLayout.LabelRole, self.modelTimeLabel)
        self.statsFrameLayout.setWidget(0, QFormLayout.FieldRole, self.modelTimeLineEdit)

        self.verticalLayout.addWidget(self.videoFrame)
        self.verticalLayout.addWidget(self.inputsFrame)
        self.verticalLayout.addWidget(self.statsFrame)

        self.setCentralWidget(self.centralwidget)


    def handleRun(self):
    
        '''
            - Disconnect signals to stop Graph execution from being started.
            - Wait for current Graph execution to finish
            - Exit Thread
                -create new image_processor QObject
                -Load neural net to image_processor
                -initialize image_processor session
                -set up video stream
        '''
        

        
        if self.th:

            self.image_processor.changePixmap.disconnect(self.setImage)
            self.pixmapChanged.disconnect(self.image_processor.loadFrame)
            time.sleep(0.5) 
            

            self.image_processor.image_detector.session.close()
            while not self.image_processor.image_detector.session._closed:
                time.sleep(0.1)

            print("here")
            self.th.quit()

            
            # self.th = None

            self.handleLoadModule()
        
        
        self.th = QtCore.QThread()
        self.image_processor.moveToThread(self.th)
        
        self.pixmapChanged.connect(self.image_processor.loadFrame)
        self.image_processor.changePixmap.connect(self.setImage)

        self.th.start()
        self.image_processor.setupVideoStream(self.videoLineEdit.text())

     


    def handleLoadModule(self):
        ''' 
            - Checks whether image_processor is currently running to determine whether or not is the fist run
            -
        '''

        if self.image_processor:
            if self.image_processor.receivers(self.image_processor.changePixmap):

                self.handleRun()

                return

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
        self.pixmapChanged.emit()

 

    def loadModule(self, fileDirName):

        '''
        - Looks for unpacked model, if found opens and loads inference graph into memory
        - Downloads model if link is provided, unpacks model and loads graph into memory
        '''
        
        graphPath = Path(self.modelsDirPath + os.sep + str(fileDirName) + os.sep + 'frozen_inference_graph.pb')

        if 'frozen_inference_graph.pb' in os.path.basename(graphPath):
            self.image_processor = FrameProcessor(graphPath)
            self.categoriesComboBox.initComboBox(self.image_processor.category_index)

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

    def loadLabelMaps(self):

        pass




class FrameProcessor(QtCore.QObject):


    changePixmap = QtCore.pyqtSignal(QtGui.QImage, float)

    def __init__(self, graphPath):
        '''
        misc
        '''
        super().__init__()

        self.NUM_CLASSES = 90

        self.image_detector = ImageDetector(graphPath)
        self.image_detector.session =  tf.Session(graph=self.image_detector.detectionGraph)

        '''Categories'''
        self.PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

        self.rgbImage = None
        self.convertToQtFormat = None
        self.p = None
        self.frame = None

        self.label_map = label_map_util.load_labelmap(r'..\data\mscoco_label_map.pbtxt')
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def setupVideoStream(self, videoFilePath):

        if self.isNumber(videoFilePath)[0]: self.cap = cv2.VideoCapture(int(videoFilePath))
        else: self.cap = cv2.VideoCapture(videoFilePath)

        self.loadFrame()

    def runDetection(self, image, sess):

        

        image_expanded = np.expand_dims(image, axis=0)

        # # Actual detection.
        startTime = time.time()
        (boxes, scores, classes) = sess.run( [self.image_detector.boxes, self.image_detector.scores, self.image_detector.classes], 
        feed_dict={self.image_detector.image_tensor: image_expanded})
        endTime = time.time()
        #print(( '\n\nboxes\n\n', boxes, '\n\nscores\n\n', scores ,'\n\nclasses\n\n', classes, '\n\nnum_detections\n\n', num_detections))
        # Visualization of the results of a detection.

        # (boxes, scores, classes) = self.filter_boxes(0.2, np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32), [1])
        # if list(boxes):
        #     print(scores)
        #     vis_util.visualize_boxes_and_labels_on_image_array(
        #         image,
        #         boxes,
        #         classes,
        #         scores,
        #         self.category_index,
        #         use_normalized_coordinates=True,
        #         line_thickness=8)


        vis_util.visualize_boxes_and_labels_on_image_array(
          image,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          self.category_index,
          use_normalized_coordinates=True,
          line_thickness=8)

        return image, endTime - startTime

 
    def loadFrame(self):

            ret, self.frame = self.cap.read()
            self.frame, time = self.runDetection(self.frame, self.image_detector.session)

            self.rgbImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.convertToQtFormat = QtGui.QImage(self.rgbImage.data, self.rgbImage.shape[1], self.rgbImage.shape[0], QtGui.QImage.Format_RGB888)
            self.p = self.convertToQtFormat.scaled(800, 640, QtCore.Qt.KeepAspectRatio)
            self.changePixmap.emit(self.p, time)




    def filter_boxes(self, min_score, boxes, scores, classes, categories):

        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)

        idxs = []
        for i in range(n):

            if classes[i] in categories and scores[i] >= min_score:
                idxs.append(i)
        
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]

        return filtered_boxes, filtered_scores, filtered_classes

    def isNumber(self, s):
        ''' 
        Implemented in validating sample calculation inputs
        '''
        try:
            int(s)
            return (True, None)
        except Exception as e:
            return (False, e)
            
class ImageDetector():

    def __init__(self, graphPath):


        '''session'''
        self.session = None

        '''Graph'''

        self.graphPath = graphPath
        self.detectionGraph = tf.Graph()
        self.od_graph_def = tf.GraphDef()

        ''' tensors '''

        self.image_tensor = None
        self.boxes = None
        self.classes = None
        self.scores = None
        self.num_detections = None 

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

        self.image_tensor = self.detectionGraph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.boxes = self.detectionGraph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.scores = self.detectionGraph.get_tensor_by_name('detection_scores:0')

        self.classes = self.detectionGraph.get_tensor_by_name('detection_classes:0')

        # num_detections = self.detectionGraph.get_tensor_by_name('num_detections:0')
        self.num_detections = self.detectionGraph.get_tensor_by_name('num_detections:0')


