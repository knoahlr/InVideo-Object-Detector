from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QWidget, QFormLayout, QMainWindow, QGroupBox, QVBoxLayout, QHBoxLayout, \
QLabel, QTextEdit, QLineEdit, QPushButton, QFrame

from errorWindow import ErrorWindow
from frameProcessor import FrameProcessor

import cv2

import sys, os, time

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

    editFilter = QtCore.pyqtSignal(dict)

    def __init__(self):

        super().__init__()
        self.checkedItems = None
        self.view().pressed.connect(self.handleItemPressed)
        self.setModel(QtGui.QStandardItemModel(self))
        self.itemCount = 0

    def initComboBox(self, categoriesIndex):
        
        elemIndex = 0
        self.addItem("All")
        item = self.model().item(elemIndex, 0)
        item.setCheckState(QtCore.Qt.Unchecked)
        elemIndex += 1

        for elem in categoriesIndex:

            self.addItem(categoriesIndex[elem]["name"])
            item = self.model().item(elemIndex, 0)
            item.setCheckState(QtCore.Qt.Unchecked)
            # item.setFlags()
            elemIndex += 1
        self.itemCount = elemIndex

    def handleItemPressed(self, index):


        item = self.model().itemFromIndex(index)


        if item.text() == "All":

            if item.checkState() == QtCore.Qt.Unchecked:

                model = self.model()

                for i in range(self.itemCount):
                    item = model.item(i, 0)
                    item.setCheckState(QtCore.Qt.Checked)
                    # model.itemChanged(item)

            else:

                model = self.model()

                for i in range(self.itemCount):
                    item = model.item(i, 0)
                    item.setCheckState(QtCore.Qt.Unchecked)


        if item.checkState() == QtCore.Qt.Checked:

            item.setCheckState(QtCore.Qt.Unchecked)

        else:

            item.setCheckState(QtCore.Qt.Checked)

        self.getAllCheckedItems()
        
    def getAllCheckedItems(self):

        filter = {}

        model = self.model()

        for i in range(model.rowCount()):
            item = model.item(i, 0)

            if item.checkState() == QtCore.Qt.Checked: filter[i] = item.text()

        self.editFilter.emit(filter)
            
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
        
        self.statsFrame = QGroupBox('Statistics')
        self.inputsFrame = QGroupBox('Specifications')

        self.videoModelFrame = QGroupBox()
        self.dataCategories = QGroupBox()

        ''' Frame Size Policy'''
        self.videoFrame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.videoFrame.setAlignment(QtCore.Qt.AlignTop)

        ''' Frame layout '''

        self.videoFrameLayout = QVBoxLayout(self.videoFrame)

        self.inputsFrameLayout = QHBoxLayout(self.inputsFrame)
        self.videoModelFrameLayout = QFormLayout(self.videoModelFrame)
        self.dataCategoriesLayout = QFormLayout(self.dataCategories)

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
        # self.categoriesComboBoxPolicy = QtWidgets.QSizePolicy()
        # self.categoriesComboBoxPolicy.setControlType(QtWidgets.QSizePolicy.GroupBox)

        # self.categoriesComboBox.setSizePolicy(self.categoriesComboBoxPolicy)
  
        ''' Load video and models '''
        self.videoFilePath = QLabel('Path to Video')
        self.modelImportButton = QPushButton('Load Module')
        self.modelImportButton.clicked.connect(self.handleLoadModule)
        

        self.videoLineEdit = QLineEdit(self.inputsFrame)
        self.modelLineEdit = QLineEdit(self.inputsFrame)
        self.modelLineEdit.setText('ssd_mobilenet_v1_coco_2018_01_28')
        # self.modelLineEditPolicy = QtWidgets.QSizePolicy()
        # self.modelLineEditPolicy.setControlType(QtWidgets.QSizePolicy.LineEdit)
        # self.modelLineEdit.setSizePolicy(self.modelLineEditPolicy)

        self.videoLineEdit.setText(r"D:\OneDrive - Carleton University\Noah\Movies\Avatar - the last Airbender - Season 2 Complete - NXOR\S02E01 Avatar - The Last Airbender - Book 2 - Chapter 01 - The Avatar State.avi")

        ''' load Input Labels '''
        self.LoadLabels = QPushButton("Load Labels")
        self.LoadLabelsLineEdit = QLineEdit("mscoco_label_map.pbtxt")

        ''' stat Fields '''
        self.modelTimeLabel = QLabel('Time')
        self.modelTimeLineEdit = QLineEdit(self.statsFrame)
        self.modelTimePolicy = QtWidgets.QSizePolicy()
        self.modelTimePolicy.setControlType(QtWidgets.QSizePolicy.LineEdit)
        self.modelTimeLineEdit.setSizePolicy(self.modelTimePolicy)

        '''run button'''
        self.runButton = QPushButton('Run')
        self.runButtonPolicy = QtWidgets.QSizePolicy()
        self.runButtonPolicy.setControlType(QtWidgets.QSizePolicy.ButtonBox)
        self.runButton.setSizePolicy(self.runButtonPolicy)
        self.runButton.clicked.connect(self.handleRun)
        self.runButton.setEnabled(False)


        ''' video Widget'''

        self.videoWidget = QtWidgets.QGraphicsView(self.videoFrame)
        self.videoFrameLayout.addWidget(self.videoWidget)

        ''' Add Labels and line edits to group boxes '''
        # self.inputsFrameLayout.addWidget(self.info)
        self.videoModelFrameLayout.setWidget(0, QFormLayout.LabelRole, self.videoFilePath)
        self.videoModelFrameLayout.setWidget(0, QFormLayout.FieldRole, self.videoLineEdit)

        self.videoModelFrameLayout.setWidget(1, QFormLayout.LabelRole, self.modelImportButton)
        self.videoModelFrameLayout.setWidget(1, QFormLayout.FieldRole,  self.modelLineEdit)

        self.dataCategoriesLayout.setWidget(0, QFormLayout.LabelRole, self.categoriesLabel)
        self.dataCategoriesLayout.setWidget(0, QFormLayout.FieldRole,self.categoriesComboBox)

        self.dataCategoriesLayout.setWidget(1, QFormLayout.LabelRole, self.LoadLabels)
        self.dataCategoriesLayout.setWidget(1, QFormLayout.FieldRole, self.LoadLabelsLineEdit)

        self.dataCategoriesLayout.addWidget(self.runButton)

        self.statsFrameLayout.setWidget(0, QFormLayout.LabelRole, self.modelTimeLabel)
        self.statsFrameLayout.setWidget(0, QFormLayout.FieldRole, self.modelTimeLineEdit)

        self.inputsFrameLayout.addWidget(self.videoModelFrame)
        self.inputsFrameLayout.addWidget(self.dataCategories)
  
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
            self.categoriesComboBox.editFilter.disconnect(self.image_processor.editFilter)
            self.pixmapChanged.disconnect(self.image_processor.loadFrame)
            
            time.sleep(0.5) 
            

            self.image_processor.image_detector.session.close()
            while not self.image_processor.image_detector.session._closed:
                time.sleep(0.1)

            self.th.quit()

            
            # self.th = None

            self.handleLoadModule()
        
        
        self.th = QtCore.QThread()
        self.image_processor.moveToThread(self.th)
        
        self.pixmapChanged.connect(self.image_processor.loadFrame)
        self.image_processor.changePixmap.connect(self.setImage)
        self.categoriesComboBox.editFilter.connect(self.image_processor.editFilter)

        self.th.start()
        self.image_processor.setupVideoStream(self.videoLineEdit.text())

     


    def handleLoadModule(self):
        ''' 
            - Checks whether image_processor is currently running to determine whether or not is the fist run
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

        size = self.videoFrame.size()
        height = size.height()
        width = size.width()

        image = image.scaled(height - 20, width - 20, QtCore.Qt.KeepAspectRatio)
        # image = cv2.resize()
        # print(height, width, image.size().height(), image.size().width())
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


