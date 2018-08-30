from PyQt5 import QtGui, QtCore, QtWidgets
import cv2
import tensorflow as tf
import numpy as np

import os, sys, time
from pathlib import Path
sys.path.append(os.path.abspath(r"../"))

try:
    from utils import label_map_util
    from utils import visualization_utils as vis_util
except ModuleNotFoundError as e:
    print(str(e))

class FrameProcessor(QtCore.QObject):


    changePixmap = QtCore.pyqtSignal(QtGui.QImage, float)

    def __init__(self, graphPath, labelMapFile):
        '''
        misc
        '''
        super().__init__()

        self.NUM_CLASSES = 90

        self.image_detector = ImageDetector(graphPath)
        self.image_detector.session =  tf.Session(graph=self.image_detector.detectionGraph)

        '''Categories'''
        self.PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

        ''' Image variables'''
        self.rgbImage = None
        self.convertToQtFormat = None
        self.p = None
        self.frame = None
        
        
        ''' Labeling and visualization utilities Utilities '''
        
        labelFilePath = Path(r"../data"+ os.sep + labelMapFile)
        self.label_map = label_map_util.load_labelmap(str(labelFilePath))
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        self.filter = {key:item["name"] for key,item in self.category_index.items()}
        

    def loadLabels(self):

        pass


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
        endTime =time.time()
        boxes, scores, classes, num_detections = self.filter_boxes(0.2, np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32), self.filter)

        #print('\n\nboxes\n\n', boxes, '\n\nscores\n\n', scores ,'\n\nclasses\n\n', classes)
        # print("\n\nboxes\n\n", boxes)

        try:  
            
            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                boxes,
                classes,
                scores,
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
        except IndexError:
            sys.exit()

        return image, endTime - startTime

 
    def loadFrame(self):

            ret, self.frame = self.cap.read()
            self.frame, time = self.runDetection(self.frame, self.image_detector.session)

            self.rgbImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.p =  QtGui.QImage(self.rgbImage.data, self.rgbImage.shape[1], self.rgbImage.shape[0], QtGui.QImage.Format_RGB888)
            
            self.changePixmap.emit(self.p, time)

            
    def filter_boxes(self, min_score, boxes, scores, classes, categories):

        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        num_detections = 0

        idxs = []
        for i in range(n):

            if classes[i] in categories.keys() and scores[i] >= min_score:
                idxs.append(i)
                num_detections += 1
        
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]

        # print(type(filtered_boxes), type(filtered_scores), type(filtered_classes))
        return filtered_boxes, filtered_scores, filtered_classes, num_detections

    def editFilter(self, filter):

        self.filter = filter
        print(self.filter)



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

