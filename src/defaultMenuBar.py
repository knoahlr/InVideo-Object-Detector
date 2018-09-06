from PyQt5 import QtCore, QtGui, QtWidgets

class opacitySlider(QtWidgets.QSlider):
    """
    Args:
        param initTuple: Tuple containing slider minimum, maximum and singleStepSize i.e (1, 100, 1)
    """
    def __init__(self, initTuple):

        super().__init__()
        self.setOrientation(QtCore.Qt.Horizontal)
        self.setRange(initTuple[0], initTuple[1])
        self.setSingleStep(initTuple[2])
        self.setValue(self.maximum())


class DefaultMenuBar(QtWidgets.QMenuBar):

    closeWindow = QtCore.pyqtSignal()

    def __init__(self, parentWindow):

        super().__init__()
        
        ''' Menu Properties '''
        self.font = QtGui.QFont()
        self.font.setBold(True)
        self.font.setFamily("Comic Sans MS")
        self.setFont(self.font)

        ''' Signals '''
        self.closeWindow.connect(parentWindow.close)

        
        ''' Adding Basic Menus '''
        self.fileMenu = self.addMenu('&File')
        self.editMenu = self.addMenu('&Edit')
        self.toolsMenu = self.addMenu("&Tools")

        ''' Adding common actions to menu '''
        
        "File Menu"
        self.fileMenu.addAction("Exit",self.closeWindow)


        """ Tools Menu """
        """ opacity sub menu added to tools menu """
            
        self.opacityMenu = self.toolsMenu.addMenu("&Opacity")
        self.opacityWidget = QtWidgets.QWidgetAction(self.toolsMenu)
        self.opacitySlider = opacitySlider((1,100,1))
        self.opacitySlider.valueChanged.connect(parentWindow.setWindowOpacity)
        self.opacityWidget.setDefaultWidget(self.opacitySlider)
        self.opacityMenu.addAction(self.opacityWidget)

 