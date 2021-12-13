from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMessageBox, QLabel, QTextEdit, QLineEdit, \
QPushButton 

class ErrorWindow(QWidget):
    
    def __init__(self, errorInfo, icon):

        super().__init__()
        self.setWindowTitle('Error Message')
        self.icon = icon
        self.resize(300, 50)

        if self.icon: self.setWindowIcon(self.icon)

        self.verticalLayout = QVBoxLayout()
        self.setLayout(self.verticalLayout)
        self.verticalLayout.setAlignment(QtCore.Qt.AlignCenter)

        self.label = QLabel(errorInfo)
        self.okButton = QPushButton('Ok')
              
        self.verticalLayout.addWidget(self.label)
        self.verticalLayout.addWidget(self.okButton)

        self.okButton.clicked.connect(self.closeWindow)

    def closeWindow(self):

        self.close()

class InformationBox(QMessageBox):
    
    def __init__(self, message, buttons):
        ''' Consider passing in message type, error or question '''
        super().__init__()
        
