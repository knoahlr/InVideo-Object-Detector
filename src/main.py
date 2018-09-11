import sys, time, ctypes, re, argparse
from videoWindow import VideoWindow
# from secondaryWindow import secondaryWindow

from PyQt5 import QtCore
from PyQt5.QtWidgets import QCommonStyle, QApplication
# ICON = r'articles\atom.png'
import os

sys.path.append(os.path.abspath(r"../"))

from pathlib import Path

if __name__ == "__main__":
    
    if not Path(r"../logs").is_dir():
        os.makedirs(r"../logs")
        
    logFile = open(Path(r"../logs/mainLog.log"), 'w')
    sys.stdout = logFile

    myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
    ''' https://stackoverflow.com/questions/1551605/how-to-set-applications-taskbar-icon-in-windows-7/1552105#1552105 '''
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    app = QApplication(sys.argv)

    styleSheetFile = QtCore.QFile(r"../css/materialDesign.qss")
    styleSheetFile.open(QtCore.QFile.ReadOnly)
    
    styleSheetFileString = str(styleSheetFile.readAll(), "utf-8") 
    print(styleSheetFileString)

    app.setStyleSheet(styleSheetFileString)

    #app.setStyle(QCommonStyle())

    window =VideoWindow()
    window.show()
    sys.exit(app.exec_())