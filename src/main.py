import sys, time, ctypes, re, argparse
from videoWindow import VideoWindow
# from secondaryWindow import secondaryWindow

from PyQt5.QtWidgets import QCommonStyle, QApplication
# ICON = r'articles\atom.png'



if __name__ == "__main__":


    myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
    ''' https://stackoverflow.com/questions/1551605/how-to-set-applications-taskbar-icon-in-windows-7/1552105#1552105 '''
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    app = QApplication(sys.argv)

    #app.setStyle(QCommonStyle())

    window =VideoWindow()
    window.show()
    sys.exit(app.exec_())