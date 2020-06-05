from PyQt5.QtWidgets import ( QMainWindow, QWidget, QStackedWidget,
                              QApplication, QPushButton, QGridLayout,
                              QFileDialog, QLabel )
from PyQt5.QtGui     import ( QIcon, QPixmap )
from PyQt5.QtCore    import ( QTimer )

import cv2
import numpy as np

from tensorflow.keras.models import load_model

class MainWindow( QMainWindow ):
    """MainWindow holds a widget
    that holds all other subwidgets.
    Subclasses QMainWindow"""

    def __init__( self ):
        """Constructor for mainwindow."""

        #Invoke QMainWindow.
        super( QMainWindow, self ).__init__( )

        #Filename=""
        self.filename = ""

        #Choose an initial size for the window.
        self.setGeometry( 0, 0, 500, 300 )
        #Set title.
        self.setWindowTitle( "Malware classification." )
        #Set icon.
        self.setWindowIcon ( QIcon( "icon.png" ) )

        #Initialize introductory widget.
        self.intro_wid = QWidget( )
        self.intro_wid . setObjectName( "intro_wid" )

        #Set background image.
        self.intro_wid . setStyleSheet( "QWidget#intro_wid{border-image:url('background.jpg');}" )

        #Initialize main_wid.
        self.main_wid = QWidget( )
        self.main_wid . setObjectName( "main_wid" )

        #Set background image.
        self.main_wid .setStyleSheet( "QWidget#main_wid{border-image:url('background.jpg');}" )

        #Label to display image.
        self.lab_im = QLabel( )
        self.lab_im . setScaledContents( True )

        #Push button to choose file.
        self.browse = QPushButton( "Browse" )
        self.browse . clicked.connect( self.select )

        #Push button to set image.
        self.set_image = QPushButton( "Set image" )
        self.set_image . clicked.connect( self.load_and_process )

        #Load model.
        self.ld_model = QPushButton( "Load model" )
        self.ld_model . clicked.connect( self.load_model )

        #A layout for main widget.
        self.lout = QGridLayout( self.main_wid )

        #Add widgets to layout.
        self.lout.addWidget( self.ld_model )
        self.lout.addWidget( self.browse )
        self.lout.addWidget( self.lab_im )
        self.lout.addWidget( self.set_image )

        #Create stacked widget.
        self.stack_wid = QStackedWidget( )
        #Add widgets to stacked widget.
        self.stack_wid . addWidget( self.intro_wid )
        self.stack_wid . addWidget( self.main_wid  )
        #After 3s change to main_wid.
        QTimer.singleShot( 3000, lambda : self.stack_wid.setCurrentIndex( 1 ) )

        #Set stacked widget as central widget.
        self.setCentralWidget( self.stack_wid )

    def load_and_process( self ):
        if self.filename != "":
            self.image = cv2.imread( self.filename )
            self.image = cv2.resize( self.image, ( 32, 32 ) )
            self.image = self.image / 255.0
            self.image = np.expand_dims( self.image, 0 )

    def select( self ):
        """Select the filename."""
        self.filename = QFileDialog.getOpenFileName( )[ 0 ]
        self.pmap     = QPixmap( self.filename )
        self.lab_im   . setPixmap( self.pmap )

    def load_model( self ):
        """Load model to memory."""
        self.model = load_model( "model/weights.18-0.39.hdf5" )

#Test.
if __name__ == "__main__":

    app = QApplication( [ ] )
    mw  = MainWindow  ( )
    mw  . show        ( )
    app . exec_       ( )
