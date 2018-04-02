#-------------------------------------------------
#
# Project created by QtCreator 2017-12-18T13:11:04
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = cv_test
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
        mainwindow.cpp \
    dialog_erosion.cpp \
    dialog_erosion.cpp \
    main.cpp \
    mainwindow.cpp \
    dialog_erosion.cpp \
    main.cpp \
    mainwindow.cpp

HEADERS += \
        mainwindow.h \
    dialog_erosion.h \
    dialog_erosion.h \
    mainwindow.h \
    dialog_erosion.h \
    mainwindow.h

FORMS += \
        mainwindow.ui \
    dialog_erosion.ui \
    dialog_erosion.ui \
    dialog_erosion.ui \
    mainwindow.ui

INCLUDEPATH += C:\opencv2413\include
INCLUDEPATH += C:\zbar\include

CONFIG(debug, debug|release): {
LIBS += -LC:\zbar\lib \
-llibzbar-0

LIBS += -LC:\opencv2413\x86\vc14\lib \
-lopencv_core2413d \
-lopencv_imgproc2413d \
-lopencv_highgui2413d \
-lopencv_ml2413d \
-lopencv_video2413d \
-lopencv_features2d2413d \
-lopencv_calib3d2413d \
-lopencv_objdetect2413d \
-lopencv_contrib2413d \
-lopencv_legacy2413d \
-lopencv_flann2413d
} else:CONFIG(release, debug|release): {
LIBS += -LC:\zbar\lib \
-llibzbar-0

LIBS += -LC:\opencv2413\x86\vc14\lib \
-lopencv_core2413 \
-lopencv_imgproc2413 \
-lopencv_highgui2413 \
-lopencv_ml2413 \
-lopencv_video2413 \
-lopencv_features2d2413 \
-lopencv_calib3d2413 \
-lopencv_objdetect2413 \
-lopencv_contrib2413 \
-lopencv_legacy2413 \
-lopencv_flann2413
}

SUBDIRS += \
    cv_test.pro
