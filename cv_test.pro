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
    mainwindow.cpp \
    imgfunc.cpp

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
INCLUDEPATH += C:\opencv341p\include
INCLUDEPATH += C:\zbar\include
INCLUDEPATH += \
$$(PYLON_DEV_DIR)/include \
$$(PYLON_DEV_DIR)/include/pylon



CONFIG(debug, debug|release): {
LIBS += -LC:\zbar\lib \
-llibzbar-0

LIBS += \
-L$$(PYLON_DEV_DIR)/lib/win32

LIBS += -LC:\opencv341p\x86\vc14\lib \
-lopencv_aruco341d \
-lopencv_bgsegm341d \
-lopencv_bioinspired341d \
-lopencv_calib3d341d \
-lopencv_ccalib341d \
-lopencv_core341d \
-lopencv_datasets341d \
-lopencv_dnn341d \
-lopencv_dnn_objdetect341d \
-lopencv_dpm341d \
-lopencv_face341d \
-lopencv_features2d341d \
-lopencv_flann341d \
-lopencv_fuzzy341d \
-lopencv_hfs341d \
-lopencv_highgui341d \
-lopencv_imgcodecs341d \
-lopencv_imgproc341d \
-lopencv_img_hash341d \
-lopencv_line_descriptor341d \
-lopencv_ml341d \
-lopencv_objdetect341d \
-lopencv_optflow341d \
-lopencv_phase_unwrapping341d \
-lopencv_photo341d \
-lopencv_plot341d \
-lopencv_reg341d \
-lopencv_rgbd341d \
-lopencv_saliency341d \
-lopencv_shape341d \
-lopencv_stereo341d \
-lopencv_stitching341d \
-lopencv_structured_light341d \
-lopencv_superres341d \
-lopencv_surface_matching341d \
-lopencv_text341d \
-lopencv_tracking341d \
-lopencv_video341d \
-lopencv_videoio341d \
-lopencv_videostab341d \
-lopencv_xfeatures2d341d \
-lopencv_ximgproc341d \
-lopencv_xobjdetect341d \
-lopencv_xphoto341d
} else:CONFIG(release, debug|release): {
LIBS += -LC:\zbar\lib \
-llibzbar-0

LIBS += \
-L$$(PYLON_DEV_DIR)/lib/win32

LIBS += -LC:\opencv341p\x86\vc14\lib \
-lopencv_aruco341 \
-lopencv_bgsegm341 \
-lopencv_bioinspired341 \
-lopencv_calib3d341 \
-lopencv_ccalib341 \
-lopencv_core341 \
-lopencv_datasets341 \
-lopencv_dnn341 \
-lopencv_dnn_objdetect341 \
-lopencv_dpm341 \
-lopencv_face341 \
-lopencv_features2d341 \
-lopencv_flann341 \
-lopencv_fuzzy341 \
-lopencv_hfs341 \
-lopencv_highgui341 \
-lopencv_imgcodecs341 \
-lopencv_imgproc341 \
-lopencv_img_hash341 \
-lopencv_line_descriptor341 \
-lopencv_ml341 \
-lopencv_objdetect341 \
-lopencv_optflow341 \
-lopencv_phase_unwrapping341 \
-lopencv_photo341 \
-lopencv_plot341 \
-lopencv_reg341 \
-lopencv_rgbd341 \
-lopencv_saliency341 \
-lopencv_shape341 \
-lopencv_stereo341 \
-lopencv_stitching341 \
-lopencv_structured_light341 \
-lopencv_superres341 \
-lopencv_surface_matching341 \
-lopencv_text341 \
-lopencv_tracking341 \
-lopencv_video341 \
-lopencv_videoio341 \
-lopencv_videostab341 \
-lopencv_xfeatures2d341 \
-lopencv_ximgproc341 \
-lopencv_xobjdetect341 \
-lopencv_xphoto341
}

SUBDIRS += \
    cv_test.pro
