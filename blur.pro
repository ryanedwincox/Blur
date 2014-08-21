TEMPLATE = app
CONFIG += console
CONFIG -= qt

SOURCES += main.cpp

INCLUDEPATH += '/opt/AMDAPP/include'
LIBS += -lOpenCL

LIBS += `pkg-config opencv --libs`

OTHER_FILES += \
    cl/gaussian_blur.cl

