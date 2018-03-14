#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMainWindow>
#include <QResizeEvent>
#include <QMessageBox>
#include <QTextCodec>

#include <QCoreApplication>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <QLabel>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/ml/ml.hpp>
#include "zbar.h"


using namespace std;
using namespace cv;
using namespace zbar;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    QString curr_picname;

    QLabel *aixLabel;
    QLabel *aixLabel_pencentage;

    float mat_de[10];

    Mat opened_mat;
    Mat curr_mat;
    Mat erzhihua;
    Mat canny_mat;
    Mat mat_end;
        Mat mat_mid;
  //  vector<Point>  max_contours;

    double pecentage;

    void mat2label_pic(Mat mat);
    void scaled_mat(Mat mat);
    void winload();

public slots:
/*    void erosion(int erosion_elem ,int erosion_size ,int );
    void dilation(int dilation_elem , int dilation_size ,int);
    void light(int alpha0, int beta ,int);
    void Morphology_Operations(int morph_elem ,int morph_size , int morph_operator);
    void curr_canny();
    void onekey_canny();
*/
private slots:
    void on_actionOpen_triggered();

    void on_actionGrey_triggered();



private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
