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

    Mat mat_opened;

    Mat mat_current;
    Mat mat_cut;
    Mat mat_contour;
    Mat mat_last;




    Mat canny_mat;


    Mat mat_mid;



  //  vector<Point>  max_contours;

    double pecentage;
    void scaledmat2label(Mat mat, QLabel* label);
//    void mat2label_current(Mat mat);
//    void scaled_mat(Mat mat);
    void winload();

public slots:
    void erosion(int erosion_elem ,int erosion_size ,int );
    void dilation(int dilation_elem , int dilation_size ,int);
    void light(int alpha0, int beta ,int);
    void Morphology_Operations(int morph_elem ,int morph_size , int morph_operator);
    void curr_canny();
    void onekey_canny();

private slots:
    void on_actionOpen_triggered();

    void on_actionGrey_triggered();

    void on_actionBilateral_Blur_triggered();

    void on_actionHomogeneous_Blur_triggered();

    void on_actionmedianBlur_triggered();

    void on_actionGaussian_Blur_triggered();

    void on_actionExit_triggered();

    void on_actionCanny_triggered();

//    void on_actionFilter_triggered();

    void resizeEvent ( QResizeEvent * event );

    void on_actionErosion_triggered();

    void on_actionDilation_triggered();

    void on_actionLight_triggered();

    void on_actionMorphology_triggered();

    void on_actionzbar_triggered();

  //  void on_actionQRall_triggered();

 //   void on_actionmat_cut_triggered();

    void on_spinBox_bry_valueChanged(const QString &arg1);


    void on_pushButton_check_clicked();

    void on_Button_info_clicked();


//    void on_horizontalSlider_canny_valueChanged(int value);

//    void on_spinBox_canny_kernel_editingFinished();

    void on_spinBox_canny_kernel_valueChanged(int arg1);

    void on_spinBox_canny_low_valueChanged(int arg1);

    void on_spinBox_canny_high_valueChanged(int arg1);

//    void on_horizontalSlider_gassblur_actionTriggered(int action);

//    void on_horizontalSlider_gassblur_valueChanged(int value);

//    void on_spinBox_gassblur_valueChanged(const QString &arg1);

//    void on_spinBox_gassblur_valueChanged(int arg1);

    void on_pushButton_circumference_clicked();

    void on_pushButton_setcanny_clicked();

    void on_pushButton_largeaera_clicked();

    void on_pushButton_openmat_clicked();

    void on_pushButton_onekey_clicked();

/*    void on_spinBox_guss_ken_editingFinished();

    void on_spinBox_cannylow_editingFinished();

    void on_spinBox_cannymax_editingFinished();

    void on_spinBox_cannyken_editingFinished();

    void on_spinBox_enzhi_editingFinished();

    void on_spinBox_dia_editingFinished();

    void on_spinBox_ero_editingFinished();
    */

    void on_spinBox_guss_ken_valueChanged(int arg1);

    void on_spinBox_cannylow_valueChanged(int arg1);

    void on_spinBox_cannymax_valueChanged(int arg1);

    void on_spinBox_cannyken_valueChanged(int arg1);

    void on_spinBox_enzhi_valueChanged(int arg1);

    void on_spinBox_dia_valueChanged(int arg1);

    void on_spinBox_ero_valueChanged(int arg1);

    void on_spinBox_catgory_valueChanged(const QString &arg1);

    void on_spinBox_picno_valueChanged(const QString &arg1);

//    void on_output_mattrain_clicked();

    void on_spinBox_test_valueChanged(const QString &arg1);

    void on_pushButton_save_clicked();

    void on_actionoutput_triggered();

    void on_actionerBinaryzation_triggered();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
