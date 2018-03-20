#include "cvfun.h"

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "dialog_erosion.h"
#include "ui_dialog_erosion.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QTextCodec>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <QResizeEvent>
using namespace std;
//using namespace cv;
//using namespace zbar;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::resizeEvent ( QResizeEvent * event )
{
    scaledmat2label(curr_mat, ui->label_pic);
    statusBar()->removeWidget(aixLabel_pencentage);
    aixLabel_pencentage = new QLabel(QString::number(pecentage*100,10,2)+"%");
    statusBar()->addWidget(aixLabel_pencentage, 2);

}

void MainWindow::scaledmat2label(Mat mat, QLabel* label)
{
    if(mat.data)
    {
      //  Mat image;
        pecentage=(double)ui->label_pic->width()/mat.cols;
        if((double)ui->label_pic->height()/mat.rows<pecentage)
            pecentage =(double)ui->label_pic->height()/mat.rows;
        cv::resize(mat,mat,cv::Size(0,0),pecentage,pecentage,INTER_AREA );
        label->clear();
        label->setPixmap(QPixmap::fromImage(cvMat2QImage(mat)));
        label->show();
    }
}

void MainWindow::on_actionOpen_triggered()
{
//    ui->textBrowser->clear();
    QFileDialog *fileDialog = new QFileDialog(this);//创建一个QFileDialog对象，构造函数中的参数可以有所添加。
    fileDialog->setWindowTitle(tr("Open"));//设置文件保存对话框的标题
    fileDialog->setAcceptMode(QFileDialog::AcceptOpen);//设置文件对话框为保存模式
    fileDialog->setFileMode(QFileDialog::AnyFile);//设置文件对话框弹出的时候显示任何文件，不论是文件夹还是文件
    fileDialog->setViewMode(QFileDialog::Detail);//文件以详细的形式显示，显示文件名，大小，创建日期等信息；
    //还有另一种形式QFileDialog::List，这个只是把文件的文件名以列表的形式显示出来
    fileDialog->setGeometry(10,30,300,200);//设置文件对话框的显示位置
    fileDialog->setDirectory(".");//设置文件对话框打开时初始打开的位置
    // fileDialog->setFilter(tr("Image Files(*.jpg *.png)"));//设置文件类型过滤器
    if(fileDialog->exec() == QDialog::Accepted)//注意使用的是QFileDialog::Accepted或者QDialog::Accepted,不是QFileDialog::Accept
    {
        curr_picname = fileDialog->selectedFiles()[0];//得到用户选择的文件名
        opened_mat=imread(curr_picname.toLocal8Bit().constData());
        curr_mat = opened_mat;
        if(opened_mat.data)
        {
            scaledmat2label(opened_mat, ui->label_pic);
            statusBar()->setVisible(true);
            scaledmat2label(curr_mat, ui->label_pic);

            statusBar()->removeWidget(aixLabel);
            statusBar()->removeWidget(aixLabel_pencentage);
            aixLabel = new QLabel(curr_picname);
            aixLabel_pencentage = new QLabel(QString::number(pecentage*100,10,2)+"%");
            statusBar()->setStyleSheet(QString("QStatusBar::item{border: 0px}")); // 设置不显示label的边框
            statusBar()->addWidget(aixLabel, 1);
            statusBar()->addWidget(aixLabel_pencentage, 2);
        }
    }
}

void MainWindow::on_actionGrey_triggered()
{
    if(opened_mat.channels()==1){
        QMessageBox msg;
        msg.setText(u8"已是灰度图像");
        msg.exec();
    }
    else{
        cvtColor(opened_mat, curr_mat, CV_BGR2GRAY);
        //  // mat2pixmap(curr_mat);
        scaledmat2label(curr_mat, ui->label_pic);
    }
}

void MainWindow::on_actioninput_triggered()
{
    Mat image = imread("D:\\1.jpg");
    imshow("cvf", image);
    image = rgb2grey(image);
    ui->label_pic->clear();
    ui->label_pic->setPixmap(QPixmap::fromImage(cvMat2QImage(image)));
    ui->label_pic->show();
}
