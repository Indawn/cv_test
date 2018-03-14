#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "dialog_erosion.h"
#include "ui_dialog_erosion.h"
#include <iostream>
#include <QFileDialog>
#include <QLabel>
#include <qtextcodec.h>

using namespace std;
using namespace cv;
using namespace zbar;

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

void MainWindow::winload(){

}
/*
void MainWindow::resizeEvent ( QResizeEvent * event )
{
    scaled_mat(curr_mat);  
    statusBar()->removeWidget(aixLabel_pencentage);
    aixLabel_pencentage = new QLabel(QString::number(pecentage*100,10,2)+"%");
    statusBar()->addWidget(aixLabel_pencentage, 2);

}
*/
Mat QImage2cvMat(QImage image)
{
    cv::Mat mat;
    switch(image.format())
    {
    case QImage::Format_ARGB32:
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32_Premultiplied:
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.bits(), image.bytesPerLine());
        break;
    case QImage::Format_RGB888:
        mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.bits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, CV_BGR2RGB);
        break;
    case QImage::Format_Indexed8:
        mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.bits(), image.bytesPerLine());
        break;
    }
    return mat;
}

QImage cvMat2QImage(const cv::Mat& mat)
{
    // 8-bits unsigned, NO. OF CHANNELS = 1
    if(mat.type() == CV_8UC1)
    {
        QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
        // Set the color table (used to translate colour indexes to qRgb values)
        image.setColorCount(256);
        for(int i = 0; i < 256; i++)
        {
            image.setColor(i, qRgb(i, i, i));
        }
        // Copy input Mat
        uchar *pSrc = mat.data;
        for(int row = 0; row < mat.rows; row ++)
        {
            uchar *pDest = image.scanLine(row);
            memcpy(pDest, pSrc, mat.cols);
            pSrc += mat.step;
        }
        return image;
    }
    // 8-bits unsigned, NO. OF CHANNELS = 3
    else if(mat.type() == CV_8UC3)
    {
        // Copy input Mat
        const uchar *pSrc = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return image.rgbSwapped();
    }
    else if(mat.type() == CV_8UC4)
    {
        // Copy input Mat
        const uchar *pSrc = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
        return image.copy();
    }
    else
    {
        return QImage();
    }
}

void MainWindow::mat2label_pic(Mat mat)
{
    ui->label_pic->clear();
    ui->label_pic->setPixmap(QPixmap::fromImage(cvMat2QImage(mat)));
    ui->label_pic->show();
}

void MainWindow::scaled_mat(Mat mat)
{
    if(mat.data)
    {
        Mat image2;
        pecentage=(double)ui->label_pic->width()/mat.cols;
        if((double)ui->label_pic->height()/mat.rows<pecentage)
            pecentage =(double)ui->label_pic->height()/mat.rows;
        cv::resize(mat,image2,cv::Size(0,0),pecentage,pecentage,INTER_AREA );
        mat2label_pic(image2);
    }
}

void MainWindow::on_actionOpen_triggered()
{
    ui->textBrowser->clear();
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
            Mat image2;
            pecentage=(double)ui->label_origin->width()/opened_mat.cols;
            if((double)ui->label_origin->height()/opened_mat.rows<pecentage)
                pecentage =(double)ui->label_origin->height()/opened_mat.rows;
            cv::resize(opened_mat,image2,cv::Size(0,0),pecentage,pecentage,INTER_AREA );
            ui->label_origin->clear();
            ui->label_origin->setPixmap(QPixmap::fromImage(cvMat2QImage(image2)));
            ui->label_origin->show();
            //mat2label_pic(image2);
        }


        statusBar()->setVisible(true);
        scaled_mat(curr_mat);

        statusBar()->removeWidget(aixLabel);
        statusBar()->removeWidget(aixLabel_pencentage);
        aixLabel = new QLabel(curr_picname);
        aixLabel_pencentage = new QLabel(QString::number(pecentage*100,10,2)+"%");
        statusBar()->setStyleSheet(QString("QStatusBar::item{border: 0px}")); // 设置不显示label的边框
        statusBar()->addWidget(aixLabel, 1);
        statusBar()->addWidget(aixLabel_pencentage, 2);

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
        scaled_mat(curr_mat);
    }
}
