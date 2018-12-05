#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "dialog_erosion.h"
#include "ui_dialog_erosion.h"
#include <iostream>
#include <QFileDialog>
#include <QLabel>
#include <qtextcodec.h>
#include <fstream>
#include <QDir>
#include <QTextStream>
#include <opencv/ml.h>
#include <opencv/cxcore.h>
using namespace std;
using namespace cv;
using namespace zbar;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    pic_dir = u8"D:\\钢板缺陷\\";
   // trainingData = 0;
    for (int i = 0; i < 120; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            trainingData[i][j] = i/20 + 1;
            labeldata[i] = i/20 + 1;
        }

    }
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::winload(){

}

void MainWindow::on_actionExit_triggered()
{
    this->close();
}

void MainWindow::resizeEvent ( QResizeEvent * event )
{

    scaledmat2label(mat_current, ui->label_current);
    scaledmat2label(mat_cut, ui->label_cut);
    scaledmat2label(mat_contour, ui->label_contour);
    scaledmat2label(mat_last, ui->label_last);
    statusBar()->removeWidget(aixLabel_pencentage);
    aixLabel_pencentage = new QLabel(QString::number(pecentage*100,10,2)+"%");
    statusBar()->addWidget(aixLabel_pencentage, 2);
}


/*

void MainWindow::mat2label_current(Mat mat)
{
    ui->label_current->clear();
    ui->label_current->setPixmap(QPixmap::fromImage(cvMat2QImage(mat)));
    ui->label_current->show();
}

void MainWindow::scaled_mat(Mat mat)
{
    if(mat.data)
    {
        Mat image2;
        pecentage=(double)ui->label_current->width()/mat.cols;
        if((double)ui->label_current->height()/mat.rows<pecentage)
            pecentage =(double)ui->label_current->height()/mat.rows;
        cv::resize(mat,image2,cv::Size(0,0),pecentage,pecentage,INTER_AREA );
        mat2label_current(image2);
    }
}
*/
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
        mat_opened=imread(curr_picname.toLocal8Bit().constData());
        mat_current = mat_opened;
        mat_cut = mat_opened;
        mat_contour= mat_opened;
        if(mat_opened.data)
        {
            scaledmat2label(mat_current, ui->label_current);
        }

        statusBar()->setVisible(true);
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
    if(mat_opened.channels()==1){
        QMessageBox msg;
        msg.setText(u8"已是灰度图像");
        msg.exec();
    }
    else{
        cvtColor(mat_opened, mat_cut, CV_BGR2GRAY);
        scaledmat2label(mat_cut, ui->label_cut);
    }
}

void MainWindow::on_actionBilateral_Blur_triggered()
{
    int i=19;
    bilateralFilter(mat_opened, mat_cut, i, i * 2, i / 2);
    scaledmat2label(mat_cut, ui->label_cut);
}

void MainWindow::on_actionHomogeneous_Blur_triggered()
{
    blur(mat_opened, mat_cut, Size(19, 19), Point(-1, -1));
    scaledmat2label(mat_cut, ui->label_cut);
}

void MainWindow::on_actionmedianBlur_triggered()
{
    int i=19;
    medianBlur(mat_opened, mat_cut, i);
scaledmat2label(mat_cut, ui->label_cut);
}

void MainWindow::on_actionGaussian_Blur_triggered()
{
    int i=3;
    GaussianBlur(mat_opened, mat_cut, Size(i, i), 0, 0);
scaledmat2label(mat_cut, ui->label_cut);
}

void MainWindow::on_actionCanny_triggered()
{
    if(mat_opened.channels()==1)
    {
        QMessageBox msg;
        msg.setText(u8"已是灰度图像");
        msg.exec();
    }
    else
    {
        Mat src, src_gray;
        Mat dst, detected_edges;

        int edgeThresh = 1;
        int lowThreshold=7;
        int const max_lowThreshold = 100;
        int ratio = 3;
        int kernel_size = 3;
        //    char* window_name = "Edge Map";

        /**
        * @函数 CannyThreshold
        * @简介： trackbar 交互回调 - Canny阈值输入比例1:3
        */
        /** @函数 main */

        src = mat_opened;
        //    imshow("origin pic", src);

        /// 创建与src同类型和大小的矩阵(dst)
        dst.create(src.size(), src.type());

        /// 原图像转换为灰度图像
        cvtColor(src, src_gray, CV_BGR2GRAY);

        /// 创建显示窗口
        //    namedWindow(window_name, CV_WINDOW_AUTOSIZE);

        /// 创建trackbar
        //    createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);

        /// 显示图像
        //    CannyThreshold(0, 0);
        /// 使用 3x3内核降噪
        blur(src_gray, detected_edges, Size(3, 3));

        /// 运行Canny算子
        Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

        /// 使用 Canny算子输出边缘作为掩码显示原图像
        dst = Scalar::all(0);

        src.copyTo(dst, detected_edges);
        // imshow(window_name, dst);
        // mat2pixmap(dst);
        mat_cut=dst;
        scaledmat2label(mat_cut, ui->label_cut);
    }
}

void MainWindow::on_actionErosion_triggered()
{
    Dialog_erosion *q = new Dialog_erosion();
    q->uii->label_elem->setText("Erosion element");
    q->uii->label_3->clear();
    q->uii->horizontalSlider_3->close();
    q->show();
    connect(q, SIGNAL(erosion_dilation(int ,int ,int)), this,SLOT(erosion(int ,int ,int)));
}

void MainWindow::on_actionDilation_triggered()
{
    Dialog_erosion *q = new Dialog_erosion();
    q->uii->label_elem->setText("Dilation element");
    q->uii->label_3->clear();
    q->uii->horizontalSlider_3->close();
    q->show();
    connect(q, SIGNAL(erosion_dilation(int ,int ,int)), this,SLOT(dilation(int ,int ,int)));
}

void MainWindow::on_actionLight_triggered()
{
    Dialog_erosion *q = new Dialog_erosion();
    q->uii->label_elem->setText("Light");
    q->uii->horizontalSlider_elem->setMinimum(10);
    q->uii->horizontalSlider_elem->setMaximum(30);

    q->uii->horizontalSlider_size->setMinimum(-100);
    q->uii->horizontalSlider_size->setMaximum(100);
    q->uii->label_size->setText("Contrast");

    q->uii->label_3->clear();
    q->uii->horizontalSlider_3->close();

    q->show();
    connect(q, SIGNAL(erosion_dilation(int ,int ,int)), this,SLOT(light(int ,int ,int)));
}

void MainWindow::on_actionMorphology_triggered()
{
    Dialog_erosion *q = new Dialog_erosion();
    q->uii->label_elem->setText("morph_elem");
    q->uii->horizontalSlider_elem->setMinimum(0);
    q->uii->horizontalSlider_elem->setMaximum(2);

    q->uii->label_3->setText("morph operator");
    q->uii->horizontalSlider_3->setMinimum(0);
    q->uii->horizontalSlider_3->setMaximum(4);

    q->show();
    connect(q, SIGNAL(erosion_dilation(int ,int ,int)), this,SLOT(Morphology_Operations(int ,int ,int)));
}

void MainWindow::on_actionzbar_triggered()
{
    zbar::ImageScanner scanner;

    //ImageScanner scanner;
    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);
    Mat image = mat_opened;
    if (!image.data)
    {
        QMessageBox msg;
        msg.setText(u8"图片为空");
        msg.exec();
    }
    Mat imageGray;
    if(mat_opened.channels()!=1){
        cvtColor(image, imageGray, CV_RGB2GRAY);
    }
    else
        imageGray=image;
    int width = imageGray.cols;
    int height = imageGray.rows;
    uchar *raw = (uchar *)imageGray.data;
    Image imageZbar(width, height, "Y800", raw, width * height);
    scanner.scan(imageZbar); //扫描条码
    Image::SymbolIterator symbol = imageZbar.symbol_begin();
    if (imageZbar.symbol_begin() == imageZbar.symbol_end())
    {
        QMessageBox msg;
        msg.setText(u8"查询条码失败，请检查图片！");
        msg.exec();
    }
    for (; symbol != imageZbar.symbol_end(); ++symbol)
    {
        ui->textBrowser->setText(QString::fromStdString(symbol->get_data()));
        QMessageBox msg;
        msg.setText(u8"条码类型："+ QString::fromStdString(symbol->get_type_name())+"\n"+u8"条码内容："+QString::fromStdString(symbol->get_data()));
        //  msg.setText("  ");
        msg.exec();
    }
    imageZbar.set_data(NULL, 0);

}
/*
void MainWindow::on_actionQRall_triggered()
{
    ImageScanner scanner;
    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);
    Mat imageSource=mat_current;
    Mat image;
    Mat imageGray;
    imageSource.copyTo(image);
    GaussianBlur(image,image,Size(3,3),0);  //滤波
    if(mat_current.channels()!=1){
        cvtColor(image, image, CV_BGR2GRAY);
    }
    threshold(image,image,100,255,CV_THRESH_BINARY);  //二值化
    //imshow("二值化",image);
    Mat element=getStructuringElement(2,Size(5,5));  //膨胀腐蚀核
    //morphologyEx(image,image,MORPH_OPEN,element);
    for(int i=0;i<10;i++)
    {
        erode(image,image,element);
        i++;
    }
    //imshow("腐蚀s",image);
    Mat image1;
    erode(image,image1,element);
    image1=image-image1;
    //imshow("边界",image1);
    //寻找直线 边界定位也可以用findContours实现
    vector<Vec2f>lines;
    HoughLines(image1,lines,1,CV_PI/150,110,0,0);
    Mat DrawLine=Mat::zeros(image1.size(),CV_8UC1);
    for(int i=0;i<lines.size();i++)
    {
        float rho=lines[i][0];
        float theta=lines[i][1];
        Point pt1,pt2;
        double a=cos(theta),b=sin(theta);
        double x0=a*rho,y0=b*rho;
        pt1.x=cvRound(x0+1000*(-b));
        pt1.y=cvRound(y0+1000*a);
        pt2.x=cvRound(x0-1000*(-b));
        pt2.y=cvRound(y0-1000*a);
        line(DrawLine,pt1,pt2,Scalar(255),1,CV_AA);
    }
    //imshow("直线",DrawLine);
    Point2f P1[4];
    Point2f P2[4];
    vector<Point2f>corners;
    goodFeaturesToTrack(DrawLine,corners,4,0.1,10,Mat()); //角点检测
    for(int i=0;i<corners.size();i++)//corners.size()
    {
        circle(DrawLine,corners[i],3,Scalar(255),3);
        P1[i]=corners[i];
    }
    imshow("交点",DrawLine);//*
    int width0=P1[0].x-P1[1].x;
    int height0=P1[2].y-P1[0].y;//width;
    P2[3]=P1[3];
    P2[0]=Point2f(P2[3].x+width0,P2[3].y);
    P2[1]=Point2f(P2[3].x,P2[3].y+height0);
    P2[2]=Point2f(P2[0].x,P2[1].y);
    Mat elementTransf;
    elementTransf=  getAffineTransform(P1,P2);
    warpAffine(imageSource,imageSource,elementTransf,imageSource.size(),1,0,Scalar(255));
    //imshow("校正",imageSource);
    imageSource = imageSource(Rect(P2[3].x, P2[3].y, width0, height0));
    imageGray=imageSource;
    mat_current=imageGray;
    // mat2pixmap(imageGray);
     scaledmat2label(mat_current, ui->label_current);
    //imshow("裁剪",mat_current);
    cvtColor(imageGray, imageGray, CV_RGB2GRAY);
    int width = imageGray.cols;
    int height = imageGray.rows;
    uchar *raw = (uchar *)imageGray.data;
    Image imageZbar(width, height, "Y800", raw, width * height);
    scanner.scan(imageZbar);//扫描条码
    Image::SymbolIterator symbol = imageZbar.symbol_begin();
    if (imageZbar.symbol_begin() == imageZbar.symbol_end())
    {
        QMessageBox msg;
        msg.setText(u8"查询条码失败，请检查图片！");
        msg.exec();
    }
    for (; symbol != imageZbar.symbol_end(); ++symbol)
    {
        ui->textBrowser->setText(QString::fromStdString(symbol->get_data()));
        QMessageBox msg;
        msg.setText(u8"条码类型："+ QString::fromStdString(symbol->get_type_name())+"\n"+u8"条码内容："+QString::fromStdString(symbol->get_data()));
      //  msg.setText("  ");
        msg.exec();
    }
    imageZbar.set_data(NULL, 0);
}

void MainWindow::on_actionmat_cut_triggered()
{
    mat_current=mat_cut;
}
*/
void MainWindow::on_spinBox_bry_valueChanged(const QString &arg1)
{
    mat_opened.copyTo(mat_cut);
    GaussianBlur(mat_cut,mat_cut,Size(3,3),0);  //滤波
    scaledmat2label(mat_cut, ui->label_cut);
    cvtColor(mat_cut, mat_contour, CV_BGR2GRAY);
    scaledmat2label(mat_contour, ui->label_contour);
    threshold(mat_contour,mat_last,ui->spinBox_bry->value(),255,CV_THRESH_BINARY);  //二值化
    scaledmat2label(mat_last, ui->label_last);
}

void MainWindow::on_pushButton_check_clicked()
{
    mat_opened=mat_current;
}

void MainWindow::on_Button_info_clicked()
{
    imshow("mat_contour", mat_contour);
    Mat img = mat_current;
    Mat gray,color;
    if(img.channels()!=1)
        cvtColor(img, gray, CV_RGB2GRAY);

    Mat tmp_m, tmp_sd;
    double m = 0, sd = 0;

    m = mean(gray)[0];
    cout << "Mean: " << m << endl;

    meanStdDev(gray, tmp_m, tmp_sd);
    m = tmp_m.at<double>(0,0);
    sd = tmp_sd.at<double>(0,0);
    ui->textBrowser->setText("Mean: "+QString::number(m, 10, 4)+" , StdDev: "+QString::number(sd, 10, 4));
}
/*
void MainWindow::on_horizontalSlider_canny_valueChanged(int value)
{
    if(value%2==1)
        ui->spinBox_canny_kernel->setValue(value);
}
*/
void MainWindow::on_spinBox_canny_kernel_valueChanged(int arg1)
{
    curr_canny();
}
<<<<<<< HEAD
=======
void MainWindow::curr_canny()
{
    try
    {
        if(ui->spinBox_canny_kernel->value()%2==1)
        {
            if(mat_opened.channels()==1)
            {
                QMessageBox msg;
                msg.setText(u8"已是灰度图像");
                msg.exec();
            }
            else
            {
                Mat src, src_gray;
                Mat dst, detected_edges;

                int edgeThresh = 1;
                int lowThreshold= ui->spinBox_canny_low->value();;
                int const max_lowThreshold = 100;
               // int ratio = ui->spinBox_canny_kernel->value();
                int kernel_size = ui->spinBox_canny_kernel->value();
                //    char* window_name = "Edge Map";

                /**
                * @函数 CannyThreshold
                * @简介： trackbar 交互回调 - Canny阈值输入比例1:3
                */
                /** @函数 main */

                src = mat_opened;
                //    imshow("origin pic", src);



                /// 创建与src同类型和大小的矩阵(dst)
                dst.create(src.size(), src.type());

                /// 原图像转换为灰度图像
                cvtColor(src, src_gray, CV_BGR2GRAY);

                /// 创建显示窗口
                //    namedWindow(window_name, CV_WINDOW_AUTOSIZE);

                /// 创建trackbar
                //    createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);

                /// 显示图像
                //    CannyThreshold(0, 0);
                /// 使用 3x3内核降噪
                blur(src_gray, mat_cut, Size(3, 3));
                scaledmat2label(mat_cut, ui->label_cut);
                /// 运行Canny算子
                Canny(mat_cut, detected_edges, lowThreshold,  ui->spinBox_canny_high->value(), kernel_size);
                mat_contour = detected_edges;
                scaledmat2label(mat_contour, ui->label_contour);
                /// 使用 Canny算子输出边缘作为掩码显示原图像
                dst = Scalar::all(0);

                src.copyTo(dst, detected_edges);
                // imshow(window_name, dst);
                // mat2pixmap(dst);
                mat_last=dst;
                 scaledmat2label(mat_last, ui->label_last);
            }
        }
        throw  1;
    }
    catch(int i)
    {
        if(i==1)
        ui->textBrowser->setText("error");
    }
>>>>>>> parent of 2cd8d70... 0402


void MainWindow::on_spinBox_canny_low_valueChanged(int arg1)
{
    curr_canny();
}

void MainWindow::on_spinBox_canny_high_valueChanged(int arg1)
{
    curr_canny();
}

void MainWindow::on_pushButton_circumference_clicked()
{
    RNG rng(12345);
    vector<vector<Point> > contours;

    vector<Vec4i> hierarchy;
    findContours( canny_mat, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    double max_length = 0;
    double sum_length = 0;
    double avg_length = 0;
    int max_i =0;
    // 画出轮廓
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        Mat drawing = Mat::zeros( canny_mat.size(), CV_8UC3);
        drawContours( drawing, contours, (int)i, color, 1, 8, hierarchy, 0, Point() );
        // cv::arcLength()
        double l=arcLength(contours[i],true);//后面参数0表示轮廓不闭合，正数表示闭合；负数表示计算序列组成的面积；提取的角点以list形式时，用负数。
        double s = contourArea(contours[i],false);

        cout<<"第"<<i<<"条轮廓"<<"周长为："<<l<<"；面积为："<<s<<endl;
       // cout<<l<<endl;
       // cout<<"第"<<i<<"条轮廓"<<
        //  ostringstream oss;
        //  oss  <<"第"<<i<<"条轮廓"<<"周长为："<<l<<endl;
        std::string oss ="Contous";
        oss = oss + to_string(i) + "周长为："+to_string(l)+"；面积为："+to_string(s);
        //const char* namess ="contous"+i;
        //  ui->textBrowser->setText(oss);
        namedWindow( oss, WINDOW_NORMAL );
        imshow( oss, drawing );
        sum_length += l;
        if(l > max_length)
        {
            max_length = l;
            max_i = i;
        }

    }
    avg_length = sum_length / contours.size();

  //
    //max_contours = contours[max_i];
/*    Mat ssdrawing = Mat::zeros( mat_opened.size(), CV_8UC3);
     Scalar colorz = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    drawContours( ssdrawing, contours,max_i, colorz, 1, 8, hierarchy, 0, Point() );
    namedWindow( "LARGEST", WINDOW_NORMAL );
    imshow( "LARGEST", ssdrawing );
*/
    double sum_lee=0;
    double variance_length = 0;
    for( size_t i = 0; i< contours.size(); i++ )
    {
        double lee=arcLength(contours[i],true);
        sum_lee += (lee-avg_length)*(lee-avg_length);
    }
    variance_length = sum_lee/contours.size();


    cout<<"共"<<contours.size()<<"条轮廓；"<<"最大周长为："<<max_length<<
          "；平均周长为："<<avg_length<<"；周长均方差为："<<variance_length<<endl;

    Mat I = imread(curr_picname.toLocal8Bit().constData());
    int sum_grey = 0;
    int sum_greynum = 0;

    if(I.channels()!=1)
        cvtColor(I, I, CV_BGR2GRAY);
    for (int i=0;i<I.rows;i++)
    {
        for (int j=0;j<I.cols;j++)
        {
            if (pointPolygonTest(contours[max_i],cv::Point(j,i),false) == 1)
            {
               sum_greynum ++;
                sum_grey += I.at<uchar>(i,j);
            }
            else
            {
                 I.at<uchar>(i,j)=0;//对灰度图像素操作赋
            }

        }
    }

    double va_grey = 0;
    double avg_gery = sum_grey / sum_greynum;

    for (int i=0;i<I.rows;i++)
    {
        for (int j=0;j<I.cols;j++)
        {
            if (pointPolygonTest(contours[max_i],cv::Point(j,i),false) == 1)
            {
                //double lee=arcLength(contours[i],true);
                va_grey += (I.at<uchar>(i,j)-avg_gery)*(I.at<uchar>(i,j)-avg_gery);

            }
        }
    }
   double variance_grey = va_grey/sum_greynum;
   cout<<"最大轮廓内共"<<sum_greynum<<"个像素；"<<"灰度平均值为："<<avg_gery<<
         "；灰度方差为："<<variance_grey<<endl;



    namedWindow( "最大轮廓区域", WINDOW_NORMAL );
    imshow( "最大轮廓区域", I );


}

void MainWindow::on_pushButton_setcanny_clicked()
{
    canny_mat = mat_current;
}

void MainWindow::on_pushButton_largeaera_clicked()
{
    Mat I = mat_opened.clone();
 /*   cvtColor(I, I, CV_BGR2GRAY);
    for (int i=0;i<I.rows;i++)
    {
        for (int j=0;j<I.cols;j++)
        {
            if (pointPolygonTest(max_contours,cv::Point(i,j),false) == -1)
            {
                I.at<uchar>(i,j)=0;//对灰度图像素操作赋
            }
        }
    }
*/
    namedWindow( "最大轮廓区域", WINDOW_NORMAL );
    imshow( "最大轮廓区域", I );
   // return I;
  //  cv::Point p1(x,y);

}

void MainWindow::on_pushButton_openmat_clicked()
{
    namedWindow( "mat_opened", WINDOW_NORMAL );
    imshow( "mat_opened", mat_opened );
}

void MainWindow::on_pushButton_onekey_clicked()
{



}
<<<<<<< HEAD
=======
void MainWindow::onekey_canny()
{
    mat_opened=imread(curr_picname.toLocal8Bit().constData());
    try
    {
        mat_current = mat_opened;
        scaledmat2label(mat_current, ui->label_current);
        GaussianBlur(mat_opened, mat_cut, Size(ui->spinBox_guss_ken->value(), ui->spinBox_guss_ken->value()), 0, 0);
        scaledmat2label(mat_cut, ui->label_cut);

        Mat img = mat_opened;
        Mat src_gray = img;
        if(img.channels()!=1)
            cvtColor(img, src_gray, CV_RGB2GRAY);

        Mat tmp_m, tmp_sd;
        double m = 0, sd = 0;
        m = mean(src_gray)[0];
        meanStdDev(src_gray, tmp_m, tmp_sd);
        m = tmp_m.at<double>(0,0);
        sd = tmp_sd.at<double>(0,0);
        //   cout << "Mean: " << m << "tmp_sd: " << sd << endl;
        ui->textBrowser_info->setText("Mean: "+QString::number(m, 10, 4)+" , StdDev: "+QString::number(sd, 10, 4));
/*
        // canny--------------------------------------------------------        
        Mat src;
        Mat dst, detected_edges;

        int edgeThresh = 1;
        int lowThreshold= ui->spinBox_cannylow->value();
        int max_Threshold= ui->spinBox_cannymax->value();
        int const max_lowThreshold = 100;
        // int ratio = ui->spinBox_canny_kernel->value();
        int kernel_size =ui->spinBox_cannyken->value();
        //    char* window_name = "Edge Map";

        ///**
        * @函数 CannyThreshold
        * @简介： trackbar 交互回调 - Canny阈值输入比例1:3
        /
        /** @函数 main /

        src = mat_opened;
        //    imshow("origin pic", src);
        /// 创建与src同类型和大小的矩阵(dst)
        dst.create(src.size(), src.type());

        /// 创建显示窗口
        //    namedWindow(window_name, CV_WINDOW_AUTOSIZE);

        /// 创建trackbar
        //    createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);

        /// 显示图像
        //    CannyThreshold(0, 0);
        /// 使用 3x3内核降噪
        blur(src_gray, detected_edges, Size(3, 3));

        /// 运行Canny算子
        Canny(detected_edges, detected_edges, lowThreshold,  max_Threshold, kernel_size);
        canny_mat = detected_edges;

        /// 使用 Canny算子输出边缘作为掩码显示原图像
        dst = Scalar::all(0);

        src.copyTo(dst, detected_edges);
        // imshow(window_name, dst);
        // mat2pixmap(dst);
        mat_current=dst;
         scaledmat2label(mat_current, ui->label_current);


*/
        //---------------------------------------------------------------------------------------
        mat_cut.copyTo(mat_contour);
        scaledmat2label(mat_contour, ui->label_contour );
        //  GaussianBlur(mat_cut,mat_cut,Size(3,3),0);  //滤波
        if(mat_contour.channels() != 1)
            cvtColor(mat_contour, mat_contour, CV_BGR2GRAY);
        threshold(mat_contour,mat_contour,(int)m + (int)sd ,255,CV_THRESH_BINARY);  //二值化ui->spinBox_enzhi->value()
        // mat2pixmap(mat_cut);
   //     mat_current=mat_cut;
        //scaled_mat(mat_opened);
        dilation(0, ui->spinBox_dia->value(), 0);
        // mat_opened = mat_current;
        erosion(0, ui->spinBox_ero->value(), 0);
        // mat_opened = mat_current;
        canny_mat = mat_contour;




        RNG rng(12345);
        vector<vector<Point> > contours;

        vector<Vec4i> hierarchy;
        findContours( canny_mat, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0) );
        //contours.resize();
        double max_length = 0;
        double sum_length = 0;
        double avg_length = 0;
        int max_i =0; int contour_num = 0;
        // 画出轮廓
        for( size_t i = 0; i< contours.size(); i++ )
        {
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            Mat drawing = Mat::zeros( canny_mat.size(), CV_8UC3);
            drawContours( drawing, contours, (int)i, color, 1, 8, hierarchy, 0, Point() );
            // cv::arcLength()
            double l=arcLength(contours[i],true);//后面参数0表示轮廓不闭合，正数表示闭合；负数表示计算序列组成的面积；提取的角点以list形式时，用负数。
            if(contourArea(contours[i],false)>50)
            {
                sum_length += l;
                contour_num++;
            }
            if(l > max_length)
            {
                max_length = l;
                max_i = i;
            }

        }
       // scaledmat2label(drawing, ui->label_contour );
        avg_length = sum_length / contour_num;//contours.size();
        double s = contourArea(contours[max_i],false);
        //
        //   max_contours = contours[max_i];
        double sum_lee=0;
        double variance_length = 0;
        int half_num = 0;
        for( size_t i = 0; i< contours.size(); i++ )
        {

            if(contourArea(contours[i],false)>s/2)
            {

                double lee=arcLength(contours[i],true);
                sum_lee += (lee-avg_length)*(lee-avg_length);
                half_num++;
            }
            /*
            double lee=arcLength(contours[i],true);
            sum_lee += (lee-avg_length)*(lee-avg_length);
            if(contourArea(contours[i],false)>50)
            {
                contour_num++;
            }
*/
        }
        variance_length = sum_lee/contour_num;//contours.size();

        //   cout<<"共"<<contours.size()<<"条轮廓；"<<"最大周长为："<<max_length<<
        //         "最大面积为："<<s<<<<curr_picname.toLocal8Bit().constData()
        //        "；平均周长为："<<avg_length<<"；周长均方差为："<<variance_length<<endl;
        //  cout<<u8"轮空数";
        cout<<" { "<<half_num/*<<"--"<<contour_numcontours.size()*/<<", "<<(int)max_length<<
              ", "<<(int)s<<
              ", "<<(int)avg_length<<", "<<(int)variance_length<<", ";
        mat_de[0] = contours.size();
        mat_de[1] =(int)max_length;
        mat_de[2] =(int)s;
        mat_de[3] =(int)avg_length;
        mat_de[4] =(int)variance_length;
               // mat_de[0] =


        Mat I = imread(curr_picname.toLocal8Bit().constData());
        int sum_grey = 0;
        int sum_greynum = 0;


        if(I.channels()!=1)
            cvtColor(I, I, CV_BGR2GRAY);
        for (int i=0;i<I.rows;i++)
        {
            for (int j=0;j<I.cols;j++)
            {
                if (pointPolygonTest(contours[max_i],cv::Point(j,i),false) == 1)
                {
                    sum_greynum ++;
                    sum_grey += I.at<uchar>(i,j);
                }
                else
                {
                    I.at<uchar>(i,j)=0;//对灰度图像素操作赋
                }

            }
        }

        double va_grey = 0;
        double avg_gery = sum_grey / sum_greynum;

        for (int i=0;i<I.rows;i++)
        {
            for (int j=0;j<I.cols;j++)
            {
                if (pointPolygonTest(contours[max_i],cv::Point(j,i),false) == 1)
                {
                    //double lee=arcLength(contours[i],true);
                    va_grey += (I.at<uchar>(i,j)-avg_gery)*(I.at<uchar>(i,j)-avg_gery);

                }
            }
        }
        double variance_grey = va_grey/sum_greynum;
   //     cout<<"最大轮廓内共"<<sum_greynum<<"个像素；"<<"灰度平均值为："<<avg_gery<<
   //           "；灰度方差为："<<variance_grey<<endl;
        cout<<(int)avg_gery<<", "<<(int)variance_grey<<", ";
        mat_de[5] =(int)avg_gery;
         mat_de[6] =(int)variance_grey;

        Mat imageContours=Mat::zeros(mat_opened.size(),CV_8UC1); //最小外接矩形画布
        Mat imageContours1=Mat::zeros(mat_opened.size(),CV_8UC1); //最小外结圆画布
        //绘制轮廓的最小外结矩形
        RotatedRect rect=minAreaRect(contours[max_i]);
        Point2f P[4];
        rect.points(P);
        for(int j=0;j<=3;j++)
        {
            line(imageContours,P[j],P[(j+1)%4],Scalar(255),2);

        }
        double distance1, distance2;
        distance1 = powf((P[0].x - P[1%4].x),2) + powf((P[0].y - P[1%4].y),2);
        distance1 = sqrtf(distance1);
        distance2 = powf((P[1].x - P[2%4].x),2) + powf((P[1].y - P[2%4].y),2);
        distance2 = sqrtf(distance2);
        if(distance1<distance2)
        {   double temp_dis = distance1;
            distance1 = distance2;
            distance2 = temp_dis;
        }
     //  int int_distance1 = (int)distance1/100;
       //    int int_distance2 = (int)distance2/100;
        cout<<(int)distance1<<", "<<(int)distance2<<", ";
        mat_de[7] =(int)distance1;
        mat_de[8] =(int)distance2;

        //绘制轮廓的最小外结圆
        Point2f center; float radius;
        minEnclosingCircle(contours[max_i],center,radius);
        circle(imageContours1,center,radius,Scalar(255),2);

        cout<<(int)radius<<" },   "<<endl;
        mat_de[9] =(int)radius;
        imshow("MinAreaRect",imageContours);
        imshow("MinAreaCircle",imageContours1);
        if(I.data)
        {
            mat_last = I;
       /*     double pecentage1;
            Mat image2;
            pecentage1=(double)ui->label_last->width()/I.cols;
            if((double)ui->label_last->height()/I.rows<pecentage1)
                pecentage1 =(double)ui->label_last->height()/I.rows;
            cv::resize(I,image2,cv::Size(0,0),pecentage1,pecentage1,INTER_AREA );
            ui->label_last->clear();
            ui->label_last->setPixmap(QPixmap::fromImage(cvMat2QImage(image2)));
            ui->label_last->show();
            //mat2label_current(image2);*/
            scaledmat2label(mat_last, ui->label_last);
        }
        // namedWindow( "最大轮廓区域", WINDOW_NORMAL );
        // imshow( "最大轮廓区域", I );
        CvSVM svm;
        svm.clear();
        svm.load( "D:\\钢板缺陷\\svm0204_5.xml");


     //   float a[10] =   { 19, 105, 42, 29, 407, 158, 1152, 46, 6, 24 }  ;//{ 7, 686, 1973, 137, 50846, 115, 344, 203, 62, 106 };
        CvMat sampleMat1;
        cvInitMatHeader(&sampleMat1,1,10,CV_32FC1, mat_de);
        float response1 = svm.predict(&sampleMat1);

        ui->textBrowser_matde->setText(QString::number(response1,10,6));

        throw 1;
    }
    catch(int i)
    {
        if(i == 1)
        {
            ui->textBrowser->setText("错误"+ QString::number(i,10));
           }
    }
}
>>>>>>> parent of 2cd8d70... 0402

void MainWindow::on_spinBox_guss_ken_valueChanged(int arg1)
{
    onekey_canny();
}

void MainWindow::on_spinBox_cannylow_valueChanged(int arg1)
{
     onekey_canny();
}

void MainWindow::on_spinBox_cannymax_valueChanged(int arg1)
{
    onekey_canny();
}

void MainWindow::on_spinBox_cannyken_valueChanged(int arg1)
{
    onekey_canny();
}

void MainWindow::on_spinBox_enzhi_valueChanged(int arg1)
{
    onekey_canny();
}

void MainWindow::on_spinBox_dia_valueChanged(int arg1)
{
    onekey_canny();
}

void MainWindow::on_spinBox_ero_valueChanged(int arg1)
{
    onekey_canny();
   // delete onekey_canny();
}

void MainWindow::on_spinBox_catgory_valueChanged(const QString &arg1)
{
    cout<<endl<<"###缺陷"<<arg1.toLocal8Bit().constData()<<":"<<endl;
    curr_picname = pic_dir+arg1+"\\"+ arg1 +" (" + QString::number( ui->spinBox_picno->value(), 10)+").JPG";//得到用户选择的文件名
    mat_opened=imread(curr_picname.toLocal8Bit().constData());  
    if(mat_opened.empty())
        cout<<"empty:"<<curr_picname.toLocal8Bit().constData()<<endl;
    else
    {
        mat_current = mat_opened;
        scaledmat2label(mat_current, ui->label_current);
        onekey_canny();
    }
}

void MainWindow::on_spinBox_picno_valueChanged(const QString &arg1)
{
    curr_picname = pic_dir+ QString::number( ui->spinBox_catgory->value(), 10)+"\\"+QString::number( ui->spinBox_catgory->value(), 10)+" ("+ arg1 +").JPG";//得到用户选择的文件名
    mat_opened=imread(curr_picname.toLocal8Bit().constData());
    if(mat_opened.empty())
        cout<<"empty:"<<curr_picname.toLocal8Bit().constData()<<endl;
    else
    {
        mat_current = mat_opened;
        scaledmat2label(mat_current, ui->label_current);
        onekey_canny();
    }
}

void MainWindow::on_spinBox_test_valueChanged(const QString &arg1)
{
  //  cout<<endl<<"###缺陷"<<arg1.toLocal8Bit().constData()<<":"<<endl;
    curr_picname = pic_dir +u8"1\\1 ("+arg1+").JPG";//得到用户选择的文件名
    mat_opened=imread(curr_picname.toLocal8Bit().constData());

    if(mat_opened.empty())
        cout<<"empty:"<<curr_picname.toLocal8Bit().constData()<<endl;
    else
    {
        mat_current = mat_opened;
        scaledmat2label(mat_current, ui->label_current);
        onekey_canny();
    }
}

void MainWindow::on_pushButton_save_clicked()
{
     // imwrite("C:\\Users\\william\\Desktop\\save\\0.jpg", mat_mid);
      ui->label_cut->pixmap()->toImage().save("456.png","JPG");
}

void MainWindow::on_actionoutput_triggered()
{
    for(int i = 1; i < 7 ; i++)
    {
        cout<<" /* Category "<<i<<" */"<<endl;
        for(int j = 1; j < 21; j++)
        {
<<<<<<< HEAD
            curr_picname = pic_dir + QString::number( i, 10)+"\\"+ QString::number( j, 10) +".JPG";//得到用户选择的文件名
=======
            curr_picname = u8"D:\\钢板缺陷\\"+ QString::number( i, 10)+"\\"+ QString::number( j, 10) +".bmp";//得到用户选择的文件名
            //     curr_picname = u8"D:\\钢板缺陷\\old\\1 ("+arg1+").bmp";//得到用户选择的文件名
>>>>>>> parent of 2cd8d70... 0402
            mat_opened=imread(curr_picname.toLocal8Bit().constData());
            mat_current = mat_opened;
            if(mat_opened.empty())
                cout<<"empty:"<<curr_picname.toLocal8Bit().constData()<<endl;
            else
            {
                onekey_canny();
            }
        }
    }
}

void MainWindow::on_actionerBinaryzation_triggered()
{

}
<<<<<<< HEAD
void MainWindow::on_spinBox_cannyken_valueChanged(int arg1)
{
    curr_canny();
}
void MainWindow::on_spinBox_canny_guss_ken_valueChanged(int arg1)
{
    curr_canny();
}

void MainWindow::on_actionbmp2jpg_triggered()
{
    int i = 3;
        cout<<" /* Category "<<i<<" */"<<endl;
        for(int j = 1; j < 21; j++)
        {
            QString picname = u8"D:\\钢板缺陷\\"+ QString::number( i, 10)+"\\"+QString::number( i, 10)+" ("+ QString::number( j, 10)+")";
            QString bmp_picname = picname +".bmp";//得到用户选择的文件名
            Mat bmp_mat=imread(bmp_picname.toLocal8Bit().constData());

            if(bmp_mat.empty())
                cout<<"empty:"<<bmp_picname.toLocal8Bit().constData()<<endl;
            else
            {
                QString jpg_picname = picname +".JPG";//得到用户选择的文件名
                imwrite(jpg_picname.toLocal8Bit().constData(), bmp_mat);;
            }
        }

}

void MainWindow::on_actionSVMtraining_triggered()
{
    Mat labeldata_Mat (120, 1, CV_32FC1, labeldata);
    Mat trainingData_Mat (120, 10, CV_32FC1, trainingData);
    // step 2:
    //训练参数设定
    CvSVMParams params;
    params.svm_type = CvSVM::C_SVC;                 //SVM类型
    params.kernel_type = CvSVM::LINEAR  ;             //核函数的类型
//    params.p = 5e-3;
//    params.C = 1;
    //SVM训练过程的终止条件, max_iter:最大迭代次数  epsilon:结果的精确性
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON );

    // step 3:
    //启动训练过程
    CvSVM SVM;
    SVM.train( trainingData_Mat, labeldata_Mat, Mat(),Mat(), params);
    SVM.save((pic_dir+"svm0519.xml").toLocal8Bit().constData());


}

void MainWindow::on_actionOutput_traindata_triggered()
{
    cout<<"Output trainingdata to trainingdata.txt"<<endl;
   // cout<<trainingData<<endl;
    QFile file(pic_dir+"\\trainingdata.txt");
     if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
        return;
     QTextStream out(&file);
     for (int i = 0; i <120; i++)
     {
         for(int j = 0; j < 10; j++)
         {
             out << " " << trainingData[i][j];
         }
         out<<endl;
     }

     file.close();
}

void MainWindow::on_actionOutput_labeldata_triggered()
{
    cout<<"Output labeldata to labeldata.txt "<<endl;
    QFile file(pic_dir+"\\labeldata.txt");
     if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
        return;
     QTextStream out(&file);
     for (int i = 0; i <120; i++)
     {
        out << " " << labeldata[i];
        out<<endl;
     }
     file.close();
}

void MainWindow::on_actionbuild_traindata_triggered()
{
    build_trainingdata();
}

void MainWindow::on_actionSVMtesting_triggered()
{
    cv::ml::SVM svm;
    svm.clear();
    svm.load( "D:\\钢板缺陷\\svm0519.xml");
    int correct_num[6] = {0, 0, 0, 0, 0, 0};
    for(int cate = 0; cate < 6; cate++)
    {
       // int correct_num = 0;
      //  qsrand(QTime(0,0,0).secsTo(QTime::currentTime()));
        for(int picno = 0; picno < 100; picno++)
        {
         //   int test =qrand()%100;
            curr_picname = pic_dir+QString::number( cate+1, 10)+"\\"+ QString::number( cate+1, 10) +" (" + QString::number( picno+1, 10)+").JPG";//得到用户选择的文件名
            onekey_canny();

            CvMat sampleMat1;
            cvInitMatHeader(&sampleMat1,1,10,CV_32FC1, mat_de);
            float response1 = svm.predict(&sampleMat1);
            if(response1 == cate+1)
                correct_num[cate]++;

         //   for(int nopp=0; nopp <10; nopp++)
           //     trainingData[cate*20+picno][nopp]=mat_de[nopp];
          //  cout<< test<<endl;
        }

    }
    for(int cate = 0; cate < 6; cate++)
    {
        cout<<"cate"<<cate+1<<":"<<correct_num[cate]<<endl;

    }
}

void MainWindow::on_actionBPtraining_triggered()
{
        CvANN_MLP bp;
        // Set up BPNetwork's parameters
        CvANN_MLP_TrainParams params;
        params.train_method=CvANN_MLP_TrainParams::BACKPROP;
        params.bp_dw_scale=0.1;
        params.bp_moment_scale=0.1;
        //params.train_method=CvANN_MLP_TrainParams::RPROP;
        //params.rp_dw0 = 0.1;
        //params.rp_dw_plus = 1.2;
        //params.rp_dw_minus = 0.5;
        //params.rp_dw_min = FLT_EPSILON;
        //params.rp_dw_max = 50.;

        // Set up training data
      //  float labels[3][5] = {{0,0,0,0,0},{1,1,1,1,1},{0,0,0,0,0}};
        Mat labelsMat(120, 1, CV_32FC1, labeldata);

      //  float trainingData[3][5] = { {1,2,3,4,5},{111,112,113,114,115}, {21,22,23,24,25} };
        Mat trainingDataMat(120, 10, CV_32FC1, trainingData);
        Mat layerSizes=(Mat_<int>(1,5) << 5,2,2,2,5);
        bp.create(layerSizes,CvANN_MLP::SIGMOID_SYM);//CvANN_MLP::SIGMOID_SYM
                                                   //CvANN_MLP::GAUSSIAN
                                                   //CvANN_MLP::IDENTITY
        bp.train(trainingDataMat, labelsMat, Mat(),Mat(), params);
/*
        int correct_num[6] = {0, 0, 0, 0, 0, 0};
        for(int cate = 0; cate < 6; cate++)
        {
           // int correct_num = 0;
          //  qsrand(QTime(0,0,0).secsTo(QTime::currentTime()));
            for(int picno = 0; picno < 100; picno++)
            {
             //   int test =qrand()%100;
                curr_picname = pic_dir+QString::number( cate+1, 10)+"\\"+ QString::number( cate+1, 10) +" (" + QString::number( picno+1, 10)+").JPG";//得到用户选择的文件名
                onekey_canny();

                CvMat sampleMat1;
                cvInitMatHeader(&sampleMat1,1,10,CV_32FC1, mat_de);
                float response1 = bp.predict(&sampleMat1);
                 bp.predict(sampleMat,responseMat);
                if(responseMat == cate+1)
                    correct_num[cate]++;

             //   for(int nopp=0; nopp <10; nopp++)
               //     trainingData[cate*20+picno][nopp]=mat_de[nopp];
              //  cout<< test<<endl;
            }

        }
        for(int cate = 0; cate < 6; cate++)
        {
            cout<<"cate"<<cate+1<<":"<<correct_num[cate]<<endl;

        }
*/
}

void MainWindow::on_actionBP_testing_triggered()
{
 /*   int correct_num[6] = {0, 0, 0, 0, 0, 0};
    for(int cate = 0; cate < 6; cate++)
    {
       // int correct_num = 0;
      //  qsrand(QTime(0,0,0).secsTo(QTime::currentTime()));
        for(int picno = 0; picno < 100; picno++)
        {
         //   int test =qrand()%100;
            curr_picname = pic_dir+QString::number( cate+1, 10)+"\\"+ QString::number( cate+1, 10) +" (" + QString::number( picno+1, 10)+").JPG";//得到用户选择的文件名
            onekey_canny();

            CvMat sampleMat1;
            cvInitMatHeader(&sampleMat1,1,10,CV_32FC1, mat_de);
            float response1 = svm.predict(&sampleMat1);
            if(response1 == cate+1)
                correct_num[cate]++;

         //   for(int nopp=0; nopp <10; nopp++)
           //     trainingData[cate*20+picno][nopp]=mat_de[nopp];
          //  cout<< test<<endl;
        }

    }
    for(int cate = 0; cate < 6; cate++)
    {
        cout<<"cate"<<cate+1<<":"<<correct_num[cate]<<endl;

    }*/
}
=======
>>>>>>> parent of 2cd8d70... 0402
