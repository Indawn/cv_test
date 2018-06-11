#include "mainwindow.h"
#include "ui_mainwindow.h"
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
#include <QTime>
#include <iostream>
using namespace std;
using namespace cv;
using namespace zbar;



Mat MainWindow::QImage2cvMat(QImage image)
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

QImage MainWindow::cvMat2QImage(const cv::Mat& mat)
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

void MainWindow::scaledmat2label(Mat mat, QLabel* label)
{
    if(mat.data)
    {
      //  Mat image;
        pecentage=(double)label->width()/mat.cols;
        if((double)label->height()/mat.rows<pecentage)
            pecentage =(double)label->height()/mat.rows;
        cv::resize(mat,mat,cv::Size(0,0),pecentage,pecentage,INTER_AREA );
        label->clear();
        label->setPixmap(QPixmap::fromImage(cvMat2QImage(mat)));
        label->show();
    }
}

void MainWindow::Morphology_Operations(int morph_elem ,int morph_size , int morph_operator)
{
    Mat dst;
    int operation = morph_operator + 2;
    Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
    /// 运行指定形态学操作
    morphologyEx(mat_opened, dst, operation, element);
    // mat2pixmap(dst);
    mat_cut=dst;
   // scaledmat2label(mat_cut, uilabel_cut);
}

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
                GaussianBlur(src_gray, mat_cut, Size(ui->spinBox_canny_guss_ken->value(), ui->spinBox_canny_guss_ken->value()), 0, 0);
                //   blur(src_gray, mat_cut, Size(ui->spinBox_canny_guss_ken->value(), ui->spinBox_canny_guss_ken->value()));
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

}

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
        cout<<" { "<< contours.size()/*half_num/*<<"--"<<contour_numcontours.size()*/<<", "<<(int)max_length<<
              ", "<<(int)s<<
              ", "<<(int)avg_length<<", "<<(int)variance_length<<", ";
        mat_de[0] =contours.size();
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
        for(int nopp=0; nopp <10; nopp++)
            trainingData[(ui->spinBox_catgory->value()-1)*20+ui->spinBox_picno->value()][nopp]=mat_de[nopp];
      //  imshow("MinAreaCircle",imageContours1);
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
        cv::ml::SVM svm;
        svm.clear();
        svm.load( "D:\\钢板缺陷\\svm0519.xml");


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


void MainWindow::build_trainingdata()
{

    for(int cate = 0; cate < 6; cate++)
    {
        qsrand(QTime(0,0,0).secsTo(QTime::currentTime()));
        for(int picno = 0; picno < 20; picno++)
        {
            int test =qrand()%100;
            curr_picname = pic_dir+QString::number( cate+1, 10)+"\\"+ QString::number( cate+1, 10) +" (" + QString::number( test+1, 10)+").JPG";//得到用户选择的文件名
            onekey_canny();
            for(int nopp=0; nopp <10; nopp++)
                trainingData[cate*20+picno][nopp]=mat_de[nopp];
          //  cout<< test<<endl;
        }
    }
    /*
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

        }
        variance_length = sum_lee/contour_num;//contours.size();

        //   cout<<"共"<<contours.size()<<"条轮廓；"<<"最大周长为："<<max_length<<
        //         "最大面积为："<<s<<<<curr_picname.toLocal8Bit().constData()
        //        "；平均周长为："<<avg_length<<"；周长均方差为："<<variance_length<<endl;
        //  cout<<u8"轮空数";
        cout<<" { "<< contours.size()/*half_num/*<<"--"<<contour_numcontours.size()*<<", "<<(int)max_length<<
              ", "<<(int)s<<
              ", "<<(int)avg_length<<", "<<(int)variance_length<<", ";
        mat_de[0] =contours.size();
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
        for(int nopp=0; nopp <10; nopp++)
            trainingData[(ui->spinBox_catgory->value()-1)*20+ui->spinBox_picno->value()][nopp]=mat_de[nopp];
      //  imshow("MinAreaCircle",imageContours1);
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
            //mat2label_current(image2);
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
*/
}



void MainWindow::erosion(int erosion_elem ,int erosion_size ,int)
{
    int erosion_type;
    if (erosion_elem == 0){ erosion_type = MORPH_RECT; }
    else if (erosion_elem == 1){ erosion_type = MORPH_CROSS; }
    else if (erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
    Mat element = getStructuringElement(erosion_type,
                                        Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                        Point(erosion_size, erosion_size));
    cv::erode(mat_contour, mat_contour, element);    /// 腐蚀操作
    scaledmat2label(mat_contour, ui->label_contour);
}
void MainWindow::dilation(int dilation_elem ,int dilation_size ,int)
{
    int dilation_type;
    if (dilation_elem == 0){ dilation_type = MORPH_RECT; }
    else if (dilation_elem == 1){ dilation_type = MORPH_CROSS; }
    else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
    Mat element = getStructuringElement(dilation_type,
                                        Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                        Point(dilation_size, dilation_size));
   // Mat dilation_dst;
    cv::dilate(mat_contour, mat_contour, element);    /// 膨胀操作
    scaledmat2label(mat_contour, ui->label_contour);
}
void MainWindow::light(int alpha0, int beta ,int)
{
    double alpha =(double)alpha0/10;
    mat_cut = Mat::zeros(mat_opened.size(), mat_opened.type());
    /// 执行运算 new_image(i,j) = alpha*image(i,j) + beta
    for (int y = 0; y < mat_opened.rows; y++)
    {
        for (int x = 0; x < mat_opened.cols; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                mat_cut.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(alpha*(mat_opened.at<Vec3b>(y, x)[c]) + beta);
            }
        }
    }
    // mat2pixmap(new_image);
 //   mat_current=new_image;
    scaledmat2label(mat_cut, ui->label_cut);
}

