/*
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

Vec3b RandomColor(int value);  //生成随机颜色函数

int main( int argc, char* argv[] )
{
    Mat image=imread("C:\\Users\\william\\Desktop\\钢板缺陷\\6\\1.bmp");    //载入RGB彩色图像
    imshow("Source Image",image);

    //灰度化，滤波，Canny边缘检测
    Mat imageGray;
    cvtColor(image,imageGray,CV_RGB2GRAY);//灰度转换
    GaussianBlur(imageGray,imageGray,Size(3,3),2);   //高斯滤波
    imshow("Gray Image",imageGray);
    Canny(imageGray,imageGray,80,150);
    imshow("Canny Image",imageGray);

    //查找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(imageGray,contours,hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE,Point());
    Mat imageContours=Mat::zeros(image.size(),CV_8UC1);  //轮廓
    Mat marks(image.size(),CV_32S);   //Opencv分水岭第二个矩阵参数
    marks=Scalar::all(0);
    int index = 0;
    int compCount = 0;
    for( ; index >= 0; index = hierarchy[index][0], compCount++ )
    {
        //对marks进行标记，对不同区域的轮廓进行编号，相当于设置注水点，有多少轮廓，就有多少注水点
        drawContours(marks, contours, index, Scalar::all(compCount+1), 1, 8, hierarchy);
        drawContours(imageContours,contours,index,Scalar(255),1,8,hierarchy);
    }

    //我们来看一下传入的矩阵marks里是什么东西
    Mat marksShows;
    convertScaleAbs(marks,marksShows);
    imshow("marksShow",marksShows);
    imshow("轮廓",imageContours);
    watershed(image,marks);

    //我们再来看一下分水岭算法之后的矩阵marks里是什么东西
    Mat afterWatershed;
    convertScaleAbs(marks,afterWatershed);
    imshow("After Watershed",afterWatershed);

    //对每一个区域进行颜色填充
    Mat PerspectiveImage=Mat::zeros(image.size(),CV_8UC3);
    for(int i=0;i<marks.rows;i++)
    {
        for(int j=0;j<marks.cols;j++)
        {
            int index=marks.at<int>(i,j);
            if(marks.at<int>(i,j)==-1)
            {
                PerspectiveImage.at<Vec3b>(i,j)=Vec3b(255,255,255);
            }
            else
            {
                PerspectiveImage.at<Vec3b>(i,j) =RandomColor(index);
            }
        }
    }
    imshow("After ColorFill",PerspectiveImage);

    //分割并填充颜色的结果跟原始图像融合
    Mat wshed;
    addWeighted(image,0.4,PerspectiveImage,0.6,0,wshed);
    imshow("AddWeighted Image",wshed);

    waitKey();
}

Vec3b RandomColor(int value)  //  <span style="line-height: 20.8px; font-family: sans-serif;">//生成随机颜色函数</span>
{
    value=value%255;  //生成0~255的随机数
    RNG rng;
    int aa=rng.uniform(0,value);
    int bb=rng.uniform(0,value);
    int cc=rng.uniform(0,value);
    return Vec3b(aa,bb,cc);
}

*/
#include <stdio.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <io.h>


using namespace std;
using namespace cv;

void getFiles( string path, vector<string>& files );

int main(int argc,char *argv[])

{



       CvSVM svm;
       svm.clear();
       svm.load( "C:\\Users\\william\\Desktop\\钢板缺陷\\svm0204.xml");


       float a[10] =   { 59, 105, 42, 29, 407, 158, 1152, 46, 6, 24 }  ;//{ 7, 686, 1973, 137, 50846, 115, 344, 203, 62, 106 };
       CvMat sampleMat1;
       cvInitMatHeader(&sampleMat1,1,10,CV_32FC1,a);
       float response1 = svm.predict(&sampleMat1);
  /*
       for (int i = 0;i < number;i++)
       {
           Mat inMat = imread(files[i].c_str());
           Mat p = inMat.reshape(1, 1);
           p.convertTo(p, CV_32FC1);
           int response = (int)svm.predict(p);
           if (response == 1)
            {
                result++;
            }
       }
       */

       cout<<response1<<endl;
     //  getchar();
       return  0;
   }
   void getFiles( string path, vector<string>& files )
   {
       long   hFile   =   0;
       struct _finddata_t fileinfo;
       string p;
       if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) !=  -1)
       {
           do
           {
               if((fileinfo.attrib &  _A_SUBDIR))
               {
                   if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
                       getFiles( p.assign(path).append("\\").append(fileinfo.name), files );
               }
               else
               {       files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
               }
           }while(_findnext(hFile, &fileinfo)  == 0);
           _findclose(hFile);
       }
   }

  /*

    Mat imageSource=imread("C:\\Users\\william\\Desktop\\钢板缺陷\\2\\112.bmp",0);
    imshow("Source Image",imageSource);
    Mat image;
    blur(imageSource,image,Size(3,3));
    threshold(image,image,0,255,CV_THRESH_OTSU);
    imshow("Threshold Image",image);

    //寻找最外层轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(image,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_NONE,Point());

    Mat imageContours=Mat::zeros(image.size(),CV_8UC1); //最小外接矩形画布
    Mat imageContours1=Mat::zeros(image.size(),CV_8UC1); //最小外结圆画布
    for(int i=0;i<contours.size();i++)
    {
        //绘制轮廓
        drawContours(imageContours,contours,i,Scalar(255),1,8,hierarchy);
        drawContours(imageContours1,contours,i,Scalar(255),1,8,hierarchy);


        //绘制轮廓的最小外结矩形
        RotatedRect rect=minAreaRect(contours[i]);
        Point2f P[4];
        rect.points(P);
        for(int j=0;j<=3;j++)
        {
            line(imageContours,P[j],P[(j+1)%4],Scalar(255),2);
        }

        //绘制轮廓的最小外结圆
        Point2f center; float radius;
        minEnclosingCircle(contours[i],center,radius);
        circle(imageContours1,center,radius,Scalar(255),2);

    }
    imshow("MinAreaRect",imageContours);
    imshow("MinAreaCircle",imageContours1);
    waitKey(0);
    return 0;
}

*/
/*
#include<iostream>
#include"opencv2\opencv.hpp"
#include<math.h>

using namespace std;
using namespace cv;

RNG g_rng(12345);
int g_nElementShape=MORPH_RECT;
Mat srcImage, dstImage;

void centerPoints(vector<Point>contour);

int main()
{
    srcImage = imread("C:\\Users\\william\\Desktop\\钢板缺陷\\Image__2018-01-24__19-45-22.bmp", 0);//读取文件，可以是文件目录
    if (!srcImage.data){ printf("图片读取错误！\n"); return false; }
    namedWindow("原图");
    imshow("原图", srcImage);
    //进行开运算平滑
    //namedWindow("【开运算/闭运算】", 1);
    Mat dstImage = Mat::zeros(srcImage.rows, srcImage.cols, CV_8UC3);

    Mat element = getStructuringElement(g_nElementShape,
        Size(5, 5), Point(-1, -1));
    morphologyEx(srcImage, dstImage, MORPH_OPEN, element, Point(-1, -1),2);
    imshow("【开运算/闭运算】", dstImage);

    vector<vector<Point>>contour;//用来储存轮廓
    vector<Vec4i>hierarchy;

    findContours(dstImage, contour, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (unsigned int i = 0; i < contour.size(); ++i)
    {
        centerPoints(contour[i]);
    }

    Mat drawing = Mat::zeros(dstImage.size(), CV_8UC3);
    for (int unsigned i = 0; i < contour.size(); i++)
    {
        Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));
        drawContours(drawing, contour, i, color, 1, 8, vector<Vec4i>(), 0, Point());

    }
    imshow("轮廓图", drawing);//画出轮廓线,在轮廓线上显示坐标
    //计算矩
    vector<Moments>mu(contour.size());
    for (unsigned int i = 0; i < contour.size(); i++)
    {
        mu[i] = moments(contour[i], false);
    }
    //计算矩中心
    vector<Point2f>mc(contour.size());
    for (unsigned int i = 0; i < contour.size(); i++)
    {
        mc[i] = Point2f(static_cast<float>(mu[i].m10 / mu[i].m00), static_cast<float>(mu[i].m01 / mu[i].m00));
    }
    for (unsigned int i = 0; i< contour.size(); ++i)
    {
        circle(drawing, mc[i], 5, Scalar(0, 0, 255), -1, 8, 0);
        rectangle(drawing, boundingRect(contour.at(i)), Scalar(0, 255, 0));
        char tam[100];
        sprintf_s(tam, "(%0.0f,%0.0f)", mc[i].x, mc[i].y);
        putText(drawing, tam, Point(mc[i].x, mc[i].y), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 0, 255), 1);
        //计算质心 坐标
        cout << "质心点坐标:" << "(" << (int)mc[i].x << "." << (int)mc[i].y << ")" << endl;
        //下标输出
    }
    namedWindow("Contours", WINDOW_AUTOSIZE);
    imshow("Contours", drawing);
    waitKey(0);
    return 0;
}
void centerPoints(vector<Point>contour)
{
    double factor = (contourArea(contour) * 4 * CV_PI) /
            (pow(arcLength(contour, true), 2));
    cout << "factor:" << factor << endl;  //计算出圆形度factor

}

*/


/*
//#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;

Mat src,src_gray;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

void thresh_callback(int, void* );

int main( int, char** argv )
{
  // 读图
  src = imread("C:\\Users\\william\\Desktop\\钢板缺陷\\6\\1.bmp", IMREAD_COLOR);
  if (src.empty())
      return -1;

  // 转化为灰度图
  cvtColor(src, src_gray, COLOR_BGR2GRAY );
  blur(src_gray, src_gray, Size(3,3) );

  // 显示
  namedWindow("Source", WINDOW_NORMAL );
  imshow( "Source", src );

  // 滑动条
  createTrackbar("Canny thresh:", "Source", &thresh, max_thresh, thresh_callback );

  // 回调函数
  thresh_callback( 0, 0 );

  waitKey(0);
}

// 回调函数
void thresh_callback(int, void* )
{
  Mat canny_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  // canny 边缘检测
  Canny(src_gray, canny_output, thresh, thresh*2, 3);

  // 寻找轮廓
  findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );



  // 画出轮廓
  for( size_t i = 0; i< contours.size(); i++ ) {
      Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3);
      drawContours( drawing, contours, (int)i, color, 1, 8, hierarchy, 0, Point() );
     // double l=arcLength(contours,true);//后面参数0表示轮廓不闭合，正数表示闭合；负数表示计算序列组成的面积；提取的角点以list形式时，用负数。
      cout<<"第"<<i<<"条轮廓"<<"周长为："<<"l"<<endl;
    //  ostringstream oss;
     // oss << "Contous" << i;
      std::string oss ="Contous";
      oss = oss + to_string(i);
      //const char* namess ="contous"+i;
      namedWindow( oss, WINDOW_NORMAL );
      imshow( oss, drawing );
  }

}
*/

/*
#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main(){

    Mat image = cvLoadImage("C:\\Users\\william\\Desktop\\钢板缺陷\\1\\99.jpg");
    Mat grayImage;
    cvtColor(image, grayImage, CV_BGR2GRAY);
    //转换为二值图
    Mat binaryImage;
    threshold(grayImage, binaryImage, 90, 255, CV_THRESH_BINARY);

    //二值图 这里进行了像素反转，因为一般我们用255白色表示前景（物体），用0黑色表示背景
    Mat reverseBinaryImage;
    bitwise_not(binaryImage, reverseBinaryImage);

    vector <vector<Point>>contours;
    findContours(reverseBinaryImage,
        contours,   //轮廓的数组
        CV_RETR_EXTERNAL,   //获取外轮廓
        CV_CHAIN_APPROX_NONE);  //获取每个轮廓的每个像素
    //在白色图像上绘制黑色轮廓
    Mat result(reverseBinaryImage.size(), CV_8U, Scalar(255));
    drawContours(result, contours,
        -1, //绘制所有轮廓
        Scalar(0),  //颜色为黑色
        2); //轮廓线的绘制宽度为2

    namedWindow("contours");
    imshow("contours", result);

    //移除过长或过短的轮廓
  /*  int cmin = 100; //最小轮廓长度
    int cmax = 1000;    //最大轮廓
    vector<vector<Point>>::const_iterator itc = contours.begin();
    while (itc!=contours.end())
    {
        if (itc->size() < cmin || itc->size() > cmax)
            itc = contours.erase(itc);
        else
            ++itc;
    }

    //在白色图像上绘制黑色轮廓
    Mat result_erase(binaryImage.size(), CV_8U, Scalar(255));
    drawContours(result_erase, contours,
        -1, //绘制所有轮廓
        Scalar(0),  //颜色为黑色
        2); //轮廓线的绘制宽度为2

    namedWindow("contours_erase");
    imshow("contours_erase", result_erase);
    waitKey(0);
    return 0;
}

*/

//*/

/*#include <QCoreApplication>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main()
{
    Mat srcImage = imread("C:\\Users\\william\\Desktop\\钢板缺陷\\2.bmp");
    imshow("【原图】", srcImage);

    //首先对图像进行空间的转换
    Mat grayImage;
    cvtColor(srcImage, grayImage, CV_BGR2GRAY);
    //对灰度图进行滤波
    GaussianBlur(grayImage, grayImage, Size(3, 3), 0, 0);
    imshow("【滤波后的图像】", grayImage);

    //为了得到二值图像，对灰度图进行边缘检测
    Mat cannyImage;
    Canny(grayImage, cannyImage, 128, 255, 3);
       namedWindow("【canny后的图像】",WINDOW_NORMAL);
    imshow("【canny后的图像】",cannyImage);

    //在得到的二值图像中寻找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(cannyImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

    //绘制轮廓
    for (int i = 0; i < (int)contours.size(); i++)
    {
        drawContours(cannyImage, contours, i, Scalar(255), 1, 8);
    }
    namedWindow("【处理后的图像】",WINDOW_NORMAL);
    imshow("【处理后的图像】", cannyImage);

    //计算轮廓的面积
    for (int i = 0; i < (int)contours.size(); i++)
    {
        double g_dConArea = contourArea(contours[i], true);
        cout << "【用轮廓面积计算函数计算出来的第" << i << "个轮廓的面积为：】" << g_dConArea << endl;
    }

    waitKey(0);

    return 0;
}
*/
