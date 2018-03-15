#ifndef CVFUN_H
#define CVFUN_H

#include <opencv2/core/core.hpp>
#include <QImage>
using namespace cv;

Mat rgb2grey(Mat mat);
QImage cvMat2QImage(const cv::Mat& mat);






#endif // CVFUN_H



/*
//#include <QObject>
//#include <opencv2/imgproc/imgproc.hpp>

#include <QCoreApplication>
#include <QMainWindow>

using namespace std;
class cvfun
{
public:
    cvfun();
    ~cvfun();
    static Mat rgb2grey(Mat mat);
    static QImage cvMat2QImage(const cv::Mat& mat);
};
*/
