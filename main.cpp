#include "mainwindow.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
  w.winload();
    w.show();
 //   w.winload();
    return a.exec();
}
