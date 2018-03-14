#ifndef DIALOG_EROSION_H
#define DIALOG_EROSION_H

#include <QDialog>
#include <QTextCodec>

namespace Ui {
class Dialog_erosion;
}

class Dialog_erosion : public QDialog
{
    Q_OBJECT

public:
    explicit Dialog_erosion(QWidget *parent = 0);
    ~Dialog_erosion();

private slots:
    void on_horizontalSlider_elem_valueChanged(int value);

    void on_horizontalSlider_size_valueChanged(int value);

    void on_horizontalSlider_3_actionTriggered(int action);

    void on_horizontalSlider_3_valueChanged(int value);

public:
    Ui::Dialog_erosion *uii;

signals:
    void erosion_dilation(int ,int ,int);//不需要实现
   // void SIGNAL_dilation(int ,int );
};

#endif // DIALOG_EROSION_H
