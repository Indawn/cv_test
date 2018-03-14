#include "dialog_erosion.h"
#include "ui_dialog_erosion.h"
Dialog_erosion::Dialog_erosion(QWidget *parent) :
    QDialog(parent),
    uii(new Ui::Dialog_erosion)
{
    uii->setupUi(this);
}

Dialog_erosion::~Dialog_erosion()
{
    delete uii;
}

void Dialog_erosion::on_horizontalSlider_elem_valueChanged(int value)
{
    emit erosion_dilation(uii->horizontalSlider_elem->value(),uii->horizontalSlider_size->value(),0);
}

void Dialog_erosion::on_horizontalSlider_size_valueChanged(int value)
{
    emit erosion_dilation(uii->horizontalSlider_elem->value(),uii->horizontalSlider_size->value(),0);
}

void Dialog_erosion::on_horizontalSlider_3_actionTriggered(int action)
{

}

void Dialog_erosion::on_horizontalSlider_3_valueChanged(int value)
{
    emit erosion_dilation(uii->horizontalSlider_elem->value(),
                          uii->horizontalSlider_size->value(),uii->horizontalSlider_3->value());

}
