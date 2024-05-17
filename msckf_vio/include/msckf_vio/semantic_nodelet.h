#ifndef SEMANTIC_NODELET_H
#define SEMANTIC_NODELET_H

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <msckf_vio/semantic.h>

namespace msckf_vio{
class SemanticNodelet : public nodelet::Nodelet{

public:
    SemanticNodelet() {return; }
    ~SemanticNodelet() {return; }

private:
    virtual void onInit();
    SemanticPtr semantic_ptr;
};
}

#endif