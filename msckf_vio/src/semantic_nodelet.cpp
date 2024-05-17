#include <msckf_vio/semantic_nodelet.h>

namespace msckf_vio
{
void SemanticNodelet::onInit()
{
    semantic_ptr.reset(new Semantic(getPrivateNodeHandle()));
    if (!semantic_ptr->initialize())
    {
        ROS_ERROR("Cannot initialize Semantic...");
        return;
    }
    return;
}
PLUGINLIB_EXPORT_CLASS(msckf_vio::SemanticNodelet, nodelet::Nodelet);
}