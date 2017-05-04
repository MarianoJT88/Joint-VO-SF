
#include <Eigen/Core>
#include <OpenNI.h>
#include <iostream>


class RGBD_Camera {
public:

    RGBD_Camera(unsigned int res_factor);

	unsigned int cam_mode;
	float max_distance;

    openni::Status		rc;
    openni::Device		device;
    openni::VideoMode	options;
    openni::VideoStream rgb,dimage;

    bool openCamera();
    void closeCamera();
    void loadFrame(Eigen::MatrixXf &depth_wf, Eigen::MatrixXf &color_wf);
    void disableAutoExposureAndWhiteBalance();
};




