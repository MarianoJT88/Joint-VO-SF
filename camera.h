
#include <mrpt/utils.h>
#include <Eigen/Dense>
#include <OpenNI.h>

#include <iostream>
#include <fstream>

using namespace mrpt;
using namespace std;
using namespace Eigen;


class RGBD_Camera {
public:

    RGBD_Camera();

	unsigned int cam_mode;
	float max_distance;

    openni::Status		rc;
    openni::Device		device;
    openni::VideoMode	options;
    openni::VideoStream rgb,dimage;

    bool openCamera();
    void closeCamera();
    void loadFrame(MatrixXf &depth_wf, MatrixXf &color_wf);
    void waitForAutoExposure();
    void saveImages();

};




