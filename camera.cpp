
#include "camera.h"

#include <PS1080.h>

RGBD_Camera::RGBD_Camera()
{
    cam_mode = 1; // (1 - 640 x 480, 2 - 320 x 240)
	max_distance = 4.f;

}


bool RGBD_Camera::openCamera()
{
    rc = openni::STATUS_OK;

    const char* deviceURI = openni::ANY_DEVICE;

    rc = openni::OpenNI::initialize();

    printf("Opening camera...\n %s\n", openni::OpenNI::getExtendedError());
    rc = device.open(deviceURI);
    if (rc != openni::STATUS_OK)
    {
        printf("Device open failed:\n%s\n", openni::OpenNI::getExtendedError());
        openni::OpenNI::shutdown();
        return 1;
    }

    //								Create RGB and Depth channels
    //========================================================================================
    rc = dimage.create(device, openni::SENSOR_DEPTH);
    rc = rgb.create(device, openni::SENSOR_COLOR);


    //                            Configure some properties (resolution)
    //========================================================================================
    rc = device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
    device.setDepthColorSyncEnabled(true);


    options = rgb.getVideoMode();
    if (cam_mode == 1)
        options.setResolution(640,480);
    else
        options.setResolution(320,240);

    rc = rgb.setVideoMode(options);
    rc = rgb.setMirroringEnabled(false);

    options = dimage.getVideoMode();
    if (cam_mode == 1)
        options.setResolution(640,480);
    else
        options.setResolution(320,240);

    rc = dimage.setVideoMode(options);
    rc = dimage.setMirroringEnabled(false);
    rc = dimage.setProperty(XN_STREAM_PROPERTY_GMC_MODE, false);

//    //Turn off autoExposure
//    rgb.getCameraSettings()->setAutoExposureEnabled(false);
//    printf("Auto Exposure: %s \n", rgb.getCameraSettings()->getAutoExposureEnabled() ? "ON" : "OFF");

//    //Turn off White balance
//    rgb.getCameraSettings()->setAutoWhiteBalanceEnabled(false);
//    printf("Auto White balance: %s \n", rgb.getCameraSettings()->getAutoWhiteBalanceEnabled() ? "ON" : "OFF");

    //Check final resolution
    options = rgb.getVideoMode();
    printf("Resolution (%d, %d) \n", options.getResolutionX(), options.getResolutionY());

    //								Start channels
    //===================================================================================
    rc = dimage.start();
    if (rc != openni::STATUS_OK)
    {
        printf("Couldn't start depth stream:\n%s\n", openni::OpenNI::getExtendedError());
        dimage.destroy();
    }

    rc = rgb.start();
    if (rc != openni::STATUS_OK)
    {
        printf("Couldn't start rgb stream:\n%s\n", openni::OpenNI::getExtendedError());
        rgb.destroy();
    }

    if (!dimage.isValid() || !rgb.isValid())
    {
        printf("Camera: No valid streams. Exiting\n");
        openni::OpenNI::shutdown();
        return 1;
    }

    return 0;
}

void RGBD_Camera::closeCamera()
{
    rgb.destroy();
    openni::OpenNI::shutdown();
}

void RGBD_Camera::loadFrame(MatrixXf &depth_wf, MatrixXf &color_wf)
{
    const float norm_factor = 1.f/255.f;
    openni::VideoFrameRef framergb, framed;
    dimage.readFrame(&framed);
    rgb.readFrame(&framergb);

    const int height = framergb.getHeight();
    const int width = framergb.getWidth();

    if ((framed.getWidth() != framergb.getWidth()) || (framed.getHeight() != framergb.getHeight()))
        cout << endl << "The RGB and the depth frames don't have the same size.";

    else
    {
        //Read new frame
        const openni::DepthPixel* pDepthRow = (const openni::DepthPixel*)framed.getData();
        const openni::RGB888Pixel* pRgbRow = (const openni::RGB888Pixel*)framergb.getData();
        int rowSize = framergb.getStrideInBytes() / sizeof(openni::RGB888Pixel);
		const float max_dist_mm = 1000.f*max_distance;

        for (int yc = height-1; yc >= 0; --yc)
        {
            const openni::RGB888Pixel* pRgb = pRgbRow;
            const openni::DepthPixel* pDepth = pDepthRow;
            for (int xc = width-1; xc >= 0; --xc, ++pRgb, ++pDepth)
            {
                color_wf(yc,xc) = norm_factor*(0.299*pRgb->r + 0.587*pRgb->g + 0.114*pRgb->b);
                depth_wf(yc,xc) = 0.001f*(*pDepth)*(*pDepth < max_dist_mm);
            }
            pRgbRow += rowSize;
            pDepthRow += rowSize;
        }
    }
}

void RGBD_Camera::waitForAutoExposure()
{
    MatrixXf aux_depth, aux_color;
	if (cam_mode == 1)	{aux_depth.resize(480,640); aux_color.resize(480,640);}
	else				{aux_depth.resize(240,320); aux_color.resize(240,320);}
	
	for (unsigned int i=1; i<=10; i++)
        loadFrame(aux_depth, aux_color);

    //Turn off autoExposure
    rgb.getCameraSettings()->setAutoExposureEnabled(false);
    printf("Auto Exposure: %s \n", rgb.getCameraSettings()->getAutoExposureEnabled() ? "ON" : "OFF");

    //Turn off White balance
    rgb.getCameraSettings()->setAutoWhiteBalanceEnabled(false);
    printf("Auto White balance: %s \n", rgb.getCameraSettings()->getAutoWhiteBalanceEnabled() ? "ON" : "OFF");
}

void RGBD_Camera::saveImages()
{
//    char	filename[100];

//    //Save colour and depth old frames
//    sprintf(filename, "/home/mariano/Desktop/intensity0.png");
//    cv::Mat intensity(height, width, CV_8U);
//    for (unsigned int v=0; v<height; v++)
//        for (unsigned int u=0; u<width; u++)
//            intensity.at<unsigned char>(v,u) = round(255.f*color_old[0](height-1-v,u));
//    cv::imwrite(filename, intensity);

//    sprintf(filename, "/home/mariano/Desktop/depth0.png");
//    cv::Mat depth_cv(height, width, CV_16U);
//    for (unsigned int v=0; v<height; v++)
//        for (unsigned int u=0; u<width; u++)
//            depth_cv.at<unsigned short>(v,u) = round(5000.f*depth_old[0](height-1-v,u));
//    cv::imwrite(filename, depth_cv);

//    //Save colour and depth artificial frames
//    sprintf(filename, "/home/mariano/Desktop/intensity1.png");
//    for (unsigned int v=0; v<height; v++)
//        for (unsigned int u=0; u<width; u++)
//            intensity.at<unsigned char>(v,u) = round(255.f*color[0](height-1-v,u));
//    cv::imwrite(filename, intensity);

//    sprintf(filename, "/home/mariano/Desktop/depth1.png");
//    for (unsigned int v=0; v<height; v++)
//        for (unsigned int u=0; u<width; u++)
//            depth_cv.at<unsigned short>(v,u) = round(5000.f*depth[0](height-1-v,u));
//    cv::imwrite(filename, depth_cv);
}

