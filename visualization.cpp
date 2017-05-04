
#include "joint_vo_sf.h"
//#include "dvo/opencv_ext.hpp"


using namespace mrpt;
using namespace mrpt::poses;
using namespace mrpt::opengl;
using namespace mrpt::utils;
using namespace std;
using namespace Eigen;


void VO_SF::initializeSceneCamera()
{
    CPose3D pose_show(0,0,1.5,0,0,0);
	CPose3D pose_segm_pc(0,3.5,1.5,0,0,0);

	global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 10000000;
	window.resize(1000,900);
	window.setPos(900,0);
	window.setCameraZoom(16);
    window.setCameraAzimuthDeg(180);
	window.setCameraElevationDeg(90);
	window.setCameraPointingToPoint(0,0,1);
	//window.getDefaultViewport()->setCustomBackgroundColor(TColorf(1,1,1));

	scene = window.get3DSceneAndLock();

	//Grid (ground)
	opengl::CGridPlaneXYPtr ground = opengl::CGridPlaneXY::Create();
	scene->insert( ground );

	//Reference
	opengl::CSetOfObjectsPtr reference = opengl::stock_objects::CornerXYZ();
	scene->insert( reference );

	//Reference tk
	opengl::CSetOfObjectsPtr referencetk = opengl::stock_objects::CornerXYZ();
	referencetk->setScale(0.2);
	scene->insert( referencetk );

	//Kinect points
	opengl::CPointCloudColouredPtr kin_points = opengl::CPointCloudColoured::Create();
	kin_points->setColor(1,0,0);
	kin_points->setPointSize(4);
	kin_points->enablePointSmooth(1);
    kin_points->setPose(pose_show);
	scene->insert( kin_points );

	//Selected point
    opengl::CPointCloudPtr sel_point = opengl::CPointCloud::Create();
    sel_point->setColor(0,0.6,0);
    sel_point->setPointSize(15.f);
    sel_point->setPose(pose_show);
    scene->insert( sel_point );

    //Scene Flow (includes initial point cloud)
    opengl::CVectorField3DPtr sf = opengl::CVectorField3D::Create();
    sf->setPointSize(3.0f);
    sf->setLineWidth(2.0f);
    sf->setPointColor(1,0,0);
    sf->setVectorFieldColor(0,0,1);
    sf->enableAntiAliasing();
    sf->setPose(pose_show);
    scene->insert( sf );

	//Labels
	COpenGLViewportPtr vp_labels = scene->createViewport("labels");
    vp_labels->setViewportPosition(0.7,0.05,240,180);

	COpenGLViewportPtr vp_backg = scene->createViewport("background");
    vp_backg->setViewportPosition(0.1,0.05,240,180);


	window.unlockAccess3DScene();
	window.repaint();
}

void VO_SF::initializeSceneDatasetVideo()
{

	global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 10000000;
	window.resize(1600,800);
	window.setPos(300,0);
	window.setCameraZoom(8);
    window.setCameraAzimuthDeg(180);
	window.setCameraElevationDeg(90);
	window.setCameraPointingToPoint(0,0,1);
	window.getDefaultViewport()->setCustomBackgroundColor(TColorf(1,1,1));

	scene = window.get3DSceneAndLock();

	//Reference gt
	opengl::CSetOfObjectsPtr reference_gt = opengl::stock_objects::CornerXYZ();
	reference_gt->setScale(0.15);
	scene->insert( reference_gt );

	//Reference est
	opengl::CSetOfObjectsPtr reference_est = opengl::stock_objects::CornerXYZ();
	reference_est->setScale(0.15);
	scene->insert( reference_est );

	//Segmented points (original)
	opengl::CPointCloudColouredPtr seg_points = opengl::CPointCloudColoured::Create();
	seg_points->setColor(1,0,0);
	seg_points->setPointSize(2);
	seg_points->enablePointSmooth(1);
	scene->insert( seg_points );

	//Estimated trajectory
	opengl::CSetOfLinesPtr estimated_traj = opengl::CSetOfLines::Create();
	estimated_traj->setColor(1.f, 0.f, 0.f);
	estimated_traj->setLineWidth(5.f);
	scene->insert(estimated_traj);

	//GT trajectory
	opengl::CSetOfLinesPtr gt_traj = opengl::CSetOfLines::Create();
	gt_traj->setColor(0.f, 0.f, 0.f);
	gt_traj->setLineWidth(5.f);
	scene->insert(gt_traj);

	//Point cloud for the scene flow
	opengl::CPointCloudPtr sf_points = opengl::CPointCloud::Create();
	sf_points->setColor(0,1,1);
	sf_points->setPointSize(3);
	sf_points->enablePointSmooth(1);
	scene->insert( sf_points );

    //Scene Flow (includes initial point cloud)
    opengl::CVectorField3DPtr sf = opengl::CVectorField3D::Create();
    sf->setPointSize(3.0f);
    sf->setLineWidth(2.0f);
    sf->setPointColor(1,0,0);
    sf->setVectorFieldColor(0,0,1);
    sf->enableAntiAliasing();
    scene->insert( sf );


	//Labels
	COpenGLViewportPtr vp_image = scene->createViewport("image");
    vp_image->setViewportPosition(0.775,0.675,320,240);

	COpenGLViewportPtr vp_labels = scene->createViewport("labels");
    vp_labels->setViewportPosition(0.775,0.350,320,240);

	COpenGLViewportPtr vp_backg = scene->createViewport("background");
	vp_backg->setViewportPosition(0.775,0.025,320,240);


	window.unlockAccess3DScene();
	window.repaint();
}

void VO_SF::initializeSceneSequencesVideo()
{
	const unsigned int repr_level = round(log2(width/cols));

	global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 10000000;
	window.resize(1600,800);
	window.setPos(300,0);
	window.setCameraZoom(8);
    window.setCameraAzimuthDeg(180);
	window.setCameraElevationDeg(90);
	window.setCameraPointingToPoint(0,0,1);
	window.getDefaultViewport()->setCustomBackgroundColor(TColorf(1,1,1));
	window.captureImagesStart();
	scene = window.get3DSceneAndLock();


	//Segmented points (original)
	opengl::CPointCloudColouredPtr seg_points = opengl::CPointCloudColoured::Create();
	seg_points->setColor(1,0,0);
	seg_points->setPointSize(2);
	seg_points->enablePointSmooth(1);
	scene->insert( seg_points );


    //Scene Flow (includes initial point cloud)
    opengl::CVectorField3DPtr sf = opengl::CVectorField3D::Create();
    sf->setPointSize(3.f);
    sf->setLineWidth(1.f);
    sf->setPointColor(1,0,0);
    sf->setVectorFieldColor(0,0,1);
    sf->enableAntiAliasing();
	sf->enableShowPoints(false);
	sf->enableColorFromModule(true);
	sf->setMaxSpeedForColor(0.05f);
	sf->setMotionFieldColormap(0,0,1,1,0,0);
    scene->insert( sf );

	//Labels
	COpenGLViewportPtr vp_image = scene->createViewport("image");
    vp_image->setViewportPosition(0.775,0.675,320,240);

	COpenGLViewportPtr vp_labels = scene->createViewport("labels");
    vp_labels->setViewportPosition(0.775,0.350,320,240);

	COpenGLViewportPtr vp_backg = scene->createViewport("background");
	vp_backg->setViewportPosition(0.775,0.025,320,240);


	window.unlockAccess3DScene();
	window.repaint();
}


void VO_SF::updateSceneCamera(bool &clean_sf)
{
	const unsigned int repr_level = round(log2(width/cols));
	CImage image;

	//Refs
	const MatrixXf &depth_ref = depth[repr_level];
	const MatrixXf &yy_ref = yy[repr_level];
	const MatrixXf &xx_ref = xx[repr_level];

	const MatrixXf &depth_old_ref = depth_old[repr_level];
	const MatrixXf &yy_old_ref = yy_old[repr_level];
	const MatrixXf &xx_old_ref = xx_old[repr_level];
	
	scene = window.get3DSceneAndLock();

	opengl::CPointCloudColouredPtr kin_points = scene->getByClass<CPointCloudColoured>(0);
	kin_points->clear();
	for (unsigned int u=0; u<cols; u++)
		for (unsigned int v=0; v<rows; v++)
            if (depth_ref(v,u) != 0.f)
                kin_points->push_back(depth_ref(v,u), xx_ref(v,u), yy_ref(v,u), 0.f, 1.f, 1.f);


    //Scene flow
    if (clean_sf == true)
    {
        motionfield[0].assign(0.f);
        motionfield[1].assign(0.f);
        motionfield[2].assign(0.f);
		clean_sf = false;
    }

	opengl::CVectorField3DPtr sf = scene->getByClass<CVectorField3D>(0);
    sf->setPointCoordinates(depth_old[repr_level], xx_old[repr_level], yy_old[repr_level]);	
	sf->setVectorField(motionfield[0], motionfield[1], motionfield[2]);

	//Labels
	COpenGLViewportPtr vp_labels = scene->getViewport("labels");
    image.setFromRGBMatrices(olabels_image[0], olabels_image[1], olabels_image[2], true);
	//image.setFromMatrix(color_wf, true);
    image.flipVertical();
    vp_labels->setImageView(image);

	//Background
	COpenGLViewportPtr vp_backg = scene->getViewport("background");
    image.setFromRGBMatrices(backg_image[0], backg_image[1], backg_image[2], true);
    image.flipVertical();
    vp_backg->setImageView(image);
			

	window.unlockAccess3DScene();
	window.repaint();
}

void VO_SF::updateSceneDatasetVideo(const CPose3D &gt, const CPose3D &gt_old)
{
	const unsigned int repr_level = round(log2(width/cols));
	CImage image;

	//Refs
	const MatrixXf &depth_ref = depth[repr_level];
	const MatrixXf &yy_ref = yy[repr_level];
	const MatrixXf &xx_ref = xx[repr_level];
	
	scene = window.get3DSceneAndLock();

	//Cameras
	opengl::CSetOfObjectsPtr reference_gt = scene->getByClass<CSetOfObjects>(0);
	reference_gt->setPose(gt);
	scene->insert( reference_gt );

	opengl::CSetOfObjectsPtr reference_est = scene->getByClass<CSetOfObjects>(1);
	reference_est->setPose(cam_pose);
	scene->insert( reference_est );

	//Segmented points
	opengl::CPointCloudColouredPtr seg_points = scene->getByClass<CPointCloudColoured>(0);
	seg_points->clear();
	seg_points->setPose(gt);
	for (unsigned int y=0; y<cols; y++)
		for (unsigned int z=0; z<rows; z++)
            if (depth_ref(z,y) != 0.f)
				seg_points->push_back(depth_ref(z,y), xx_ref(z,y), yy_ref(z,y), im_r(z,y), im_g(z,y), im_b(z,y));


	//Trajectories
	opengl::CSetOfLinesPtr estimated_traj = scene->getByClass<CSetOfLines>(0);
	estimated_traj->appendLine(cam_pose[0], cam_pose[1], cam_pose[2], cam_oldpose[0], cam_oldpose[1], cam_oldpose[2]);

	opengl::CSetOfLinesPtr gt_traj = scene->getByClass<CSetOfLines>(1);
	gt_traj->appendLine(gt[0], gt[1], gt[2], gt_old[0], gt_old[1], gt_old[2]);



	//Scene flow
	//opengl::CPointCloudPtr points_sf = scene->getByClass<CPointCloud>(0);
	//CPose3D aux_pose_sf = dataset.gt_pose + CPose3D(4,4,0,0,0,0);
	//points_sf->clear();
	//points_sf->setPose(aux_pose_sf);
	//for (unsigned int y=0; y<cols; y++)
	//	for (unsigned int z=0; z<rows; z++)
 //           if (depth[repr_level](z,y) > 0.f)
 //               points_sf->insertPoint(depth[repr_level](z,y), xx[repr_level](z,y), yy[repr_level](z,y));

	opengl::CVectorField3DPtr sf = scene->getByClass<CVectorField3D>(0);
	//sf->setPose(aux_pose_sf);
 //   sf->setPointCoordinates(depth_old[repr_level], xx_old[repr_level], yy_old[repr_level]);
 //   sf->setVectorField(motionfield[0], motionfield[1], motionfield[2]);


	//Image
	COpenGLViewportPtr vp_image = scene->getViewport("image");
    image.setFromRGBMatrices(im_r_old, im_g_old, im_b_old, true);
    image.flipVertical();
	image.flipHorizontal();
    vp_image->setImageView(image);

	//Labels
	COpenGLViewportPtr vp_labels = scene->getViewport("labels");
    image.setFromRGBMatrices(olabels_image[0], olabels_image[1], olabels_image[2], true);
    image.flipVertical();
	image.flipHorizontal();
    vp_labels->setImageView(image);

	//Background
	COpenGLViewportPtr vp_backg = scene->getViewport("background");
	image.setFromRGBMatrices(backg_image[0], backg_image[1], backg_image[2], true);
    image.flipVertical();
	image.flipHorizontal();
    vp_backg->setImageView(image);
			
	window.unlockAccess3DScene();
	window.repaint();

}

void VO_SF::updateSceneSequencesVideo()
{
	const unsigned int repr_level = round(log2(width/cols));
	CImage image;

	//Refs
	const MatrixXf &depth_old_ref = depth_old[repr_level];
	const MatrixXf &yy_old_ref = yy_old[repr_level];
	const MatrixXf &xx_old_ref = xx_old[repr_level];
	const MatrixXi &labels_ref = labels[repr_level];
	
	scene = window.get3DSceneAndLock();

	//Segmented points
	opengl::CPointCloudColouredPtr seg_points = scene->getByClass<CPointCloudColoured>(0);
	seg_points->clear();
	const float brigthing_fact = 0.7f;
	for (unsigned int y=0; y<cols; y++)
		for (unsigned int z=0; z<rows; z++)
            if (depth_old_ref(z,y) != 0.f)
			{
				
				float mult;
				if (bf_segm[labels_ref(z,y)] < 0.333f)
					mult = 0.15f;
				else
					mult = brigthing_fact + (1.f - brigthing_fact)*0.f; //bf_segm[labels[repr_level](z,y)];

				const float red = mult*(im_r_old(z,y)-1.f)+1.f;
				const float green = mult*(im_g_old(z,y)-1.f)+1.f;
				const float blue = mult*(im_b_old(z,y)-1.f)+1.f;

				seg_points->push_back(depth_old_ref(z,y), xx_old_ref(z,y), yy_old_ref(z,y), red, green, blue);			
			}


	//Scene flow
	const unsigned int sf_level = repr_level + 0;
	const unsigned int s = pow(2.f,int(sf_level - repr_level));
	cols_i = cols/s; rows_i = rows/s;

	MatrixXf mx(rows_i, cols_i), my(rows_i, cols_i), mz(rows_i, cols_i);
	for (unsigned int u=0; u<cols_i; u++)
		for (unsigned int v=0; v<rows_i; v++)
		{
			mx(v,u) = motionfield[0](s*v, s*u);
			my(v,u) = motionfield[1](s*v, s*u);
			mz(v,u) = motionfield[2](s*v, s*u);		
		}

	opengl::CVectorField3DPtr sf = scene->getByClass<CVectorField3D>(0);
    sf->setPointCoordinates(depth_old[sf_level], xx_old[sf_level], yy_old[sf_level]);
    sf->setVectorField(mx, my, mz);


	//Image
	COpenGLViewportPtr vp_image = scene->getViewport("image");
    image.setFromRGBMatrices(im_r_old, im_g_old, im_b_old, true);
    image.flipVertical();
	image.flipHorizontal();
    vp_image->setImageView(image);

	//Labels
	COpenGLViewportPtr vp_labels = scene->getViewport("labels");
    image.setFromRGBMatrices(olabels_image[0], olabels_image[1], olabels_image[2], true);
    image.flipVertical();
	image.flipHorizontal();
    vp_labels->setImageView(image);

	//Background
	COpenGLViewportPtr vp_backg = scene->getViewport("background");
	image.setFromRGBMatrices(backg_image[0], backg_image[1], backg_image[2], true);
    image.flipVertical();
	image.flipHorizontal();
    vp_backg->setImageView(image);
			
	window.unlockAccess3DScene();
	window.repaint();
}


