
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
    //Initialize camera for a good visualization
	cam_pose.setFromValues(0,0,1.5,0,0,0);

	global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 10000000;
	window.resize(1000,900);
	window.setPos(900,0);
	window.setCameraZoom(8);
    window.setCameraAzimuthDeg(180);
	window.setCameraElevationDeg(40);
	window.setCameraPointingToPoint(1,0,1.5);
	//window.getDefaultViewport()->setCustomBackgroundColor(TColorf(1,1,1));

	scene = window.get3DSceneAndLock();

	//Grid (ground)
	opengl::CGridPlaneXYPtr ground = opengl::CGridPlaneXY::Create();
	scene->insert( ground );

	//Camera
	CPose3D rel_lenspose(0,-0.022,0,0,0,0);
	CBoxPtr camera = CBox::Create(math::TPoint3D(-0.02,-0.1,-0.01),math::TPoint3D(0.02,0.1,0.01));
	camera->setPose(cam_pose + rel_lenspose);
	camera->setColor(0,1,0);
	scene->insert( camera );

	//Frustum
	opengl::CFrustumPtr FOV = opengl::CFrustum::Create(0.3f, 2.f, 57.3*fovh, 57.3*fovv, 1.f, true, false);
	FOV->setColor(0.7,0.7,0.7);
	FOV->setPose(cam_pose);
	scene->insert( FOV );

	//Reference tk
	opengl::CSetOfObjectsPtr reference_cam = opengl::stock_objects::CornerXYZ();
	reference_cam->setScale(0.2);
	reference_cam->setPose(cam_pose);
	scene->insert( reference_cam );

	//3D Points (last frame)
	opengl::CPointCloudPtr points_lf = opengl::CPointCloud::Create();
	points_lf->setColor(0.f, 1.f, 1.f);
	points_lf->setPointSize(3);
	points_lf->enablePointSmooth();
    points_lf->setPose(cam_pose);
	scene->insert( points_lf );

    //Scene Flow (includes initial point cloud)
    opengl::CVectorField3DPtr sf = opengl::CVectorField3D::Create();
    sf->setPointSize(3.0f);
    sf->setLineWidth(2.0f);
    sf->setPointColor(1,0,0);
    sf->setVectorFieldColor(0,0,1);
    sf->enableAntiAliasing();
    sf->setPose(cam_pose);
    scene->insert( sf );

	//Labels
	COpenGLViewportPtr vp_labels = scene->createViewport("labels");
    vp_labels->setViewportPosition(0.7,0.05,240,180);

	COpenGLViewportPtr vp_backg = scene->createViewport("background");
    vp_backg->setViewportPosition(0.1,0.05,240,180);

	window.unlockAccess3DScene();
	window.repaint();
}

void VO_SF::initializeSceneDatasets()
{

	global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 10000000;
	window.resize(1600,800);
	window.setPos(300,0);
	window.setCameraZoom(8);
    window.setCameraAzimuthDeg(180);
	window.setCameraElevationDeg(30);
	window.setCameraPointingToPoint(0,0,0);
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

void VO_SF::initializeSceneImageSeq()
{
	const unsigned int repr_level = round(log2(width/cols));

	global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 10000000;
	window.resize(1600,800);
	window.setPos(300,0);
	window.setCameraZoom(6);
    window.setCameraAzimuthDeg(180);
	window.setCameraElevationDeg(15);
	window.setCameraPointingToPoint(0,-1,0);
	window.getDefaultViewport()->setCustomBackgroundColor(TColorf(1,1,1));
	scene = window.get3DSceneAndLock();

	//Camera
	CPose3D rel_lenspose(0,-0.022,0,0,0,0);
	CBoxPtr camera = CBox::Create(math::TPoint3D(-0.02,-0.1,-0.01),math::TPoint3D(0.02,0.1,0.01));
	camera->setPose(cam_pose + rel_lenspose);
	camera->setColor(0,1,0);
	scene->insert( camera );

	//Frustum
	opengl::CFrustumPtr FOV = opengl::CFrustum::Create(0.3f, 2.f, 57.3*fovh, 57.3*fovv, 1.f, true, false);
	FOV->setColor(0.7,0.7,0.7);
	FOV->setPose(cam_pose);
	scene->insert( FOV );

	//Reference tk
	opengl::CSetOfObjectsPtr reference_cam = opengl::stock_objects::CornerXYZ();
	reference_cam->setScale(0.2);
	reference_cam->setPose(cam_pose);
	scene->insert( reference_cam );


	//3D points (last frame)
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


void VO_SF::updateSceneCamera(bool clean_sf)
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

	//Camera
	CPose3D rel_lenspose(0,-0.022,0,0,0,0);
	CBoxPtr camera = scene->getByClass<CBox>(0);
	camera->setPose(cam_pose + rel_lenspose);
	scene->insert( camera );

	//Frustum
	opengl::CFrustumPtr FOV = scene->getByClass<CFrustum>(0);
	FOV->setPose(cam_pose);
	scene->insert( FOV );

	//Reference tk
	opengl::CSetOfObjectsPtr reference_cam = scene->getByClass<CSetOfObjects>(0);
	reference_cam->setPose(cam_pose);
	scene->insert( reference_cam );

	//Points of the last frame
	opengl::CPointCloudPtr kin_points = scene->getByClass<CPointCloud>(0);
	kin_points->clear();
	for (unsigned int u=0; u<cols; u++)
		for (unsigned int v=0; v<rows; v++)
            if (depth_ref(v,u) != 0.f)
                kin_points->insertPoint(depth_ref(v,u), xx_ref(v,u), yy_ref(v,u));


    //Scene flow
    if (clean_sf == true)
    {
        motionfield[0].assign(0.f);
        motionfield[1].assign(0.f);
        motionfield[2].assign(0.f);
    }

	opengl::CVectorField3DPtr sf = scene->getByClass<CVectorField3D>(0);
    sf->setPointCoordinates(depth_old[repr_level], xx_old[repr_level], yy_old[repr_level]);	
	sf->setVectorField(motionfield[0], motionfield[1], motionfield[2]);

	//Labels
	COpenGLViewportPtr vp_labels = scene->getViewport("labels");
    image.setFromRGBMatrices(labels_image[0], labels_image[1], labels_image[2], true);
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

void VO_SF::updateSceneDatasets(const CPose3D &gt, const CPose3D &gt_old)
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

	//Points
	opengl::CPointCloudColouredPtr points = scene->getByClass<CPointCloudColoured>(0);
	points->clear();
	points->setPose(gt);
	const unsigned int size_factor = width/cols;
	for (unsigned int u=0; u<cols; u++)
		for (unsigned int v=0; v<rows; v++)
            if (depth_ref(v,u) != 0.f)
				points->push_back(depth_ref(v,u), xx_ref(v,u), yy_ref(v,u),
								im_r(size_factor*v,size_factor*u), im_g(size_factor*v,size_factor*u), im_b(size_factor*v,size_factor*u));


	//Trajectories
	opengl::CSetOfLinesPtr estimated_traj = scene->getByClass<CSetOfLines>(0);
	estimated_traj->appendLine(cam_pose[0], cam_pose[1], cam_pose[2], cam_oldpose[0], cam_oldpose[1], cam_oldpose[2]);

	opengl::CSetOfLinesPtr gt_traj = scene->getByClass<CSetOfLines>(1);
	gt_traj->appendLine(gt[0], gt[1], gt[2], gt_old[0], gt_old[1], gt_old[2]);


	//Image
	COpenGLViewportPtr vp_image = scene->getViewport("image");
    image.setFromRGBMatrices(im_r_old, im_g_old, im_b_old, true);
    image.flipVertical();
	image.flipHorizontal();
    vp_image->setImageView(image);

	//Labels
	COpenGLViewportPtr vp_labels = scene->getViewport("labels");
    image.setFromRGBMatrices(labels_image[0], labels_image[1], labels_image[2], true);
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

void VO_SF::updateSceneImageSeq()
{
	const unsigned int repr_level = round(log2(width/cols));
	CImage image;

	//Refs
	const MatrixXf &depth_old_ref = depth_old[repr_level];
	const MatrixXf &yy_old_ref = yy_old[repr_level];
	const MatrixXf &xx_old_ref = xx_old[repr_level];
	const MatrixXi &labels_ref = labels[repr_level];
	
	scene = window.get3DSceneAndLock();

	//Camera
	CPose3D rel_lenspose(0,-0.022,0,0,0,0);
	CBoxPtr camera = scene->getByClass<CBox>(0);
	camera->setPose(cam_pose + rel_lenspose);
	scene->insert( camera );

	//Frustum
	opengl::CFrustumPtr FOV = scene->getByClass<CFrustum>(0);
	FOV->setPose(cam_pose);
	scene->insert( FOV );

	//Reference tk
	opengl::CSetOfObjectsPtr reference_cam = scene->getByClass<CSetOfObjects>(0);
	reference_cam->setPose(cam_pose);
	scene->insert( reference_cam );

	//3D points (last frame)
	opengl::CPointCloudColouredPtr points = scene->getByClass<CPointCloudColoured>(0);
	points->setPose(cam_pose);
	points->clear();
	const float brigthing_fact = 0.7f;
	const unsigned int size_factor = width/cols;
	for (unsigned int u=0; u<cols; u++)
		for (unsigned int v=0; v<rows; v++)
            if (depth_old_ref(v,u) != 0.f)
			{		
				const float mult = (bf_segm[labels_ref(v,u)] < 0.333f) ? 0.25f : brigthing_fact;
				const float red = mult*(im_r_old(size_factor*v,size_factor*u)-1.f)+1.f;
				const float green = mult*(im_g_old(size_factor*v,size_factor*u)-1.f)+1.f;
				const float blue = mult*(im_b_old(size_factor*v,size_factor*u)-1.f)+1.f;

				points->push_back(depth_old_ref(v,u), xx_old_ref(v,u), yy_old_ref(v,u), red, green, blue);			
			}


	//Scene flow
	opengl::CVectorField3DPtr sf = scene->getByClass<CVectorField3D>(0);
	sf->setPose(cam_pose);
    sf->setPointCoordinates(depth_old[repr_level], xx_old[repr_level], yy_old[repr_level]);
    sf->setVectorField(motionfield[0], motionfield[1], motionfield[2]);


	//Image
	COpenGLViewportPtr vp_image = scene->getViewport("image");
    image.setFromRGBMatrices(im_r_old, im_g_old, im_b_old, true);
    image.flipVertical();
	image.flipHorizontal();
    vp_image->setImageView(image);

	//Labels
	COpenGLViewportPtr vp_labels = scene->getViewport("labels");
    image.setFromRGBMatrices(labels_image[0], labels_image[1], labels_image[2], true);
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


