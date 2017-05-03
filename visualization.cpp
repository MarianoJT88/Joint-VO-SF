
#include "joint_vo_sf.h"
#include "dvo/opencv_ext.hpp"


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
	m_window.resize(1000,900);
	m_window.setPos(900,0);
	m_window.setCameraZoom(16);
    m_window.setCameraAzimuthDeg(180);
	m_window.setCameraElevationDeg(90);
	m_window.setCameraPointingToPoint(0,0,1);
	//m_window.getDefaultViewport()->setCustomBackgroundColor(TColorf(1,1,1));

	m_scene = m_window.get3DSceneAndLock();

	//Grid (ground)
	opengl::CGridPlaneXYPtr ground = opengl::CGridPlaneXY::Create();
	m_scene->insert( ground );

	//Reference
	opengl::CSetOfObjectsPtr reference = opengl::stock_objects::CornerXYZ();
	m_scene->insert( reference );

	//Reference tk
	opengl::CSetOfObjectsPtr referencetk = opengl::stock_objects::CornerXYZ();
	referencetk->setScale(0.2);
	m_scene->insert( referencetk );

	//Kinect points
	opengl::CPointCloudColouredPtr kin_points = opengl::CPointCloudColoured::Create();
	kin_points->setColor(1,0,0);
	kin_points->setPointSize(4);
	kin_points->enablePointSmooth(1);
    kin_points->setPose(pose_show);
	m_scene->insert( kin_points );

	//Selected point
    opengl::CPointCloudPtr sel_point = opengl::CPointCloud::Create();
    sel_point->setColor(0,0.6,0);
    sel_point->setPointSize(15.f);
    sel_point->setPose(pose_show);
    m_scene->insert( sel_point );

	//Connectivity and Segmented point cloud
	if (draw_segmented_pcloud)
	{
		opengl::CSetOfLinesPtr connectivity = opengl::CSetOfLines::Create();
		connectivity->setLineWidth(4.f);
		connectivity->setPose(pose_segm_pc);
		m_scene->insert( connectivity );	

		opengl::CPointCloudColouredPtr seg_points = opengl::CPointCloudColoured::Create();
		seg_points->setPointSize(4);
		seg_points->enablePointSmooth(1);
		seg_points->setPose(pose_segm_pc);
		m_scene->insert( seg_points );		
	}

    //Scene Flow (includes initial point cloud)
    opengl::CVectorField3DPtr sf = opengl::CVectorField3D::Create();
    sf->setPointSize(3.0f);
    sf->setLineWidth(2.0f);
    sf->setPointColor(1,0,0);
    sf->setVectorFieldColor(0,0,1);
    sf->enableAntiAliasing();
    sf->setPose(pose_show);
    m_scene->insert( sf );

	//Labels
	COpenGLViewportPtr vp_labels = m_scene->createViewport("labels");
    vp_labels->setViewportPosition(0.7,0.05,240,180);

	COpenGLViewportPtr vp_backg = m_scene->createViewport("background");
    vp_backg->setViewportPosition(0.1,0.05,240,180);


	m_window.unlockAccess3DScene();
	m_window.repaint();
}

void VO_SF::initializeSceneDataset()
{
	global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 10000000;
	m_window.resize(1000,900);
	m_window.setPos(900,0);
	m_window.setCameraZoom(16);
    m_window.setCameraAzimuthDeg(180);
	m_window.setCameraElevationDeg(90);
	m_window.setCameraPointingToPoint(0,0,1);

	m_scene = m_window.get3DSceneAndLock();

	//Grid (ground)
	opengl::CGridPlaneXYPtr ground = opengl::CGridPlaneXY::Create();
	m_scene->insert( ground );

	//Reference
	opengl::CSetOfObjectsPtr reference = opengl::stock_objects::CornerXYZ();
	m_scene->insert( reference );

	//Reference gt
	opengl::CSetOfObjectsPtr reference_gt = opengl::stock_objects::CornerXYZ();
	reference_gt->setScale(0.15);
	m_scene->insert( reference_gt );

	//Reference est
	opengl::CSetOfObjectsPtr reference_est = opengl::stock_objects::CornerXYZ();
	reference_est->setScale(0.15);
	m_scene->insert( reference_est );

	//Segmented points (original)
	opengl::CPointCloudColouredPtr seg_points = opengl::CPointCloudColoured::Create();
	seg_points->setColor(1,0,0);
	seg_points->setPointSize(4);
	seg_points->enablePointSmooth(1);
	m_scene->insert( seg_points );

	//Segmented points (warped)
	opengl::CPointCloudColouredPtr warped_seg_points = opengl::CPointCloudColoured::Create();
	warped_seg_points->setColor(1,0,0);
	warped_seg_points->setPointSize(4);
	warped_seg_points->enablePointSmooth(1);
	m_scene->insert( warped_seg_points );

	//Estimated trajectory
	opengl::CSetOfLinesPtr estimated_traj = opengl::CSetOfLines::Create();
	estimated_traj->setColor(1.f, 0.f, 0.f);
	estimated_traj->setLineWidth(5.f);
	m_scene->insert(estimated_traj);

	//GT trajectory
	opengl::CSetOfLinesPtr gt_traj = opengl::CSetOfLines::Create();
	gt_traj->setColor(0.f, 0.f, 0.f);
	gt_traj->setLineWidth(5.f);
	m_scene->insert(gt_traj);

	//Point cloud for the scene flow
	opengl::CPointCloudPtr sf_points = opengl::CPointCloud::Create();
	sf_points->setColor(0,1,1);
	sf_points->setPointSize(4);
	sf_points->enablePointSmooth(1);
	m_scene->insert( sf_points );

    //Scene Flow (includes initial point cloud)
    opengl::CVectorField3DPtr sf = opengl::CVectorField3D::Create();
    sf->setPointSize(3.0f);
    sf->setLineWidth(2.0f);
    sf->setPointColor(1,0,0);
    sf->setVectorFieldColor(0,0,1);
    sf->enableAntiAliasing();
    m_scene->insert( sf );

	//Points showing label smoothness
	opengl::CPointCloudColouredPtr smt_points = opengl::CPointCloudColoured::Create();
	smt_points->setColor(1,0,0);
	smt_points->setPointSize(4);
	smt_points->enablePointSmooth(1);
	m_scene->insert( smt_points );

	//Number of the labels
	for (unsigned int l = 0; l < NUM_LABELS; l++)
	{
		opengl::CText3DPtr labels_id = opengl::CText3D::Create();
		labels_id->setString(std::to_string(l));
		labels_id->setScale(0.1f);
		labels_id->setColor(0.5, 0, 0);
		m_scene->insert(labels_id);
	}

	//Connectivity
	opengl::CSetOfLinesPtr connectivity = opengl::CSetOfLines::Create();
	connectivity->setColor(0,0,0.6);
	connectivity->setLineWidth(4.f);
	m_scene->insert( connectivity );	


	//Labels
	COpenGLViewportPtr vp_labels = m_scene->createViewport("labels");
    vp_labels->setViewportPosition(0.7,0.05,240,180);

	COpenGLViewportPtr vp_backg = m_scene->createViewport("background");
	vp_backg->setViewportPosition(0.1,0.05,240,180);


	m_window.unlockAccess3DScene();
	m_window.repaint();
}

void VO_SF::initializeSceneDatasetVideo()
{

	global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 10000000;
	m_window.resize(1600,800);
	m_window.setPos(300,0);
	m_window.setCameraZoom(8);
    m_window.setCameraAzimuthDeg(180);
	m_window.setCameraElevationDeg(90);
	m_window.setCameraPointingToPoint(0,0,1);
	m_window.getDefaultViewport()->setCustomBackgroundColor(TColorf(1,1,1));

	m_scene = m_window.get3DSceneAndLock();

	//Reference gt
	opengl::CSetOfObjectsPtr reference_gt = opengl::stock_objects::CornerXYZ();
	reference_gt->setScale(0.15);
	m_scene->insert( reference_gt );

	//Reference est
	opengl::CSetOfObjectsPtr reference_est = opengl::stock_objects::CornerXYZ();
	reference_est->setScale(0.15);
	m_scene->insert( reference_est );

	//Segmented points (original)
	opengl::CPointCloudColouredPtr seg_points = opengl::CPointCloudColoured::Create();
	seg_points->setColor(1,0,0);
	seg_points->setPointSize(2);
	seg_points->enablePointSmooth(1);
	m_scene->insert( seg_points );

	//Estimated trajectory
	opengl::CSetOfLinesPtr estimated_traj = opengl::CSetOfLines::Create();
	estimated_traj->setColor(1.f, 0.f, 0.f);
	estimated_traj->setLineWidth(5.f);
	m_scene->insert(estimated_traj);

	//GT trajectory
	opengl::CSetOfLinesPtr gt_traj = opengl::CSetOfLines::Create();
	gt_traj->setColor(0.f, 0.f, 0.f);
	gt_traj->setLineWidth(5.f);
	m_scene->insert(gt_traj);

	//Point cloud for the scene flow
	opengl::CPointCloudPtr sf_points = opengl::CPointCloud::Create();
	sf_points->setColor(0,1,1);
	sf_points->setPointSize(3);
	sf_points->enablePointSmooth(1);
	m_scene->insert( sf_points );

    //Scene Flow (includes initial point cloud)
    opengl::CVectorField3DPtr sf = opengl::CVectorField3D::Create();
    sf->setPointSize(3.0f);
    sf->setLineWidth(2.0f);
    sf->setPointColor(1,0,0);
    sf->setVectorFieldColor(0,0,1);
    sf->enableAntiAliasing();
    m_scene->insert( sf );


	//Labels
	COpenGLViewportPtr vp_image = m_scene->createViewport("image");
    vp_image->setViewportPosition(0.775,0.675,320,240);

	COpenGLViewportPtr vp_labels = m_scene->createViewport("labels");
    vp_labels->setViewportPosition(0.775,0.350,320,240);

	COpenGLViewportPtr vp_backg = m_scene->createViewport("background");
	vp_backg->setViewportPosition(0.775,0.025,320,240);


	m_window.unlockAccess3DScene();
	m_window.repaint();
}

void VO_SF::initializeSceneSequencesVideo()
{
	const unsigned int repr_level = round(log2(width/cols));

	global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 10000000;
	m_window.resize(1600,800);
	m_window.setPos(300,0);
	m_window.setCameraZoom(8);
    m_window.setCameraAzimuthDeg(180);
	m_window.setCameraElevationDeg(90);
	m_window.setCameraPointingToPoint(0,0,1);
	m_window.getDefaultViewport()->setCustomBackgroundColor(TColorf(1,1,1));
	m_window.captureImagesStart();
	m_scene = m_window.get3DSceneAndLock();


	//Segmented points (original)
	opengl::CPointCloudColouredPtr seg_points = opengl::CPointCloudColoured::Create();
	seg_points->setColor(1,0,0);
	seg_points->setPointSize(2);
	seg_points->enablePointSmooth(1);
	m_scene->insert( seg_points );


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
    m_scene->insert( sf );

	//Labels
	COpenGLViewportPtr vp_image = m_scene->createViewport("image");
    vp_image->setViewportPosition(0.775,0.675,320,240);

	COpenGLViewportPtr vp_labels = m_scene->createViewport("labels");
    vp_labels->setViewportPosition(0.775,0.350,320,240);

	COpenGLViewportPtr vp_backg = m_scene->createViewport("background");
	vp_backg->setViewportPosition(0.775,0.025,320,240);


	m_window.unlockAccess3DScene();
	m_window.repaint();
}

void VO_SF::showOriginalPointCloud()
{
	m_scene = m_window.get3DSceneAndLock();

	//Kinect points
	opengl::CPointCloudPtr kin_points = opengl::CPointCloud::Create();
	kin_points->setColor(1,0,0);
	kin_points->setPose(CPose3D(0,-4,0,0,0,0));
	kin_points->setPointSize(4);
	kin_points->enablePointSmooth(1);
	m_scene->insert( kin_points );

	const float inv_f_i = 2.f*tan(0.5f*fovh)/float(width);
    const float disp_u_i = 0.5f*(width-1);
    const float disp_v_i = 0.5f*(height-1);

	for (unsigned int u=0; u<width; u++)
		for (unsigned int v=0; v<height; v++)
		{
			const float x = (u - disp_u_i)*depth_wf(v,u)*inv_f_i;
			const float y = (v - disp_v_i)*depth_wf(v,u)*inv_f_i;
			kin_points->insertPoint(depth_wf(v,u), x, y);
		}

	m_window.unlockAccess3DScene();
	m_window.repaint();
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
	
	m_scene = m_window.get3DSceneAndLock();

	opengl::CPointCloudColouredPtr kin_points = m_scene->getByClass<CPointCloudColoured>(0);
	kin_points->clear();
	for (unsigned int u=0; u<cols; u++)
		for (unsigned int v=0; v<rows; v++)
            if (depth_ref(v,u) != 0.f)
                kin_points->push_back(depth_ref(v,u), xx_ref(v,u), yy_ref(v,u), 0.f, 1.f, 1.f);

	//Connectivity and Segmented point cloud
	if (draw_segmented_pcloud)
	{
		opengl::CSetOfLinesPtr connectivity = m_scene->getByClass<CSetOfLines>(0);
		connectivity->clear();
		for (unsigned int i=0; i<NUM_LABELS; i++)
			for (unsigned int j=i+1; j<NUM_LABELS; j++)
				if (this->connectivity[i][j] == true)
					connectivity->appendLine(kmeans(0,i), kmeans(1,i), kmeans(2,i), kmeans(0,j), kmeans(1,j), kmeans(2,j));

		opengl::CPointCloudColouredPtr seg_points = m_scene->getByClass<CPointCloudColoured>(1);
		seg_points->clear();
		for (unsigned int y=0; y<cols; y++)
			for (unsigned int z=0; z<rows; z++)
				if (depth_old_ref(z,y) != 0.f)
					seg_points->push_back(depth_old_ref(z,y), xx_old_ref(z,y), yy_old_ref(z,y),
											olabels_image[0](z,y), olabels_image[1](z,y), olabels_image[2](z,y));	
	}

    //Scene flow
    if (clean_sf == true)
    {
        motionfield[0].assign(0.f);
        motionfield[1].assign(0.f);
        motionfield[2].assign(0.f);
		clean_sf = false;
    }

	opengl::CVectorField3DPtr sf = m_scene->getByClass<CVectorField3D>(0);
    sf->setPointCoordinates(depth_old[repr_level], xx_old[repr_level], yy_old[repr_level]);	
	sf->setVectorField(motionfield[0], motionfield[1], motionfield[2]);

	//Labels
	COpenGLViewportPtr vp_labels = m_scene->getViewport("labels");
    image.setFromRGBMatrices(olabels_image[0], olabels_image[1], olabels_image[2], true);
	//image.setFromMatrix(color_wf, true);
    image.flipVertical();
    vp_labels->setImageView(image);

	//Background
	COpenGLViewportPtr vp_backg = m_scene->getViewport("background");
    image.setFromRGBMatrices(backg_image[0], backg_image[1], backg_image[2], true);
    image.flipVertical();
    vp_backg->setImageView(image);
			

	m_window.unlockAccess3DScene();
	m_window.repaint();
}

void VO_SF::updateSceneDataset()
{
	const unsigned int repr_level = round(log2(width/cols));
	CImage image;

	//Refs
	const MatrixXf &depth_old_ref = depth_old[repr_level];
	const MatrixXf &yy_old_ref = yy_old[repr_level];
	const MatrixXf &xx_old_ref = xx_old[repr_level];
	
	m_scene = m_window.get3DSceneAndLock();

	//Cameras
	opengl::CSetOfObjectsPtr reference_gt = m_scene->getByClass<CSetOfObjects>(1);
	reference_gt->setPose(dataset.gt_pose);
	m_scene->insert( reference_gt );

	opengl::CSetOfObjectsPtr reference_est = m_scene->getByClass<CSetOfObjects>(2);
	reference_est->setPose(cam_pose);
	m_scene->insert( reference_est );

	//Segmented points
	opengl::CPointCloudColouredPtr seg_points = m_scene->getByClass<CPointCloudColoured>(0);
	seg_points->clear();
	seg_points->setPose(dataset.gt_pose);
	for (unsigned int y=0; y<cols; y++)
		for (unsigned int z=0; z<rows; z++)
            if (depth_old_ref(z,y) != 0.f)
			{
				const float intensity = 1.f - backg_image[0](z,y);
				seg_points->push_back(depth_old_ref(z,y), xx_old_ref(z,y), yy_old_ref(z,y), intensity, intensity, intensity);
			}


	//Warped segmented points
	//opengl::CPointCloudColouredPtr warped_seg_points = m_scene->getByClass<CPointCloudColoured>(1);
	//CPose3D aux_pose = dataset.gt_pose + CPose3D(0,4,0,0,0,0);
	//warped_seg_points->clear();
	//warped_seg_points->setPose(aux_pose);
	//for (unsigned int y=0; y<cols; y++)
	//	for (unsigned int z=0; z<rows; z++)
 //           if (depth[repr_level](z,y) > 0.f)
	//		{
	//			const float intensity = 1.f - bf_segm_image_warped(z,y);
 //               warped_seg_points->push_back(depth[repr_level](z,y), xx[repr_level](z,y), yy[repr_level](z,y), intensity, intensity, intensity);
	//		}


	//Trajectories
	opengl::CSetOfLinesPtr estimated_traj = m_scene->getByClass<CSetOfLines>(0);
	estimated_traj->appendLine(cam_pose[0], cam_pose[1], cam_pose[2], cam_oldpose[0], cam_oldpose[1], cam_oldpose[2]);

	opengl::CSetOfLinesPtr gt_traj = m_scene->getByClass<CSetOfLines>(1);
	gt_traj->appendLine(dataset.gt_pose[0], dataset.gt_pose[1], dataset.gt_pose[2], dataset.gt_oldpose[0], dataset.gt_oldpose[1], dataset.gt_oldpose[2]);

	//Number of the labels
	for (unsigned int l = 0; l < NUM_LABELS; l++)
	{
		opengl::CText3DPtr labels_id = m_scene->getByClass<CText3D>(l);
		CPose3D pose_label = dataset.gt_pose + CPose3D(kmeans(0,l), kmeans(1,l), kmeans(2,l), DEG2RAD(-90), 0, DEG2RAD(90));
		labels_id->setPose(pose_label);
		m_scene->insert(labels_id);
	}

	//Connectivity
	opengl::CSetOfLinesPtr connectivity = m_scene->getByClass<CSetOfLines>(2);
	connectivity->clear();
	connectivity->setPose(dataset.gt_pose);
	for (unsigned int i=0; i<NUM_LABELS; i++)
		for (unsigned int j=i+1; j<NUM_LABELS; j++)
			if (this->connectivity[i][j] == true)
				connectivity->appendLine(kmeans(0,i), kmeans(1,i), kmeans(2,i), kmeans(0,j), kmeans(1,j), kmeans(2,j));


	//Scene flow
	//opengl::CPointCloudPtr points_sf = m_scene->getByClass<CPointCloud>(0);
	//CPose3D aux_pose_sf = dataset.gt_pose + CPose3D(4,4,0,0,0,0);
	//points_sf->clear();
	//points_sf->setPose(aux_pose_sf);
	//for (unsigned int y=0; y<cols; y++)
	//	for (unsigned int z=0; z<rows; z++)
 //           if (depth[repr_level](z,y) > 0.f)
 //               points_sf->insertPoint(depth[repr_level](z,y), xx[repr_level](z,y), yy[repr_level](z,y));

	//opengl::CVectorField3DPtr sf = m_scene->getByClass<CVectorField3D>(0);
	//sf->setPose(aux_pose_sf);
 //   sf->setPointCoordinates(depth_old[repr_level], xx_old[repr_level], yy_old[repr_level]);
 //   sf->setVectorField(motionfield[0], motionfield[1], motionfield[2]);


	//Labels
	COpenGLViewportPtr vp_labels = m_scene->getViewport("labels");
    image.setFromRGBMatrices(olabels_image[0], olabels_image[1], olabels_image[2], true);
    image.flipVertical();
    vp_labels->setImageView(image);

	//Background
	COpenGLViewportPtr vp_backg = m_scene->getViewport("background");
	image.setFromRGBMatrices(backg_image[0], backg_image[1], backg_image[2], true);
    image.flipVertical();
    vp_backg->setImageView(image);
			

	m_window.unlockAccess3DScene();
	m_window.repaint();
}

void VO_SF::updateSceneDatasetVideo()
{
	const unsigned int repr_level = round(log2(width/cols));
	CImage image;

	//Refs
	const MatrixXf &depth_ref = depth[repr_level];
	const MatrixXf &yy_ref = yy[repr_level];
	const MatrixXf &xx_ref = xx[repr_level];
	
	m_scene = m_window.get3DSceneAndLock();

	//Cameras
	opengl::CSetOfObjectsPtr reference_gt = m_scene->getByClass<CSetOfObjects>(0);
	reference_gt->setPose(dataset.gt_pose);
	m_scene->insert( reference_gt );

	opengl::CSetOfObjectsPtr reference_est = m_scene->getByClass<CSetOfObjects>(1);
	reference_est->setPose(cam_pose);
	m_scene->insert( reference_est );

	//Segmented points
	opengl::CPointCloudColouredPtr seg_points = m_scene->getByClass<CPointCloudColoured>(0);
	seg_points->clear();
	seg_points->setPose(dataset.gt_pose);
	for (unsigned int y=0; y<cols; y++)
		for (unsigned int z=0; z<rows; z++)
            if (depth_ref(z,y) != 0.f)
				seg_points->push_back(depth_ref(z,y), xx_ref(z,y), yy_ref(z,y), im_r(z,y), im_g(z,y), im_b(z,y));


	//Trajectories
	opengl::CSetOfLinesPtr estimated_traj = m_scene->getByClass<CSetOfLines>(0);
	estimated_traj->appendLine(cam_pose[0], cam_pose[1], cam_pose[2], cam_oldpose[0], cam_oldpose[1], cam_oldpose[2]);

	opengl::CSetOfLinesPtr gt_traj = m_scene->getByClass<CSetOfLines>(1);
	gt_traj->appendLine(dataset.gt_pose[0], dataset.gt_pose[1], dataset.gt_pose[2], dataset.gt_oldpose[0], dataset.gt_oldpose[1], dataset.gt_oldpose[2]);



	//Scene flow
	//opengl::CPointCloudPtr points_sf = m_scene->getByClass<CPointCloud>(0);
	//CPose3D aux_pose_sf = dataset.gt_pose + CPose3D(4,4,0,0,0,0);
	//points_sf->clear();
	//points_sf->setPose(aux_pose_sf);
	//for (unsigned int y=0; y<cols; y++)
	//	for (unsigned int z=0; z<rows; z++)
 //           if (depth[repr_level](z,y) > 0.f)
 //               points_sf->insertPoint(depth[repr_level](z,y), xx[repr_level](z,y), yy[repr_level](z,y));

	opengl::CVectorField3DPtr sf = m_scene->getByClass<CVectorField3D>(0);
	//sf->setPose(aux_pose_sf);
 //   sf->setPointCoordinates(depth_old[repr_level], xx_old[repr_level], yy_old[repr_level]);
 //   sf->setVectorField(motionfield[0], motionfield[1], motionfield[2]);


	//Image
	COpenGLViewportPtr vp_image = m_scene->getViewport("image");
    image.setFromRGBMatrices(im_r_old, im_g_old, im_b_old, true);
    image.flipVertical();
	image.flipHorizontal();
    vp_image->setImageView(image);

	//Labels
	COpenGLViewportPtr vp_labels = m_scene->getViewport("labels");
    image.setFromRGBMatrices(olabels_image[0], olabels_image[1], olabels_image[2], true);
    image.flipVertical();
	image.flipHorizontal();
    vp_labels->setImageView(image);

	//Background
	COpenGLViewportPtr vp_backg = m_scene->getViewport("background");
	image.setFromRGBMatrices(backg_image[0], backg_image[1], backg_image[2], true);
    image.flipVertical();
	image.flipHorizontal();
    vp_backg->setImageView(image);
			
	m_window.unlockAccess3DScene();
	m_window.repaint();

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
	
	m_scene = m_window.get3DSceneAndLock();

	//Segmented points
	opengl::CPointCloudColouredPtr seg_points = m_scene->getByClass<CPointCloudColoured>(0);
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

	opengl::CVectorField3DPtr sf = m_scene->getByClass<CVectorField3D>(0);
    sf->setPointCoordinates(depth_old[sf_level], xx_old[sf_level], yy_old[sf_level]);
    sf->setVectorField(mx, my, mz);


	//Image
	COpenGLViewportPtr vp_image = m_scene->getViewport("image");
    image.setFromRGBMatrices(im_r_old, im_g_old, im_b_old, true);
    image.flipVertical();
	image.flipHorizontal();
    vp_image->setImageView(image);

	//Labels
	COpenGLViewportPtr vp_labels = m_scene->getViewport("labels");
    image.setFromRGBMatrices(olabels_image[0], olabels_image[1], olabels_image[2], true);
    image.flipVertical();
	image.flipHorizontal();
    vp_labels->setImageView(image);

	//Background
	COpenGLViewportPtr vp_backg = m_scene->getViewport("background");
	image.setFromRGBMatrices(backg_image[0], backg_image[1], backg_image[2], true);
    image.flipVertical();
	image.flipHorizontal();
    vp_backg->setImageView(image);
			
	m_window.unlockAccess3DScene();
	m_window.repaint();
}

void VO_SF::showResidualsUsedForSegmentation()
{
	gui::CDisplayWindow3D	window;
	COpenGLScenePtr			scene;

	CPose3D pose_flow(0,0,0,0,0,0);
	CPose3D pose_segm(7,0,0,0,0,0);
	CPose3D pose_res(3.5,0,0,0,0,0);
	CPose3D pose_warp(10.5,0,0,0,0,0);
	CPose3D pose_imag(14,0,0,0,0,0);

	const unsigned int cols_i_gl = cols, rows_i_gl = rows;
    const unsigned int image_level_gl = round(log2(width/cols));


	global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 10000000;
	window.resize(1000,900);
	window.setPos(900,0);
	window.setCameraZoom(16);
    window.setCameraAzimuthDeg(180);
	window.setCameraElevationDeg(90);

	scene = window.get3DSceneAndLock();

	//Reference
	opengl::CSetOfObjectsPtr reference = opengl::stock_objects::CornerXYZ();
	scene->insert( reference );
		
	//Warped points
	opengl::CPointCloudColouredPtr kin_points = opengl::CPointCloudColoured::Create();
	kin_points->setColor(1,0,0);
	kin_points->setPointSize(4);
	kin_points->enablePointSmooth(1);
	kin_points->setPose(pose_flow);
	scene->insert( kin_points );

	for (unsigned int y=0; y<cols_i_gl; y++)
		for (unsigned int z=0; z<rows_i_gl; z++)
			if (depth_warped[image_level_gl](z,y) > 0.f)
				kin_points->push_back(depth_warped[image_level_gl](z,y), xx_warped[image_level_gl](z,y), yy_warped[image_level_gl](z,y), 0.f, 1.f, 1.f);

	//Old points
	opengl::CPointCloudColouredPtr old_points = opengl::CPointCloudColoured::Create();
	old_points->setPose(pose_flow);
	old_points->setPointSize(4);
	scene->insert( old_points );
	for (unsigned int y=0; y<cols_i_gl; y++)
		for (unsigned int z=0; z<rows_i_gl; z++)
			if (depth_old[image_level_gl](z,y) > 0.f)
				old_points->push_back(depth_old[image_level_gl](z,y), xx_old[image_level_gl](z,y), yy_old[image_level_gl](z,y), 1.f, 0.f, 0.f);


	//Lines connecting associated points
	opengl::CSetOfLinesPtr lines_error = opengl::CSetOfLines::Create();
	lines_error->setLineWidth(2.0f);
	lines_error->enableAntiAliasing();
	lines_error->setPose(pose_flow);
	scene->insert( lines_error );
	for (unsigned int y=0; y<cols_i_gl; y++)
		for (unsigned int z=0; z<rows_i_gl; z++)
			if ((depth_old[image_level_gl](z,y) > 0.f)&&(depth_warped[image_level_gl](z,y) > 0.f))
				lines_error->appendLine(depth_old[image_level_gl](z,y), xx_old[image_level_gl](z,y), yy_old[image_level_gl](z,y),
										depth_warped[image_level_gl](z,y), xx_warped[image_level_gl](z,y), yy_warped[image_level_gl](z,y));




	////Segmented point clouds
	////--------------------------------------------------------------------------------------------------------
	////Associate colors to labels
	//float r[NUM_LABELS], g[NUM_LABELS], b[NUM_LABELS];
	//for (unsigned int l=0; l<num_labels_now; l++)
	//{
	//	const float indx = float(l)/float(num_labels_now-1);
	//	mrpt::utils::colormap(mrpt::utils::cmJET, indx, r[l], g[l], b[l]);
	//}

	////Compute the color for every pixel according to the estimated labeling
	//MatrixXf colours[3];
	//colours[0].resize(rows_i_gl, cols_i_gl); colours[1].resize(rows_i_gl, cols_i_gl); colours[2].resize(rows_i_gl, cols_i_gl); 
	//colours[0].assign(0.f); colours[1].assign(0.f); colours[2].assign(0.f);

	//for (unsigned int u=0; u<cols_i_gl; u++)
	//	for (unsigned int v=0; v<rows_i_gl; v++)
	//		if (labels_opt[num_labels_now][image_level_gl](v,u) == 0.f)
	//			for (unsigned int l=0; l<num_labels_now; l++)
	//			{
	//				const float lab = labels_opt[l][image_level_gl](v,u);
	//				colours[0](v,u) += lab*r[l];
	//				colours[1](v,u) += lab*g[l];
	//				colours[2](v,u) += lab*b[l];
	//			}


	//opengl::CSetOfLinesPtr connectivity = opengl::CSetOfLines::Create();
	//connectivity->setPose(pose_segm);
	//for (unsigned int i=0; i<num_labels_now; i++)
	//	for (unsigned int j=i+1; j<num_labels_now; j++)
	//		if (this->connectivity[i][j] == true)
	//			connectivity->appendLine(depth_kmeans[i], x_kmeans[i], y_kmeans[i], depth_kmeans[j], x_kmeans[j], y_kmeans[j]);
	//scene->insert( connectivity );

	//opengl::CPointCloudColouredPtr seg_points = opengl::CPointCloudColoured::Create();
	//seg_points->setPose(pose_segm);
	//seg_points->setPointSize(4);
	//for (unsigned int y=0; y<cols_i_gl; y++)
	//	for (unsigned int z=0; z<rows_i_gl; z++)
	//		if (depth_old[image_level_gl](z,y) > 0.f)
	//			seg_points->push_back(depth_old[image_level_gl](z,y), xx_old[image_level_gl](z,y), yy_old[image_level_gl](z,y),
	//									colours[0](z,y), colours[1](z,y), colours[2](z,y));	
	//scene->insert( seg_points );

	////Point clouds showing residuals (ddt and ddc)
	//opengl::CPointCloudColouredPtr res_points = opengl::CPointCloudColoured::Create();
	//res_points->setPointSize(4);
	//res_points->enablePointSmooth(1);
	//res_points->setPose(pose_res);
	//scene->insert( res_points );
	//const float res_threshold = 0.1f;

	//for (unsigned int y=0; y<cols_i_gl; y++)
	//	for (unsigned int z=0; z<rows_i_gl; z++)
	//		if (depth_inter[image_level_gl](z,y) > 0.f)
	//		{
	//			const float depth_res = abs(depth_old[image_level_gl](z,y) - depth_warped[image_level_gl](z,y))*occ_weights[image_level_gl](z,y);
	//			const float color_res = abs(color_old[image_level_gl](z,y) - color_warped[image_level_gl](z,y))*occ_weights[image_level_gl](z,y);
	//			const float red = max(0.f, min(1.f, depth_res/res_threshold));
	//			const float green = max(0.f, min(1.f, color_res/res_threshold));
	//			const float blue = max(0.f, min(1.f, 1.f - (depth_res + color_res)/res_threshold));
	//			res_points->push_back(depth_old[image_level_gl](z,y), xx_old[image_level_gl](z,y), yy_old[image_level_gl](z,y), red, green, blue);

	//			//if ((green > 0.7f)&&(depth_old[image_level_gl](z,y) > 1.5f))
	//			//	printf("\n level = %d, depth_old = %.3f, depth_warped = %.3f, res_depth = %.3f, res_color = %.3f, occ_weight = %.3f", image_level_gl, depth_old[image_level_gl](z,y), depth_warped[image_level_gl](z,y), depth_res, color_res, occ_weights[image_level_gl](z,y));
	//		}


	window.unlockAccess3DScene();
	window.repaint();

	system::os::getch();
}

