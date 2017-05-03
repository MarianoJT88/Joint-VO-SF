
#include "datasets.h"

using namespace mrpt;
using namespace mrpt::obs;
using namespace mrpt::utils;
using namespace mrpt::math;
using namespace std;
using namespace Eigen;

Datasets::Datasets()
{
    downsample = 1; // (1 - 640 x 480, 2 - 320 x 240)
	max_distance = 6.f;
	dataset_finished = false;
}


void Datasets::openRawlog()
{

	//						Open Rawlog File
	//==================================================================
	if (!dataset.loadFromRawLogFile(filename))
		throw std::runtime_error("\nCouldn't open rawlog dataset file for input...");

	rawlog_count = 0;

	// Set external images directory:
	const string imgsPath = CRawlog::detectImagesDirectory(filename);
	CImage::IMAGES_PATH_BASE = imgsPath;

	//					Load ground-truth
	//=========================================================
	filename = system::extractFileDirectory(filename);
	filename.append("/groundtruth.txt");
	f_gt.open(filename.c_str());
	if (f_gt.fail())
		throw std::runtime_error("\nError finding the groundtruth file: it should be contained in the same folder than the rawlog file");

	//Count number of lines
	unsigned int number_of_lines = 0;
    std::string line;

    while (std::getline(f_gt, line))
        ++number_of_lines;

	gt_matrix.resize(number_of_lines-3, 8);
    f_gt.clear();
	f_gt.seekg(0, ios::beg);


	char aux[100];
	f_gt.getline(aux, 100);
	f_gt.getline(aux, 100);
	f_gt.getline(aux, 100);
	for (unsigned int k=0; k<number_of_lines-3; k++)
	{
		f_gt >> gt_matrix(k,0);
		f_gt >> gt_matrix(k,1); f_gt >> gt_matrix(k,2); f_gt >> gt_matrix(k,3);
		f_gt >> gt_matrix(k,4); f_gt >> gt_matrix(k,5); f_gt >> gt_matrix(k,6); f_gt >> gt_matrix(k,7);
		f_gt.ignore(10,'\n');	
	}

	f_gt.close();
	last_gt_row = 0;
	
}

void Datasets::loadFrameAndPoseNoInterpolation(Eigen::MatrixXf &depth_wf, Eigen::MatrixXf &color_wf, Eigen::MatrixXf &im_r, Eigen::MatrixXf &im_g,Eigen::MatrixXf &im_b)
{
	if (dataset_finished)
	{
		printf("\n End of the dataset reached. Stop estimating stuff!");
		return;
	}
	
	//Images
	//-------------------------------------------------------
	CObservationPtr alfa = dataset.getAsObservation(rawlog_count);

	while (!IS_CLASS(alfa, CObservation3DRangeScan))
	{
		rawlog_count++;
		if (dataset.size() <= rawlog_count)
		{
			dataset_finished = true;
			return;
		}
		alfa = dataset.getAsObservation(rawlog_count);
	}

	CObservation3DRangeScanPtr obs3D = CObservation3DRangeScanPtr(alfa);
	obs3D->load();
	const MatrixXf range = obs3D->rangeImage;
	const CImage int_image =  obs3D->intensityImage;
	CMatrixFloat color; int_image.getAsMatrix(color);
	CMatrixFloat r,g,b; int_image.getAsRGBMatrices(r, g, b);
	const unsigned int height = range.getRowCount();
	const unsigned int width = range.getColCount();
	const unsigned int cols = width/downsample, rows = height/downsample;

	for (unsigned int j = 0; j<cols; j++)
		for (unsigned int i = 0; i<rows; i++)
		{
			color_wf(i,j) = color(height-downsample*i-1, width-downsample*j-1);
			im_r(i,j) = b(height-downsample*i-1, width-downsample*j-1);
			im_g(i,j) = g(height-downsample*i-1, width-downsample*j-1);
			im_b(i,j) = r(height-downsample*i-1, width-downsample*j-1);
			const float z = range(height-downsample*i-1, width-downsample*j-1);
			if (z < max_distance)	depth_wf(i,j) = z;
			else					depth_wf(i,j) = 0.f;
		}


	timestamp_obs = mrpt::system::timestampTotime_t(obs3D->timestamp);

	obs3D->unload();
	rawlog_count++;

	if (dataset.size() <= rawlog_count)
		dataset_finished = true;

	//Groundtruth
	//--------------------------------------------------

	//Check whether the current gt is the closest one or we should read new gt
	const float current_dif_tim = abs(gt_matrix(last_gt_row,0) - timestamp_obs);
	const float next_dif_tim = abs(gt_matrix(last_gt_row+1,0) - timestamp_obs);

	while (abs(gt_matrix(last_gt_row,0) - timestamp_obs) > abs(gt_matrix(last_gt_row+1,0) - timestamp_obs))
	{
		last_gt_row++;
		if (last_gt_row >= gt_matrix.rows())
		{
			dataset_finished = true;
			return;		
		}
	}

	double x,y,z,qx,qy,qz,w;
	x = gt_matrix(last_gt_row,1); y = gt_matrix(last_gt_row,2); z = gt_matrix(last_gt_row,3);
	qx = gt_matrix(last_gt_row,4); qy = gt_matrix(last_gt_row,5); qz = gt_matrix(last_gt_row,6);
	w = gt_matrix(last_gt_row,7);

	CMatrixDouble33 mat;
	mat(0,0) = 1- 2*qy*qy - 2*qz*qz;
	mat(0,1) = 2*(qx*qy - w*qz);
	mat(0,2) = 2*(qx*qz + w*qy);
	mat(1,0) = 2*(qx*qy + w*qz);
	mat(1,1) = 1 - 2*qx*qx - 2*qz*qz;
	mat(1,2) = 2*(qy*qz - w*qx);
	mat(2,0) = 2*(qx*qz - w*qy);
	mat(2,1) = 2*(qy*qz + w*qx);
	mat(2,2) = 1 - 2*qx*qx - 2*qy*qy;

	poses::CPose3D gt, transf;
	gt.setFromValues(x,y,z,0,0,0);
	gt.setRotationMatrix(mat);
	transf.setFromValues(0,0,0,0.5*M_PI, -0.5*M_PI, 0);

	//Alternative - directly quaternions
	//vector<float> quat;
	//quat[0] = x, quat[1] = y; quat[2] = z;
	//quat[3] = w, quat[4] = qx; quat[5] = qy; quat[6] = qz;
	//gt.setFromXYZQ(quat);

	gt_oldpose = gt_pose;
	gt_pose = gt + transf;
}


void Datasets::CreateResultsFile()
{
	try
	{
		// Open file, find the first free file-name.
		char	aux[100];
		int     nFile = 0;
		bool    free_name = false;

		system::createDirectory("./odometry_results");

		while (!free_name)
		{
			nFile++;
			sprintf(aux, "./odometry_results/experiment_%03u.txt", nFile );
			free_name = !system::fileExists(aux);
		}

		// Open log file:
		f_res.open(aux);
		printf(" Saving results to file: %s \n", aux);
	}
	catch (...)
	{
		printf("Exception found trying to create the 'results file' !!\n");
	}
}

void Datasets::writeTrajectoryFile(poses::CPose3D &cam_pose, MatrixXf &ddt)
{	
	//Don't take into account those iterations with consecutive equal depth images
	if (abs(ddt.sumAll()) > 0)
	{		
		mrpt::math::CQuaternionDouble quat;
		poses::CPose3D auxpose, transf;
		transf.setFromValues(0,0,0,0.5*M_PI, -0.5*M_PI, 0);

		auxpose = cam_pose - transf;
		auxpose.getAsQuaternion(quat);
	
		char aux[24];
		sprintf(aux,"%.04f", timestamp_obs);
		f_res << aux << " ";
		f_res << cam_pose[0] << " ";
		f_res << cam_pose[1] << " ";
		f_res << cam_pose[2] << " ";
		f_res << quat(2) << " ";
		f_res << quat(3) << " ";
		f_res << -quat(1) << " ";
		f_res << -quat(0) << endl;
	}
}




