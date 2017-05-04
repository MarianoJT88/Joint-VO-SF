
#include <mrpt/utils.h>
#include <mrpt/obs/CRawlog.h>
#include <mrpt/utils/CConfigFileBase.h>
#include <mrpt/system/filesystem.h>
#include <mrpt/obs/CObservation3DRangeScan.h>
#include <Eigen/Core>
#include <iostream>
#include <fstream>


class Datasets {
public:

    Datasets(unsigned int res_factor);

	unsigned int rawlog_count;
	unsigned int last_gt_row;
	unsigned int downsample;
	float max_distance;

	mrpt::obs::CRawlog	dataset;
	std::ifstream		f_gt;
	std::ofstream		f_res;
	std::string			filename;

	Eigen::MatrixXd gt_matrix;
	mrpt::poses::CPose3D gt_pose;		//!< Groundtruth camera pose
	mrpt::poses::CPose3D gt_oldpose;	//!< Groundtruth camera previous pose
	double timestamp_obs;				//!< Timestamp of the last observation
	bool dataset_finished;

    void openRawlog();
	void loadFrameAndPoseFromDataset(Eigen::MatrixXf &depth_wf, Eigen::MatrixXf &color_wf, Eigen::MatrixXf &im_r, Eigen::MatrixXf &im_g,Eigen::MatrixXf &im_b);
	void CreateResultsFile();
	void writeTrajectoryFile(mrpt::poses::CPose3D &cam_pose, Eigen::MatrixXf &ddt);
};




