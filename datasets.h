/*********************************************************************************
**Fast Odometry and Scene Flow from RGB-D Cameras based on Geometric Clustering	**
**------------------------------------------------------------------------------**
**																				**
**	Copyright(c) 2017, Mariano Jaimez Tarifa, University of Malaga & TU Munich	**
**	Copyright(c) 2017, Christian Kerl, TU Munich								**
**	Copyright(c) 2017, MAPIR group, University of Malaga						**
**	Copyright(c) 2017, Computer Vision group, TU Munich							**
**																				**
**  This program is free software: you can redistribute it and/or modify		**
**  it under the terms of the GNU General Public License (version 3) as			**
**	published by the Free Software Foundation.									**
**																				**
**  This program is distributed in the hope that it will be useful, but			**
**	WITHOUT ANY WARRANTY; without even the implied warranty of					**
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the				**
**  GNU General Public License for more details.								**
**																				**
**  You should have received a copy of the GNU General Public License			**
**  along with this program. If not, see <http://www.gnu.org/licenses/>.		**
**																				**
*********************************************************************************/

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
	void loadFrameAndPoseFromDataset(Eigen::MatrixXf &depth_wf, Eigen::MatrixXf &intensity_wf, Eigen::MatrixXf &im_r, Eigen::MatrixXf &im_g,Eigen::MatrixXf &im_b);
	void CreateResultsFile();
	void writeTrajectoryFile(mrpt::poses::CPose3D &cam_pose, Eigen::MatrixXf &ddt);
};




