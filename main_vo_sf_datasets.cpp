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

#include <stdio.h>
#include <joint_vo_sf.h>
#include <datasets.h>


// -------------------------------------------------------------------------------
//								Instructions:
// You need to click on the window of the 3D Scene to be able to interact with it.
// 'n' - Load new frame and solve
// 's' - Turn on/off continuous estimation
// 'e' - Finish/exit
//
// Set the flag "save_results" to true if you want to save the estimated trajectory
// -------------------------------------------------------------------------------

int main()
{	
	const bool save_results = true;
    unsigned int res_factor = 2;
	VO_SF cf(res_factor);
	Datasets dataset(res_factor);


	//Set dir of the Rawlog file
	//dataset.filename = "D:/TUM datasets/rawlog_rgbd_dataset_freiburg3_walking_static/rgbd_dataset_freiburg3_walking_static.rawlog";
	//dataset.filename = "D:/TUM datasets/rawlog_rgbd_dataset_freiburg3_walking_xyz/rgbd_dataset_freiburg3_walking_xyz.rawlog";
	//dataset.filename = "D:/TUM datasets/rawlog_rgbd_dataset_freiburg3_walking_halfsphere/rgbd_dataset_freiburg3_walking_halfsphere.rawlog";
	//dataset.filename = "D:/TUM datasets/rawlog_rgbd_dataset_freiburg3_sitting_static/rgbd_dataset_freiburg3_sitting_static.rawlog";
	//dataset.filename = "D:/TUM datasets/rawlog_rgbd_dataset_freiburg3_sitting_xyz/rgbd_dataset_freiburg3_sitting_xyz.rawlog";
	//dataset.filename = "D:/TUM datasets/rawlog_rgbd_dataset_freiburg3_sitting_halfsphere/rgbd_dataset_freiburg3_sitting_halfsphere.rawlog";
	dataset.filename = "D:/TUM datasets/rawlog_rgbd_dataset_freiburg1_desk/rgbd_dataset_freiburg1_desk.rawlog";
	//dataset.filename = "D:/TUM datasets/rawlog_rgbd_dataset_freiburg1_desk2/rgbd_dataset_freiburg1_desk2.rawlog";
	//dataset.filename = "D:/TUM datasets/rawlog_rgbd_dataset_freiburg1_teddy/rgbd_dataset_freiburg1_teddy.rawlog";

	//Create the 3D Scene
	cf.initializeSceneDatasets();

	//Initialize
	if (save_results)
		dataset.CreateResultsFile();
    dataset.openRawlog();
	dataset.loadFrameAndPoseFromDataset(cf.depth_wf, cf.intensity_wf, cf.im_r, cf.im_g, cf.im_b);
	cf.cam_pose = dataset.gt_pose; cf.cam_oldpose = dataset.gt_pose;
    cf.createImagePyramid();


	//Auxiliary variables
	int pushed_key = 0, stop = 0;
	bool anything_new = false, continuous_exec = false;
	
	while (!stop)
	{	

        if (cf.window.keyHit())
            pushed_key = cf.window.getPushedKey();
        else
            pushed_key = 0;

		switch (pushed_key) {
			
        //Load new frame and solve
		case  'n':
            dataset.loadFrameAndPoseFromDataset(cf.depth_wf, cf.intensity_wf, cf.im_r, cf.im_g, cf.im_b);
            cf.run_VO_SF(true);
            cf.createImagesOfSegmentations();
			if (save_results)
				dataset.writeTrajectoryFile(cf.cam_pose, cf.ddt);
            anything_new = 1;
			break;

        //Turn on/off continuous estimation
        case 's':
            continuous_exec = !continuous_exec;
            break;
		
		//Close the program
		case 'e':
			stop = 1;
			break;
		}

        if (continuous_exec)
        {
			dataset.loadFrameAndPoseFromDataset(cf.depth_wf, cf.intensity_wf, cf.im_r, cf.im_g, cf.im_b);
            cf.run_VO_SF(true);
            cf.createImagesOfSegmentations();
			if (save_results)
				dataset.writeTrajectoryFile(cf.cam_pose, cf.ddt);
            anything_new = 1;

			if (dataset.dataset_finished)
				continuous_exec = false;
        }
	
		if (anything_new)
		{
			bool aux = false;
			cf.updateSceneDatasets(dataset.gt_pose, dataset.gt_oldpose);
			anything_new = 0;
		}
	}

	if (save_results)
		dataset.f_res.close();

	return 0;
}

