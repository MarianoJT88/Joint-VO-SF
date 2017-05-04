
//*******************************************************
// Authors: Mariano Jaimez Tarifa & Christian Kerl
// Organizations: MAPIR, University of Malaga
//				  Computer Vision group, TUM
// Dates: September 2015 - present
// License: GNU GPL3
//*******************************************************

#include <stdio.h>
#include <string.h>
#include <joint_vo_sf.h>
#include <datasets.h>


// ------------------------------------------------------
//						MAIN
// ------------------------------------------------------

int main()
{	
    unsigned int res_factor = 2;
	VO_SF cf(res_factor);
	Datasets dataset(res_factor);

	//Flags and parameters
	const bool save_results = true;


	//Create the 3D Scene
	cf.initializeSceneDatasetVideo();

	//Open Rawlog
	//dataset.filename = "D:/TUM datasets/rawlog_rgbd_dataset_freiburg3_walking_static/rgbd_dataset_freiburg3_walking_static.rawlog";
	//dataset.filename = "D:/TUM datasets/rawlog_rgbd_dataset_freiburg3_walking_xyz/rgbd_dataset_freiburg3_walking_xyz.rawlog";
	//dataset.filename = "D:/TUM datasets/rawlog_rgbd_dataset_freiburg3_walking_halfsphere/rgbd_dataset_freiburg3_walking_halfsphere.rawlog";
	//dataset.filename = "D:/TUM datasets/rawlog_rgbd_dataset_freiburg3_sitting_static/rgbd_dataset_freiburg3_sitting_static.rawlog";
	//dataset.filename = "D:/TUM datasets/rawlog_rgbd_dataset_freiburg3_sitting_xyz/rgbd_dataset_freiburg3_sitting_xyz.rawlog";
	//dataset.filename = "D:/TUM datasets/rawlog_rgbd_dataset_freiburg3_sitting_halfsphere/rgbd_dataset_freiburg3_sitting_halfsphere.rawlog";

	dataset.filename = "D:/TUM datasets/rawlog_rgbd_dataset_freiburg1_desk/rgbd_dataset_freiburg1_desk.rawlog";
	//dataset.filename = "D:/TUM datasets/rawlog_rgbd_dataset_freiburg1_desk2/rgbd_dataset_freiburg1_desk2.rawlog";
	//dataset.filename = "D:/TUM datasets/rawlog_rgbd_dataset_freiburg1_teddy/rgbd_dataset_freiburg1_teddy.rawlog";

	if (save_results)
		dataset.CreateResultsFile();
    dataset.openRawlog();
	dataset.loadFrameAndPoseFromDataset(cf.depth_wf, cf.color_wf, cf.im_r, cf.im_g, cf.im_b);
	cf.cam_pose = dataset.gt_pose; cf.cam_oldpose = dataset.gt_pose;
    cf.createImagePyramid();


	//Auxiliary variables
	int pushed_key = 0;
	bool anything_new = 0;
    bool realtime = 0;
	int stop = 0;
	
	while (!stop)
	{	

        if (cf.window.keyHit())
            pushed_key = cf.window.getPushedKey();
        else
            pushed_key = 0;

		switch (pushed_key) {
			
        //Capture a new frame and solve
		case  'n':
			cf.im_r_old.swap(cf.im_r);
			cf.im_g_old.swap(cf.im_g);
			cf.im_b_old.swap(cf.im_b);
            dataset.loadFrameAndPoseFromDataset(cf.depth_wf, cf.color_wf, cf.im_r, cf.im_g, cf.im_b);
			cf.createImagePyramid();
            cf.mainIteration();
            cf.createOptLabelImage();
			if (save_results)
				dataset.writeTrajectoryFile(cf.cam_pose, cf.ddt);
            anything_new = 1;
			break;

        //Only solve
        case 'a':
            cf.mainIteration();
            cf.createOptLabelImage();
            anything_new = 1;
            break;

        //Turn on/off continuous estimation
        case 's':
            realtime = !realtime;
            break;

		//Save segmentation in color
        case 'g':
            cf.createOptLabelImage();
			cf.saveSegmentationImage();
            fflush(stdout);
            break;

        //Compute percentage of moving and uncertain pixels
        case 'e':
		{
			const float perc_moving_pixels = float(cf.num_mov_pixels)/float(cf.num_valid_pixels);
			const float perc_uncertain_pixels = float(cf.num_uncertain_pixels)/float(cf.num_valid_pixels);
			const float perc_valid_pixels = float(cf.num_valid_pixels)/float(cf.num_images*cf.rows*cf.cols);
			const float min_perc_valid_pixels = float(cf.min_num_valid_pixels)/float(cf.rows*cf.cols);
			printf("\n Percentage of moving pixels = %f, Percentage of uncertain pixels = %f", perc_moving_pixels, perc_uncertain_pixels);
			printf("\n Percentage of valid pixels (aver) = %f, min percentage of valid pixels = %f", perc_valid_pixels, min_perc_valid_pixels);
			break;
		}
		
		//Close the program
		case 'p':
			stop = 1;
			break;
		}

        if (realtime)
        {
            cf.im_r_old.swap(cf.im_r);
			cf.im_g_old.swap(cf.im_g);
			cf.im_b_old.swap(cf.im_b);
			dataset.loadFrameAndPoseFromDataset(cf.depth_wf, cf.color_wf, cf.im_r, cf.im_g, cf.im_b);
			cf.createImagePyramid();
            cf.mainIteration();
            cf.createOptLabelImage();
			if (save_results)
				dataset.writeTrajectoryFile(cf.cam_pose, cf.ddt);
            anything_new = 1;

			if (dataset.dataset_finished)
				realtime = false;
        }
	
		if (anything_new)
		{
			bool aux = false;
			cf.updateSceneDatasetVideo(dataset.gt_pose, dataset.gt_oldpose);
			anything_new = 0;
		}
	}

	if (save_results)
		dataset.f_res.close();

	return 0;
}

