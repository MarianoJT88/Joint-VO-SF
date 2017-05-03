
//*******************************************************
// Authors: Mariano Jaimez Tarifa & Christian Kerl
// Organizations: MAPIR, University of Malaga
//				  Computer Vision group, TUM
// Dates: September 2015 - present
// License: GNU GPL3
//*******************************************************

#include <stdio.h>
#include <string.h>
#include "joint_vo_sf.h"


// ------------------------------------------------------
//						MAIN
// ------------------------------------------------------

int main()
{	
    const unsigned int res_factor = 2;
	VO_SF cf(res_factor);

	//Flags and parameters
	cf.draw_segmented_pcloud = true;
	cf.iter_irls = 10; //15 without 2w, 5 with 2w :)
	cf.max_iter_per_level = 3;
	cf.k_photometric_res = 0.15f;
    cf.irls_chi2_decrement_threshold = 0.98f;
    cf.irls_var_delta_threshold = 1e-6f;
	cf.use_backg_temp_reg = true;


	//Load images
	unsigned int im_count;
	const unsigned int decimation = 5;
	//string dir = "D:/My RGBD sequences/Giraff loop/"; im_count = 200;
	string dir = "D:/My RGBD sequences/Giraff sinusoidal/"; im_count = 320; //250
	//string dir = "D:/My RGBD sequences/Giraff straight/"; im_count = 200;
	//string dir = "D:/My RGBD sequences/Me sitting 1/"; im_count = 1; 
	//string dir = "D:/My RGBD sequences/Me sitting 2/"; im_count = 1; 
	//string dir = "D:/My RGBD sequences/Me standing moving cam 1/"; im_count = 1; - This is bad
	//string dir = "D:/My RGBD sequences/Me standing moving cam 2/"; im_count = 1; 
	//string dir = "D:/My RGBD sequences/Me opening door cam slow 1/"; im_count = 1; 
	//string dir = "D:/My RGBD sequences/Me cleaning whiteboard 1/"; im_count = 1; 
	//string dir = "D:/My RGBD sequences/two people moving 1/"; im_count = 1;
	cf.loadImageFromSequence(dir, im_count, res_factor);

	//Create the 3D Scene
	cf.initializeSceneSequencesVideo();

	//Auxiliary variables
	int pushed_key = 0;
	bool anything_new = 1;
    bool clean_sf = 0;
	bool continuous_run = 0;
	int stop = 0;
    utils::CTicTac	main_clock, aux_clock;

	
	while (!stop)
	{	

        if (cf.m_window.keyHit())
            pushed_key = cf.m_window.getPushedKey();
        else
            pushed_key = 0;

		switch (pushed_key) {

        //Read new image and solve
        case 'n':
			printf("\n Old image = %d, new_image = %d", im_count, im_count + decimation);
			im_count += decimation;

			cf.im_r_old.swap(cf.im_r);
			cf.im_g_old.swap(cf.im_g);
			cf.im_b_old.swap(cf.im_b);
			cf.loadImageFromSequence(dir, im_count, res_factor);
            cf.mainIteration();
            cf.createOptLabelImage();
            anything_new = 1;
            break;

		//Save flow to file
		case 'q':
			cf.saveFlowAndSegmToFile(dir);
			break;

		//Start/Stop continuous estimation
		case 's':
			continuous_run = !continuous_run;
			break;


        //Print the velocities
        case 'v':
            for (unsigned int l=0; l<NUM_LABELS; l++)
            {
                printf("\nKai[%d] = ",l);
                cout << cf.kai_loc[l].transpose();
            }
            cout << endl;
            break;

		//Show original point cloud
		case 'b':
			cf.showOriginalPointCloud();
			break;

			
		//Close the program
		case 'p':
			stop = 1;
			break;
		}
	
		if (continuous_run)
		{
			im_count += decimation;

			cf.im_r_old.swap(cf.im_r);
			cf.im_g_old.swap(cf.im_g);
			cf.im_b_old.swap(cf.im_b);
			cf.loadImageFromSequence(dir, im_count, res_factor);
            cf.mainIteration();
            cf.createOptLabelImage();
            anything_new = 1;
		}

		if (anything_new)
		{
			cf.updateSceneSequencesVideo();
			anything_new = 0;
		}
	}

    cf.camera.closeCamera();
	return 0;
}

