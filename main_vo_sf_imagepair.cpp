
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
	cf.max_iter_per_level = 2;
	cf.k_photometric_res = 0.15f;
    cf.irls_chi2_decrement_threshold = 0.98f;
    cf.irls_var_delta_threshold = 1e-6f;
	cf.use_backg_temp_reg = false;

	//Load images
	//string dir = "C:/Users/jaimez/Dropbox/Cluster-Flow/Experiments/Mo and Christian/";
	//string dir = "C:/Users/jaimez/Dropbox/Cluster-Flow/Experiments/Person Moving 1/";
	//string dir = "C:/Users/jaimez/Dropbox/Cluster-Flow/Experiments/Person Moving 2/";
	//string dir = "C:/Users/jaimez/Dropbox/Cluster-Flow/Experiments/Person Moving 3/";
	//string dir = "C:/Users/jaimez/Dropbox/Cluster-Flow/Experiments/Person Moving 4/";
	//string dir = "C:/Users/jaimez/Dropbox/Cluster-Flow/Experiments/Person Moving 5/";
	//string dir = "C:/Users/jaimez/Dropbox/Cluster-Flow/Experiments/Giraff 1/";
	//string dir = "C:/Users/jaimez/Dropbox/Cluster-Flow/Experiments/Giraff 2/";
	//string dir = "C:/Users/jaimez/Dropbox/Cluster-Flow/Experiments/Giraff 3/";
	//string dir = "C:/Users/jaimez/Dropbox/Cluster-Flow/Experiments/Giraff 4/";
	//string dir = "C:/Users/jaimez/Dropbox/Cluster-Flow/Experiments/Me moving 1/";
	//string dir = "C:/Users/jaimez/Dropbox/Cluster-Flow/Experiments/Me moving 2/";
	string dir = "C:/Users/jaimez/Dropbox/Cluster-Flow/Experiments/Two people 1/"; //*****************************
	//string dir = "C:/Users/jaimez/Dropbox/Cluster-Flow/Experiments/Whiteboard 1/";
	//string dir = "C:/Users/jaimez/Dropbox/Cluster-Flow/Experiments/Door 1/";
	//string dir = "C:/Users/jaimez/Dropbox/Cluster-Flow/Experiments/Door 2/";

	//string dir = "C:/Users/jaimez/Dropbox/Cluster-Flow/Experiments/Quiroga hand/";
	//string dir = "C:/Users/jaimez/Dropbox/Cluster-Flow/Experiments/Quiroga person/";
	//string dir = "C:/Users/jaimez/Dropbox/Cluster-Flow/Experiments/Quiroga tree/";

	cf.loadImagePairFromFiles(dir, false, res_factor);

	//Create the 3D Scene
	cf.initializeSceneCamera();

	//Auxiliary variables
	int pushed_key = 0;
	bool anything_new = 0;
    bool clean_sf = 0;
	int stop = 0;
    utils::CTicTac	main_clock, aux_clock;

	
	while (!stop)
	{	

        if (cf.m_window.keyHit())
            pushed_key = cf.m_window.getPushedKey();
        else
            pushed_key = 0;

		switch (pushed_key) {

        //Compute the solution (CPU)
        case 'a':
            cf.mainIteration();
            cf.createOptLabelImage();
            anything_new = 1;
            break;

		//Save flow to file
		case 's':
			cf.saveFlowAndSegmToFile(dir);
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
	
		if (anything_new)
		{
			cf.updateSceneCamera(clean_sf);
			anything_new = 0;
		}
	}

    cf.camera.closeCamera();
	return 0;
}

