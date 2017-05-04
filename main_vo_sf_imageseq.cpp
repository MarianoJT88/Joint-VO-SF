
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

using namespace std;


// ------------------------------------------------------
//						MAIN
// ------------------------------------------------------

int main()
{	
    const unsigned int res_factor = 2;
	VO_SF cf(res_factor);


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
	cf.createImagePyramid();

	//Create the 3D Scene
	cf.initializeSceneSequencesVideo();

	//Auxiliary variables
	int pushed_key = 0;
	bool anything_new = 1;
    bool clean_sf = 0;
	bool continuous_run = 0;
	int stop = 0;

	
	while (!stop)
	{	

        if (cf.window.keyHit())
            pushed_key = cf.window.getPushedKey();
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
            cf.mainIteration(true);
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
            cf.mainIteration(true);
            cf.createOptLabelImage();
            anything_new = 1;
		}

		if (anything_new)
		{
			cf.updateSceneSequencesVideo();
			anything_new = 0;
		}
	}

	return 0;
}

