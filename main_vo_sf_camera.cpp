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

#include <joint_vo_sf.h>
#include <camera.h>


// ------------------------------------------------------
//						MAIN
// ------------------------------------------------------

int main()
{	
    unsigned int res_factor = 1;
	VO_SF cf(res_factor);
	RGBD_Camera camera(res_factor);


	//Create the 3D Scene
	cf.initializeSceneCamera();

	//Initialize camera and method
    camera.openCamera();
    camera.disableAutoExposureAndWhiteBalance();
	camera.loadFrame(cf.depth_wf, cf.intensity_wf);
    cf.createImagePyramid();
	camera.loadFrame(cf.depth_wf, cf.intensity_wf);
    cf.createImagePyramid();
	cf.initializeKMeans();

	//Auxiliary variables for the interface
	int pushed_key = 0;
	bool anything_new = false, stop = false;
    bool clean_sf = false, continuous_exec = false;

	
	while (!stop)
	{	

        if (cf.window.keyHit())
            pushed_key = cf.window.getPushedKey();
        else
            pushed_key = 0;

		switch (pushed_key) {
			
        //Capture a new frame
		case  'n':
            camera.loadFrame(cf.depth_wf, cf.intensity_wf);
			cf.createImagePyramid();
			cf.kMeans3DCoordLowRes();
            cf.createOptLabelImage();

            anything_new = true;
            clean_sf = true;
			break;

        //Compute the solution
        case 'a':
            cf.mainIteration(false);
            cf.createOptLabelImage();

            anything_new = true;
            break;

        //Turn on/off continuous estimation
        case 's':
            continuous_exec = !continuous_exec;
            break;

		//Reset the camera pose
		case 'r':
			cf.cam_pose.setFromValues(0,0,1.5,0,0,0);
			anything_new = true;
			break;
			
		//Close the program
		case 'p':
			stop = true;
			break;
		}

        if (continuous_exec)
        {
            camera.loadFrame(cf.depth_wf, cf.intensity_wf);
            cf.mainIteration(true);
            cf.createOptLabelImage();
            anything_new = 1;
        }
	
		if (anything_new)
		{
			cf.updateSceneCamera(clean_sf);
			clean_sf = false;
			anything_new = 0;
		}
	}

    camera.closeCamera();
	return 0;
}

