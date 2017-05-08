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

#include <string.h>
#include <joint_vo_sf.h>


// ------------------------------------------------------
//						MAIN
// ------------------------------------------------------

int main()
{	
    const unsigned int res_factor = 2;
	VO_SF cf(res_factor);


	//Load images
	unsigned int im_count;
	const unsigned int decimation = 5; //5
	//string dir = "D:/My RGBD sequences/Giraff loop/"; im_count = 200;
	std::string dir = "D:/My RGBD sequences/Giraff sinusoidal/"; im_count = 320; //250
	//string dir = "D:/My RGBD sequences/Giraff straight/"; im_count = 200;
	//string dir = "D:/My RGBD sequences/Me sitting 1/"; im_count = 1; 
	//string dir = "D:/My RGBD sequences/Me sitting 2/"; im_count = 1; 
	//string dir = "D:/My RGBD sequences/Me standing moving cam 1/"; im_count = 1; //- This is bad
	//string dir = "D:/My RGBD sequences/Me standing moving cam 2/"; im_count = 1; 
	//string dir = "D:/My RGBD sequences/Me opening door cam slow 1/"; im_count = 1; 
	//string dir = "D:/My RGBD sequences/Me cleaning whiteboard 1/"; im_count = 1; 
	//string dir = "D:/My RGBD sequences/two people moving 1/"; im_count = 1;
	cf.loadImageFromSequence(dir, im_count, res_factor);
	cf.createImagePyramid();

	//Create the 3D Scene
	cf.initializeSceneImageSeq();

	//Auxiliary variables
	int pushed_key = 0;
	bool continuous_exec = false, stop = false;
	
	while (!stop)
	{	
        if (cf.window.keyHit())
            pushed_key = cf.window.getPushedKey();
        else
            pushed_key = 0;

		switch (pushed_key) {

        //Read new image and solve
        case 'n':
			im_count += decimation;
			cf.im_r_old.swap(cf.im_r);
			cf.im_g_old.swap(cf.im_g);
			cf.im_b_old.swap(cf.im_b);
			stop = cf.loadImageFromSequence(dir, im_count, res_factor);
            cf.mainIteration(true);
            cf.createImagesOfSegmentations();
            cf.updateSceneImageSeq();
            break;

		//Start/Stop continuous estimation
		case 's':
			continuous_exec = !continuous_exec;
			break;
			
		//Close the program
		case 'p':
			stop = true;
			break;
		}
	
		if ((continuous_exec)&&(!stop))
		{
			im_count += decimation;
			cf.im_r_old.swap(cf.im_r);
			cf.im_g_old.swap(cf.im_g);
			cf.im_b_old.swap(cf.im_b);
			stop = cf.loadImageFromSequence(dir, im_count, res_factor);
            cf.mainIteration(true);
            cf.createImagesOfSegmentations();
			cf.updateSceneImageSeq();
		}
	}

	return 0;
}

