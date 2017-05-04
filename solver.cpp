
#include "joint_vo_sf.h"
#include "structs_parallelization.h"

using namespace mrpt;
using namespace mrpt::math;
using namespace mrpt::poses;
using namespace mrpt::utils;
using namespace std;
using namespace Eigen;


VO_SF::VO_SF(unsigned int res_factor) : T(NUM_LABELS), ws_foreground(640*480), ws_background(640*480)
{
    rows = 240; //I should also change the amount of pixels to remove a label
    cols = 320;
	fovh = M_PI*62.5/180.0;
    fovv = M_PI*48.5/180.0;
    width = 640/res_factor;
    height = 480/res_factor;
	ctf_levels = log2(cols/40) + 2;
	k_photometric_res = 0.15f;
	irls_chi2_decrement_threshold = 0.98f;
    irls_var_delta_threshold = 1e-6f;
	iter_irls = 10;
	max_iter_per_level = 3;
	use_backg_temp_reg = false;

	//Velocities and poses
	cam_pose.setFromValues(0,0,0,0,0,0);
	cam_oldpose = cam_pose;

	//Resize matrices which are not in a "pyramid"
	depth_wf.setSize(height,width);
	intensity_wf.setSize(height,width);
    motionfield[0].setSize(rows,cols);
    motionfield[1].setSize(rows,cols);
    motionfield[2].setSize(rows,cols);
	dct.resize(rows,cols); ddt.resize(rows,cols);
    dcu.resize(rows,cols); ddu.resize(rows,cols);
    dcv.resize(rows,cols); ddv.resize(rows,cols);
	im_r.resize(rows,cols); im_g.resize(rows,cols); im_b.resize(rows,cols);
	im_r_old.resize(rows,cols); im_g_old.resize(rows,cols); im_b_old.resize(rows,cols);
    Null.resize(rows,cols);
    weights_c.setSize(rows,cols);
    weights_d.setSize(rows,cols);


	//Resize matrices in a "pyramid"
    const unsigned int pyr_levels = round(log2(width/cols)) + ctf_levels;
    intensity.resize(pyr_levels); intensity_old.resize(pyr_levels); intensity_inter.resize(pyr_levels);
    depth.resize(pyr_levels); depth_old.resize(pyr_levels); depth_inter.resize(pyr_levels);
    xx.resize(pyr_levels); xx_inter.resize(pyr_levels); xx_old.resize(pyr_levels);
    yy.resize(pyr_levels); yy_inter.resize(pyr_levels); yy_old.resize(pyr_levels);
    intensity_warped.resize(pyr_levels);
    depth_warped.resize(pyr_levels);
    xx_warped.resize(pyr_levels);
    yy_warped.resize(pyr_levels);
	labels.resize(pyr_levels);

	for (unsigned int i = 0; i<pyr_levels; i++)
    {
        const unsigned int s = pow(2.f,int(i));
        cols_i = width/s; rows_i = height/s;
        intensity[i].resize(rows_i, cols_i); intensity_old[i].resize(rows_i, cols_i); intensity_inter[i].resize(rows_i, cols_i);
        depth[i].resize(rows_i, cols_i); depth_inter[i].resize(rows_i, cols_i); depth_old[i].resize(rows_i, cols_i);
        depth[i].assign(0.f); depth_old[i].assign(0.f);
        xx[i].resize(rows_i, cols_i); xx_inter[i].resize(rows_i, cols_i); xx_old[i].resize(rows_i, cols_i);
        xx[i].assign(0.f); xx_old[i].assign(0.f);
        yy[i].resize(rows_i, cols_i); yy_inter[i].resize(rows_i, cols_i); yy_old[i].resize(rows_i, cols_i);
        yy[i].assign(0.f); yy_old[i].assign(0.f);

		if (cols_i <= cols)
		{
            intensity_warped[i].resize(rows_i,cols_i);
            depth_warped[i].resize(rows_i,cols_i);
            xx_warped[i].resize(rows_i,cols_i);
            yy_warped[i].resize(rows_i,cols_i);
			labels[i].resize(rows_i, cols_i);
		}
    }

    //Compute gaussian and "fast-symmetric" mask
    const Vector4f v_mask(1.f, 2.f, 2.f, 1.f);
    for (unsigned int i=0; i<4; i++)
        for (unsigned int j=0; j<4; j++)
            f_mask(i,j) = v_mask(i)*v_mask(j)/36.f;


    //                      Labels
    //=========================================================
	bf_segm_image_warped.setSize(rows,cols);
	bf_segm_image_warped.fill(0.f);
	label_in_backg.fill(false);
	label_in_foreg.fill(true);
	backg_image[0].resize(rows,cols);
	backg_image[1].resize(rows,cols);
	backg_image[2].resize(rows,cols);

    for (unsigned int c=0; c<3; c++)
        olabels_image[c].resize(rows,cols);

    labels_opt.resize(pyr_levels);
    for (unsigned int i = 0; i<pyr_levels; i++)
    {
        const unsigned int s = pow(2.f,int(i));
        cols_i = width/s; rows_i = height/s;
        if (cols_i <= cols)
        {
            labels_opt[i].resize(NUM_LABELS+1, rows_i*cols_i);
            labels_opt[i].assign(0.f);
        }
    }

	//Statistics for the segmentation
	num_valid_pixels = 0;
	num_mov_pixels = 0;
	num_uncertain_pixels = 0;
	min_num_valid_pixels = rows*cols;
	num_images = 0;
}

void VO_SF::loadImagePairFromFiles(string files_dir, bool is_Quiroga, unsigned int res_factor)
{
    const float norm_factor = 1.f/255.f;
    char aux[30];

    //                              Load the first frame
    //==============================================================================
    sprintf(aux, "intensity0.png");
    string name = files_dir + aux;

    cv::Mat intensity = cv::imread(name.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    for (unsigned int v=0; v<height; v++)
        for (unsigned int u=0; u<width; u++)
            intensity_wf(height-1-v,u) = norm_factor*intensity.at<unsigned char>(res_factor*v+1,res_factor*u);

    sprintf(aux, "depth0.png");
    name = files_dir + aux;

    cv::Mat depth = cv::imread(name, -1);
    cv::Mat depth_float;
    depth.convertTo(depth_float, CV_32FC1, 1.0 / 5000.0);

    for (unsigned int v=0; v<height; v++)
        for (unsigned int u=0; u<width; u++)
            depth_wf(height-1-v,u) = depth_float.at<float>(res_factor*v+1,res_factor*u);

	if (is_Quiroga)
		depth_wf *= 5.f;

	createImagePyramid();



    //                              Load the second frame
    //==============================================================================
    sprintf(aux, "intensity1.png");
    name = files_dir + aux;

    intensity = cv::imread(name.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    for (unsigned int v=0; v<height; v++)
        for (unsigned int u=0; u<width; u++)
            intensity_wf(height-1-v,u) = norm_factor*intensity.at<unsigned char>(res_factor*v+1,res_factor*u);

    sprintf(aux, "depth1.png");
    name = files_dir + aux;

    depth = cv::imread(name, -1);
    depth.convertTo(depth_float, CV_32FC1, 1.0 / 5000.0);
    for (unsigned int v=0; v<height; v++)
        for (unsigned int u=0; u<width; u++)
            depth_wf(height-1-v,u) = depth_float.at<float>(res_factor*v+1,res_factor*u);

	if (is_Quiroga)
		depth_wf *= 5.f;

	createImagePyramid();
}

void VO_SF::loadImageFromSequence(string files_dir, unsigned int index, unsigned int res_factor)
{
    const float norm_factor = 1.f/255.f;
    char aux[30];

    //                              Load the first frame
    //==============================================================================
    sprintf(aux, "i%d.png", index);
    string name = files_dir + aux;

	cv::Mat color = cv::imread(name.c_str(), CV_LOAD_IMAGE_COLOR);
    for (unsigned int v=0; v<height; v++)
        for (unsigned int u=0; u<width; u++)
		{
			cv::Vec3b color_here = color.at<cv::Vec3b>(res_factor*v,res_factor*u);
			im_r(height-1-v,u) = norm_factor*color_here[2];
			im_g(height-1-v,u) = norm_factor*color_here[1];
			im_b(height-1-v,u) = norm_factor*color_here[0];
			intensity_wf(height-1-v,u) = 0.299f*im_r(height-1-v,u) + 0.587f*im_g(height-1-v,u) + 0.114f*im_b(height-1-v,u);
		}

    sprintf(aux, "d%d.png", index);
    name = files_dir + aux;

    cv::Mat depth = cv::imread(name, -1);
    cv::Mat depth_float;
    depth.convertTo(depth_float, CV_32FC1, 1.0 / 5000.0);

    for (unsigned int v=0; v<height; v++)
        for (unsigned int u=0; u<width; u++)
            depth_wf(height-1-v,u) = depth_float.at<float>(res_factor*v,res_factor*u);
}

void VO_SF::saveFlowAndSegmToFile(string files_dir)
{
    char aux[30];
	string name;
	sprintf(aux, "ClusterFlow.xml");
    name = files_dir + aux;

	cv::FileStorage SFlow;
    SFlow.open(name, cv::FileStorage::WRITE);

    //Stored in camera coordinates (z -> pointing to the front)
	cv::Mat rmx(rows, cols, CV_32FC1), rmy(rows, cols, CV_32FC1), rmz(rows, cols, CV_32FC1);
	cv::Mat segm(rows, cols, CV_8U), segm_col(rows, cols, CV_8UC3), kmeans(rows, cols, CV_8UC3);
    for (unsigned int v=0; v<rows; v++)
        for (unsigned int u=0; u<cols; u++)
        {
            rmx.at<float>(v,u) = motionfield[1](rows-1-v,u);
            rmy.at<float>(v,u) = motionfield[2](rows-1-v,u);
            rmz.at<float>(v,u) = motionfield[0](rows-1-v,u);
			segm.at<unsigned char>(v,u) = int(255.f*min(1.f, bf_segm[labels[image_level](rows-1-v,u)]));
			segm_col.at<cv::Vec3b>(v,u) = cv::Vec3b(255.f*backg_image[2](rows-1-v,u), 255.f*backg_image[1](rows-1-v,u), 255.f*backg_image[0](rows-1-v,u));
			kmeans.at<cv::Vec3b>(v,u) = cv::Vec3b(255.f* olabels_image[2](rows-1-v,u), 255.f*olabels_image[1](rows-1-v,u), 255.f*olabels_image[0](rows-1-v,u));
        }

    SFlow << "SFx" << rmx;
    SFlow << "SFy" << rmy;
    SFlow << "SFz" << rmz;
    SFlow.release();
	cout << endl << "Scene flow saved in " << name;

	//Save segmentations
	sprintf(aux, "Segmentation_backg.png");
    name = files_dir + aux;
	cv::imwrite(name, segm);
	cout << endl << "Segmentation (grayscale) saved in " << name;

	sprintf(aux, "Segmentation_backg_color.png");
    name = files_dir + aux;
	cv::imwrite(name, segm_col);
	cout << endl << "Segmentation (color) saved in " << name;

	sprintf(aux, "Segmentation_kmeans.png");
    name = files_dir + aux;
	cv::imwrite(name, kmeans);
	cout << endl << "Segmentation (kmeans) saved in " << name;
}

void VO_SF::saveSegmentationImage()
{
	cv::Mat segm(rows, cols, CV_8UC3);
    for (unsigned int v=0; v<rows; v++)
        for (unsigned int u=0; u<cols; u++)
			segm.at<cv::Vec3b>(v,u) = cv::Vec3b(255.f*backg_image[2](rows-1-v,u), 255.f*backg_image[1](rows-1-v,u), 255.f*backg_image[0](rows-1-v,u));

	string name = ".../segmentation_color.png"; //Set the directory where it should be saved here
	cv::imwrite(name, segm);
	cout << endl << "Segmentation image saved in " << name;
}


void VO_SF::createImagePyramid()
{	
	//Threshold to use (or not) neighbours in the filter
	const float max_depth_dif = 0.1f;

    //Push the frames back
    intensity_old.swap(intensity);
    depth_old.swap(depth);
    xx_old.swap(xx);
    yy_old.swap(yy);

    //The number of levels of the pyramid does not match the number of levels used
    //in the odometry computation (because we sometimes want to finish with lower resolutions)
    unsigned int pyr_levels = round(log2(width/cols)) + ctf_levels;

    //Generate levels
    for (unsigned int i = 0; i<pyr_levels; i++)
    {
        unsigned int s = pow(2.f,int(i));
        cols_i = width/s;
        rows_i = height/s;
        const unsigned int i_1 = i-1;
		MatrixXf &depth_here = depth[i];
		MatrixXf &intensity_here = intensity[i];
		MatrixXf &xx_here = xx[i];
		MatrixXf &yy_here = yy[i];

        if (i == 0)
        {
            depth_here.swap(depth_wf);
            intensity_here.swap(intensity_wf);
        }

        //                              Downsampling
        //-----------------------------------------------------------------------------
        else
        {
            const MatrixXf &depth_prev = depth[i_1];
			const MatrixXf &intensity_prev = intensity[i_1];
			
			for (unsigned int u = 0; u < cols_i; u++)
                for (unsigned int v = 0; v < rows_i; v++)
                {
                    const int u2 = 2*u;
                    const int v2 = 2*v;

                    //Inner pixels
                    if ((v>0)&&(v<rows_i-1)&&(u>0)&&(u<cols_i-1))
                    {
                        const Matrix4f depth_block = depth_prev.block<4,4>(v2-1,u2-1);
                        const Matrix4f intensity_block = intensity_prev.block<4,4>(v2-1,u2-1);
                        float depths[4] = {depth_block(5), depth_block(6), depth_block(9), depth_block(10)};

                        //Find the "second maximum" value of the central block
						if (depths[1] < depths[0]) {std::swap(depths[1], depths[0]);}
						if (depths[3] < depths[2]) {std::swap(depths[3], depths[2]);}
						const float dcenter = (depths[3] < depths[1]) ? max(depths[3], depths[0]) : max(depths[1], depths[2]);

                        if (dcenter != 0.f)
                        {
                            float sum_d = 0.f;
                            float sum_c = 0.f;
                            float weight = 0.f;

                            for (unsigned char k=0; k<16; k++)
                            {
                                const float abs_dif = abs(depth_block(k)-dcenter);
                                if (abs_dif < max_depth_dif)
                                {
                                    const float aux_w = f_mask(k)*(max_depth_dif - abs_dif);
                                    weight += aux_w;
                                    sum_d += aux_w*depth_block(k);
                                    sum_c += aux_w*intensity_block(k);
                                }
                            }
                            depth_here(v,u) = sum_d/weight;
                            intensity_here(v,u) = sum_c/weight;
                        }
                        else
                        {
                            float sum_c = 0.f;
                            for (unsigned char k=0; k<16; k++)
                                sum_c += f_mask(k)*intensity_block(k);

                            depth_here(v,u) = 0.f;
                            intensity_here(v,u) = sum_c;
                        }
                    }

                    //Boundary
                    else
                    {
                        const Matrix2f depth_block = depth_prev.block<2,2>(v2,u2);
                        const Matrix2f intensity_block = intensity_prev.block<2,2>(v2,u2);

						intensity_here(v,u) = 0.25f*intensity_block.sumAll();

						float new_d = 0.f;
						unsigned int cont = 0;
                        for (unsigned int k=0; k<4;k++)
							if (depth_block(k) != 0.f)
							{
								new_d += depth_block(k);
								cont++;
							}

                        if (cont != 0)	depth_here(v,u) = new_d/float(cont);
                        else		    depth_here(v,u) = 0.f;
                    }
                }
        }

        //Calculate coordinates "xy" of the points
        const float inv_f_i = 2.f*tan(0.5f*fovh)/float(cols_i);
        const float disp_u_i = 0.5f*(cols_i-1);
        const float disp_v_i = 0.5f*(rows_i-1);

        for (unsigned int u = 0; u < cols_i; u++)
            for (unsigned int v = 0; v < rows_i; v++)
                if (depth_here(v,u) != 0.f)
                {
                    xx_here(v,u) = (u - disp_u_i)*depth_here(v,u)*inv_f_i;
                    yy_here(v,u) = (v - disp_v_i)*depth_here(v,u)*inv_f_i;
                }
                else
                {
                    xx_here(v,u) = 0.f;
                    yy_here(v,u) = 0.f;
                }
    }
}

void VO_SF::calculateCoord()
{
    calculateCoord(cv::Rect(0, 0, cols_i, rows_i));
}

void VO_SF::calculateCoord(cv::Rect region)
{		
    unsigned int x = region.tl().x, y = region.tl().y, w = region.width, h = region.height;

    Null.block(y,x,h,w).assign(false);

	//Refs
	const MatrixXf &depth_old_ref = depth_old[image_level];
	const MatrixXf &depth_warped_ref = depth_warped[image_level];

	MatrixXf &depth_inter_ref = depth_inter[image_level];
	MatrixXf &intensity_inter_ref = intensity_inter[image_level];
	MatrixXf &xx_inter_ref = xx_inter[image_level];
	MatrixXf &yy_inter_ref = yy_inter[image_level];

    for (unsigned int u = x; u < x+w; u++)
        for (unsigned int v = y; v < y+h; v++)
		{
			if ((depth_old_ref(v,u) != 0.f)&&(depth_warped_ref(v,u) != 0.f))
            {
				depth_inter_ref(v,u) = 0.5f*(depth_old_ref(v,u) + depth_warped_ref(v,u));
                xx_inter_ref(v,u) = 0.5f*(xx_old[image_level](v,u) + xx_warped[image_level](v,u));
                yy_inter_ref(v,u) = 0.5f*(yy_old[image_level](v,u) + yy_warped[image_level](v,u));
            }
			else
			{
                Null(v,u) = true;
                depth_inter_ref(v,u) = 0.f;
                xx_inter_ref(v,u) = 0.f;
                yy_inter_ref(v,u) = 0.f;
			}

            intensity_inter_ref(v,u) = 0.5f*(intensity_old[image_level](v,u) + intensity_warped[image_level](v,u));
		}
}

void VO_SF::calculateDerivatives()
{
	//Compute connectivity
	MatrixXf rx(rows_i,cols_i), ry(rows_i,cols_i);
    rx.fill(1.f); ry.fill(1.f);

	MatrixXf rx_intensity(rows_i,cols_i), ry_intensity(rows_i, cols_i);
    rx_intensity.fill(1.f); ry_intensity.fill(1.f);

	const MatrixXf &depth_ref = depth_inter[image_level];
	const MatrixXf &intensity_ref = intensity_inter[image_level];
	const MatrixXf &xx_ref = xx_inter[image_level];
	const MatrixXf &yy_ref = yy_inter[image_level];

    const float epsilon_intensity = 1e-6f;
	const float epsilon_depth = 0.005f;

    for (unsigned int u = 0; u < cols_i-1; u++)
        for (unsigned int v = 0; v < rows_i; v++)
            if (Null(v,u) == false)
            {
                //rx(v,u) = sqrtf(square(xx_ref(v,u+1) - xx_ref(v,u)) + square(depth_ref(v,u+1) - depth_ref(v,u)));
				rx(v,u) = abs(depth_ref(v,u+1) - depth_ref(v,u)) + epsilon_depth;
				rx_intensity(v,u) = abs(intensity_ref(v,u+1) - intensity_ref(v,u)) + epsilon_intensity;
            }

    for (unsigned int u = 0; u < cols_i; u++)
        for (unsigned int v = 0; v < rows_i-1; v++)
            if (Null(v,u) == false)
            {
                //ry(v,u) = sqrtf(square(yy_ref(v+1,u) - yy_ref(v,u)) + square(depth_ref(v+1,u) - depth_ref(v,u)));
				ry(v,u) = abs(depth_ref(v+1,u) - depth_ref(v,u)) + epsilon_depth;
				ry_intensity(v,u) = abs(intensity_ref(v+1,u) - intensity_ref(v,u)) + epsilon_intensity;
            }

	//Alternative using block operations (same speed in my test with few null pixels)
	//rx.block(0,0, rows_i, cols_i-1) = (depth_ref.block(0,1,rows_i,cols_i-1) - depth_ref.block(0,0,rows_i,cols_i-1)).array().abs() + epsilon_depth;
	//ry.block(0,0, rows_i-1, cols_i) = (depth_ref.block(1,0,rows_i-1,cols_i) - depth_ref.block(0,0,rows_i-1,cols_i)).array().abs() + epsilon_depth;
	//rx_intensity.block(0,0, rows_i, cols_i-1) = (intensity_ref.block(0,1,rows_i,cols_i-1) - intensity_ref.block(0,0,rows_i,cols_i-1)).array().abs() + epsilon_intensity;
	//ry_intensity.block(0,0, rows_i-1, cols_i) = (intensity_ref.block(1,0,rows_i-1,cols_i) - intensity_ref.block(0,0,rows_i-1,cols_i)).array().abs() + epsilon_intensity;


    //Spatial derivatives
    for (unsigned int v = 0; v < rows_i; v++)
        for (unsigned int u = 1; u < cols_i-1; u++)
            if (Null(v,u) == false)
            {
                dcu(v,u) = (rx_intensity(v,u-1)*(intensity_ref(v,u+1)-intensity_ref(v,u)) + rx_intensity(v,u)*(intensity_ref(v,u) - intensity_ref(v,u-1)))/(rx_intensity(v,u)+rx_intensity(v,u-1));
                ddu(v,u) = (rx(v,u-1)*(depth_ref(v,u+1)-depth_ref(v,u)) + rx(v,u)*(depth_ref(v,u) - depth_ref(v,u-1)))/(rx(v,u)+rx(v,u-1));
            }

	dcu.col(0) = dcu.col(1);
	dcu.col(cols_i-1) = dcu.col(cols_i-2);
	ddu.col(0) = ddu.col(1);
	ddu.col(cols_i-1) = ddu.col(cols_i-2);

    for (unsigned int u = 0; u < cols_i; u++)
        for (unsigned int v = 1; v < rows_i-1; v++)
            if (Null(v,u) == false)
            {
                dcv(v,u) = (ry_intensity(v-1,u)*(intensity_ref(v+1,u)-intensity_ref(v,u)) + ry_intensity(v,u)*(intensity_ref(v,u) - intensity_ref(v-1,u)))/(ry_intensity(v,u)+ry_intensity(v-1,u));
                ddv(v,u) = (ry(v-1,u)*(depth_ref(v+1,u)-depth_ref(v,u)) + ry(v,u)*(depth_ref(v,u) - depth_ref(v-1,u)))/(ry(v,u)+ry(v-1,u));
            }

    dcv.row(0) = dcv.row(1);
    dcv.row(rows_i-1) = dcv.row(rows_i-2);
    ddv.row(0) = ddv.row(1);
    ddv.row(rows_i-1) = ddv.row(rows_i-2);

	//Temporal derivative
	dct = intensity_warped[image_level] - intensity_old[image_level];
    ddt = depth_warped[image_level] - depth_old[image_level];
}

void VO_SF::computeWeights()
{
    weights_c.resize(rows_i, cols_i);
    weights_c.assign(0.f);
	weights_d.resize(rows_i, cols_i);
    weights_d.assign(0.f);
	const MatrixXi &labels_ref = labels[image_level];

	//Parameters for error_measurement
    //const float km = 1.f;
    //const float kz2 = 0.01f;
	
	//Parameters for error_linearization
    const float kduvt_c = 10.f;
	const float kduvt_d = 200.f;
	
    for (unsigned int u = 1; u < cols_i-1; u++)
		for (unsigned int v = 1; v < rows_i-1; v++)
            if (Null(v,u) == false)
			{
				//Compute error_measurement
                //const float error_m_c = flag_test ? 1.f : km*square(square(depth_inter[image_level](v,u)));
				const float error_m_c = 1.f; 

				//const float error_m = kz2*square(square(depth_inter[image_level](v,u)));
				const float error_m_d = 0.01f; //kz2*depth_inter[image_level](v,u);


				//Compute error linearization
				const float error_l_c = kduvt_c*(square(dct(v,u)) + square(dcu(v,u)) + square(dcv(v,u)));

				//const float ini_du = depth_old_ref(v,u+1) - depth_old_ref(v,u-1);
                //const float ini_dv = depth_old_ref(v+1,u) - depth_old_ref(v-1,u);
                //const float final_du = depth_warped_ref(v,u+1) - depth_warped_ref(v,u-1);
                //const float final_dv = depth_warped_ref(v+1,u) - depth_warped_ref(v-1,u);
                //const float dut = ini_du - final_du;
                //const float dvt = ini_dv - final_dv;
                //const float duu = ddu(v,u+1) - ddu(v,u-1);
                //const float dvv = ddv(v+1,u) - ddv(v-1,u);
                //const float dvu = ddu(v+1,u) - ddu(v-1,u); //Completely equivalent to compute duv
                const float error_l_d = kduvt_d*(square(ddt(v,u)) + square(ddu(v,u)) + square(ddv(v,u))); // + k2dt*(square(dut) + square(dvt)) + k2duv*(square(duu) + square(dvv) + square(dvu));

				const float w_dinobj = label_in_backg[labels_ref(v,u)] ? max(0.f, 1.f - bf_segm[labels_ref(v,u)]) : 1.f;

                weights_c(v,u) = sqrtf(w_dinobj/(error_m_c + error_l_c));
				weights_d(v,u) = sqrtf(w_dinobj/(error_m_d + error_l_d)); 
			}

    const float inv_max_c = 1.f/weights_c.maximum();
    weights_c = inv_max_c*weights_c;

	const float inv_max_d = 1.f/weights_d.maximum();
    weights_d = inv_max_d*weights_d;
}


void VO_SF::solveRobustOdometryCauchy()
{
    SolveForMotionWorkspace &ws = ws_foreground;
    ws.indices.clear();

    //Create list of pixels&constraints
    for (unsigned int u = 1; u < cols_i-1; u++)
        for (unsigned int v = 1; v < rows_i-1; v++)
            if (Null(v,u) == false)
                ws.indices.push_back(std::make_pair(v, u));

    size_t valid_points = ws.indices.size();
    float *A = ws.A, *B = ws.B;

	//initialize A and B for the first computation of residuals
    JacobianElementForRobustOdometryFn fn_ini(ws,*this);
    JacobianElementForRobustOdometryFn::Range range_ini(0, ws.indices.size(), 32);
    const float sum_of_residuals = tbb::parallel_reduce(range_ini, 0.f, fn_ini, std::plus<float>()); // parallel version
    //float mean_res = fn(range, 0.f); // linear version


    //Solve IRLS - Cauchy kernel
    //===================================================================
	float chi2_last = numeric_limits<float>::max(), chi2;
	Vector6f robust_odo = Vector6f::Zero();
    NormalEquation::MatrixA AtA; NormalEquation::VectorB AtB;

	//Aux structure for the solver
	IrlsContext ctx;
	ctx.residuals.resize(2*valid_points, 1);
	ctx.num_pixels = valid_points;
	ctx.A = A; ctx.B = B;
	ctx.Cauchy_factor = 16.f; //25 before
	
	for (unsigned int iter=0; iter<=iter_irls; iter++)
    {
        //Recompute residuals and update the Cauchy parameter
		ctx.Var = robust_odo;
		ctx.computeNewResiduals();
		
		//Build the system with the new weights
        IrlsElementFn fn(ctx);
		IrlsElementFn::Range range(0, ws.indices.size(), 32);
        NormalEquationAndChi2 nes_and_chi2 = tbb::parallel_reduce(range, NormalEquationAndChi2(), fn, NormalEquationAndChi2::Reduce());

        nes_and_chi2.nes.get(AtA, AtB);
        const Vector6f new_sol = AtA.ldlt().solve(-AtB);
		const Vector6f delta_sol = robust_odo - new_sol;
		robust_odo = new_sol;

        chi2 = nes_and_chi2.chi2;

		//Check convergence - It is using the old residuals to check convergence, not the very last one.
		const float chi2_ratio = chi2/max(1e-10f, chi2_last);
		chi2_last = chi2;
		if ((chi2_ratio > irls_chi2_decrement_threshold)||(delta_sol.lpNorm<Infinity>() < irls_var_delta_threshold))
		{
			//printf("\n Number of iterations = %d", iter+1);
			break;	
		}
    }

	//Compute covariance of the estimate and filter the motion
	kai_loc_level_odometry = robust_odo;
	computeTransformationFromTwist();
}


void VO_SF::solveMotionForIndices(vector<pair<int, int> > const&indices, Vector6f &Var, SolveForMotionWorkspace &ws, bool is_background)
{
	const size_t num_points = indices.size();
	float *A = ws.A, *B = ws.B;

	JacobianElementFn fn_ini(ws,*this);
	JacobianElementFn::Range range_ini(0, indices.size(), 32);
	NormalEquation::MatrixA AtA; NormalEquation::VectorB AtB;

	//Solve it once only with pre-weighting
	NormalEquationAndChi2 nes_and_chi2_ini = tbb::parallel_reduce(range_ini, NormalEquationAndChi2(), fn_ini, NormalEquationAndChi2::Reduce()); // parallel version
	//NormalEquationAndChi2 nes_and_chi2 = fn(range, NormalEquationAndChi2()); // linear version
	nes_and_chi2_ini.nes.get(AtA, AtB);
	Var = AtA.ldlt().solve(-AtB);


	//Solve iteratively reweighted least squares
	//===================================================================
	float chi2_last = numeric_limits<float>::max(); 

	//Aux structure for the solver
	IrlsContext ctx;
	ctx.residuals.resize(2*indices.size(), 1);
	ctx.num_pixels = indices.size();
	ctx.A = A; ctx.B = B;
	ctx.Cauchy_factor = is_background ? 0.25f : 1.f;

	for (unsigned int it=1; it<=iter_irls; it++)
	{	
		//Recompute residuals and update the Cauchy parameter
		ctx.Var = Var;
		ctx.computeNewResiduals();
		
		//Build the system with the new weights
		IrlsElementFn fn(ctx);
		IrlsElementFn::Range range(0, indices.size(), 32);
		NormalEquationAndChi2 nes_and_chi2 = tbb::parallel_reduce(range, NormalEquationAndChi2(), fn, NormalEquationAndChi2::Reduce());
		
		//Solve the linear system of equations using a minimum least squares method
		nes_and_chi2.nes.get(AtA, AtB);
		const Vector6f Var_new = AtA.ldlt().solve(-AtB);
		const Vector6f Var_delta = Var - Var_new;
		Var = Var_new;

		//Check convergence
		const float chi2_ratio = nes_and_chi2.chi2/max(1e-10f, chi2_last);
		if (chi2_ratio > irls_chi2_decrement_threshold || Var_delta.lpNorm<Infinity>() < irls_var_delta_threshold)
			break;
		
		chi2_last = nes_and_chi2.chi2;
	}

	//If it is background -> Update odometry
	if (is_background)
	{
		kai_loc_level_odometry = Var;
		computeTransformationFromTwist();
		Var = kai_loc_level_odometry;
	}
}


void VO_SF::solveMotionForegroundAndBackground()
{
    MemberFunctor<VO_SF, &VO_SF::solveMotionForeground> solve_motion_foreground(*this);
    MemberFunctor<VO_SF, &VO_SF::solveMotionBackground> solve_motion_background(*this);

    // only helps if there is more than one motion ;)
    //tbb::parallel_invoke(solve_motion_background, solve_motion_foreground);

    if (level > 0) solve_motion_foreground(); //Not at the very first level of the pyramid, it is too small
    solve_motion_background();
}

void VO_SF::solveMotionForeground()
{
    const float in_threshold = 0.2f;
    Matrix <float,6,1> Var;
    vector<pair<int,int> > &indices = ws_foreground.indices;
	const Matrix<float, NUM_LABELS+1, Dynamic> &labels_ref = labels_opt[image_level];

    for (unsigned int l=0; l<NUM_LABELS; l++)
    {
        if (!label_in_foreg[l])
			continue;
		
        indices.clear();

        for (unsigned int u = 1; u < cols_i-1; u++)
            for (unsigned int v = 1; v < rows_i-1; v++)
                if ((Null(v,u) == false)&&(labels_ref(l,v+u*rows_i) > in_threshold))
                    indices.push_back(make_pair(v,u));

		//Solve
        solveMotionForIndices(indices, Var, ws_foreground, false);

        //Save the solution
		updateVelocitiesAndTransformations(Var, l);
    }
}

void VO_SF::solveMotionBackground()
{
    const float in_threshold = 0.2f;
    Vector6f Var;
    vector<pair<int,int> > &indices = ws_background.indices;
    indices.clear();
	const Matrix<float, NUM_LABELS+1, Dynamic> &labels_ref = labels_opt[image_level];

	//Create the indices for the elements in the background
    for (unsigned int l=0; l<NUM_LABELS; l++)
    {
        if (!label_in_backg[l])
			continue;

        for (unsigned int u = 1; u < cols_i-1; u++)
            for (unsigned int v = 1; v < rows_i-1; v++)
                if ((Null(v,u) == false)&&(labels_ref(l,v+u*rows_i) > in_threshold))
                    indices.push_back(make_pair(v,u));
	}

    //Solve - The odometry is updated inside this method too (that's why it needs the flag)
    solveMotionForIndices(indices, Var, ws_background, true);

    //Save the solution
	for (unsigned int l=0; l<NUM_LABELS; l++)
		if ((label_in_backg[l])&&(!label_in_foreg[l])) 
			updateVelocitiesAndTransformations(Var, l);
}


void VO_SF::updateVelocitiesAndTransformations(Matrix<float,6,1> &last_sol, unsigned int label)
{
	const unsigned int l = label;
    kai_loc_level[l] = last_sol;

    Matrix<float,4,4> local_mat; local_mat.assign(0.f);
    local_mat(0,1) = -last_sol(5); local_mat(1,0) = last_sol(5);
    local_mat(0,2) = last_sol(4); local_mat(2,0) = -last_sol(4);
    local_mat(1,2) = -last_sol(3); local_mat(2,1) = last_sol(3);
    local_mat(0,3) = last_sol(0); local_mat(1,3) = last_sol(1); local_mat(2,3) = last_sol(2);
    T[l] = local_mat.exp()*T[l];

    Matrix<float, 4, 4> log_trans = T[l].log();
    kai_loc[l](0) = log_trans(0,3); kai_loc[l](1) = log_trans(1,3); kai_loc[l](2) = log_trans(2,3);
    kai_loc[l](3) = -log_trans(1,2); kai_loc[l](4) = log_trans(0,2); kai_loc[l](5) = -log_trans(0,1);	
}

void VO_SF::getCameraPoseFromBackgroundEstimate()
{
	cam_oldpose = cam_pose;
	CMatrixDouble44 aux_acu = T_odometry;
	poses::CPose3D pose_aux(aux_acu);
	cam_pose = cam_pose + pose_aux;
}

void VO_SF::computeTransformationFromTwist()
{
	//Compute the rigid transformation
	Matrix4f local_mat; local_mat.assign(0.f); 
	local_mat(0,1) = -kai_loc_level_odometry(5); local_mat(1,0) = kai_loc_level_odometry(5);
	local_mat(0,2) = kai_loc_level_odometry(4); local_mat(2,0) = -kai_loc_level_odometry(4);
	local_mat(1,2) = -kai_loc_level_odometry(3); local_mat(2,1) = kai_loc_level_odometry(3);
	local_mat(0,3) = kai_loc_level_odometry(0);
	local_mat(1,3) = kai_loc_level_odometry(1);
	local_mat(2,3) = kai_loc_level_odometry(2);
	T_odometry = local_mat.exp()*T_odometry;

    Matrix4f log_trans = T_odometry.log();
    kai_loc_odometry(0) = log_trans(0,3); kai_loc_odometry(1) = log_trans(1,3); kai_loc_odometry(2) = log_trans(2,3);
    kai_loc_odometry(3) = -log_trans(1,2); kai_loc_odometry(4) = log_trans(0,2); kai_loc_odometry(5) = -log_trans(0,1);	
}


void VO_SF::warpImagesParallel()
{
    ImageDomain domain(0, rows_i, 30, 0, cols_i, 40);

    typedef VO_SF_RegionFunctor<&VO_SF::warpImages> WarpImagesDelegate;
    WarpImagesDelegate warp_images(*this);
    tbb::parallel_for(domain, warp_images);
}

void VO_SF::computeCoordsParallel()
{
    ImageDomain domain(0, rows_i, 30, 0, cols_i, 40);

    typedef VO_SF_RegionFunctor<&VO_SF::calculateCoord> Delegate;
    Delegate delegate(*this);
    tbb::parallel_for(domain, delegate);
}

void VO_SF::warpImages()
{
    warpImages(cv::Rect(0,0, cols_i, rows_i));
}

void VO_SF::warpImages(cv::Rect region)
{
    unsigned int x = region.tl().x, y = region.tl().y, w = region.width, h = region.height;

    //Camera parameters (which also depend on the level resolution)
    const float f = float(cols_i)/(2.f*tan(0.5f*fovh));
    const float inv_f_i = 1.f/f;
    const float disp_u_i = 0.5f*float(cols_i-1);
    const float disp_v_i = 0.5f*float(rows_i-1);

	//Refs
	MatrixXf &depth_warped_ref = depth_warped[image_level];
	MatrixXf &intensity_warped_ref = intensity_warped[image_level];
	MatrixXf &xx_warped_ref = xx_warped[image_level];
	MatrixXf &yy_warped_ref = yy_warped[image_level];
	const MatrixXf &depth_old_ref = depth_old[image_level];
	const MatrixXf &xx_old_ref = xx_old[image_level];
	const MatrixXf &yy_old_ref = yy_old[image_level];
	const Matrix<float, NUM_LABELS+1, Dynamic> &labels_ref = labels_opt[image_level];

	//Initialize
	depth_warped_ref.block(y, x, h, w).assign(0.f);
    xx_warped_ref.block(y, x, h, w).assign(0.f);
    yy_warped_ref.block(y, x, h, w).assign(0.f);
    intensity_warped_ref.block(y, x, h, w).assign(0.f);

    //Compute the rigid transformation associated to the labels
    Matrix4f mytrans[NUM_LABELS];
    for (unsigned int l=0; l<NUM_LABELS; l++)
		mytrans[l] = T[l].inverse();


    for (unsigned int j = x; j < x + w; j++)
        for (unsigned int i = y; i< y + h; i++)
        {
            const int pixel_label = i+j*rows_i;
			const float z = depth_old_ref(i,j);
            if ((z > 0.f)&&(labels_ref(NUM_LABELS, pixel_label) < 1.f))
            {
                //Interpolate between the transformations
                Matrix4f trans; trans.assign(0.f);
                for (unsigned int l=0; l<=NUM_LABELS; l++)
                    if (labels_ref(l,pixel_label) != 0.f)
                        trans += labels_ref(l,pixel_label)*mytrans[l];

                //Transform point to the warped reference frame
                const float depth_w = trans(0,0)*z + trans(0,1)*xx_old_ref(i,j) + trans(0,2)*yy_old_ref(i,j) + trans(0,3);
                const float x_w = trans(1,0)*z + trans(1,1)*xx_old_ref(i,j) + trans(1,2)*yy_old_ref(i,j) + trans(1,3);
                const float y_w = trans(2,0)*z + trans(2,1)*xx_old_ref(i,j) + trans(2,2)*yy_old_ref(i,j) + trans(2,3);

                //Calculate warping
                const float uwarp = f*x_w/depth_w + disp_u_i;
                const float vwarp = f*y_w/depth_w + disp_v_i;
                interpolateColorAndDepthAcu(intensity_warped_ref(i,j), depth_warped_ref(i,j), uwarp, vwarp);
                if (depth_warped_ref(i,j) != 0.f)
                    depth_warped_ref(i,j) -= (depth_w-z);

                xx_warped_ref(i,j) = (j - disp_u_i)*depth_warped_ref(i,j)*inv_f_i;
                yy_warped_ref(i,j) = (i - disp_v_i)*depth_warped_ref(i,j)*inv_f_i;
            }
        }
}

void VO_SF::warpImagesOld()
{
	//Camera parameters (which also depend on the level resolution)
	const float f = float(cols_i)/(2.f*tan(0.5f*fovh));
	const float disp_u_i = 0.5f*float(cols_i-1);
    const float disp_v_i = 0.5f*float(rows_i-1);

	//Refs
	MatrixXf &depth_warped_ref = depth_warped[image_level];
	MatrixXf &intensity_warped_ref = intensity_warped[image_level];
	MatrixXf &xx_warped_ref = xx_warped[image_level];
	MatrixXf &yy_warped_ref = yy_warped[image_level];
	const MatrixXf &depth_ref = depth[image_level];
	const MatrixXf &intensity_ref = intensity[image_level];
	const MatrixXf &xx_ref = xx[image_level];
	const MatrixXf &yy_ref = yy[image_level];
	depth_warped_ref.assign(0.f);
	intensity_warped_ref.assign(0.f);

	//Rigid transformation and weights
	Matrix4f acu_trans = T_odometry; 
	MatrixXf wacu(rows_i,cols_i);
	wacu.assign(0.f);

	const float cols_lim = float(cols_i-1);
	const float rows_lim = float(rows_i-1);

	//						Warping loop
	//---------------------------------------------------------
	for (unsigned int j = 0; j<cols_i; j++)
		for (unsigned int i = 0; i<rows_i; i++)
		{		
			const float z = depth_ref(i,j);
			
			if (z != 0.f)
			{
				//Transform point to the warped reference frame
				const float intensity_w = intensity_ref(i,j);
				const float depth_w = acu_trans(0,0)*z + acu_trans(0,1)*xx_ref(i,j) + acu_trans(0,2)*yy_ref(i,j) + acu_trans(0,3);
				const float x_w = acu_trans(1,0)*z + acu_trans(1,1)*xx_ref(i,j) + acu_trans(1,2)*yy_ref(i,j) + acu_trans(1,3);
				const float y_w = acu_trans(2,0)*z + acu_trans(2,1)*xx_ref(i,j) + acu_trans(2,2)*yy_ref(i,j) + acu_trans(2,3);

				//Calculate warping
				const float uwarp = f*x_w/depth_w + disp_u_i;
				const float vwarp = f*y_w/depth_w + disp_v_i;

				//The warped pixel (which is not integer in general) contributes to all the surrounding ones
				if (( uwarp >= 0.f)&&( uwarp < cols_lim)&&( vwarp >= 0.f)&&( vwarp < rows_lim))
				{
					const int uwarp_l = uwarp;
					const int uwarp_r = uwarp_l + 1;
					const int vwarp_d = vwarp;
					const int vwarp_u = vwarp_d + 1;
					const float delta_r = float(uwarp_r) - uwarp;
					const float delta_l = uwarp - float(uwarp_l);
					const float delta_u = float(vwarp_u) - vwarp;
					const float delta_d = vwarp - float(vwarp_d);

					//Warped pixel very close to an integer value
					if (abs(round(uwarp) - uwarp) + abs(round(vwarp) - vwarp) < 0.05f)
					{
						depth_warped_ref(round(vwarp), round(uwarp)) += depth_w;
						intensity_warped_ref(round(vwarp), round(uwarp)) += intensity_w;
						wacu(round(vwarp), round(uwarp)) += 1.f;
					}
					else
					{
						const float w_ur = square(delta_l) + square(delta_d);
						depth_warped_ref(vwarp_u,uwarp_r) += w_ur*depth_w;
						intensity_warped_ref(vwarp_u,uwarp_r) += w_ur*intensity_w;
						wacu(vwarp_u,uwarp_r) += w_ur;

						const float w_ul = square(delta_r) + square(delta_d);
						depth_warped_ref(vwarp_u,uwarp_l) += w_ul*depth_w;
						intensity_warped_ref(vwarp_u,uwarp_l) += w_ul*intensity_w;
						wacu(vwarp_u,uwarp_l) += w_ul;

						const float w_dr = square(delta_l) + square(delta_u);
						depth_warped_ref(vwarp_d,uwarp_r) += w_dr*depth_w;
						intensity_warped_ref(vwarp_d,uwarp_r) += w_dr*intensity_w;
						wacu(vwarp_d,uwarp_r) += w_dr;

						const float w_dl = square(delta_r) + square(delta_u);
						depth_warped_ref(vwarp_d,uwarp_l) += w_dl*depth_w;
						intensity_warped_ref(vwarp_d,uwarp_l) += w_dl*intensity_w;
						wacu(vwarp_d,uwarp_l) += w_dl;
					}
				}
			}
		}

	//Scale the averaged depth and compute spatial coordinates
    const float inv_f_i = 1.f/f;
	for (unsigned int u = 0; u<cols_i; u++)
		for (unsigned int v = 0; v<rows_i; v++)
		{	
			if (wacu(v,u) != 0.f)
			{
				intensity_warped_ref(v,u) /= wacu(v,u);
				depth_warped_ref(v,u) /= wacu(v,u);
				xx_warped_ref(v,u) = (u - disp_u_i)*depth_warped_ref(v,u)*inv_f_i;
				yy_warped_ref(v,u) = (v - disp_v_i)*depth_warped_ref(v,u)*inv_f_i;
			}
			else
			{
				xx_warped_ref(v,u) = 0.f;
				yy_warped_ref(v,u) = 0.f;
			}
		}
}


void VO_SF::mainIteration(bool create_image_pyr)
{
	CTicTac clock; clock.Tic();
	
	//          Create the image pyramid if it has not been computed yet
    //----------------------------------------------------------------------------------
	if (create_image_pyr) 
		createImagePyramid();

    //                                Create labels
    //----------------------------------------------------------------------------------
    //Kmeans
	kMeans3DCoordLowRes();

	//Create the pyramid for the labels
	createLabelsPyramidUsingKMeans();

	//Compute warped bf_segmentation (necessary for the robust estimation)
	computeBackgTemporalRegValues();


    //Solve a robust odometry problem to segment the background (coarse-to-fine)
    //---------------------------------------------------------------------------------
    //Initialize the overall transformations to 0
	T_odometry.setIdentity();
    for (unsigned int l=0; l<NUM_LABELS; l++)
    {
        T[l].setIdentity();
        kai_loc[l].assign(0.f);
    }

    //Solve the system (coarse-to-fine)
    for (unsigned int i=0; i<ctf_levels; i++)
		for (unsigned int k=0; k<max_iter_per_level; k++)
		{
			//Previous computations
			level = i;
			unsigned int s = pow(2.f,int(ctf_levels-(i+1)));
			cols_i = cols/s; rows_i = rows/s;
			image_level = ctf_levels - i + round(log2(width/cols)) - 1;


			//1. Perform warping
			if (i == 0)
			{
				depth_warped[image_level] = depth[image_level];
				intensity_warped[image_level] = intensity[image_level];
				xx_warped[image_level] = xx[image_level];
				yy_warped[image_level] = yy[image_level];
			}
			else 
                warpImagesOld(); // forward warping, more precise
                //warpImagesParallel(); // inverse warping


			//2. Compute inter coords (better linearization of the range and optical flow constraints)
			computeCoordsParallel();

			//4. Compute derivatives
			calculateDerivatives();

			//6. Solve odometry
			solveRobustOdometryCauchy();

			//Convergence of nonlinear iterations
			if (kai_loc_level_odometry.norm() < 0.04f)
				break;
		}

	//Segment static and dynamic parts
	segmentBackgroundForeground();


	//Solve the multi-odometry problem (coarse-to-fine)
	//-------------------------------------------------------------------------------------
	//Set the overall transformations to 0
	T_odometry.setIdentity();
    for (unsigned int l=0; l<NUM_LABELS; l++)
    {
        T[l].setIdentity();
        kai_loc[l].assign(0.f);
    }

    for (unsigned int i=0; i<ctf_levels; i++)
    {
        //Previous computations
        level = i;
        unsigned int s = pow(2.f,int(ctf_levels-(i+1)));
        cols_i = cols/s; rows_i = rows/s;
        image_level = ctf_levels - i + round(log2(width/cols)) - 1;


		//1. Perform warping
		//Info: The accuracy of the odometry is slightly better using the other warping but I cannot use in general because
		// the labels are defined in the old image (better about 7% for the only sequence I have tested it)
		if (i == 0)
		{
			depth_warped[image_level] = depth[image_level];
			intensity_warped[image_level] = intensity[image_level];
			xx_warped[image_level] = xx[image_level];
			yy_warped[image_level] = yy[image_level];
		}
		else
			warpImagesParallel();

		//2. Compute inter coords
		computeCoordsParallel();

		//4. Compute derivatives
		calculateDerivatives();

		//5. Compute weights
		computeWeights();

		//6. Solve odometry
		solveMotionForegroundAndBackground();
    }

	//Get Pose (Odometry) from the background motion estimate
	getCameraPoseFromBackgroundEstimate();

	// Refine static/dynamic segmentation and warp it to use it in the next iteration
	segmentBackgroundForeground();
	warpBackgForegSegmentation();
	//countMovingAndUncertainPixels();

    //Compute the scene flow from the rigid motions and the labels
	computeSceneFlowFromRigidMotions();

	const float runtime = 1000.f*clock.Tac();
	printf("\n Runtime = %f (ms) ", runtime);
	if (create_image_pyr)	printf("including the image pyramid");
	else					printf("without including the image pyramid");
}

void VO_SF::computeSceneFlowFromRigidMotions()
{
    const unsigned int repr_level = round(log2(width/cols));

    //Compute the rigid transformation associated to the labels
    Matrix4f mytrans[NUM_LABELS];
    for (unsigned int l=0; l<NUM_LABELS; l++)
        mytrans[l] = T[l].inverse();

	//Refs
	const MatrixXf &depth_old_ref = depth_old[repr_level];
	const MatrixXf &xx_old_ref = xx_old[repr_level];
	const MatrixXf &yy_old_ref = yy_old[repr_level];
	const Matrix<float, NUM_LABELS+1, Dynamic> &labels_ref = labels_opt[image_level];

	Matrix4f trans; 
    for (unsigned int u = 0; u<cols; u++)
        for (unsigned int v = 0; v<rows; v++)
        {
            const float z = depth_old_ref(v,u);
			const int pixel_label = v + u*rows;

			if (z != 0.f)
            {			
				//Interpolate between the transformations
                trans.fill(0.f);
                for (unsigned int l=0; l<NUM_LABELS; l++)
                    if (labels_ref(l,pixel_label) != 0.f)
                        trans += labels_ref(l,pixel_label)*mytrans[l];

                //Transform point to the warped reference frame
                motionfield[0](v,u) = trans(0,0)*z + trans(0,1)*xx_old_ref(v,u) + trans(0,2)*yy_old_ref(v,u) + trans(0,3) - z;
                motionfield[1](v,u) = trans(1,0)*z + trans(1,1)*xx_old_ref(v,u) + trans(1,2)*yy_old_ref(v,u) + trans(1,3) - xx_old_ref(v,u);
                motionfield[2](v,u) = trans(2,0)*z + trans(2,1)*xx_old_ref(v,u) + trans(2,2)*yy_old_ref(v,u) + trans(2,3) - yy_old_ref(v,u);
            }
            else
            {
				motionfield[0](v,u) = 0.f;
                motionfield[1](v,u) = 0.f;
                motionfield[2](v,u) = 0.f;
            }
        }
}


void VO_SF::interpolateColorAndDepthAcu(float &c, float &d, const float ind_u, const float ind_v)
{
    //const float depth_threshold = 0.03f; // 4 times distances of 10 cm (they are squared below)
	const float depth_threshold = 0.3f;
	const float null_threshold = 0.5f;

    if (ind_u <= 0.f) { c = 0.f; d = 0.f; return; }
    else if (ind_u >= float(cols_i - 1)) { c = 0.f; d = 0.f; return; }
    if (ind_v <= 0.f) { c = 0.f; d = 0.f; return; }
    else if (ind_v >= float(rows_i - 1)) { c = 0.f; d = 0.f; return; }

    const float inf_u = floor(ind_u);
    const float sup_u = inf_u + 1.f;
    const float inf_v = floor(ind_v);
    const float sup_v = inf_v + 1.f;

    const Matrix2f cmat = intensity[image_level].block<2,2>(inf_v,inf_u);
    const Matrix2f dmat = depth[image_level].block<2,2>(inf_v,inf_u);


	if ((sup_u != inf_u)&&(sup_v != inf_v))
    {
		//const float depth_dist = square(dmat(0)-dmat(1)) + square(dmat(1)-dmat(2)) + square(dmat(2)-dmat(3));
		const float depth_dist = abs(dmat(0)-dmat(1)) + abs(dmat(1)-dmat(2)) + abs(dmat(2)-dmat(3));

		if (depth_dist > depth_threshold)
		{
            //Nearest neighbour
			const int round_v = int(round(ind_v-inf_v)), round_u = int(round(ind_u-inf_u));
			d = dmat(round_v, round_u);
			c = cmat(round_v, round_u);
		}
		else
		{
			//Gaussian filter
			const float weight_dl = (sup_u - ind_u)*(sup_v - ind_v);
			const float weight_ul = (sup_u - ind_u)*(ind_v - inf_v);
			const float weight_dr = (ind_u - inf_u)*(sup_v - ind_v);
			const float weight_ur = (ind_u - inf_u)*(ind_v - inf_v);
			d = weight_dl*dmat(0,0) + weight_ul*dmat(1,0) + weight_dr*dmat(0,1) + weight_ur*dmat(1,1);
			c = weight_dl*cmat(0,0) + weight_ul*cmat(1,0) + weight_dr*cmat(0,1) + weight_ur*cmat(1,1);
		}	
    }
    else if ((sup_u == inf_u)&&(sup_v == inf_v))
    {
        c = cmat(0,0);
        d = dmat(0,0);
    }
    else if (sup_u == inf_u)
    {
        c = (sup_v - ind_v)*cmat(0,0) + (ind_v - inf_v)*cmat(1,0);

        const float weight_d = (sup_v - ind_v)*(dmat(0,0)>0.f);
        const float weight_u = (ind_v - inf_v)*(dmat(1,0)>0.f);
        const float sum_w = weight_d + weight_u;
        if (sum_w < null_threshold)
            d = 0.f;
        else if (sum_w > 0.99f)
            d = weight_d*dmat(0,0) + weight_u*dmat(1,0);
        else
            d = (weight_d*dmat(0,0) + weight_u*dmat(1,0))/sum_w;
    }

    else if (sup_v == inf_v)
    {
        c = (sup_u - ind_u)*cmat(0,0) + (ind_u - inf_u)*cmat(0,1);

        const float weight_l = (sup_u - ind_u)*(dmat(0,0)>0.f);
        const float weight_r = (ind_u - inf_u)*(dmat(0,1)>0.f);
        const float sum_w = weight_l + weight_r;
        if (sum_w < null_threshold)
            d = 0.f;
        else if (sum_w > 0.99f)
            d = weight_l*dmat(0,0) + weight_r*dmat(0,1);
        else
            d = (weight_l*dmat(0,0) + weight_r*dmat(0,1))/sum_w;
    }
}


