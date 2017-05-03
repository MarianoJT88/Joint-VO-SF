/*********************************************************************************
**Fast Odometry and Scene Flow from RGB-D Cameras based on Geometric Clustering	**
**------------------------------------------------------------------------------**
**																				**
**	Copyright(c) 2017, Mariano Jaimez Tarifa, University of Malaga				**
**	Copyright(c) 2017, Christian Kerl, Technical University of Munich			**
**	Copyright(c) 2017, MAPIR group, University of Malaga						**
**	Copyright(c) 2017, Computer Vision group, Tech. University of Munich		**
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
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.		**
**																				**
*********************************************************************************/

#ifndef joint_VO_SF_H
#define joint_VO_SF_H


#include <mrpt/system.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/utils.h>
#include <mrpt/gui/CDisplayWindow3D.h>
#include <mrpt/opengl.h>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <unsupported/Eigen/MatrixFunctions>
#include <opencv2/opencv.hpp>
#include <camera.h>
#include <datasets.h>


#define NUM_LABELS 24

typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef Eigen::Matrix<float, 2, 6> JacobianT;
typedef Eigen::Matrix<float, 2, 1> ResidualT;

static const int JacobianElements = JacobianT::RowsAtCompileTime * JacobianT::ColsAtCompileTime;
static const int ResidualElements = ResidualT::RowsAtCompileTime * ResidualT::ColsAtCompileTime;

struct SolveForMotionWorkspace
{
    float *A, *B;
    vector<pair<int,int> > indices;

    SolveForMotionWorkspace(int max_npoints)
    {
        A = new float[max_npoints * JacobianElements];
        B = new float[max_npoints * ResidualElements];
        indices.reserve(max_npoints);
    }
    ~SolveForMotionWorkspace()
    {
        delete[] A;
        delete[] B;
    }
};


class VO_SF {
public:

    vector<MatrixXf> color, color_old, color_inter, color_warped;
    vector<MatrixXf> depth, depth_old, depth_inter, depth_warped;
	vector<MatrixXf> xx, xx_inter, xx_old, xx_warped;
	vector<MatrixXf> yy, yy_inter, yy_old, yy_warped;

    //They store the accumulated value up to the present level
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > T;
    Vector6f kai_loc[NUM_LABELS], kai_loc_level[NUM_LABELS];
	Eigen::Matrix<float, 6, 6> all_kai_levels[NUM_LABELS];

	//Odometry
	Eigen::Matrix<float, 6, 6> est_cov;
	Eigen::Matrix4f T_odometry;
	Vector6f kai_loc_odometry, kai_loc_odometry_old, kai_loc_level_odometry;
	mrpt::poses::CPose3D cam_pose, cam_oldpose;

    Eigen::MatrixXf motionfield[3];
    Eigen::MatrixXf depth_wf, color_wf;
    Eigen::MatrixXf dcu, dcv, dct;
    Eigen::MatrixXf ddu, ddv, ddt;
	Eigen::MatrixXf im_r, im_g, im_b;
	Eigen::MatrixXf im_r_old, im_g_old, im_b_old;

    Eigen::MatrixXf weights_c, weights_d;
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> Null;
	Eigen::MatrixXf final_residual_color, final_residual_depth;

	float k_photometric_res;
	float previous_speed_const_weight;
	float previous_speed_eig_weight;
    float irls_chi2_decrement_threshold, irls_var_delta_threshold;
    float fovh, fovv;
    unsigned int rows, cols;
    unsigned int rows_i, cols_i;
    unsigned int width, height;
	unsigned int ctf_levels;
	unsigned int image_level, level;
	unsigned int iter_irls;
	unsigned int max_iter_per_level;
	bool use_backg_temp_reg;
    Eigen::Array44f f_mask;

    SolveForMotionWorkspace ws_foreground, ws_background;

    VO_SF(unsigned int res_factor);
    void createImagePyramid();
    void warpImages();
    void warpImagesParallel();
    void warpImages(cv::Rect region);
	void warpImagesOld();
    void calculateCoord();
	void computeCoordsParallel();
    void calculateCoord(cv::Rect region);
	void calculateDerivatives();
    void computeWeights();

	void solveMotionForIndices(vector<pair<int, int> > const&indices, Vector6f &Var, SolveForMotionWorkspace &ws, bool is_background);
	void solveMotionForeground();
	void solveMotionBackground();
    void solveMotionForegroundAndBackground();
	void solveRobustOdometryCauchy();

    void mainIteration();
    void computeSceneFlowFromMotionFast();
    //void interpolateColorAndDepth(float &c, float &d, float ind_u, float ind_v);
	void interpolateColorAndDepthAcu(float &c, float &d, const float ind_u, const float ind_v);
	void updateVelocitiesAndTransformations(Eigen::Matrix<float,6,1> &last_sol, unsigned int label);
	void getCameraPoseFromBackgroundEstimate();
	void computeTransformationFromTwist();


    //							Scene
	//--------------------------------------------------------------
	mrpt::gui::CDisplayWindow3D		m_window;
	mrpt::opengl::COpenGLScenePtr	m_scene;
	bool	draw_segmented_pcloud;

	void initializeSceneCamera();
	void initializeSceneDataset();
	void initializeSceneDatasetVideo();
	void initializeSceneSequencesVideo();
	void updateSceneCamera(bool &clean_sf);
	void updateSceneDataset();
	void updateSceneDatasetVideo();
	void updateSceneSequencesVideo();
	void showOriginalPointCloud();
	void showResidualsUsedForSegmentation();

    mrpt::utils::CTicTac	clock;
	float	time_im_pyr;


    //	Camera and dataset (they still belong to the class)
	//--------------------------------------------------------------
	RGBD_Camera camera;
	Datasets	dataset;
	void loadImagePairFromFiles(string files_dir, bool is_Quiroga, unsigned int res_factor);
	void loadImageFromSequence(string files_dir, unsigned int index, unsigned int res_factor);
	void saveFlowAndSegmToFile(string files_dir);


    //					Geometric clustering
    //--------------------------------------------------------------   
    Eigen::MatrixXf olabels_image[3], backg_image[3];
	std::vector<Eigen::MatrixXi> labels;
    std::vector<Eigen::Matrix<float, NUM_LABELS+1, Eigen::Dynamic> > labels_opt;
	Eigen::Matrix<float, 3, NUM_LABELS> kmeans;
	Eigen::Matrix<int, NUM_LABELS, 1> size_kmeans_maxres;
	Eigen::Matrix<bool, NUM_LABELS, 1> label_in_backg, label_in_foreg;
	Eigen::Matrix<float, NUM_LABELS, 1> bf_segm, bf_segm_warped;
	Eigen::MatrixXf bf_segm_image_warped;
	bool connectivity[NUM_LABELS][NUM_LABELS];

	void createLabelsPyramidUsingKMeans();
	void initializeKMeans();
	void kMeans3DCoordLowRes();
    void createOptLabelImage();
    void computeRegionConnectivity();
    void smoothRegions(unsigned int image_level);


	//					Background-Foreground segmentation
	//--------------------------------------------------------------------------------
	void segmentBackgroundForeground();
	void optimizeSegmentation(Eigen::Matrix<float, NUM_LABELS, 1> &r);
	void warpBackgForegSegmentation();
	void computeBackgTemporalRegValues();
	void saveSegmentationImage();
	void countMovingAndUncertainPixels();
	unsigned int num_valid_pixels, num_mov_pixels, num_uncertain_pixels;
	unsigned int min_num_valid_pixels, num_images;
};

#endif





