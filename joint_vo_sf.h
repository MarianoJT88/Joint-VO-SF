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


#define NUM_LABELS 24

typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef Eigen::Matrix<float, 2, 6> JacobianT;
typedef Eigen::Matrix<float, 2, 1> ResidualT;

static const int JacobianElements = JacobianT::RowsAtCompileTime * JacobianT::ColsAtCompileTime;
static const int ResidualElements = ResidualT::RowsAtCompileTime * ResidualT::ColsAtCompileTime;

struct SolveForMotionWorkspace
{
    float *A, *B;
    std::vector<std::pair<int,int> > indices;

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

	//						General
	//----------------------------------------------------------------
    std::vector<Eigen::MatrixXf> intensity, intensity_old, intensity_inter, intensity_warped;
    std::vector<Eigen::MatrixXf> depth, depth_old, depth_inter, depth_warped;
	std::vector<Eigen::MatrixXf> xx, xx_inter, xx_old, xx_warped;
	std::vector<Eigen::MatrixXf> yy, yy_inter, yy_old, yy_warped;

	Eigen::MatrixXf depth_wf, intensity_wf;
    Eigen::MatrixXf dcu, dcv, dct;
    Eigen::MatrixXf ddu, ddv, ddt;
	Eigen::MatrixXf im_r, im_g, im_b;
	Eigen::MatrixXf im_r_old, im_g_old, im_b_old;
    Eigen::MatrixXf weights_c, weights_d;
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> Null;
	Eigen::MatrixXf motionfield[3];
	Eigen::Array44f f_mask;

    //Velocities, transformations and poses
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > T;
    Vector6f kai_loc[NUM_LABELS], kai_loc_level[NUM_LABELS];
	Eigen::Matrix4f T_odometry;
	Vector6f kai_loc_odometry, kai_loc_level_odometry;
	mrpt::poses::CPose3D cam_pose, cam_oldpose;

	//Parameters
	float k_photometric_res;
    float fovh, fovv;
    unsigned int rows, cols;
    unsigned int rows_i, cols_i;
    unsigned int width, height;
	unsigned int ctf_levels;
	unsigned int image_level, level;


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

    void mainIteration(bool create_image_pyr);
    void computeSceneFlowFromRigidMotions();
	void interpolateColorAndDepthAcu(float &c, float &d, const float ind_u, const float ind_v);
	void updateVelocitiesAndTransformations(Eigen::Matrix<float,6,1> &last_sol, unsigned int label);
	void getCameraPoseFromBackgroundEstimate();
	void computeTransformationFromTwist();


	//							Solver
	//--------------------------------------------------------------
	unsigned int iter_irls;
	unsigned int max_iter_per_level;
	float irls_chi2_decrement_threshold, irls_var_delta_threshold;
	SolveForMotionWorkspace ws_foreground, ws_background;

	void solveMotionForIndices(std::vector<std::pair<int, int> > const&indices, Vector6f &Var, SolveForMotionWorkspace &ws, bool is_background);
	void solveMotionForeground();
	void solveMotionBackground();
    void solveMotionForegroundAndBackground();
	void solveRobustOdometryCauchy();


    //							Scene
	//--------------------------------------------------------------
	mrpt::gui::CDisplayWindow3D		window;
	mrpt::opengl::COpenGLScenePtr	scene;

	void initializeSceneCamera();
	void initializeSceneDatasetVideo();
	void initializeSceneSequencesVideo();
	void updateSceneCamera(bool &clean_sf);
	void updateSceneDatasetVideo(const mrpt::poses::CPose3D &gt, const mrpt::poses::CPose3D &gt_old);
	void updateSceneSequencesVideo();



    //Input / Output
	//--------------------------------------------------------------
	void loadImagePairFromFiles(std::string files_dir, bool is_Quiroga, unsigned int res_factor);
	void loadImageFromSequence(std::string files_dir, unsigned int index, unsigned int res_factor);
	void saveFlowAndSegmToFile(std::string files_dir);


    //					Geometric clustering
    //--------------------------------------------------------------   
    Eigen::MatrixXf olabels_image[3], backg_image[3];
	std::vector<Eigen::MatrixXi> labels;
    std::vector<Eigen::Matrix<float, NUM_LABELS+1, Eigen::Dynamic> > labels_opt;
	Eigen::Matrix<float, 3, NUM_LABELS> kmeans;
	Eigen::Matrix<int, NUM_LABELS, 1> size_kmeans_maxres;
	bool connectivity[NUM_LABELS][NUM_LABELS];

	void createLabelsPyramidUsingKMeans();
	void initializeKMeans();
	void kMeans3DCoordLowRes();
    void createOptLabelImage();
    void computeRegionConnectivity();
    void smoothRegions(unsigned int image_level);


	//					Background-Foreground segmentation
	//--------------------------------------------------------------------------------
	Eigen::Matrix<bool, NUM_LABELS, 1> label_in_backg, label_in_foreg;
	Eigen::Matrix<float, NUM_LABELS, 1> bf_segm, bf_segm_warped;
	Eigen::MatrixXf bf_segm_image_warped;
	bool use_backg_temp_reg;

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



