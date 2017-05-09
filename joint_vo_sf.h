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
    std::vector<Eigen::MatrixXf> intensity, intensity_old, intensity_inter, intensity_warped;	//Intensity images
    std::vector<Eigen::MatrixXf> depth, depth_old, depth_inter, depth_warped;					//Depth images
	std::vector<Eigen::MatrixXf> xx, xx_inter, xx_old, xx_warped;								//x coordinates of points (proportional to the col index of the pixels)
	std::vector<Eigen::MatrixXf> yy, yy_inter, yy_old, yy_warped;								//y coordinates of points (proportional to the row index of the pixels)

	Eigen::MatrixXf depth_wf, intensity_wf;							//Original images read from the camera, dataset or file
    Eigen::MatrixXf dcu, dcv, dct;									//Gradients of the intensity images
    Eigen::MatrixXf ddu, ddv, ddt;									//Gradients of the depth images
	Eigen::MatrixXf im_r, im_g, im_b;								//Last color image used only for visualization
	Eigen::MatrixXf im_r_old, im_g_old, im_b_old;					//Prev color image used only for visualization
    Eigen::MatrixXf weights_c, weights_d;							//Pre-weighting used in the solver
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> Null;		//Mask for pixels with null depth measurments
	Eigen::MatrixXf motionfield[3];									//Per-pixel scene flow (coordinates [0] -> depth, [1] -> x, [2] -> y)
	Eigen::Array44f f_mask;											//Convolutional kernel used to build the image pyramid

    //Velocities, transformations and poses
	Eigen::Matrix4f T_clusters[NUM_LABELS];					//Rigid transformations estimated for each cluster
	Eigen::Matrix4f T_odometry;								//Rigid transformation of the camera motion (odometry)
	Vector6f twist_odometry, twist_level_odometry;			//Twist encoding the odometry (accumulated and local for the pyramid level)	
	mrpt::poses::CPose3D cam_pose, cam_oldpose;				//Estimated camera poses (current and prev)

	//Parameters
    float fovh, fovv;							//Field of view of the camera (intrinsic calibration)
    unsigned int rows, cols;					//Max resolution used for the solver (240 x 320 by default)
    unsigned int rows_i, cols_i;				//Aux variables
    unsigned int width, height;					//Resolution of the input images
	unsigned int ctf_levels;					//Number of coarse-to-fine levels
	unsigned int image_level, level;			//Aux variables


    VO_SF(unsigned int res_factor);
    void createImagePyramid();					//Create image pyramids (intensity and depth)
    void warpImages();							//Fast warping (last image towards the prev one)
    void warpImagesParallel();
    void warpImages(cv::Rect region);
	void warpImagesAccurate();					//Accurate warping (last image towards the prev one)
    void calculateCoord();						//Compute so-called "intermediate coordinates", related to a more precise linearization of optical and range flow
	void computeCoordsParallel();
    void calculateCoord(cv::Rect region);
	void calculateDerivatives();				//Compute the image gradients
    void computeWeights();						//Compute pre-weighting functions for the solver
	void computeSceneFlowFromRigidMotions();	//Compute dense scene flow from rigid motions
	void updateCameraPoseFromOdometry();		//Update the camera pose
	void computeTransformationFromTwist(Vector6f &twist, bool is_odometry, unsigned int label = 0);	//Compute rigid transformation from twist
	void interpolateColorAndDepthAcu(float &c, float &d, const float ind_u, const float ind_v);		//Interpolate in images (necessary for warping)

    void run_VO_SF(bool create_image_pyr);		//Main method to run whole algorithm



	//							Solver
	//--------------------------------------------------------------
	unsigned int max_iter_irls;				//Max number of iterations for the IRLS solver
	unsigned int max_iter_per_level;		//Max number of complete iterations for every level of the pyramid
	float k_photometric_res;				//Weight of the photometric residuals (against geometric ones)
	float irls_chi2_decrement_threshold;	//Convergence threshold for the IRLS solver (change in chi2)	
	float irls_delta_threshold;				//Convergence threshold for the IRLS solver (change in the solution)	
	SolveForMotionWorkspace ws_foreground, ws_background;		//Structures for efficient solver

	//Estimate rigid motion for a set of pixels (given their indices)
	void solveMotionForIndices(std::vector<std::pair<int, int> > const&indices, Vector6f &twist, SolveForMotionWorkspace &ws, bool is_background);	
	void solveMotionDynamicClusters();			//Estimate motion of dynamic clusters
	void solveMotionStaticClusters();			//Estimate motion of static clusters
    void solveMotionAllClusters();				//Estimate motion after knowing the segmentation
	void solveRobustOdometryCauchy();			//Estimate robust odometry before knowing the segmentation

	

    //					Geometric clustering
    //--------------------------------------------------------------   
	std::vector<Eigen::MatrixXi> labels;											//Integer non-smooth labelling
    std::vector<Eigen::Matrix<float, NUM_LABELS+1, Eigen::Dynamic> > label_funct;	//Indicator funtions for the continuous labelling
	Eigen::Matrix<float, 3, NUM_LABELS> kmeans;										//Centers of the KMeans clusters
	Eigen::Matrix<int, NUM_LABELS, 1> size_kmeans;									//Size of the clusters
	bool connectivity[NUM_LABELS][NUM_LABELS];										//Connectivity between the clusters

	void createLabelsPyramidUsingKMeans();				//Create the label pyramid
	void initializeKMeans();							//Initialize KMeans by uniformly dividing the image plane
	void kMeans3DCoord();								//Segment the scene in clusters using the 3D coordinates of the points				
    void computeRegionConnectivity();					//Compute connectivity graph (which cluster is contiguous to which)
    void smoothRegions(unsigned int image_level);		//Smooth/blend clusters for a better scene flow estimation



	//						Static-Dynamic segmentation
	//--------------------------------------------------------------------------------
	Eigen::Matrix<bool, NUM_LABELS, 1> label_static, label_dynamic;			//Cluster segmentation as static, dynamic or both (uncertain)
	Eigen::Matrix<float, NUM_LABELS, 1> b_segm, b_segm_warped;				//Exact b values of the segmentation (original and warped)
	Eigen::MatrixXf b_segm_image_warped;									//Per-pixel static-dynamic segmentation (value of b per pixel, used for temporal propagation)
	bool use_b_temp_reg;													//Flag to turn on/off temporal propagation of the static/dynamic segmentation

	void segmentStaticDynamic();											//Main method to segment the clusters into static/dynamic
	void optimizeSegmentation(Eigen::Matrix<float, NUM_LABELS, 1> &r);		//Solver the optimization problem proposed for the segmentation
	void warpStaticDynamicSegmentation();									//Warp the segmentation forward
	void computeSegTemporalRegValues();										//Compute ref values for the temporal regularization



    //						3D Scene
	//--------------------------------------------------------------
	mrpt::gui::CDisplayWindow3D		window;
	mrpt::opengl::COpenGLScenePtr	scene;
	Eigen::MatrixXf labels_image[3], backg_image[3];

	void initializeSceneCamera();
	void initializeSceneDatasets();
	void initializeSceneImageSeq();
	void updateSceneCamera(bool clean_sf);
	void updateSceneDatasets(const mrpt::poses::CPose3D &gt, const mrpt::poses::CPose3D &gt_old);
	void updateSceneImageSeq();
	void createImagesOfSegmentations();



    //					Input / Output
	//--------------------------------------------------------------
	void loadImagePairFromFiles(std::string files_dir, unsigned int res_factor);
	bool loadImageFromSequence(std::string files_dir, unsigned int index, unsigned int res_factor);
	void saveFlowAndSegmToFile(std::string files_dir);	

};

#endif



