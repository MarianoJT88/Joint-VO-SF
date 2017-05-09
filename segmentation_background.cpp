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

using namespace mrpt;
using namespace mrpt::utils;
using namespace std;
using namespace Eigen;


void VO_SF::segmentStaticDynamic()
{
	//Done at the highest resolution
	cols_i = cols; rows_i = rows;
	image_level = round(log2(width/cols));
	
	//First warp images according to the estimated odometry
	warpImagesAccurate();

	//Aux variables and parameters
	Matrix<float, NUM_LABELS, 1> lab_res_c, lab_res_d, weighted_res;
	lab_res_c.fill(0.f); lab_res_d.fill(0.f); 
	const float trunc_threshold = 0.2f;
	const float res_depth_t = 0.1f;

	//Refs
	const MatrixXf &depth_old_ref = depth_old[image_level];
	const MatrixXf &depth_warped_ref = depth_warped[image_level];
	const MatrixXf &intensity_old_ref = intensity_old[image_level];
	const MatrixXf &intensity_warped_ref = intensity_warped[image_level];
	const MatrixXi &labels_ref = labels[image_level];


	//First, compute a mask of edges (to downweight their residuals, they are always high no matter what segment they belong)
	Matrix<bool, Dynamic, Dynamic> edge_mask(rows,cols); edge_mask.fill(0.f);
	const float threshold_edge = 0.3f;

	for (unsigned int u=1; u<cols-1; u++)
		for (unsigned int v=1; v<rows-1; v++)
		{
			const float d_here = depth_old_ref(v,u);
			if (d_here != 0.f)
			{	
				const float sum_dif_depth = abs(depth_old_ref(v+1,u) - d_here) + abs(depth_old_ref(v-1,u) - d_here) 
										  + abs(depth_old_ref(v,u+1) - d_here) + abs(depth_old_ref(v,u-1) - d_here);
				edge_mask(v,u) = (sum_dif_depth < threshold_edge);
			}
		}

	//Compute residuals
	for (unsigned int u=0; u<cols; u++)
		for (unsigned int v=0; v<rows; v++)
			if ((depth_old_ref(v,u) != 0.f)&&(depth_warped_ref(v,u) != 0.f))
			{
				const unsigned int pixel_label = labels_ref(v,u); //Using the binary segmentation here
				
				//Truncated Mean with occlusion handling
				const float dif_depth = depth_old_ref(v,u) - depth_warped_ref(v,u);
				if (dif_depth < res_depth_t)
				{
					lab_res_d[pixel_label] += edge_mask(v,u)*min(trunc_threshold, abs(dif_depth));
					lab_res_c[pixel_label] += edge_mask(v,u)*min(0.5f, abs(intensity_old_ref(v,u) - intensity_warped_ref(v,u)));
				}
				else if (dif_depth < 2.f*res_depth_t)
				{
					const float mult_factor = edge_mask(v,u)*(2.f*res_depth_t - dif_depth);
					lab_res_d[pixel_label] += mult_factor;
					lab_res_c[pixel_label] += mult_factor*min(0.5f, abs(intensity_old_ref(v,u) - intensity_warped_ref(v,u)));
				}		
			}

	for (unsigned int l=0; l<NUM_LABELS; l++)
	{	
		if (size_kmeans[l] != 0)
		{
			lab_res_c[l] /= size_kmeans[l];
			lab_res_d[l] /= size_kmeans[l];
		}

		//Compute the overall residual
		weighted_res[l] = k_photometric_res*lab_res_c[l] + lab_res_d[l]/max(1e-6f,kmeans(0,l)); 
	}


	//Optimization problem to improve consistency of the segmentation
	//----------------------------------------------------------------------------
	optimizeSegmentation(weighted_res);

}


void VO_SF::optimizeSegmentation(Matrix<float, NUM_LABELS, 1> &r)
{
	//Set thresholds according to the residuals obtained and the estimated velocity
	vector<float> res_sorted;
	for (unsigned int l=0; l<NUM_LABELS; l++)
		if (size_kmeans[l] != 0)
			res_sorted.push_back(r(l));
	std::sort(res_sorted.begin(), res_sorted.end());

	const float median_res = res_sorted.at(res_sorted.size()/2);
	const float trunc_res = max(0.007f, min(0.015f*(1.f + 10.f*twist_odometry.norm()), median_res));
	const float lim_nobackg = (1.f + 10.f*twist_odometry.norm())*trunc_res;
	const float lim_backg = (2.f + 10.f*twist_odometry.norm())*trunc_res;	

	//Find the number of connections between clusters (for the reg term)
	unsigned int num_connections = 0;
	for (unsigned int l=0; l<NUM_LABELS; l++)
		for (unsigned int lc=l+1; lc<NUM_LABELS; lc++)
			if (connectivity[l][lc])
				num_connections++;

	Matrix<float, NUM_LABELS, 1> background_ref;
	Matrix<float, Dynamic, Dynamic> A(NUM_LABELS + num_connections, NUM_LABELS);
	Matrix<float, Dynamic, 1> B(NUM_LABELS + num_connections, 1);
	MatrixXf AtA, AtB;
	A.fill(0.f); B.fill(0.f);

	//Find the depth range of the image (approx)
	float min_depth = 10.f, max_depth = 0.f;
	for (unsigned int l=0; l<NUM_LABELS; l++)
		if (size_kmeans[l] != 0)
		{
			min_depth = min(min_depth, kmeans(0,l));
			max_depth = max(max_depth, kmeans(0,l));
		}

	//Define/set parameters of the optimization problem
	const float lambda_reg = 0.5f; 
	const float lambda_temp = use_b_temp_reg ? 1.5f : 0.f;
	const float lambda_depth = 0.15f;
	const float w_min = 0.5f*(lim_backg - lim_nobackg);
	const float depth_threshold_backg = 0.75f*min_depth + 0.25f*max_depth;


	//Fill A and B
	//----------------------------------------------------------------
	//Data term + "depth" term + temporal regularization
	for (unsigned int l=0; l<NUM_LABELS; l++)
	{
		const float transition_error = 0.5f*(lim_nobackg + lim_backg);
		background_ref(l) = max(0.f, min(2.f, (r[l] - lim_nobackg)/(lim_backg-lim_nobackg)));
		const float w_dataterm = (1.f + 1.5f*r[l]>transition_error)*sqrtf(square((r[l] - transition_error)/w_min) + 1.f);
		const float depth_term = lambda_depth*max(0.f, exp(kmeans(0,l))-exp(depth_threshold_backg));

		A(l,l) = w_dataterm + depth_term + lambda_temp;
		B(l) = w_dataterm*background_ref(l) + lambda_temp*b_segm_warped[l];
	}

	//Spatial regularization
	unsigned int cont_reg = 0;
	for (unsigned int l=0; l<NUM_LABELS; l++)
		for (unsigned int lc=l+1; lc<NUM_LABELS; lc++)
			if (connectivity[l][lc] == true)
			{
				const float weight_reg = lambda_reg;
				A(NUM_LABELS + cont_reg, l) = weight_reg;
				A(NUM_LABELS + cont_reg, lc) = -weight_reg;
				cont_reg++;		
			}

	//Build AtA and AtB
    AtA.multiply_AtA(A);
    AtB.multiply_AtB(A,B);

	//Solve
	b_segm = AtA.ldlt().solve(AtB);	

	//Classify clusters as static, uncertain or moving
	for (unsigned int l=0; l<NUM_LABELS; l++)
	{
		if (b_segm[l] > 0.667f) 
		{
			label_static[l] = false;
			label_dynamic[l] = true;
		}
		else if (b_segm[l] < 0.333f)
		{
			label_dynamic[l] = false;
			label_static[l] = true;
		}
		else
		{
			label_dynamic[l] = true;
			label_static[l] = true;			
		}
	}
}


void VO_SF::warpStaticDynamicSegmentation()
{
	//Warp the KMeans and then compute belongings to them. 
	//-----------------------------------------------------------
	const MatrixXf &depth_ref = depth[image_level];
	const MatrixXf &xx_ref = xx[image_level];
	const MatrixXf &yy_ref = yy[image_level];

	//Warped Kmeans
	Matrix<float, 3, NUM_LABELS> kmeans_w;
	for (unsigned int l=0; l<NUM_LABELS; l++)
	{
        const Matrix4f trans = T_clusters[l].inverse();
		const Vector4f kmeans_homog(kmeans(0,l), kmeans(1,l), kmeans(2,l), 1.f);
		kmeans_w.col(l) = trans.block<3,4>(0,0)*kmeans_homog;
	}

	//Compute distance between the kmeans (to improve runtime of the next phase)
	Matrix<float, NUM_LABELS, NUM_LABELS> kmeans_dist;
	for (unsigned int la=0; la<NUM_LABELS; la++)
		for (unsigned int lb=la+1; lb<NUM_LABELS; lb++)
			kmeans_dist(la,lb) = (kmeans_w.col(la) - kmeans_w.col(lb)).squaredNorm();

	//Compute KMeans belongings
	for (unsigned int u=0; u<cols; u++)
		for (unsigned int v=0; v<rows; v++)
			if (depth_ref(v,u) != 0.f)
			{
				const Vector3f p(depth_ref(v,u), xx_ref(v,u), yy_ref(v,u));
				unsigned int label = 0;
				float min_dist = (kmeans_w.col(0) - p).squaredNorm();
				float dist_here;

				for (unsigned int l=1; l<NUM_LABELS; l++)
				{
					if (kmeans_dist(label,l) > 4.f*min_dist) continue;
					else if ((dist_here = (kmeans_w.col(l) - p).squaredNorm()) < min_dist)
					{
						label = l;
						min_dist = dist_here;
					}
				}

				b_segm_image_warped(v,u) = b_segm[label];
			}
			else
				b_segm_image_warped(v,u) = 0.f;

	//Off initially for the first iteration but must be turned on after that
	if (use_b_temp_reg == false)
		use_b_temp_reg = true;
}

void VO_SF::computeSegTemporalRegValues()
{
	b_segm_warped.fill(0.f);
	image_level = round(log2(width/cols));
	const MatrixXi &labels_ref = labels[image_level];
	const MatrixXf &depth_old_ref = depth_old[image_level];

	for (unsigned int u=0; u<cols; u++)
		for (unsigned int v=0; v<rows; v++)
			if (depth_old_ref(v,u) != 0.f)
				b_segm_warped[labels_ref(v,u)] += b_segm_image_warped(v,u);
	
	
	for (unsigned int l=0; l<NUM_LABELS; l++)
		if (size_kmeans[l] != 0)
			b_segm_warped[l] /= size_kmeans[l];
}



