
#include "joint_vo_sf.h"

using namespace mrpt;
using namespace mrpt::poses;
using namespace mrpt::utils;
using namespace std;
using namespace Eigen;


struct IndexAndDistance
{
    int idx;
    float distance;

    bool operator<(const IndexAndDistance &o) const
    {
        return distance < o.distance;
    }
};

void VO_SF::initializeKMeans()
{
	//Initialization
	rows_i = rows/2; cols_i = cols/2; //KMeans are computed at one resolution lower than the max
	image_level = round(log2(width/cols_i));
	const MatrixXf &depth_ref = depth_old[image_level];
	const MatrixXf &xx_ref = xx_old[image_level];
	const MatrixXf &yy_ref = yy_old[image_level];
	MatrixXi &labels_ref = labels[image_level];
	labels_ref.assign(NUM_LABELS);


	//Initialize from scratch at every iteration
	//-------------------------------------------------------------
	//Create seeds for the k-means by dividing the image domain
	unsigned int u_label[NUM_LABELS], v_label[NUM_LABELS];
	const unsigned int vert_div = ceil(sqrt(NUM_LABELS));
	const float u_div = float(cols_i)/float(NUM_LABELS+1);
	const float v_div = float(rows_i)/float(vert_div+1); 
	for (unsigned int i=0; i<NUM_LABELS; i++)
	{
		u_label[i] = round((i + 1)*u_div);
		v_label[i] = round((i%vert_div + 1)*v_div);
	}

	//Compute the coordinates associated to the initial seeds
	for (unsigned int u=0; u<cols_i; u++)
		for (unsigned int v=0; v<rows_i; v++)
			if (depth_ref(v,u) != 0.f)
			{
				unsigned int min_dist = 1000000.f, quad_dist;
				unsigned int ini_label = NUM_LABELS;
				for (unsigned int l=0; l<NUM_LABELS; l++)
					if ((quad_dist = square(v - v_label[l]) + square(u - u_label[l])) < min_dist)
					{
						ini_label = l;
						min_dist = quad_dist;
					}

				labels_ref(v,u) = ini_label;
			}

	//Compute the "center of mass" for each region
	std::vector<float> depth_sorted[NUM_LABELS]; //, x_sorted[NUM_LABELS], y_sorted[NUM_LABELS];

	for (unsigned int u=0; u<cols_i; u++)
		for (unsigned int v=0; v<rows_i; v++)
			if (depth_ref(v,u) != 0.f)
				{
					depth_sorted[labels_ref(v,u)].push_back(depth_ref(v,u));
					//x_sorted[labels_ref(v,u)].push_back(xx_ref(v,u));
					//y_sorted[labels_ref(v,u)].push_back(yy_ref(v,u));
						
				}

	//Compute the first KMeans values (using median to avoid getting a floating point between two regions)
	const float inv_f_i = 2.f*tan(0.5f*fovh)/float(cols_i);
    const float disp_u_i = 0.5f*(cols_i-1);
    const float disp_v_i = 0.5f*(rows_i-1);
	for (unsigned int l=0; l<NUM_LABELS; l++)
	{
		const unsigned int size_label = depth_sorted[l].size();
		const unsigned int med_pos = size_label/2;

		if (size_label > 0)
		{
			std::nth_element(depth_sorted[l].begin(), depth_sorted[l].begin() + med_pos, depth_sorted[l].end());
			//std::nth_element(x_sorted[l].begin(), x_sorted[l].begin() + med_pos, x_sorted[l].end());
			//std::nth_element(y_sorted[l].begin(), y_sorted[l].begin() + med_pos, y_sorted[l].end());
					
			kmeans(0,l) = depth_sorted[l].at(med_pos);
			//x_kmeans[l] = x_sorted[l].at(med_pos);
			//y_kmeans[l] = y_sorted[l].at(med_pos);
			kmeans(1,l) = (u_label[l]-disp_u_i)*kmeans(0,l)*inv_f_i;
			kmeans(2,l) = (v_label[l]-disp_v_i)*kmeans(0,l)*inv_f_i;
		}
		else
		{
			kmeans.col(l).fill(0.f);
			//printf("\n label %d is empty from the beginning", l);
		}
	}
}

void VO_SF::kMeans3DCoord()
{
	const unsigned int max_level = round(log2(width/cols));
    const unsigned int lower_level = max_level+1;
	const unsigned int iter_kmeans = 10;

	//Refs
	const MatrixXf &depth_ref = depth_old[lower_level];
	const MatrixXf &xx_ref = xx_old[lower_level];
	const MatrixXf &yy_ref = yy_old[lower_level];
	MatrixXi &labels_lowres = labels[lower_level];

	//Initialization
	initializeKMeans();


    //                                      Iterate 
    //=======================================================================================
	const float max_depth_dist = 1.f;
    vector<vector<IndexAndDistance> > cluster_distances(NUM_LABELS, vector<IndexAndDistance>(NUM_LABELS));

    MatrixXf centers_a(3,NUM_LABELS), centers_b(3,NUM_LABELS);
	int count[NUM_LABELS];

	//Fill centers_a (I need to do it in this way to get maximum speed...)
	//centers_a.swap(kmeans);
	for (unsigned int c=0; c<NUM_LABELS; c++)
		for (unsigned int r=0; r<3; r++)
			centers_a(r,c) = kmeans(r,c);

    for (unsigned int i=0; i<iter_kmeans-1; i++)
    {
        centers_b.setZero();

		//Compute and sort distances between the kmeans
        for (unsigned int l=0; l<NUM_LABELS; l++)
        {
            count[l] = 0;
			vector<IndexAndDistance> &distances = cluster_distances.at(l);
            for (unsigned int li=0; li<NUM_LABELS; li++)
            {
                IndexAndDistance &idx_and_distance = distances.at(li);
                idx_and_distance.idx = li;
                idx_and_distance.distance = (centers_a.col(l) - centers_a.col(li)).squaredNorm();
            }
            std::sort(distances.begin(), distances.end());
        }


        //Compute belonging to each label
        for (unsigned int u=0; u<cols_i; u++)
            for (unsigned int v=0; v<rows_i; v++)
                if (depth_ref(v,u) != 0.f)
                {
                    const int last_label = labels_lowres(v,u);
					int best_label = last_label;
                    vector<IndexAndDistance> &distances = cluster_distances.at(last_label);

                    const Vector3f p(depth_ref(v,u), xx_ref(v,u), yy_ref(v,u));

                    const float distance_to_last_label = (centers_a.col(last_label) - p).squaredNorm();
                    float best_distance = distance_to_last_label;

                    for(size_t li = 1; li < distances.size(); ++li)
                    {
                        IndexAndDistance &idx_and_distance = distances.at(li);

                        if(idx_and_distance.distance > 4.f * distance_to_last_label) break;

                        const float distance_to_label = (centers_a.col(idx_and_distance.idx) - p).squaredNorm();

                        if(distance_to_label < best_distance)
                        {
                            best_distance = distance_to_label;
                            best_label = idx_and_distance.idx;
                        }
                    }

                    labels_lowres(v,u) = best_label;
                    centers_b.col(best_label) += p;
                    count[best_label] += 1;
                }


        for (unsigned int l=0; l<NUM_LABELS; l++)
        {
            if (count[l] > 0) centers_b.col(l) /= count[l];
			else	 		  ;//printf("\n Label %d is empty", l);
        }

        const float max_diff = (centers_a - centers_b).lpNorm<Infinity>();
        //std::cout << max_diff <<  " " << changed_points << std::endl;
        centers_a.swap(centers_b);

        if (max_diff < 1e-2f) break;
    }

	//Copy solution
	//kmeans.swap(centers_a);
	for (unsigned int c=0; c<NUM_LABELS; c++)
		for (unsigned int r=0; r<3; r++)
			kmeans(r,c) = centers_a(r,c);



    //      Compute the labelling functions at the max resolution (rows,cols)
    //------------------------------------------------------------------------------------
	const MatrixXf &depth_highres = depth_old[max_level];
	const MatrixXf &xx_highres = xx_old[max_level];
	const MatrixXf &yy_highres = yy_old[max_level];
	MatrixXi &labels_ref = labels[max_level];

	//Initialize
	labels_ref.assign(NUM_LABELS);
	for(int i = 0; i < NUM_LABELS; ++i)
		count[i] = 0;


    //Find the closest kmean and set the corresponding label to 1
    for (unsigned int u=0; u<cols; u++)
        for (unsigned int v=0; v<rows; v++)
            if (depth_highres(v,u) != 0.f)
            {
                const int label_lowres_here = labels_lowres(v/2,u/2);
				const int last_label = (label_lowres_here == NUM_LABELS) ? 0 : label_lowres_here; //If it was invalid in the low res level initialize it randomly (at 0)

                int best_label = last_label;
                vector<IndexAndDistance> &distances = cluster_distances.at(last_label);
                const Vector3f p(depth_highres(v,u), xx_highres(v,u), yy_highres(v,u));

                const float distance_to_last_label = (centers_a.col(last_label) - p).squaredNorm();
                float best_distance = distance_to_last_label;

                for(size_t li = 1; li < distances.size(); ++li)
                {
                    IndexAndDistance &idx_and_distance = distances.at(li);
                    if(idx_and_distance.distance > 4.f * distance_to_last_label) break;

                    const float distance_to_label = (centers_a.col(idx_and_distance.idx) - p).squaredNorm();

                    if(distance_to_label < best_distance)
                    {
                        best_distance = distance_to_label;
                        best_label = idx_and_distance.idx;
                    }
                }
                labels_ref(v,u) = best_label;
				count[best_label]++;
            }

    //Compute connectivity
    computeRegionConnectivity();

    //Smooth regions
	rows_i = rows; cols_i = cols;
    smoothRegions(max_level);

	//Save the size of each segment (at max resolution)
	for (unsigned int l=0; l<NUM_LABELS; l++)
		size_kmeans[l] = count[l];
}

void VO_SF::computeRegionConnectivity()
{
    const unsigned int max_level = round(log2(width/cols));
    const float dist2_threshold = square(0.03f*120.f/float(rows));

	//Refs
	const MatrixXi &labels_ref = labels[max_level];
	const MatrixXf &depth_old_ref = depth_old[max_level];
	const MatrixXf &xx_old_ref = xx_old[max_level];
	const MatrixXf &yy_old_ref = yy_old[max_level];

    for (unsigned int i=0; i<NUM_LABELS; i++)
        for (unsigned int j=0; j<NUM_LABELS; j++)
		{
			if (i == j) connectivity[i][j] = true;
			else		connectivity[i][j] = false;
		}

    for (unsigned int u=0; u<cols-1; u++)
        for (unsigned int v=0; v<rows-1; v++)					
			if (depth_old_ref(v,u) != 0.f)
            {
                //Detect change in the labelling (v+1)
                if ((labels_ref(v,u) != labels_ref(v+1,u))&&(labels_ref(v+1,u) != NUM_LABELS))
                {
                    const float disty = square(depth_old_ref(v,u) - depth_old_ref(v+1,u)) + square(yy_old_ref(v,u) - yy_old_ref(v+1,u));
                    if (disty < dist2_threshold)
                    {
                        connectivity[labels_ref(v,u)][labels_ref(v+1,u)] = true;
                        connectivity[labels_ref(v+1,u)][labels_ref(v,u)] = true;
                    }
                }

                //Detect change in the labelling (u+1)
                if (labels_ref(v,u) != labels_ref(v,u+1)&&(labels_ref(v,u+1) != NUM_LABELS))
                {
                    const float distx = square(depth_old_ref(v,u) - depth_old_ref(v,u+1)) + square(xx_old_ref(v,u) - xx_old_ref(v,u+1));
                    if (distx < dist2_threshold)
                    {
                        connectivity[labels_ref(v,u)][labels_ref(v,u+1)] = true;
                        connectivity[labels_ref(v,u+1)][labels_ref(v,u)] = true;
                    }
                }
            }
}

void VO_SF::smoothRegions(unsigned int image_level)
{
	//Refs
	const MatrixXf &depth_ref = depth_old[image_level];
	const MatrixXf &xx_ref = xx_old[image_level];
	const MatrixXf &yy_ref = yy_old[image_level];
	const MatrixXi &labels_ref = labels[image_level];
	Matrix<float, NUM_LABELS+1, Dynamic> &label_funct_ref = label_funct[image_level];

    //Set all labels to zero initially
    label_funct_ref.assign(0.f);


	//Smooth
	const float k_smooth = 100.f;
	Matrix<float, NUM_LABELS+1, 1> weights;

	for (unsigned int u=0; u<cols_i; u++)
		for (unsigned int v=0; v<rows_i; v++)
		{
			const Vector3f p(depth_ref(v,u), xx_ref(v,u), yy_ref(v,u));
			const unsigned int pixel_ind = v+u*rows_i;

			if (labels_ref(v,u) < NUM_LABELS)
			{
				const unsigned int lab = labels_ref(v,u);
				weights.fill(0.f);

				//Compute distance to its original label
				const float ref_dist = (kmeans.col(lab) - p).squaredNorm();

				for (unsigned int l=0; l<NUM_LABELS; l++)
					if (connectivity[lab][l])
					{
						const float dist = (kmeans.col(l) - p).squaredNorm();
						const float ref_dist = (kmeans.col(lab) - p).squaredNorm();
						const float exponent = k_smooth*abs(ref_dist - dist);

						if (exponent < 6.f) //Otherwise it is almost zero and doesn't need to be computed
							weights(l) = exp(-exponent);
					}

				const float sum_weights_inv = 1.f/weights.sumAll();
				label_funct_ref.col(pixel_ind) = weights*sum_weights_inv;
			}
			else
				label_funct_ref(NUM_LABELS, pixel_ind) = 1.f;
		}
}

void VO_SF::createLabelsPyramidUsingKMeans()
{
	const float limit_depth_dist = 1.f;

	//Compute distance between the kmeans (to improve runtime of the next phase)
	Matrix<float, NUM_LABELS, NUM_LABELS> kmeans_dist;
	for (unsigned int la=0; la<NUM_LABELS; la++)
		for (unsigned int lb=la+1; lb<NUM_LABELS; lb++)
			kmeans_dist(la,lb) = (kmeans.col(la) - kmeans.col(lb)).squaredNorm();
	
	//Generate levels
    for (unsigned int i = 1; i<ctf_levels; i++)
    {
        unsigned int s = pow(2.f,int(i));
        cols_i = cols/s; rows_i = rows/s;
		image_level = i + round(log2(width/cols));

		//Refs
		MatrixXi &labels_ref = labels[image_level];
		const MatrixXf &depth_old_ref = depth_old[image_level];
		const MatrixXf &xx_old_ref = xx_old[image_level];
		const MatrixXf &yy_old_ref = yy_old[image_level];

		labels_ref.assign(NUM_LABELS);
	
		//Compute belonging to each label
		for (unsigned int u=0; u<cols_i; u++)
			for (unsigned int v=0; v<rows_i; v++)
				if (depth_old_ref(v,u) != 0.f)
				{			
					unsigned int label = 0;
					const Vector3f p(depth_old_ref(v,u), xx_old_ref(v,u), yy_old_ref(v,u));
					float min_dist = (kmeans.col(0) - p).squaredNorm();
					float quad_dist;

					for (unsigned int l=1; l<NUM_LABELS; l++)
					{
						if (kmeans_dist(label,l) > 4.f*min_dist) continue;

						else if ((quad_dist = (kmeans.col(l)-p).squaredNorm()) < min_dist)
						{
							label = l;
							min_dist = quad_dist;
						}
					}

					labels_ref(v,u) = label;
				}

		//Smooth regions
		smoothRegions(image_level);
	}
}


void VO_SF::createImagesOfSegmentations()
{
    image_level = round(log2(width/cols));

	//Refs
	const Matrix<float, NUM_LABELS+1, Dynamic> label_funct_ref = label_funct[image_level];
	const MatrixXf &depth_old_ref = depth_old[image_level];

    //Associate colors to labels
    float r[NUM_LABELS], g[NUM_LABELS], b[NUM_LABELS]; //20 max 
    for (unsigned int l=0; l<NUM_LABELS; l++)
    {
        const float indx = float(l)/float(NUM_LABELS-1);
        mrpt::utils::colormap(mrpt::utils::cmJET, indx, r[l], g[l], b[l]);
    }

    //Compute the color for every pixel according to the estimated labeling
	for (unsigned int c=0; c<3; c++)
	{
		labels_image[c].fill(0.f);
		backg_image[c].fill(0.f);
	}

    for (unsigned int u=0; u<cols; u++)
        for (unsigned int v=0; v<rows; v++)
            if (depth_old_ref(v,u) != 0.f)
			{
                for (unsigned int l=0; l<NUM_LABELS; l++)
                {
                    const float lab = label_funct_ref(l, v+u*rows);
                    labels_image[0](v,u) += lab*r[l];
                    labels_image[1](v,u) += lab*g[l];
                    labels_image[2](v,u) += lab*b[l];

					float aux_var;
					if (bf_segm[l] < 0.333) 		aux_var = 0.f;
					else if (bf_segm[l] > 0.667)	aux_var = 1.f;
					else							aux_var = min(1.f, 3.f*(bf_segm[l] - 0.333f));
					backg_image[0](v,u) += aux_var*lab;
					backg_image[2](v,u) += (1.f - aux_var)*lab;
                }
			}
}



