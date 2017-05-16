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

#ifndef structs_parallelization_H
#define structs_parallelization_H

#include <joint_vo_sf.h>
#include <dvo/normal_equation.hpp>
#include <dvo/opencv_ext.hpp>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range2d.h>


typedef tbb::blocked_range2d<int> ImageDomain;

inline cv::Rect toRegion(ImageDomain const &domain)
{
    int x = domain.cols().begin(), y = domain.rows().begin(), w = domain.cols().size(), h = domain.rows().size();
    return cv::Rect(x, y, w, h);
}

template<void (VO_SF::*F1)(cv::Rect)>
class VO_SF_RegionFunctor
{
private:
    VO_SF &self;
public:
    VO_SF_RegionFunctor(VO_SF &new_self) : self(new_self) {}

    void operator()(ImageDomain const &domain) const
    {
        cv::Rect r = toRegion(domain);
        (self.*F1)(r);
    }
};

typedef dvo::NormalEquation<float, 6, 2> NormalEquation;

struct NormalEquationAndChi2
{
    NormalEquation nes;
    float chi2;

    NormalEquationAndChi2()
    {
        nes.setZero();
        chi2 = 0.f;
    }

    struct Reduce
    {
        NormalEquationAndChi2 operator()(NormalEquationAndChi2 const& a, NormalEquationAndChi2 const& b) const
        {
            NormalEquationAndChi2 r;
            r.chi2 = a.chi2 + b.chi2;
            r.nes.add(a.nes);
            r.nes.add(b.nes);

            return r;
        }
    };
};

struct IrlsContext
{
    float *A, *B;
    float k_Cauchy, Cauchy_factor;
	float sum_residuals;
	unsigned int num_pixels;
    Vector6f Var;
	Eigen::VectorXf residuals;


	inline void computeNewResiduals()
	{
		//A is sorted weirdly (Jc11, Jd11, Jc12, Jd12...Jc21, Jd21...), so I can't get it complete with:
		//const MatrixXf J_aux = Map<Matrix<float, 6, Dynamic>>( A, 6, num_equations);
		//residuals = Map<VectorXf>( B, 2*num_pixels, 1);
		
		sum_residuals = 0.f;
		for (size_t i = 0; i < num_pixels; ++i)
		{
			const JacobianT::MapType J(A + i*JacobianElements); 
			const ResidualT::MapType r(B + i*ResidualElements);
			residuals.block<2,1>(2*i,0) = J*Var - r;
			sum_residuals += std::abs(residuals(2*i)) + std::abs(residuals(2*i+1));
		}

		const float mean_res = std::max(1e-5f, sum_residuals/float(2*num_pixels));
		k_Cauchy = Cauchy_factor/(mean_res*mean_res);
	}
};

struct IrlsElementFn
{
    typedef tbb::blocked_range<size_t> Range;
    IrlsContext const &ctx;

    IrlsElementFn(IrlsContext const &new_ctx) : ctx(new_ctx) {}

    inline void update(NormalEquationAndChi2 &nes_and_chi2, Eigen::Matrix2f &info, size_t i) const
    {
		const float res_c = ctx.residuals(2*i);
		const float res_d = ctx.residuals(2*i+1);

		//Intensity and depth weights
        const float res_weight_intensity = 1.f/(1.f + ctx.k_Cauchy*res_c*res_c);
        const float res_weight_depth = 1.f/(1.f + ctx.k_Cauchy*res_d*res_d);

        info(0,0) = res_weight_intensity;
        info(1,1) = res_weight_depth;

		//Update matrices
		const float *A_elem = ctx.A + i*JacobianElements;
		const float *B_elem = ctx.B + i*ResidualElements;
		nes_and_chi2.nes.update(A_elem, B_elem, info.data());

		//Update chi2
        nes_and_chi2.chi2 += res_c*res_c*res_weight_intensity + res_d*res_d*res_weight_depth;
    }

    NormalEquationAndChi2 operator()(const Range& range, const NormalEquationAndChi2 &initial) const
    {
        NormalEquationAndChi2 r(initial);
        Eigen::Matrix2f info = Eigen::Matrix2f::Identity();
        for(Range::const_iterator it = range.begin(); it != range.end(); ++it)
        {
            update(r, info, it);
        }

        return r;
    }
};

struct JacobianElementFn
{
    typedef tbb::blocked_range<size_t> Range;
    SolveForMotionWorkspace const &ws;
    VO_SF const &self;

    JacobianElementFn(SolveForMotionWorkspace const &new_ws, VO_SF const &new_self) : ws(new_ws), self(new_self) {}

    NormalEquationAndChi2 operator()(const Range& range, const NormalEquationAndChi2 &initial) const
    {
        const float f_inv = float(self.cols_i)/(2.f*tan(0.5f*self.fovh));

        NormalEquationAndChi2 result(initial);
        Eigen::Matrix2f info = Eigen::Matrix2f::Identity();

        Eigen::MatrixXf const& depth_inter_ = self.depth_inter[self.image_level];
        Eigen::MatrixXf const& xx_inter_ = self.xx_inter[self.image_level];
        Eigen::MatrixXf const& yy_inter_ = self.yy_inter[self.image_level];

        for(Range::const_iterator it = range.begin(); it != range.end(); ++it)
        {
            JacobianT::MapType J(ws.A + it*JacobianElements);
            ResidualT::MapType r(ws.B + it*ResidualElements);

            const std::pair<int, int> &vu = ws.indices[it];
            const int &v = vu.first;
            const int &u = vu.second;

            // Precomputed expressions
            const float d = depth_inter_(v,u);
            const float inv_d = 1.f/d;
            const float x = xx_inter_(v,u);
            const float y = yy_inter_(v,u);

            //                                          Intensity
            //------------------------------------------------------------------------------------------------
            const float dycomp_c = self.dcu(v,u)*f_inv*inv_d;
            const float dzcomp_c = self.dcv(v,u)*f_inv*inv_d;
            const float twc = self.weights_c(v,u)*self.k_photometric_res;

            //Fill the matrix A
            J(0,0) = twc*(dycomp_c*x*inv_d + dzcomp_c*y*inv_d);
            J(0,1) = twc*(-dycomp_c);
            J(0,2) = twc*(-dzcomp_c);
            J(0,3) = twc*(dycomp_c*y - dzcomp_c*x);
            J(0,4) = twc*(dycomp_c*inv_d*y*x + dzcomp_c*(y*y*inv_d + d));
            J(0,5) = twc*(-dycomp_c*(x*x*inv_d + d) - dzcomp_c*inv_d*y*x);
            r(0) = twc*(-self.dct(v,u));

            //                                          Geometry
            //------------------------------------------------------------------------------------------------
            const float dycomp_d = self.ddu(v,u)*f_inv*inv_d;
            const float dzcomp_d = self.ddv(v,u)*f_inv*inv_d;
            const float twd = self.weights_d(v,u);

            //Fill the matrix A
            J(1,0) = twd*(1.f + dycomp_d*x*inv_d + dzcomp_d*y*inv_d);
            J(1,1) = twd*(-dycomp_d);
            J(1,2) = twd*(-dzcomp_d);
            J(1,3) = twd*(dycomp_d*y - dzcomp_d*x);
            J(1,4) = twd*(y + dycomp_d*inv_d*y*x + dzcomp_d*(y*y*inv_d + d));
            J(1,5) = twd*(-x - dycomp_d*(x*x*inv_d + d) - dzcomp_d*inv_d*y*x);
            r(1) = twd*(-self.ddt(v,u));

            result.nes.update(J.data(), r.data(), info.data());
        }

        return result;
    }
};


struct JacobianElementForRobustOdometryFn
{
    typedef tbb::blocked_range<size_t> Range;
    SolveForMotionWorkspace const &ws;
    VO_SF const &self;

    JacobianElementForRobustOdometryFn(SolveForMotionWorkspace const &new_ws, VO_SF const &new_self) : ws(new_ws), self(new_self) {}

    float operator()(const Range& range, const float &initial_mean_residual) const
    {
        const float f_inv = float(self.cols_i)/(2.f*tan(0.5f*self.fovh));

        float result = initial_mean_residual;

        Eigen::MatrixXf const& depth_inter_ = self.depth_inter[self.image_level];
        Eigen::MatrixXf const& xx_inter_ = self.xx_inter[self.image_level];
        Eigen::MatrixXf const& yy_inter_ = self.yy_inter[self.image_level];
        Eigen::MatrixXi const& labels_ref = self.labels[self.image_level];

        for(Range::const_iterator it = range.begin(); it != range.end(); ++it)
        {
            JacobianT::MapType J(ws.A + it*JacobianElements);
            ResidualT::MapType r(ws.B + it*ResidualElements);

            const std::pair<int, int> &vu = ws.indices[it];
            const int &v = vu.first;
            const int &u = vu.second;

            // Precomputed expressions
            const float d = depth_inter_(v,u);
            const float inv_d = 1.f/d;
            const float x = xx_inter_(v,u);
            const float y = yy_inter_(v,u);

            const float w_dinobj = std::max(0.f, 1.f - self.b_segm_warped[labels_ref(v,u)]);

            //                                          Intensity
            //------------------------------------------------------------------------------------------------
            const float dycomp_c = self.dcu(v,u)*f_inv*inv_d;
            const float dzcomp_c = self.dcv(v,u)*f_inv*inv_d;
            const float twc = w_dinobj*d*self.k_photometric_res;

            //Fill the matrix A
            J(0,0) = twc*(dycomp_c*x*inv_d + dzcomp_c*y*inv_d);
            J(0,1) = twc*(-dycomp_c);
            J(0,2) = twc*(-dzcomp_c);
            J(0,3) = twc*(dycomp_c*y - dzcomp_c*x);
            J(0,4) = twc*(dycomp_c*inv_d*y*x + dzcomp_c*(y*y*inv_d + d));
            J(0,5) = twc*(-dycomp_c*(x*x*inv_d + d) - dzcomp_c*inv_d*y*x);
            r(0) = twc*(-self.dct(v,u));

            //                                          Geometry
            //------------------------------------------------------------------------------------------------
            const float dycomp_d = self.ddu(v,u)*f_inv*inv_d;
            const float dzcomp_d = self.ddv(v,u)*f_inv*inv_d;
            const float twd = w_dinobj * d;

            //Fill the matrix A
            J(1,0) = twd*(1.f + dycomp_d*x*inv_d + dzcomp_d*y*inv_d);
            J(1,1) = twd*(-dycomp_d);
            J(1,2) = twd*(-dzcomp_d);
            J(1,3) = twd*(dycomp_d*y - dzcomp_d*x);
            J(1,4) = twd*(y + dycomp_d*inv_d*y*x + dzcomp_d*(y*y*inv_d + d));
            J(1,5) = twd*(-x - dycomp_d*(x*x*inv_d + d) - dzcomp_d*inv_d*y*x);
            r(1) = twd*(-self.ddt(v,u));

            result += r.cwiseAbs().sum();
        }

        return result;
    }
};


// unfortunately we don't have boost or modern c++ :(
template<class X, void (X::*p)()>
class MemberFunctor
{
    X& _x;
public:
    MemberFunctor(X& x) : _x( x ) {}
    void operator()() const { (_x.*p)(); }
};


struct WarpImagesDelegate
{
    typedef tbb::blocked_range2d<int> ImageDomain;

    VO_SF &self;
    WarpImagesDelegate(VO_SF &new_self) : self(new_self) {}

    void operator()(ImageDomain const &domain) const
    {
        int x = domain.cols().begin(), y = domain.rows().begin(), w = domain.cols().size(), h = domain.rows().size();
        cv::Rect region(x, y, w, h);
        self.warpImages(region);
    }
};

#endif