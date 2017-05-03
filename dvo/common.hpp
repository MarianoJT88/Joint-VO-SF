/**
 *  This file is part of dvo.
 *
 *  Copyright 2014 Christian Kerl <christian.kerl@in.tum.de> (Technical University of Munich)
 *  For more information see <http://vision.in.tum.de/data/software/dvo>.
 *
 *  dvo is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  dvo is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with dvo.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <vector>

#include <opencv2/opencv.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#if (defined __GNUC__)
#define NOEXCEPT noexcept (true)
#elif (defined _MSC_VER)
#define NOEXCEPT
#else
#error Check NOEXCEPT equivalent for your compiler!
#endif

namespace dvo
{

template<typename ScalarT>
struct GeometryTypes
{
  typedef Eigen::Matrix<ScalarT, 3, 1> Point;
  typedef Eigen::Matrix<ScalarT, 4, 1> HomogenousPoint;
  typedef Eigen::Matrix<ScalarT, 4, 1> HomogeneousPoint;
  typedef Eigen::Matrix<ScalarT, 2, 1> Pixel;

  typedef Eigen::Matrix<ScalarT, 3, 3> Matrix3;
  typedef Eigen::Matrix<ScalarT, 4, 4> Matrix4;
  typedef Eigen::Transform<ScalarT, 3, Eigen::Isometry> Transformation;
};

} /* namespace dvo */

#endif /* COMMON_HPP_ */
