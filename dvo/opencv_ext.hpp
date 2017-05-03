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

#ifndef OPENCV_EXT_HPP_
#define OPENCV_EXT_HPP_

#include <dvo/common.hpp>
#include <ostream>

namespace dvo
{


template<int InsetRight, int InsetBottom, typename ScalarT>
inline bool inImage(int width, int height, ScalarT x, ScalarT y)
{
  return x > 0 && y > 0 && x < (width - InsetRight) && y < (height - InsetBottom);
}

template<int InsetRight, int InsetBottom, typename ScalarT>
inline bool inImage(int width, int height, const Eigen::Matrix<ScalarT, 2, 1> &x)
{
  return inImage<InsetRight, InsetBottom>(width, height, x(0), x(1));
}

template<int InsetRight, int InsetBottom, typename ScalarT>
inline bool inImage(const cv::Mat &m, const Eigen::Matrix<ScalarT, 2, 1> &x)
{
  return inImage<InsetRight, InsetBottom>(m.cols, m.rows, x);
}

template<typename ScalarT, int Dim>
inline bool isAnyNaN(const cv::Vec<ScalarT, Dim> &v)
{
  for(int i = 0; i < Dim; ++i)
    if(v[i] != v[i]) return true;

  return false;
}
/*
template<typename Scalar1T, typename Scalar2T>
cv::Mat_<Scalar1T> divide(const cv::Mat_<Scalar1T> &in1, const cv::Mat_<Scalar2T> &in2)
{
  cv::Mat_<Scalar1T> out;
  in1.copyTo(out);
  auto it1 = in1.begin(), it2 = in2.begin(), out_it = out.begin();

  for(; it1 != in1.end(); ++it1, ++it2, ++out_it)
    *out_it = *it1 / *it2;

  return out;
}

template<typename Scalar1T, typename Scalar2T>
void divideInPlace(cv::Mat_<Scalar1T> &in1, const cv::Mat_<Scalar2T> &in2)
{
  auto it1 = in1.begin(), it2 = in2.begin();

  for(; it1 != in1.end(); ++it1, ++it2)
    *it1 /= *it2;
}

template<typename ScalarT>
cv::Mat_<ScalarT> invert(const cv::Mat_<ScalarT> &in)
{
  cv::Mat_<ScalarT> out(in.size());
  auto it = in.begin();
  auto out_it = out.begin();

  for(; it != in.end(); ++it, ++out_it)
    *out_it = ScalarT(1.0) / *it;

  return out;
}

template<typename ScalarT>
void invertInPlace(cv::Mat_<ScalarT> &in1)
{
  auto it1 = in1.begin();

  for(; it1 != in1.end(); ++it1)
    *it1 = ScalarT(1.0) / *it1;
}
*/
bool isAnyNaN(const cv::Vec<float, 4> &v);
bool isAnyNaN(const cv::Vec<float, 8> &v);

cv::Mat pasteLeftRight(const cv::Mat &left, const cv::Mat &right);

cv::Mat sqrt(const cv::Mat &m);
cv::Mat normalize(const cv::Mat &m, const cv::Mat &mask);

void computeRgbValuesForHsvRange(float h_min, float h_max, std::vector<cv::Vec3f> &rgb);

cv::Mat remapToHsvRange(const cv::Mat &m, float h_min, float h_max, cv::Mat mask);
cv::Mat remapToHsvRangeNoNormalize(const cv::Mat &m, float h_min, float h_max, cv::Mat mask);

void imshow2(const std::string &name, const cv::Mat &m);

cv::Mat merge(const cv::Mat &c0, const cv::Mat &c1);
cv::Mat merge(const cv::Mat &c0, const cv::Mat &c1, const cv::Mat &c2);
cv::Mat merge(const cv::Mat &c0, const cv::Mat &c1, const cv::Mat &c2, const cv::Mat &c3);

void split(const cv::Mat &in, cv::Mat &c0, cv::Mat &c1);

template<typename LambdaT>
void for_each_4neighbours(int width, int height, int x, int y, const LambdaT &fn)
{
  if(x > 0)
  {
    fn(x - 1, y);
  }

  if(x + 1 < width)
  {
    fn(x + 1, y);
  }

  if(y > 0)
  {
    fn(x, y - 1);
  }

  if(y + 1 < height)
  {
    fn(x, y + 1);
  }
}
template<typename LambdaT>
void for_each_4neighbours_with_index(int width, int height, int x, int y, const LambdaT &fn)
{
  if(x > 0)
  {
    fn(0, x - 1, y);
  }

  if(x + 1 < width)
  {
    fn(1, x + 1, y);
  }

  if(y > 0)
  {
    fn(2, x, y - 1);
  }

  if(y + 1 < height)
  {
    fn(3, x, y + 1);
  }
}

template<typename LambdaT>
void for_each_8neighbours(int width, int height, int x, int y, const LambdaT &fn)
{
  if(x > 0)
  {
    if(y > 0)
    {
      fn(x - 1, y - 1);
    }

    fn(x - 1, y);

    if(y + 1 < height)
    {
      fn(x - 1, y + 1);
    }
  }

  if(x + 1 < width)
  {
    if(y > 0)
    {
      fn(x + 1, y - 1);
    }

    fn(x + 1, y);

    if(y + 1 < height)
    {
      fn(x + 1, y + 1);
    }
  }

  if(y > 0)
  {
    fn(x, y - 1);
  }

  if(y + 1 < height)
  {
    fn(x, y + 1);
  }
}

template<typename LambdaT>
void for_each_8neighbours_with_index(int width, int height, int x, int y, const LambdaT &fn)
{
  if(x > 0)
  {
    if(y > 0)
    {
      fn(0, x - 1, y - 1);
    }

    fn(1, x - 1, y);

    if(y + 1 < height)
    {
      fn(2, x - 1, y + 1);
    }
  }

  if(x + 1 < width)
  {
    if(y > 0)
    {
      fn(3, x + 1, y - 1);
    }

    fn(4, x + 1, y);

    if(y + 1 < height)
    {
      fn(5, x + 1, y + 1);
    }
  }

  if(y > 0)
  {
    fn(6, x, y - 1);
  }

  if(y + 1 < height)
  {
    fn(7, x, y + 1);
  }
}
struct PrintMinMax
{
  const cv::Mat &image;

  PrintMinMax(const cv::Mat &img) : image(img)
  {
  }
};

} /* namespace dvo */

std::ostream& operator<<(std::ostream& os, const dvo::PrintMinMax& print);

#endif /* OPENCV_EXT_HPP_ */
