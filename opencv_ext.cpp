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

#include <dvo/opencv_ext.hpp>
#include <dvo/sse_ext.hpp>

namespace dvo
{

#ifdef __AVX__
bool isAnyNaN(const cv::Vec<float, 8> &v)
{
  __m256 a = _mm256_load_ps(v.val);

  return _mm256_movemask_ps(_mm256_cmp_ps(a, a, _CMP_UNORD_Q)) != 0;
}
#else
bool isAnyNaN(const cv::Vec<float, 8> &v)
{
  __m128 a = _mm_load_ps(v.val);
  __m128 b = _mm_load_ps(v.val + 4);

  return _mm_movemask_ps(_mm_cmpunord_ps(a, b)) != 0;
}
#endif

bool isAnyNaN(const cv::Vec<float, 4> &v)
{
  __m128 a = _mm_load_ps(v.val);

  return _mm_movemask_ps(_mm_cmpunord_ps(a, a)) != 0;
}

cv::Mat sqrt(const cv::Mat &m)
{
  cv::Mat r;
  cv::sqrt(m, r);
  return r;
}

cv::Mat log(const cv::Mat &m)
{
  cv::Mat r;
  cv::log(m, r);
  return r;
}

cv::Mat normalize(const cv::Mat &m, const cv::Mat &mask)
{
  cv::Mat normalized;
  double min, max;
  int min_idx, max_idx;
  cv::minMaxIdx((m), &min, &max, &min_idx, &max_idx, mask);

  return cv::Mat((m - min) / (max - min));
}

void computeRgbValuesForHsvRange(float h_min, float h_max, std::vector<cv::Vec3f> &rgb)
{
  std::vector<cv::Vec3f> hsv;;
  float h_step = (h_max - h_min) / 255.0f;

  for(int i = 0; i < 256; ++i)
    hsv.push_back(cv::Vec3f(h_min + h_step * i, 1.0f, 1.0f));

  cv::cvtColor(hsv, rgb, CV_HSV2BGR);
}

cv::Mat remapToHsvRangeNoNormalize(const cv::Mat &m, float h_min, float h_max, cv::Mat mask)
{
  if(m.type() == CV_32FC1 || m.type() == CV_64FC1)
  {
    cv::Mat normalized_u8c1, result;

    m.convertTo(normalized_u8c1, CV_8UC1, 255);
    normalized_u8c1.setTo(255, m > 1.0);

    std::vector<cv::Vec3f> rgb;
    computeRgbValuesForHsvRange(h_min, h_max, rgb);

    result.create(normalized_u8c1.size(), CV_8UC3);
    result.setTo(cv::Vec3b(255, 255, 255));

    for(int y = 0; y < result.rows; ++y)
      for(int x = 0; x < result.cols; ++x)
      {
        if(mask.empty() || mask.at<uchar>(y, x) != 0)
        {
           uchar idx = normalized_u8c1.at<uchar>(y, x);
           const cv::Vec3f &rgbf = rgb[idx] * 255.0f;

           result.at<cv::Vec3b>(y, x) = cv::Vec3b(uchar(rgbf[0]), uchar(rgbf[1]), uchar(rgbf[2]));
        }
      }

    return result;
  }
  else
  {
    return m;
  }
}

cv::Mat remapToHsvRange(const cv::Mat &m, float h_min, float h_max, cv::Mat mask)
{
  return remapToHsvRangeNoNormalize(normalize(m, mask), h_min, h_max, mask);
}

cv::Mat pasteLeftRight(const cv::Mat &left, const cv::Mat &right)
{
  cv::Mat result;
  cv::copyMakeBorder(left, result, 0, 0, 0, right.cols, cv::BORDER_CONSTANT, 0);

  cv::Mat roi(result, cv::Rect(left.cols, 0, right.cols, right.rows));
  right.copyTo(roi);

  return result;
}

void imshow2(const std::string &name, const cv::Mat &m)
{
  if(m.type() == CV_32FC1 || m.type() == CV_64FC1)
  {
    double min, max;
    cv::minMaxIdx(m, &min, &max);

    cv::imshow(name, (m - min) / (max - min));
  }
  else
  {
    cv::imshow(name, m);
  }
}

cv::Mat merge(const cv::Mat &c0, const cv::Mat &c1)
{
  std::vector<cv::Mat> c;
  c.push_back(c0);
  c.push_back(c1);

  cv::Mat out;
  cv::merge(c, out);

  return out;
}

cv::Mat merge(const cv::Mat &c0, const cv::Mat &c1, const cv::Mat &c2)
{
  std::vector<cv::Mat> c;
  c.push_back(c0);
  c.push_back(c1);
  c.push_back(c2);

  cv::Mat out;
  cv::merge(c, out);

  return out;
}

cv::Mat merge(const cv::Mat &c0, const cv::Mat &c1, const cv::Mat &c2, const cv::Mat &c3)
{
  std::vector<cv::Mat> c;
  c.push_back(c0);
  c.push_back(c1);
  c.push_back(c2);
  c.push_back(c3);

  cv::Mat out;
  cv::merge(c, out);

  return out;
}

void split(const cv::Mat &in, cv::Mat &c0, cv::Mat &c1)
{
  std::vector<cv::Mat> c;
  cv::split(in, c);
  c0 = c[0];
  c1 = c[1];
}

} /* namespace dvo */

std::ostream& operator<<(std::ostream& os, const dvo::PrintMinMax& print)
{
  double min, max;
  cv::minMaxIdx(print.image, &min, &max);
  os << "min: " << min << " max: " << max;

  return os;
}
