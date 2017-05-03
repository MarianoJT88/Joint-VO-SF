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


#include "dvo/normal_equation.hpp"
#include "dvo/sse_ext.hpp"
#include <iostream>
#include <stdexcept>

namespace dvo
{

inline void toEigen(const float data[24], Eigen::Matrix<float, 6, 6> &m)
{
  Eigen::Matrix<float, 6, 6> tmp;
  int idx = 0;

  for(int i = 0; i < 6; i += 2)
  {
    for(int j = i; j < 6; j += 2)
    {
      tmp(i  , j  ) = data[idx++];
      tmp(i  , j+1) = data[idx++];
      tmp(i+1, j  ) = data[idx++];
      tmp(i+1, j+1) = data[idx++];
    }
  }

  tmp.selfadjointView<Eigen::Upper>().evalTo(m);
}

void assertNoNaN(const float *data, size_t length, const std::string &message)
{
  for(size_t i = 0; i < length; ++i)
  {
    if(std::isnan(data[i]))
    {
      throw std::runtime_error(message);
    }
  } 
}

void NormalEquation<float, 6, 1>::setZero()
{
  for(int idx = 0; idx < Size; idx++)
    data[idx] = 0.0f;

  for(int idx = 0; idx < SizeB; idx++)
    data_b[idx] = 0.0f;
}

void NormalEquation<float, 6, 1>::get(Traits::MatrixA &A, Traits::VectorB &b)
{
  toEigen(data, A);
  b = Traits::VectorB::MapAligned(data_b); //This might crash ************
}

void NormalEquation<float, 6, 1>::update(const Traits::JacobianMatrix &jacobian, const Traits::ResidualVector &residual, const Traits::InformationMatrix &information)
{
  __m128 s = _mm_set1_ps(information);
  __m128 r = _mm_mul_ps(s, _mm_set1_ps(residual));
  __m128 v1234 = _mm_load_ps(jacobian.data());
  __m128 v56xx = _mm_load_ps(jacobian.data() + 4);

  _mm_store_ps(data_b + 0, _mm_sub_ps(_mm_load_ps(data_b + 0), _mm_mul_ps(v1234, r)));
  _mm_store_ps(data_b + 4, _mm_sub_ps(_mm_load_ps(data_b + 4), _mm_mul_ps(v56xx, r)));

  __m128 v1212 = _mm_movelh_ps(v1234, v1234);
  __m128 v3434 = _mm_movehl_ps(v1234, v1234);
  __m128 v5656 = _mm_movelh_ps(v56xx, v56xx);

  __m128 v1122 = _mm_mul_ps(s, _mm_unpacklo_ps(v1212, v1212));

  _mm_store_ps(data + 0, _mm_add_ps(_mm_load_ps(data + 0), _mm_mul_ps(v1122, v1212)));
  _mm_store_ps(data + 4, _mm_add_ps(_mm_load_ps(data + 4), _mm_mul_ps(v1122, v3434)));
  _mm_store_ps(data + 8, _mm_add_ps(_mm_load_ps(data + 8), _mm_mul_ps(v1122, v5656)));

  __m128 v3344 = _mm_mul_ps(s, _mm_unpacklo_ps(v3434, v3434));

  _mm_store_ps(data + 12, _mm_add_ps(_mm_load_ps(data + 12), _mm_mul_ps(v3344, v3434)));
  _mm_store_ps(data + 16, _mm_add_ps(_mm_load_ps(data + 16), _mm_mul_ps(v3344, v5656)));

  __m128 v5566 = _mm_mul_ps(s, _mm_unpacklo_ps(v5656, v5656));

  _mm_store_ps(data + 20, _mm_add_ps(_mm_load_ps(data + 20), _mm_mul_ps(v5566, v5656)));
}

void NormalEquation<float, 6, 2>::setZero()
{
  for(int idx = 0; idx < Size; idx++)
    data[idx] = 0.0f;

  for(int idx = 0; idx < SizeB; idx++)
    data_b[idx] = 0.0f;
}

void NormalEquation<float, 6, 2>::get(Traits::MatrixA &A, Traits::VectorB &b)
{
  toEigen(data, A);
  b = Traits::VectorB::MapAligned(data_b); //This might crash ************
}

void NormalEquation<float, 6, 2>::add(NormalEquation<float, 6, 2> const &o)
{
    for(int i = 0; i < Size; ++i)
        data[i] += o.data[i];

    for(int i = 0; i < SizeB; ++i)
        data_b[i] += o.data_b[i];
}

void NormalEquation<float, 6, 2>::update(float const *jacobian, float const *residual, float const *information)
{
  //assertNoNaN(data, Size, "NaN in A before update");
  //assertNoNaN(data_b, SizeB, "NaN in b before update");
  __m128 r12xx = _mm_loadu_ps(residual);
  __m128 r1212 = _mm_movelh_ps(r12xx, r12xx);

  /**
   * layout of information:
   *
   *   1 2
   *   3 4
   */
  __m128 alpha1324 = _mm_load_ps(information);     // load first two columns from column major data
  __m128 alpha1313 = _mm_movelh_ps(alpha1324, alpha1324); // first column 2x
  __m128 alpha2424 = _mm_movehl_ps(alpha1324, alpha1324); // second column 2x

  /**
   * layout of jacobian/v:
   *
   *   1a 2a 3a 4a 5a 6a
   *   1b 2b 3b 4b 5b 6b
   */

  /**
   * layout of u = jacobian/v * information:
   *
   *   1a 2a 3a 4a 5a 6a
   *   1b 2b 3b 4b 5b 6b
   */
  __m128 v1a1b2a2b = _mm_load_ps(jacobian + 0); // load first and second column

  __m128 u1a2a1b2b = _mm_hadd_ps(
      _mm_mul_ps(v1a1b2a2b, alpha1313),
      _mm_mul_ps(v1a1b2a2b, alpha2424)
  );

  __m128 u1a1b1a1b = _mm_shuffle_ps(u1a2a1b2b, u1a2a1b2b, _MM_SHUFFLE(2, 0, 2, 0));
  __m128 u2a2b2a2b = _mm_shuffle_ps(u1a2a1b2b, u1a2a1b2b, _MM_SHUFFLE(3, 1, 3, 1));

  // upper left 2x2 block of A matrix in row major format
  __m128 b11 = _mm_hadd_ps(
      _mm_mul_ps(u1a1b1a1b, v1a1b2a2b),
      _mm_mul_ps(u2a2b2a2b, v1a1b2a2b)
  );
  _mm_store_ps(data + 0, _mm_add_ps(_mm_load_ps(data + 0), b11));

  __m128 v3a3b4a4b = _mm_load_ps(jacobian + 4); // load third and fourth column

  // upper center 2x2 block of A matrix in row major format
  __m128 b12 = _mm_hadd_ps(
      _mm_mul_ps(u1a1b1a1b, v3a3b4a4b),
      _mm_mul_ps(u2a2b2a2b, v3a3b4a4b)
  );
  _mm_store_ps(data + 4, _mm_add_ps(_mm_load_ps(data + 4), b12));

  __m128 v5a5b6a6b = _mm_load_ps(jacobian + 8); // load fifth and sixth column

  // upper right 2x2 block of A matrix in row major format
  __m128 b13 = _mm_hadd_ps(
      _mm_mul_ps(u1a1b1a1b, v5a5b6a6b),
      _mm_mul_ps(u2a2b2a2b, v5a5b6a6b)
  );
  _mm_store_ps(data + 8, _mm_add_ps(_mm_load_ps(data + 8), b13));

  __m128 u3a4a3b4b = _mm_hadd_ps(
      _mm_mul_ps(v3a3b4a4b, alpha1313),
      _mm_mul_ps(v3a3b4a4b, alpha2424)
  );

  // update first 4 values of b
  __m128 u1a1b2a2b = _mm_shuffle_ps(u1a2a1b2b, u1a2a1b2b, _MM_SHUFFLE(3, 1, 2, 0));
  __m128 u3a3b4a4b = _mm_shuffle_ps(u3a4a3b4b, u3a4a3b4b, _MM_SHUFFLE(3, 1, 2, 0));

  __m128 b_update1234 = _mm_hadd_ps(_mm_mul_ps(u1a1b2a2b, r1212), _mm_mul_ps(u3a3b4a4b, r1212));
  _mm_store_ps(data_b + 0, _mm_sub_ps(_mm_load_ps(data_b + 0), b_update1234));

  __m128 u3a3b3a3b = _mm_shuffle_ps(u3a4a3b4b, u3a4a3b4b, _MM_SHUFFLE(2, 0, 2, 0));
  __m128 u4a4b4a4b = _mm_shuffle_ps(u3a4a3b4b, u3a4a3b4b, _MM_SHUFFLE(3, 1, 3, 1));

  // center center 2x2 block of A matrix in row major format
  __m128 b22 = _mm_hadd_ps(
      _mm_mul_ps(u3a3b3a3b, v3a3b4a4b),
      _mm_mul_ps(u4a4b4a4b, v3a3b4a4b)
  );
  _mm_store_ps(data + 12, _mm_add_ps(_mm_load_ps(data + 12), b22));

  // center right 2x2 block of A matrix in row major format
  __m128 b23 = _mm_hadd_ps(
      _mm_mul_ps(u3a3b3a3b, v5a5b6a6b),
      _mm_mul_ps(u4a4b4a4b, v5a5b6a6b)
  );
  _mm_store_ps(data + 16, _mm_add_ps(_mm_load_ps(data + 16), b23));

  __m128 u5a6a5b6b = _mm_hadd_ps(
      _mm_mul_ps(v5a5b6a6b, alpha1313),
      _mm_mul_ps(v5a5b6a6b, alpha2424)
  );

  // update last 4 values of b
  __m128 u5a5b6a6b = _mm_shuffle_ps(u5a6a5b6b, u5a6a5b6b, _MM_SHUFFLE(3, 1, 2, 0));
  __m128 b_update56xx = _mm_hadd_ps(_mm_mul_ps(u5a5b6a6b, r1212), _mm_mul_ps(u5a5b6a6b, r1212));
  _mm_store_ps(data_b + 4, _mm_sub_ps(_mm_load_ps(data_b + 4), b_update56xx));

  __m128 u5a5b5a5b = _mm_shuffle_ps(u5a6a5b6b, u5a6a5b6b, _MM_SHUFFLE(2, 0, 2, 0));
  __m128 u6a6b6a6b = _mm_shuffle_ps(u5a6a5b6b, u5a6a5b6b, _MM_SHUFFLE(3, 1, 3, 1));

  // bottom right 2x2 block of A matrix in row major format
  __m128 b33 = _mm_hadd_ps(
      _mm_mul_ps(u5a5b5a5b, v5a5b6a6b),
      _mm_mul_ps(u6a6b6a6b, v5a5b6a6b)
  );
  _mm_store_ps(data + 20, _mm_add_ps(_mm_load_ps(data + 20), b33));

  //assertNoNaN(data, Size, "NaN in A after update");
  //assertNoNaN(data_b, SizeB, "NaN in b after update");
  //for(int i = 0; i < 6; ++i)
  //{
  //  std::cout << data_b[i] <<  " ";
  //}
  //std::cout << std::endl;
  //std::cout << (jacobian.transpose() * information * residual).transpose() << std::endl << std::endl;
  //dump("u1a2a1b2b", u1a2a1b2b);
  //std::cout << (jacobian.transpose() * information).transpose() << std::endl;
  //throw std::exception();
}

} /* namespace dvo */
