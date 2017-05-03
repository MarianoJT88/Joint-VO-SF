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

#ifndef NORMAL_EQUATIONS_HPP_
#define NORMAL_EQUATIONS_HPP_

#include <Eigen/Core>
#include <dvo/sse_ext.hpp>

namespace dvo
{

template<typename T>
struct NormalEquationTraits { };

template<typename Derived>
struct NormalEquationBase
{
  typedef NormalEquationTraits<Derived> Traits;
  typedef typename Traits::JacobianMatrix JacobianMatrix;
  typedef typename Traits::InformationMatrix InformationMatrix;
  typedef typename Traits::ParameterVector ParameterVector;
  typedef typename Traits::ResidualVector ResidualVector;

  typedef typename Traits::MatrixA MatrixA;
  typedef typename Traits::VectorB VectorB;

  inline Derived *derived()
  {
    return static_cast<Derived *>(this);
  }

  void setZero()
  {
    derived()->setZero();
  }

  void get(MatrixA &A, VectorB &b)
  {
    derived()->get(A, b);
  }

  void update(const JacobianMatrix &jacobian, const ResidualVector &residual, const InformationMatrix &information)
  {
    derived()->update(jacobian, residual, information);
  }
};

template<typename ScalarT, int ParameterDim, int ResidualDim>
struct NormalEquation : public NormalEquationBase<NormalEquation<ScalarT, ParameterDim, ResidualDim> >
{
  typedef NormalEquation<ScalarT, ParameterDim, ResidualDim> Self;
  typedef NormalEquationTraits<Self> Traits;

  typename Traits::MatrixA m_a;
  typename Traits::VectorB m_b;

  NormalEquation()
  {
  }

  void setZero()
  {
    m_a.setZero();
    m_b.setZero();
  }

  void get(typename Traits::MatrixA &A, typename Traits::VectorB &b)
  {
    A = m_a;
    b = m_b;
  }

  template<typename JacobianDerivedT, typename ResidualDerivedT, typename InformationDerivedT>
  void update(const Eigen::MatrixBase<JacobianDerivedT> &jacobian, const Eigen::MatrixBase<ResidualDerivedT> &residual, const Eigen::MatrixBase<InformationDerivedT> &information)
  {
    m_a += jacobian.transpose() * information * jacobian;
    m_b -= jacobian.transpose() * information * residual;
  }
};

template<typename ScalarT, int ParameterDim, int ResidualDim>
struct NormalEquationTraits<NormalEquation<ScalarT, ParameterDim, ResidualDim> >
{
  typedef Eigen::Matrix<ScalarT, ResidualDim, ParameterDim> JacobianMatrix;
  typedef Eigen::Matrix<ScalarT, ParameterDim, 1> ParameterVector;
  typedef Eigen::Matrix<ScalarT, ResidualDim, 1> ResidualVector;
  typedef Eigen::Matrix<ScalarT, ResidualDim, ResidualDim> InformationMatrix;

  typedef Eigen::Matrix<ScalarT, ParameterDim, ParameterDim> MatrixA;
  typedef ParameterVector VectorB;
};

template<typename ScalarT, int ParameterDim>
struct NormalEquationTraits<NormalEquation<ScalarT, ParameterDim, 1> >
{
  typedef Eigen::Matrix<ScalarT, 1, ParameterDim> JacobianMatrix;
  typedef Eigen::Matrix<ScalarT, ParameterDim, 1> ParameterVector;
  typedef ScalarT ResidualVector;
  typedef ScalarT InformationMatrix;

  typedef Eigen::Matrix<ScalarT, ParameterDim, ParameterDim> MatrixA;
  typedef ParameterVector VectorB;
};

template<>
struct NormalEquation<float, 6, 1> : public NormalEquationBase<NormalEquation<float, 6, 1> >
{
  typedef NormalEquationTraits<NormalEquation<float, 6, 1> > Traits;

  static const int Size = 24;
  static const int SizeB = 8;

  MEMORY_ALIGN16(float data[Size]);
  MEMORY_ALIGN16(float data_b[SizeB]);

  void setZero();

  void get(Traits::MatrixA &A, Traits::VectorB &b);

  void update(const Traits::JacobianMatrix &jacobian, const Traits::ResidualVector &residual, const Traits::InformationMatrix &information);
};

template<>
struct NormalEquation<float, 6, 2> : public NormalEquationBase<NormalEquation<float, 6, 2> >
{
  typedef NormalEquationTraits<NormalEquation<float, 6, 2> > Traits;

  static const int Size = 24;
  static const int SizeB = 8;

  MEMORY_ALIGN16(float data[Size]);
  MEMORY_ALIGN16(float data_b[SizeB]);

  void setZero();

  void get(Traits::MatrixA &A, Traits::VectorB &b);

  void update(const Traits::JacobianMatrix &jacobian, const Traits::ResidualVector &residual, const Traits::InformationMatrix &information)
  {
      update(jacobian.data(), residual.data(), information.data());
  }

  void update(float const *jacobian, float const *residual, float const *information);

  void add(NormalEquation<float, 6, 2> const &o);
};

} /* namespace dvo */


#endif /* NORMAL_EQUATIONS_HPP_ */
