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

#ifndef SSE_EXT_HPP_
#define SSE_EXT_HPP_

#ifdef __CDT_PARSER__
  #define __SSE3__ 1
  #define __SSSE3__ 1
  #define __SSE4_1__ 1
  #define __SSE4_2__ 1
  #define __AVX__ 1
#endif
#if (defined __GNUC__)
#include <x86intrin.h>
#elif (defined _MSC_VER)
#include <intrin.h>
#else
#error Check x86intrin.h equivalent for your compiler!
#endif
#include <iostream>

#ifdef __AVX__
#define BLOCK_SIZE 8
#else
#ifdef __SSSE3__
#define BLOCK_SIZE 4
#else
#define BLOCK_SIZE 1
#endif
#endif

#if (defined __GNUC__) || (defined __PGI) || (defined __IBMCPP__) || (defined __ARMCC_VERSION)
  #define MEMORY_ALIGN(n,...) __VA_ARGS__ __attribute__((aligned(n)))
#elif (defined _MSC_VER)
  #define MEMORY_ALIGN(n,...) __declspec(align(n)) __VA_ARGS__
#else
  #error Please tell me what is the equivalent of __attribute__((aligned(n))) for your compiler
#endif

#define MEMORY_ALIGN16(...) MEMORY_ALIGN(16,__VA_ARGS__)
#define MEMORY_ALIGN32(...) MEMORY_ALIGN(32,__VA_ARGS__)

namespace dvo
{

#ifdef __AVX__
static const __m128 ONES = _mm_set1_ps(1.0f);
static const __m256 ONES256 = _mm256_set1_ps(1.0f);
static const __m128 SIGNMASK = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
static const __m256 SIGNMASK256 =  _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

#define _mm_negate_ps(v) (_mm_xor_ps(v, SIGNMASK))
#define _mm256_negate_ps(v) (_mm256_xor_ps(v, SIGNMASK256))

static inline void dump(const char* prefix, __m128 v)
{
  MEMORY_ALIGN16(float data[4]);

  _mm_store_ps(data, v);

  std::cerr << prefix;
  for(int i = 0; i < 4; ++i)
    std::cerr << " " << data[i];
  std::cerr << std::endl;
}

static inline void dump256(const char* prefix, __m256 v)
{
  MEMORY_ALIGN32(float data[8]);

  _mm256_store_ps(data, v);

  std::cerr << prefix;
  for(int i = 0; i < 8; ++i)
    std::cerr << " " << data[i];
  std::cerr << std::endl;
}
#endif
template<int Size>
struct AlignmentSize
{
  static const int value = (Size / 4 + (Size % 4 > 0 ? 1 : 0)) * 4;
};

template<>
struct AlignmentSize<1>
{
  static const int value = 1;
};

} /* namespace dvo */


#endif /* SSE_EXT_HPP_ */
