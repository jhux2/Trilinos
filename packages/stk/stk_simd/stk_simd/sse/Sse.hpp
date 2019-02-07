// Copyright 2013 Sandia Corporation, Albuquerque, NM.

#ifndef STK_SIMD_SSE_H
#define STK_SIMD_SSE_H

// IWYU pragma: private, include <stk_simd/Simd.hpp>

#include <emmintrin.h>
#include <stdio.h>
#include <cmath>
#include <assert.h>

namespace stk {
namespace simd {
constexpr int ndoubles = 2;
constexpr int nfloats  = 4;
}
}

#include "./SseDouble.hpp"
#include "./SseFloat.hpp"
#include "./SseBool.hpp"
#include "./SseBoolF.hpp"

#include "./SseDoubleOperators.hpp"
#include "./SseDoubleLoadStore.hpp"
#include "./SseDoubleMath.hpp"

#include "./SseFloatOperators.hpp"
#include "./SseFloatLoadStore.hpp"
#include "./SseFloatMath.hpp"

namespace stk {
namespace simd {

inline double reduce_sum(const Double& x) {
  return x[0]+x[1];
}

inline float reduce_sum(const Float& x) {
  return x[0]+x[1]+x[2]+x[3];
}

inline double reduce_max(const Double& x) {
  double max = x[0];
  for (int i=1; i<2; ++i){
    max = max > x[i] ? max : x[i];
  }
  return max;
}

inline float reduce_max(const Float& x) {
  float max = x[0];
  for (int i=1; i<4; ++i){
    max = max > x[i] ? max : x[i];
  }
  return max;
}

inline double reduce_min(const Double& x) {
  double max = x[0];
  for (int i=1; i<2; ++i){
    max = max < x[i] ? max : x[i];
  }
  return max;
}

inline float reduce_min(const Float& x) {
  float max = x[0];
  for (int i=1; i<4; ++i){
    max = max < x[i] ? max : x[i];
  }
  return max;
}

}
}

#endif // STK_SIMD_SSE_H
