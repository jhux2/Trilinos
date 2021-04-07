// @HEADER
// ************************************************************************
//
//               Rapid Optimization Library (ROL) Package
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact lead developers:
//              Drew Kouri   (dpkouri@sandia.gov) and
//              Denis Ridzal (dridzal@sandia.gov)
//
// ************************************************************************
// @HEADER

#ifndef ROL_TYPEU_ALGORITHM_H
#define ROL_TYPEU_ALGORITHM_H

#include "ROL_CombinedStatusTest.hpp"
#include "ROL_Objective.hpp"
#include "ROL_Constraint.hpp"
#include "ROL_Problem.hpp"

/** \class ROL::TypeU::Algorithm
    \brief Provides an interface to run unconstrained optimization algorithms.
*/

namespace ROL {
namespace TypeU { 

template<typename Real>
struct AlgorithmState : public ROL::AlgorithmState<Real> {
  Real searchSize;
  Ptr<Vector<Real>> stepVec;
  Ptr<Vector<Real>> gradientVec;

  AlgorithmState()
    : searchSize(1),
      stepVec(nullPtr),
      gradientVec(nullPtr) {}

  void reset() {
    ROL::AlgorithmState<Real>::reset();
    searchSize = static_cast<Real>(1);
    if (stepVec != nullPtr) {
      stepVec->zero();
    }
    if (gradientVec != nullPtr) {
      gradientVec->zero();
    }
  }
};

template<typename Real>
class Algorithm {
protected:
  const Ptr<CombinedStatusTest<Real>> status_;
  const Ptr<AlgorithmState<Real>>   state_;

  void initialize(const Vector<Real> &x, const Vector<Real> &g); 

public:

  virtual ~Algorithm() {}

  /** \brief Constructor, given a step and a status test.
  */
  Algorithm();

  void setStatusTest(const Ptr<StatusTest<Real>> &status,
                     bool combineStatus = false);

  /** \brief Run algorithm on unconstrained problems (Type-U).
             This is the primary Type-U interface.
  */
  virtual void run( Problem<Real> &problem,
                    std::ostream &outStream = std::cout );

  /** \brief Run algorithm on unconstrained problems (Type-U).
             This is the primary Type-U interface.
  */
  virtual void run( Vector<Real>    &x,
                    Objective<Real> &obj,
                    std::ostream    &outStream = std::cout );

  /** \brief Run algorithm on unconstrained problems with explicit
             linear equality constraints (Type-U).
	     This is the primary Type-U with explicit linear equality
             constraints interface.
  */
  virtual void run( Vector<Real>     &x,
                    Objective<Real>  &obj,
                    Constraint<Real> &linear_con,
                    Vector<Real>     &linear_mul,
                    std::ostream     &outStream = std::cout );

  /** \brief Run algorithm on unconstrained problems with explicit
             linear equality constraints (Type-U).
             This general interface supports the use of dual optimization vector spaces,
             where the user does not define the dual() method.
  */
  virtual void run( Vector<Real>       &x,
                    const Vector<Real> &g,
                    Objective<Real>    &obj,
                    Constraint<Real>   &linear_con,
                    Vector<Real>       &linear_mul,
		    const Vector<Real> &linear_c,
                    std::ostream       &outStream = std::cout );

  /** \brief Run algorithm on unconstrained problems (Type-U).
             This general interface supports the use of dual optimization vector spaces,
             where the user does not define the dual() method.
  */
  virtual void run( Vector<Real>       &x,
                    const Vector<Real> &g, 
                    Objective<Real>    &obj,
                    std::ostream       &outStream = std::cout) = 0;

  /** \brief Print iterate header.
  */
  virtual void writeHeader( std::ostream& os ) const;

  /** \brief Print step name.
  */
  virtual void writeName( std::ostream& os ) const;

  /** \brief Print iterate status.
  */
  virtual void writeOutput( std::ostream& os, bool write_header = false ) const;

  virtual void writeExitStatus( std::ostream& os ) const;

  //Ptr<const AlgorithmState<Real>>& getState() const;
  Ptr<const AlgorithmState<Real>> getState() const;

  void reset();

}; // class ROL::TypeU::Algorithm
} // namespace TypeU
} // namespace ROL

#include "ROL_TypeU_Algorithm_Def.hpp"

#endif
