/*
//@HEADER
// ***********************************************************************
//
//       Ifpack2: Templated Object-Oriented Algebraic Preconditioner Package
//                 Copyright (2009) Sandia Corporation
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
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ***********************************************************************
//@HEADER
*/

/*! \file Ifpack2_UnitTestRILUK.cpp

\brief Ifpack2 parallel unit tests for the RILUK template.
*/


#include <Teuchos_ConfigDefs.hpp>
#include <Ifpack2_ConfigDefs.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_StackedTimer.hpp>
#include <Ifpack2_Version.hpp>
#include <iostream>

#include "Tpetra_Core.hpp"
#include "Tpetra_MatrixIO.hpp"
#include "MatrixMarket_Tpetra.hpp"
#include "TpetraExt_MatrixMatrix.hpp"

#if defined(HAVE_IFPACK2_QD) && !defined(HAVE_TPETRA_EXPLICIT_INSTANTIATION)
#include <qd/dd_real.h>
#endif

#include <Ifpack2_UnitTestHelpers.hpp>
#include <Ifpack2_RILUK.hpp>

// Xpetra / Galeri
#ifdef HAVE_IFPACK2_XPETRA
#include "Xpetra_ConfigDefs.hpp"
#include "Xpetra_DefaultPlatform.hpp"
#include "Xpetra_Parameters.hpp"
#include "Xpetra_MapFactory.hpp"
#include "Xpetra_TpetraMap.hpp"
#include "Xpetra_CrsMatrix.hpp"
#include "Xpetra_TpetraCrsMatrix.hpp"
#include "Galeri_XpetraProblemFactory.hpp"
#include "Galeri_XpetraMatrixTypes.hpp"
#include "Galeri_XpetraParameters.hpp"
#include "Galeri_XpetraUtils.hpp"
#include "Galeri_XpetraMaps.hpp"
#endif // HAVE_IFPACK2_XPETRA

namespace {
using Tpetra::global_size_t;
typedef tif_utest::Node Node;

struct IlukImplTypeDetails {
  enum Enum { Serial, KSPILUK };
};
// Single-process unit tests for RILUK are located in the file Ifpack2_UnitTestSerialRILUK.cpp.

template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal>
static Teuchos::RCP<Ifpack2::RILUK<Tpetra::RowMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > > setupTest (const IlukImplTypeDetails::Enum ilukimplType)
{
  // Test that ILU(k) can be done on a parallel sparse matrix with noncontiguous row map.
  // See bug #6033.
  typedef Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>   crs_matrix_type;
  typedef Tpetra::RowMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>   row_matrix_type;
  typedef Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node>                map_type;

  using Teuchos::RCP;
  using Teuchos::rcp;

  RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();

  int nx = 2;
  size_t numElementsPerProc = nx*nx;

  //create noncontiguous row map.
  Teuchos::Array<GlobalOrdinal> eltList;
  GlobalOrdinal offset = 2*numElementsPerProc;
  for (size_t i=0; i<numElementsPerProc; ++i)
    eltList.push_back(comm->getRank()*offset + 2*i);
  const global_size_t INVALID = Teuchos::OrdinalTraits<global_size_t>::invalid();
  RCP<const map_type> map = Teuchos::rcp(new map_type(INVALID, eltList(), 0, comm));

  //Construct a nondiagonal matrix.  It's not tridiagonal because of processor boundaries.
  const Scalar one = Teuchos::ScalarTraits<Scalar>::one();
  const Scalar two = one + one;
  RCP<crs_matrix_type > A = Teuchos::rcp(new crs_matrix_type(map, 3));
  Teuchos::Array<GlobalOrdinal> col(3);
  Teuchos::Array<Scalar>        val(3);
  size_t numLocalElts = map->getLocalNumElements();
  for(LocalOrdinal l_row = 0; (size_t) l_row < numLocalElts; l_row++) {
    GlobalOrdinal g_row = map->getGlobalElement(l_row);
    size_t i=0;
    col[i] = g_row;
    val[i++] = two;
    if (l_row>0)                      {col[i] = map->getGlobalElement(l_row-1); val[i++] = -one;}
    if ((size_t)l_row<numLocalElts-1) {col[i] = map->getGlobalElement(l_row+1); val[i++] = -one;}
    A->insertGlobalValues(g_row, col(0,i), val(0,i));
  }
  A->fillComplete();

  RCP<const crs_matrix_type> constA = A;
  auto prec = rcp(new Ifpack2::RILUK<row_matrix_type>(constA));

  Teuchos::ParameterList params;
  GlobalOrdinal lof=1;
  params.set("fact: iluk level-of-fill", lof);
  params.set("fact: iluk level-of-overlap", 0);
  if (ilukimplType == IlukImplTypeDetails::KSPILUK)
    params.set("fact: type", "KSPILUK");
  prec->setParameters(params);
  return prec;
}

template <typename Scalar, typename LocalOrdinal, typename GlobalOrdinal>
static Teuchos::RCP<Ifpack2::RILUK<Tpetra::RowMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> > > setupTest2(const IlukImplTypeDetails::Enum ilukimplType)
{
  typedef Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>   crs_matrix_type;
  typedef Tpetra::RowMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>   row_matrix_type;
  typedef Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node>                map_type;

  using Teuchos::RCP;
  using Teuchos::rcp;

  RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();

  ////////////////////////////////////////
  typedef Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>   crs_matrix_type;
  // Generate the matrix using Galeri.  Galeri wraps it in an Xpetra
  // matrix, so after it finishes, ask it for the Tpetra matrix.
  Teuchos::CommandLineProcessor clp;
  //GlobalOrdinal nx = 173, ny=173, nz=1; //30k rows
  //GlobalOrdinal nx = 273, ny=273, nz=1; //30k rows
  //Galeri::Xpetra::Parameters<GlobalOrdinal> GaleriParameters (clp, nx, ny, nz, "Laplace2D");
  //GlobalOrdinal nx = 42, ny=42, nz=42; //75k rows
  GlobalOrdinal nx = 46, ny=46, nz=46; //100k rows
  //GlobalOrdinal nx = 5, ny=5, nz=5;
  Galeri::Xpetra::Parameters<GlobalOrdinal> GaleriParameters (clp, nx, ny, nz, "Laplace3D");
  Xpetra::Parameters xpetraParameters (clp);
  Teuchos::ParameterList GaleriList = GaleriParameters.GetParameterList ();

  typedef Xpetra::Map<LocalOrdinal,GlobalOrdinal,Node>                XMapType;
  RCP<XMapType> xmap =
    Galeri::Xpetra::CreateMap<LocalOrdinal, GlobalOrdinal, Node> (xpetraParameters.GetLib (),
                                             "Cartesian3D", comm, GaleriList);
  typedef Xpetra::TpetraCrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node> XCrsType;
  typedef Xpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>     XMVectorType;
  RCP<Galeri::Xpetra::Problem<XMapType,XCrsType,XMVectorType> > Pr =
    Galeri::Xpetra::BuildProblem<Scalar,LocalOrdinal,GlobalOrdinal,XMapType,XCrsType,XMVectorType> (std::string("Laplace3D"),
                                                                           xmap, GaleriList);

  RCP<XCrsType> XA = Pr->BuildMatrix ();
  RCP<Tpetra::CrsMatrix<Scalar,LocalOrdinal,GlobalOrdinal,Node>> A = XA->getTpetra_CrsMatrixNonConst();
  //TEST_INEQUALITY(A, Teuchos::null);
  ////////////////////////////////////////
  RCP<const crs_matrix_type> constA = A;
  auto prec = rcp(new Ifpack2::RILUK<row_matrix_type>(constA));

  Teuchos::ParameterList params;
  GlobalOrdinal lof=1;
  params.set("fact: iluk level-of-fill", lof);
  params.set("fact: iluk level-of-overlap", 0);
  if (ilukimplType == IlukImplTypeDetails::KSPILUK)
    params.set("fact: type", "KSPILUK");
  prec->setParameters(params);
  return prec;
}

TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL(Ifpack2RILUK, Parallel, Scalar, LocalOrdinal, GlobalOrdinal)
{
  std::string version = Ifpack2::Version();
  out << "Ifpack2::Version(): " << version << std::endl;
  {
    auto prec = setupTest<Scalar, LocalOrdinal, GlobalOrdinal>(IlukImplTypeDetails::Serial);
    prec->initialize();
    prec->compute();
  }
  {
    auto prec = setupTest<Scalar, LocalOrdinal, GlobalOrdinal>(IlukImplTypeDetails::KSPILUK);
    prec->initialize();
    prec->compute();
  }
}

TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL(Ifpack2RILUK, ParallelReuse, Scalar, LocalOrdinal, GlobalOrdinal)
{
  Teuchos::RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm ();
#if 1
  {
    out << "IlukImplTypeDetails::Serial" << std::endl;
    Teuchos::RCP<Teuchos::StackedTimer> stacked_timer;
    stacked_timer = Teuchos::rcp(new Teuchos::StackedTimer("Ifpack2RILUK::ParallelReuse (SERIAL)"));
    Teuchos::TimeMonitor::setStackedTimer(stacked_timer);
    auto prec = setupTest2<Scalar, LocalOrdinal, GlobalOrdinal>(IlukImplTypeDetails::Serial);
    prec->initialize();
    {
    Teuchos::Time timer("RILUK::compute() 0");
    Teuchos::TimeMonitor timeMon(timer);
    prec->compute();
    }
    // Pretend we've updated some of the numbers in the matrix, but not its structure.
    {
    Teuchos::Time timer("RILUK::compute() 1");
    Teuchos::TimeMonitor timeMon(timer);
    prec->compute();
    }
    {
    Teuchos::Time timer("RILUK::compute() 2");
    Teuchos::TimeMonitor timeMon(timer);
    prec->compute();
    }
    stacked_timer->stopBaseTimer();
    Teuchos::StackedTimer::OutputOptions options;
    options.output_fraction = options.output_histogram = options.output_minmax = true;
    stacked_timer->report(out, comm, options);
  }
#endif
  {
    out << "IlukImplTypeDetails::KSPILUK" << std::endl;
    Teuchos::RCP<Teuchos::StackedTimer> stacked_timer;
    stacked_timer = Teuchos::rcp(new Teuchos::StackedTimer("Ifpack2RILUK::ParallelReuse (KSPILUK)"));
    Teuchos::TimeMonitor::setStackedTimer(stacked_timer);
    auto prec = setupTest2<Scalar, LocalOrdinal, GlobalOrdinal>(IlukImplTypeDetails::KSPILUK);
    prec->initialize();
    {
    Teuchos::Time timer("RILUK::compute() 0");
    Teuchos::TimeMonitor timeMon(timer);
    prec->compute();
    }
    // Pretend we've updated some of the numbers in the matrix, but not its structure.
    {
    Teuchos::Time timer("RILUK::compute() 1");
    Teuchos::TimeMonitor timeMon(timer);
    prec->compute();
    }
    {
    Teuchos::Time timer("RILUK::compute() 2");
    Teuchos::TimeMonitor timeMon(timer);
    prec->compute();
    }
    stacked_timer->stopBaseTimer();
    Teuchos::StackedTimer::OutputOptions options;
    options.output_fraction = options.output_histogram = options.output_minmax = true;
    stacked_timer->report(out, comm, options);
  }
}

#define UNIT_TEST_GROUP_SC_LO_GO( SC, LO, GO ) \
  TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( Ifpack2RILUK, Parallel, SC, LO, GO ) \
  TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( Ifpack2RILUK, ParallelReuse, SC, LO, GO )

#include "Ifpack2_ETIHelperMacros.h"

IFPACK2_ETI_MANGLING_TYPEDEFS()

// Test all enabled combinations of Scalar (SC), LocalOrdinal (LO),
// and GlobalOrdinal (GO) types.

IFPACK2_INSTANTIATE_SLG( UNIT_TEST_GROUP_SC_LO_GO )

}//namespace <anonymous>
