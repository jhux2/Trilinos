/*@HEADER
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
// Questions? Contact ifpack2-developers@software.sandia.gov
//
// ***********************************************************************
//@HEADER
*/

#ifndef IFPACK2_OVERLAPPINGROWMATRIX_DEF_HPP
#define IFPACK2_OVERLAPPINGROWMATRIX_DEF_HPP

#include <sstream>

#include "Kokkos_Core.hpp"
#include "std_algorithms/Kokkos_ModifyingSequenceOperations.hpp"
#include "Kokkos_Sort.hpp"
#include <Ifpack2_Details_OverlappingRowGraph.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Import.hpp>
#include "Tpetra_Map.hpp"
#include <Teuchos_CommHelpers.hpp>

namespace Ifpack2 {

template<class MatrixType>
OverlappingRowMatrix<MatrixType>::
OverlappingRowMatrix (const Teuchos::RCP<const row_matrix_type>& A,
                      const int overlapLevel) :
  A_ (Teuchos::rcp_dynamic_cast<const crs_matrix_type> (A, true)),
  OverlapLevel_ (overlapLevel)
{
  //int mypid = A_->getComm()->getRank();
  //printf("%d: starting OverlappingRowMatrix ctor\n",mypid); fflush(stdout);
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::outArg;
  using Teuchos::REDUCE_SUM;
  using Teuchos::reduceAll;
  typedef Tpetra::global_size_t GST;
  typedef Tpetra::CrsGraph<local_ordinal_type,
                           global_ordinal_type, node_type> crs_graph_type;
  TEUCHOS_TEST_FOR_EXCEPTION(
    OverlapLevel_ <= 0, std::runtime_error,
    "Ifpack2::OverlappingRowMatrix: OverlapLevel must be > 0.");
  TEUCHOS_TEST_FOR_EXCEPTION
    (A_.is_null (), std::runtime_error,
     "Ifpack2::OverlappingRowMatrix: The input matrix must be a "
     "Tpetra::CrsMatrix with the same scalar_type, local_ordinal_type, "
     "global_ordinal_type, and device_type typedefs as MatrixType.");
  TEUCHOS_TEST_FOR_EXCEPTION(
    A_->getComm()->getSize() == 1, std::runtime_error,
    "Ifpack2::OverlappingRowMatrix: Matrix must be "
    "distributed over more than one MPI process.");

  RCP<const crs_graph_type> A_crsGraph = A_->getCrsGraph ();
  const size_t numMyRowsA = A_->getLocalNumRows ();
  const global_ordinal_type global_invalid =
    Teuchos::OrdinalTraits<global_ordinal_type>::invalid ();

  Kokkos::View<global_ordinal_type*, device_type> ExtElements("all halo rows", 0);
  typename Kokkos::View<global_ordinal_type*, device_type>::HostMirror ExtElements_h = Kokkos::create_mirror_view(ExtElements);
  RCP<map_type>        TmpMap;
  RCP<crs_graph_type>  TmpGraph;
  RCP<import_type>     TmpImporter;
  RCP<const map_type>  RowMap, ColMap;
  size_t globalCount_h = 0; //count of GIDs in all halo level-sets
  size_t globalNumExtElements = 0;

  //printf("%d: starting big import loop\n",mypid); fflush(stdout);
  // The big import loop
  for (int overlap = 0 ; overlap < OverlapLevel_ ; ++overlap) {
    //printf("%d: starting import loop preliminaries\n",mypid); fflush(stdout);
    //FIXME
    //ExtHaloStarts_[overlap] = (size_t) ExtElements.size();

    // Get the current maps
    if (overlap == 0) {
      RowMap = A_->getRowMap ();
      ColMap = A_->getColMap ();
    }
    else {
      RowMap = TmpGraph->getRowMap ();
      ColMap = TmpGraph->getColMap ();
    }
    //printf("%d: finished setting maps\n",mypid); fflush(stdout);

    const size_t size = ColMap->getLocalNumElements () - RowMap->getLocalNumElements ();
    Kokkos::View<global_ordinal_type*, device_type> mylist("current halo GIDs",size);
    typename Kokkos::View<global_ordinal_type*, device_type>::HostMirror mylist_h = Kokkos::create_mirror_view(mylist);
    size_t count_h = 0; //count of GIDs in current halo level-set
    Kokkos::View<size_t,device_type> count("count of GIDS in current halo");
    //printf("%d: finished allocating mylist\n",mypid); fflush(stdout);

    size_t maxNumExtElements = ColMap()->getLocalNumElements() - RowMap()->getLocalNumElements();
    //must account for element counts from previous import rounds
    globalNumExtElements += maxNumExtElements;
    Kokkos::resize(ExtElements,globalNumExtElements);
    Kokkos::resize(ExtElements_h,globalNumExtElements);
    //printf("%d: finished resizing ExtElements\n",mypid); fflush(stdout);

    // identify the set of rows that are in ColMap but not in RowMap
    //printf("%d: populating ExtElements\n",mypid); fflush(stdout);
    for (local_ordinal_type i = 0 ; (size_t) i < ColMap->getLocalNumElements() ; ++i) {
      const global_ordinal_type GID = ColMap->getGlobalElement (i);
      if (A_->getRowMap ()->getLocalElement (GID) == global_invalid) {
        ExtElements_h(globalCount_h) = GID;
        globalCount_h++;
        mylist_h(count_h) = GID;
        count_h++;
      }
    }
    namespace KE = Kokkos::Experimental;
    if (overlap > 0) {
      //printf("%d: sort/unique\n",mypid); fflush(stdout);
      std::sort(KE::begin(ExtElements_h),KE::end(ExtElements_h));
      std::unique(KE::begin(ExtElements_h),KE::end(ExtElements_h));
      //Kokkos::sort(ExtElements_h,0,globalCount_h); //TODO FIXME
      //KE::unique(execution_space(),KE::begin(ExtElements_h),KE::end(ExtElements_h));
    }

    // On last import round, TmpMap, TmpGraph, and TmpImporter are unneeded,
    // so don't build them.
    if (overlap + 1 < OverlapLevel_) {
      //map consisting of GIDs that are in the current halo level-set
      Kokkos::deep_copy(mylist,mylist_h);
      TmpMap = rcp (new map_type (global_invalid, mylist,
                                  Teuchos::OrdinalTraits<global_ordinal_type>::zero (),
                                  A_->getComm ()));
      //graph whose rows are the current halo level-set to import
      TmpGraph = rcp (new crs_graph_type (TmpMap, 0));
      TmpImporter = rcp (new import_type (A_->getRowMap (), TmpMap));

      //import from original matrix graph to current halo level-set graph
      TmpGraph->doImport (*A_crsGraph, *TmpImporter, Tpetra::INSERT);
      TmpGraph->fillComplete (A_->getDomainMap (), TmpMap);
    }
  } // end overlap loop

  //printf("%d: finished big import loop\n",mypid); fflush(stdout);
  A_->getComm()->barrier();

  //record index of start of last halo level-set
  //FIXME
  //ExtHaloStarts_[OverlapLevel_] = (size_t) ExtElements.size();

  // build the map containing all the nodes (original matrix + extended matrix)
  Kokkos::deep_copy(ExtElements,ExtElements_h);
  Kokkos::View<global_ordinal_type*, device_type> mylist("local plus halo GIDs", numMyRowsA + ExtElements.size ());
  typename Kokkos::View<global_ordinal_type*, device_type>::HostMirror mylist_h = Kokkos::create_mirror_view(mylist);
  for (local_ordinal_type i = 0; (size_t)i < numMyRowsA; ++i) {
    mylist_h(i) = A_->getRowMap ()->getGlobalElement (i);
  }
  for (local_ordinal_type i = 0; (size_t)i < ExtElements.size (); ++i) {
    mylist_h(i + numMyRowsA) = ExtElements_h(i);
  }
  Kokkos::deep_copy(mylist,mylist_h);

  RowMap_ = rcp (new map_type (global_invalid, mylist_h(), /*mylist(),*/
                               Teuchos::OrdinalTraits<global_ordinal_type>::zero (),
                               A_->getComm ()));
  Importer_ = rcp (new import_type (A_->getRowMap (), RowMap_));
  ColMap_ = RowMap_;

  // now build the map corresponding to all the external nodes
  // (with respect to A().RowMatrixRowMap().
  ExtMap_ = rcp (new map_type (global_invalid, ExtElements_h(), /*ExtElements (),*/
                               Teuchos::OrdinalTraits<global_ordinal_type>::zero (),
                               A_->getComm ()));
  ExtImporter_ = rcp (new import_type (A_->getRowMap (), ExtMap_));

  //printf("%d: finished building ExtImporter_\n",mypid); fflush(stdout);

  {
    RCP<crs_matrix_type> ExtMatrix_nc =
      rcp (new crs_matrix_type (ExtMap_, ColMap_, 0));
    ExtMatrix_nc->doImport (*A_, *ExtImporter_, Tpetra::INSERT);
    ExtMatrix_nc->fillComplete (A_->getDomainMap (), RowMap_);
    ExtMatrix_ = ExtMatrix_nc; // we only need the const version after here
  }

  // fix indices for overlapping matrix
  const size_t numMyRowsB = ExtMatrix_->getLocalNumRows ();

  GST NumMyNonzeros_tmp = A_->getLocalNumEntries () + ExtMatrix_->getLocalNumEntries ();
  GST NumMyRows_tmp = numMyRowsA + numMyRowsB;
  {
    GST inArray[2], outArray[2];
    inArray[0] = NumMyNonzeros_tmp;
    inArray[1] = NumMyRows_tmp;
    outArray[0] = 0;
    outArray[1] = 0;
    reduceAll<int, GST> (* (A_->getComm ()), REDUCE_SUM, 2, inArray, outArray);
    NumGlobalNonzeros_ = outArray[0];
    NumGlobalRows_ = outArray[1];
  }

  MaxNumEntries_ = A_->getLocalMaxNumRowEntries ();
  if (MaxNumEntries_ < ExtMatrix_->getLocalMaxNumRowEntries ()) {
    MaxNumEntries_ = ExtMatrix_->getLocalMaxNumRowEntries ();
  }

  //printf("%d: starting to create RowGraph\n",mypid); fflush(stdout);
  // Create the graph (returned by getGraph()).
  typedef Details::OverlappingRowGraph<row_graph_type> row_graph_impl_type;
  RCP<row_graph_impl_type> graph =
    rcp (new row_graph_impl_type (A_->getGraph (),
                                  ExtMatrix_->getGraph (),
                                  RowMap_,
                                  ColMap_,
                                  NumGlobalRows_,
                                  NumGlobalRows_, // # global cols == # global rows
                                  NumGlobalNonzeros_,
                                  MaxNumEntries_,
                                  Importer_,
                                  ExtImporter_));
  graph_ = Teuchos::rcp_const_cast<const row_graph_type>
    (Teuchos::rcp_implicit_cast<row_graph_type> (graph));
  // Resize temp arrays
  Kokkos::resize(Indices_,MaxNumEntries_);
  Kokkos::resize(Values_,MaxNumEntries_);
  //printf("%d: finished OverlappingRowMatrix ctor\n",mypid); fflush(stdout);
}


template<class MatrixType>
Teuchos::RCP<const Teuchos::Comm<int> >
OverlappingRowMatrix<MatrixType>::getComm () const
{
  return A_->getComm ();
}


template<class MatrixType>
Teuchos::RCP<const Tpetra::Map<typename MatrixType::local_ordinal_type, typename MatrixType::global_ordinal_type, typename MatrixType::node_type> >
OverlappingRowMatrix<MatrixType>::getRowMap () const
{
  // FIXME (mfh 12 July 2013) Is this really the right Map to return?
  return RowMap_;
}


template<class MatrixType>
Teuchos::RCP<const Tpetra::Map<typename MatrixType::local_ordinal_type, typename MatrixType::global_ordinal_type, typename MatrixType::node_type> >
OverlappingRowMatrix<MatrixType>::getColMap () const
{
  // FIXME (mfh 12 July 2013) Is this really the right Map to return?
  return ColMap_;
}


template<class MatrixType>
Teuchos::RCP<const Tpetra::Map<typename MatrixType::local_ordinal_type, typename MatrixType::global_ordinal_type, typename MatrixType::node_type> >
OverlappingRowMatrix<MatrixType>::getDomainMap () const
{
  // The original matrix's domain map is irrelevant; we want the map associated
  // with the overlap. This can then be used by LocalFilter, for example, while
  // letting LocalFilter still filter based on domain and range maps (instead of
  // column and row maps).
  // FIXME Ideally, this would be the same map but restricted to a local
  // communicator. If replaceCommWithSubset were free, that would be the way to
  // go. That would require a new Map ctor. For now, we'll stick with ColMap_'s
  // global communicator.
  return ColMap_;
}


template<class MatrixType>
Teuchos::RCP<const Tpetra::Map<typename MatrixType::local_ordinal_type, typename MatrixType::global_ordinal_type, typename MatrixType::node_type> >
OverlappingRowMatrix<MatrixType>::getRangeMap () const
{
  return RowMap_;
}


template<class MatrixType>
Teuchos::RCP<const Tpetra::RowGraph<typename MatrixType::local_ordinal_type, typename MatrixType::global_ordinal_type, typename MatrixType::node_type> >
OverlappingRowMatrix<MatrixType>::getGraph() const
{
  return graph_;
}


template<class MatrixType>
global_size_t OverlappingRowMatrix<MatrixType>::getGlobalNumRows() const
{
  return NumGlobalRows_;
}


template<class MatrixType>
global_size_t OverlappingRowMatrix<MatrixType>::getGlobalNumCols() const
{
  return NumGlobalRows_;
}


template<class MatrixType>
size_t OverlappingRowMatrix<MatrixType>::getLocalNumRows() const
{
  return A_->getLocalNumRows () + ExtMatrix_->getLocalNumRows ();
}


template<class MatrixType>
size_t OverlappingRowMatrix<MatrixType>::getLocalNumCols() const
{
  return this->getLocalNumRows ();
}


template<class MatrixType>
typename MatrixType::global_ordinal_type
OverlappingRowMatrix<MatrixType>::getIndexBase () const
{
  return A_->getIndexBase();
}


template<class MatrixType>
Tpetra::global_size_t OverlappingRowMatrix<MatrixType>::getGlobalNumEntries() const
{
  return NumGlobalNonzeros_;
}


template<class MatrixType>
size_t OverlappingRowMatrix<MatrixType>::getLocalNumEntries() const
{
  return A_->getLocalNumEntries () + ExtMatrix_->getLocalNumEntries ();
}


template<class MatrixType>
size_t
OverlappingRowMatrix<MatrixType>::
getNumEntriesInGlobalRow (global_ordinal_type globalRow) const
{
  const local_ordinal_type localRow = RowMap_->getLocalElement (globalRow);
  if (localRow == Teuchos::OrdinalTraits<local_ordinal_type>::invalid ()) {
    return Teuchos::OrdinalTraits<size_t>::invalid();
  } else {
    return getNumEntriesInLocalRow (localRow);
  }
}


template<class MatrixType>
size_t
OverlappingRowMatrix<MatrixType>::
getNumEntriesInLocalRow (local_ordinal_type localRow) const
{
  using Teuchos::as;
  const size_t numMyRowsA = A_->getLocalNumRows ();
  if (as<size_t> (localRow) < numMyRowsA) {
    return A_->getNumEntriesInLocalRow (localRow);
  } else {
    return ExtMatrix_->getNumEntriesInLocalRow (as<local_ordinal_type> (localRow - numMyRowsA));
  }
}


template<class MatrixType>
size_t OverlappingRowMatrix<MatrixType>::getGlobalMaxNumRowEntries() const
{
  throw std::runtime_error("Ifpack2::OverlappingRowMatrix::getGlobalMaxNumRowEntries() not supported.");
}


template<class MatrixType>
size_t OverlappingRowMatrix<MatrixType>::getLocalMaxNumRowEntries() const
{
  return MaxNumEntries_;
}


template<class MatrixType>
bool OverlappingRowMatrix<MatrixType>::hasColMap() const
{
  return true;
}


template<class MatrixType>
bool OverlappingRowMatrix<MatrixType>::isLocallyIndexed() const
{
  return true;
}


template<class MatrixType>
bool OverlappingRowMatrix<MatrixType>::isGloballyIndexed() const
{
  return false;
}


template<class MatrixType>
bool OverlappingRowMatrix<MatrixType>::isFillComplete() const
{
  return true;
}

template<class MatrixType>
void
OverlappingRowMatrix<MatrixType>::
getGlobalRowView (global_ordinal_type GlobalRow,
                  global_inds_host_view_type &indices,
                  values_host_view_type &values) const {
  const local_ordinal_type LocalRow = RowMap_->getLocalElement (GlobalRow);
  if (LocalRow == Teuchos::OrdinalTraits<local_ordinal_type>::invalid())  {
    indices = global_inds_host_view_type();
    values = values_host_view_type();
  } else {
    if (Teuchos::as<size_t> (LocalRow) < A_->getLocalNumRows ()) {
      A_->getGlobalRowView (GlobalRow, indices, values);
    } else {
      ExtMatrix_->getGlobalRowView (GlobalRow, indices, values);
    }
  }
}

template<class MatrixType>
void
OverlappingRowMatrix<MatrixType>::
  getLocalRowView (local_ordinal_type LocalRow,
                   local_inds_host_view_type & indices,
                   values_host_view_type & values) const {
  using Teuchos::as;
  const size_t numMyRowsA = A_->getLocalNumRows ();
  if (as<size_t> (LocalRow) < numMyRowsA) {
    A_->getLocalRowView (LocalRow, indices, values);
  } else {
    ExtMatrix_->getLocalRowView (LocalRow - as<local_ordinal_type> (numMyRowsA),
                                 indices, values);
  }

}

template<class MatrixType>
void
OverlappingRowMatrix<MatrixType>::
getGlobalRowCopy (global_ordinal_type GlobalRow,
                  nonconst_global_inds_host_view_type &Indices,
                  nonconst_values_host_view_type &Values,
                  size_t& NumEntries) const {
  throw std::runtime_error("Ifpack2::OverlappingRowMatrix::getGlobalRowCopy() not supported.");
}

template<class MatrixType>
void
OverlappingRowMatrix<MatrixType>::
  getLocalRowCopy (local_ordinal_type LocalRow,
                   nonconst_local_inds_host_view_type &Indices,
                   nonconst_values_host_view_type &Values,
                   size_t& NumEntries) const {
  throw std::runtime_error("Ifpack2::OverlappingRowMatrix::getLocalRowCopy() not supported.");
}


#ifdef TPETRA_ENABLE_DEPRECATED_CODE
template<class MatrixType>
void
OverlappingRowMatrix<MatrixType>::
getGlobalRowCopy (global_ordinal_type GlobalRow,
                  const Teuchos::ArrayView<global_ordinal_type> &Indices,
                  const Teuchos::ArrayView<scalar_type> &Values,
                  size_t &NumEntries) const {
  throw std::runtime_error("Ifpack2::OverlappingRowMatrix::getGlobalRowCopy() with Teuchos Arrays not supported.");
}
template<class MatrixType>
void
OverlappingRowMatrix<MatrixType>::
getLocalRowCopy (local_ordinal_type LocalRow,
                 const Teuchos::ArrayView<local_ordinal_type> &Indices,
                 const Teuchos::ArrayView<scalar_type> &Values,
                 size_t &NumEntries) const {
  throw std::runtime_error("Ifpack2::OverlappingRowMatrix::getLocalRowCopy() with Teuchos Arrays not supported.");
}

template<class MatrixType>
void
OverlappingRowMatrix<MatrixType>::
getGlobalRowView (global_ordinal_type GlobalRow,
                  Teuchos::ArrayView<const global_ordinal_type> &indices,
                  Teuchos::ArrayView<const scalar_type> &values) const {
  throw std::runtime_error("Ifpack2::OverlappingRowMatrix::getGlobalRowView() with Teuchos Arrays not supported.");
}

template<class MatrixType>
void
OverlappingRowMatrix<MatrixType>::
getLocalRowView (local_ordinal_type LocalRow,
                 Teuchos::ArrayView<const local_ordinal_type> &indices,
                 Teuchos::ArrayView<const scalar_type> &values) const {
  throw std::runtime_error("Ifpack2::OverlappingRowMatrix::getLocalRowView() with Teuchos Arrays not supported.");
}
#endif

template<class MatrixType>
void
OverlappingRowMatrix<MatrixType>::
getLocalDiagCopy (Tpetra::Vector<scalar_type,local_ordinal_type,global_ordinal_type,node_type>& diag) const
{
  using Teuchos::Array;

  //extract diagonal of original matrix
  vector_type baseDiag(A_->getRowMap());         // diagonal of original matrix A_
  A_->getLocalDiagCopy(baseDiag);
  Array<scalar_type> baseDiagVals(baseDiag.getLocalLength());
  baseDiag.get1dCopy(baseDiagVals());
  //extra diagonal of ghost matrix
  vector_type extDiag(ExtMatrix_->getRowMap());
  ExtMatrix_->getLocalDiagCopy(extDiag);
  Array<scalar_type> extDiagVals(extDiag.getLocalLength());
  extDiag.get1dCopy(extDiagVals());

  Teuchos::ArrayRCP<scalar_type> allDiagVals = diag.getDataNonConst();
  if (allDiagVals.size() != baseDiagVals.size() + extDiagVals.size()) {
    std::ostringstream errStr;
    errStr << "Ifpack2::OverlappingRowMatrix::getLocalDiagCopy : Mismatch in diagonal lengths, "
           << allDiagVals.size() << " != " << baseDiagVals.size() << "+" << extDiagVals.size();
    throw std::runtime_error(errStr.str());
  }
  for (Teuchos::Ordinal i=0; i<baseDiagVals.size(); ++i)
    allDiagVals[i] = baseDiagVals[i];
  Teuchos_Ordinal offset=baseDiagVals.size();
  for (Teuchos::Ordinal i=0; i<extDiagVals.size(); ++i)
    allDiagVals[i+offset] = extDiagVals[i];
}


template<class MatrixType>
void
OverlappingRowMatrix<MatrixType>::
leftScale (const Tpetra::Vector<scalar_type, local_ordinal_type, global_ordinal_type, node_type>& /* x */)
{
  throw std::runtime_error("Ifpack2::OverlappingRowMatrix does not support leftScale.");
}


template<class MatrixType>
void
OverlappingRowMatrix<MatrixType>::
rightScale (const Tpetra::Vector<scalar_type, local_ordinal_type, global_ordinal_type, node_type>& /* x */)
{
  throw std::runtime_error("Ifpack2::OverlappingRowMatrix does not support rightScale.");
}


template<class MatrixType>
typename OverlappingRowMatrix<MatrixType>::mag_type
OverlappingRowMatrix<MatrixType>::getFrobeniusNorm () const
{
  throw std::runtime_error("Ifpack2::OverlappingRowMatrix does not support getFrobeniusNorm.");
}


template<class MatrixType>
void
OverlappingRowMatrix<MatrixType>::
apply (const Tpetra::MultiVector<scalar_type,local_ordinal_type,global_ordinal_type,node_type> &X,
       Tpetra::MultiVector<scalar_type,local_ordinal_type,global_ordinal_type,node_type> &Y,
       Teuchos::ETransp mode,
       scalar_type alpha,
       scalar_type beta) const
{
  using MV = Tpetra::MultiVector<scalar_type, local_ordinal_type,
                                 global_ordinal_type, node_type>;
  TEUCHOS_TEST_FOR_EXCEPTION
    (X.getNumVectors() != Y.getNumVectors(), std::runtime_error,
     "Ifpack2::OverlappingRowMatrix::apply: X.getNumVectors() = "
     << X.getNumVectors() << " != Y.getNumVectors() = " << Y.getNumVectors()
     << ".");
  // If X aliases Y, we'll need to copy X.
  bool aliases = X.aliases(Y);
  if (aliases) {
    MV X_copy (X, Teuchos::Copy);
    this->apply (X_copy, Y, mode, alpha, beta);
    return;
  }

  const auto& rowMap0 = * (A_->getRowMap ());
  const auto& colMap0 = * (A_->getColMap ());
  MV X_0 (X, mode == Teuchos::NO_TRANS ? colMap0 : rowMap0, 0);
  MV Y_0 (Y, mode == Teuchos::NO_TRANS ? rowMap0 : colMap0, 0);
  A_->localApply (X_0, Y_0, mode, alpha, beta);

  const auto& rowMap1 = * (ExtMatrix_->getRowMap ());
  const auto& colMap1 = * (ExtMatrix_->getColMap ());
  MV X_1 (X, mode == Teuchos::NO_TRANS ? colMap1 : rowMap1, 0);
  MV Y_1 (Y, mode == Teuchos::NO_TRANS ? rowMap1 : colMap1, A_->getLocalNumRows ());
  ExtMatrix_->localApply (X_1, Y_1, mode, alpha, beta);
}


template<class MatrixType>
void
OverlappingRowMatrix<MatrixType>::
importMultiVector (const Tpetra::MultiVector<scalar_type,local_ordinal_type,global_ordinal_type,node_type> &X,
                   Tpetra::MultiVector<scalar_type,local_ordinal_type,global_ordinal_type,node_type> &OvX,
                   Tpetra::CombineMode CM)
{
  OvX.doImport (X, *Importer_, CM);
}


template<class MatrixType>
void
OverlappingRowMatrix<MatrixType>::
exportMultiVector (const Tpetra::MultiVector<scalar_type,local_ordinal_type,global_ordinal_type,node_type> &OvX,
                   Tpetra::MultiVector<scalar_type,local_ordinal_type,global_ordinal_type,node_type> &X,
                   Tpetra::CombineMode CM)
{
  X.doExport (OvX, *Importer_, CM);
}


template<class MatrixType>
bool OverlappingRowMatrix<MatrixType>::hasTransposeApply () const
{
  return true;
}


template<class MatrixType>
bool OverlappingRowMatrix<MatrixType>::supportsRowViews () const
{
  return false;
}

template<class MatrixType>
std::string OverlappingRowMatrix<MatrixType>::description() const
{
  std::ostringstream oss;
  if (isFillComplete()) {
    oss << "{ isFillComplete: true"
        << ", global rows: " << getGlobalNumRows()
        << ", global columns: " << getGlobalNumCols()
        << ", global entries: " << getGlobalNumEntries()
        << " }";
  }
  else {
    oss << "{ isFillComplete: false"
        << ", global rows: " << getGlobalNumRows()
        << " }";
  }
  return oss.str();
}

template<class MatrixType>
void OverlappingRowMatrix<MatrixType>::describe(Teuchos::FancyOStream &out,
            const Teuchos::EVerbosityLevel verbLevel) const
{
  //TODO
}

template<class MatrixType>
Teuchos::RCP<const Tpetra::RowMatrix<typename MatrixType::scalar_type, typename MatrixType::local_ordinal_type, typename MatrixType::global_ordinal_type, typename MatrixType::node_type> >
OverlappingRowMatrix<MatrixType>::getUnderlyingMatrix() const
{
  return A_;
}

template<class MatrixType>
const typename OverlappingRowMatrix<MatrixType>::crs_matrix_type::local_matrix_device_type
OverlappingRowMatrix<MatrixType>::getExtMatrix() const
{
  return ExtMatrix_->getLocalMatrixDevice();
}

template<class MatrixType>
Teuchos::ArrayView<const size_t> OverlappingRowMatrix<MatrixType>::getExtHaloStarts() const
{
  //return ExtHaloStarts_();
  throw std::runtime_error("Ifpack2::OverlappingRowMatrix::getExtHaloStarts() not supported.");
}

} // namespace Ifpack2

#define IFPACK2_OVERLAPPINGROWMATRIX_INSTANT(S,LO,GO,N)                 \
  template class Ifpack2::OverlappingRowMatrix< Tpetra::RowMatrix<S, LO, GO, N> >;

#endif // IFPACK2_OVERLAPPINGROWMATRIX_DEF_HPP
