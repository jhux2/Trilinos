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
// Questions? Contact the Ifpack2 developers (ifpack2-developers@software.sandia.gov)
//
// ***********************************************************************
//@HEADER
*/

#ifndef IFPACK2_LOCALFILTER_KOKKOS_DEF_HPP
#define IFPACK2_LOCALFILTER_KOKKOS_DEF_HPP

#include <Ifpack2_LocalFilter_decl.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Vector.hpp>

#ifdef HAVE_MPI
#  include "Teuchos_DefaultMpiComm.hpp"
#else
#  include "Teuchos_DefaultSerialComm.hpp"
#endif

namespace Ifpack2 {


template<class MatrixType>
LocalFilter_kokkos<MatrixType>::
LocalFilter_kokkos (const Teuchos::RCP<const row_matrix_type>& A) :
  A_ (A),
  NumNonzeros_ (0),
  MaxNumEntries_ (0)
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  const size_t numRows = A_->getRangeMap()->getLocalNumElements ();

  const global_ordinal_type indexBase = static_cast<global_ordinal_type> (0);

  // tentative value for MaxNumEntries. This is the number of
  // nonzeros in the local matrix
  MaxNumEntries_  = A_->getLocalMaxNumRowEntries ();

  // now compute:
  // - the number of nonzero per row
  // - the total number of nonzeros
  // - the diagonal entries

  // compute nonzeros (total and per-row), and store the
  // diagonal entries (already modified)
  size_t ActualMaxNumEntries = 0;

  kokkos_crs_matrix_type Ak_ = A_->getMatrix;  //FIXME call the Tpetra "get the Kokkos matrix" method, whatever that is
  for (size_t i = 0; i < numRows; ++i) {
    NumEntries_[i] = 0;
    size_t Nnz, NewNnz = 0;
    //A_->getLocalRowCopy (i, localIndices_, Values_, Nnz); //JHU FIXME 1/5 Just grab the col pointer and walk through
    SparseRowView<CrsMatrix> myRow = Ak->row(i);
    //TODO now actually populate a new KokkosCrs matrix ...
    for (size_t j = 0; j < Nnz; ++j) {
      // FIXME (mfh 03 Apr 2013) This assumes the following:
      //
      // 1. Row Map, range Map, and domain Map are all the same.
      //
      // 2. The column Map's list of GIDs on this process is the
      //    domain Map's list of GIDs, followed by remote GIDs.  Thus,
      //    for any GID in the domain Map on this process, its LID in
      //    the domain Map (and therefore in the row Map, by (1)) is
      //    the same as its LID in the column Map.  (Hence the
      //    less-than test, which if true, means that localIndices_[j]
      //    belongs to the row Map.)
      if (static_cast<size_t> (localIndices_[j]) < numRows) {
        ++NewNnz;
      }
    }

    if (NewNnz > ActualMaxNumEntries) {
      ActualMaxNumEntries = NewNnz;
    }

    NumNonzeros_ += NewNnz;
    NumEntries_[i] = NewNnz;
  }

  MaxNumEntries_ = ActualMaxNumEntries;
}


template<class MatrixType>
LocalFilter_kokkos<MatrixType>::~LocalFilter_kokkos()
{}


template<class MatrixType>
Teuchos::RCP<const Tpetra::Map<typename MatrixType::local_ordinal_type,
                               typename MatrixType::global_ordinal_type,
                               typename MatrixType::node_type> >
LocalFilter_kokkos<MatrixType>::getRowMap () const
{
  return localRowMap_;
}


template<class MatrixType>
Teuchos::RCP<const Tpetra::RowGraph<typename MatrixType::local_ordinal_type,
                                     typename MatrixType::global_ordinal_type,
                                     typename MatrixType::node_type> >
LocalFilter_kokkos<MatrixType>::getGraph () const
{
  // FIXME (mfh 20 Nov 2013) This is not what the documentation says
  // this method should do!  It should return the graph of the locally
  // filtered matrix, not the original matrix's graph.
  return A_->getGraph ();
}


template<class MatrixType>
size_t LocalFilter_kokkos<MatrixType>::getLocalNumRows() const
{
  return static_cast<size_t> (A_->numRows());
}


template<class MatrixType>
size_t LocalFilter_kokkos<MatrixType>::getLocalNumCols() const
{
  return static_cast<size_t> (localDomainMap_->getLocalNumElements ());
}


template<class MatrixType>
size_t LocalFilter_kokkos<MatrixType>::getLocalNumEntries () const
{
  return NumNonzeros_;
}


template<class MatrixType>
size_t
LocalFilter_kokkos<MatrixType>::
getNumEntriesInLocalRow (local_ordinal_type localRow) const
{
  // FIXME (mfh 07 Jul 2014) Shouldn't localRow be a local row index
  // in the matrix's row Map, not in the LocalFilter_kokkos's row Map?  The
  // latter is different; it even has different global indices!
  // (Maybe _that_'s the bug.)

  if (getRowMap ()->isNodeLocalElement (localRow)) {
    return NumEntries_[localRow];
  } else {
    // NOTE (mfh 26 Mar 2014) We return zero if localRow is not in the
    // row Map on this process, since "get the number of entries in
    // the local row" refers only to what the calling process owns in
    // that row.  In this case, it owns no entries in that row, since
    // it doesn't own the row.
    return 0;
  }
}



template<class MatrixType>
size_t LocalFilter_kokkos<MatrixType>::getLocalMaxNumRowEntries() const
{
  return MaxNumEntries_;
}


template<class MatrixType>
std::string
LocalFilter_kokkos<MatrixType>::description () const
{
  using Teuchos::TypeNameTraits;
  std::ostringstream os;

  os << "Ifpack2::LocalFilter_kokkos: {";
  os << "MatrixType: " << TypeNameTraits<MatrixType>::name ();
  if (this->getObjectLabel () != "") {
    os << ", Label: \"" << this->getObjectLabel () << "\"";
  }
  /*
  TODO
  os << ", Number of rows: " << getGlobalNumRows ()
     << ", Number of columns: " << getGlobalNumCols ()
     << "}";
  */
  return os.str ();
}


template<class MatrixType>
void
LocalFilter_kokkos<MatrixType>::
describe (Teuchos::FancyOStream &out,
          const Teuchos::EVerbosityLevel verbLevel) const
{
  using Teuchos::OSTab;
  using Teuchos::TypeNameTraits;
  using std::endl;

  const Teuchos::EVerbosityLevel vl =
    (verbLevel == Teuchos::VERB_DEFAULT) ? Teuchos::VERB_LOW : verbLevel;

  if (vl > Teuchos::VERB_NONE) {
    // describe() starts with a tab, by convention.
    OSTab tab0 (out);

    out << "Ifpack2::LocalFilter_kokkos:" << endl;
    OSTab tab1 (out);
    out << "MatrixType: " << TypeNameTraits<MatrixType>::name () << endl;
    if (this->getObjectLabel () != "") {
      out << "Label: \"" << this->getObjectLabel () << "\"" << endl;
    }
    /*
    out << "Number of rows: " << getGlobalNumRows () << endl
        << "Number of columns: " << getGlobalNumCols () << endl
        << "Number of nonzeros: " << NumNonzeros_ << endl;

    */
  }
}

} // namespace Ifpack2

#define IFPACK2_LOCALFILTER_KOKKOS_INSTANT(S,LO,GO,N) \
  template class Ifpack2::LocalFilter_kokkos< Tpetra::RowMatrix<S, LO, GO, N> >;

#endif //ifndef IFPACK2_LOCALFILTER_KOKKOS_DEF_HPP
