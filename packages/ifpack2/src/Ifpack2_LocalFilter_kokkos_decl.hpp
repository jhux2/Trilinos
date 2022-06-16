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

#ifndef IFPACK2_LOCALFILTER_KOKKOS_DECL_HPP
#define IFPACK2_LOCALFILTER_KOKKOS_DECL_HPP

#include "Ifpack2_ConfigDefs.hpp"
#include <type_traits>
#include <vector>


namespace Ifpack2 {

template<class MatrixType>
class LocalFilter_kokkos :
    virtual public Teuchos::Describable
{
private:
  static_assert (std::is_same<
                   MatrixType,
                   Tpetra::RowMatrix<
                     typename MatrixType::scalar_type,
                     typename MatrixType::local_ordinal_type,
                     typename MatrixType::global_ordinal_type,
                     typename MatrixType::node_type> >::value,
                 "Ifpack2::LocalFilter_kokkos: MatrixType must be a Tpetra::RowMatrix specialization.");

public:
  //! \name Typedefs
  //@{

  //! The type of the entries of the input MatrixType.
  typedef typename MatrixType::scalar_type scalar_type;

  //! The type of local indices in the input MatrixType.
  typedef typename MatrixType::local_ordinal_type local_ordinal_type;

  //! The type of global indices in the input MatrixType.
  typedef typename MatrixType::global_ordinal_type global_ordinal_type;

  //! The Node type used by the input MatrixType.
  typedef typename MatrixType::node_type node_type;
  typedef typename MatrixType::node_type::device_type device_type;
  typedef typename device_type::execution_space execution_space;

  typedef typename MatrixType::global_inds_host_view_type global_inds_host_view_type;
  typedef typename MatrixType::local_inds_host_view_type local_inds_host_view_type;
  typedef typename MatrixType::values_host_view_type values_host_view_type;

  typedef typename MatrixType::nonconst_global_inds_host_view_type nonconst_global_inds_host_view_type;
  typedef typename MatrixType::local_inds_device_view_type local_inds_device_view_type;
  typedef typename MatrixType::values_device_view_type values_device_view_type;
  typedef typename MatrixType::nonconst_local_inds_host_view_type nonconst_local_inds_host_view_type;
  typedef typename MatrixType::nonconst_values_host_view_type nonconst_values_host_view_type;

////////////////////////////////////////
  typedef KokkosSparse::CrsMatrix<scalar_type, local_ordinal_type, execution_space> kokkos_crs_matrix_type;
////////////////////////////////////////

  typedef typename Kokkos::View<local_ordinal_type *, typename node_type::device_type>::non_const_type nonconst_local_inds_device_view_type;
  typedef typename Kokkos::View<scalar_type *, typename node_type::device_type>::non_const_type nonconst_values_device_view_type;

  //! The type of the magnitude (absolute value) of a matrix entry.
  typedef typename Teuchos::ScalarTraits<scalar_type>::magnitudeType magnitude_type;

  //! Type of the Tpetra::RowMatrix specialization that this class uses.
  typedef Tpetra::RowMatrix<scalar_type,
                            local_ordinal_type,
                            global_ordinal_type,
                            node_type> row_matrix_type;

  //! Type of the Tpetra::Map specialization that this class uses.
  typedef Tpetra::Map<local_ordinal_type,
                      global_ordinal_type,
                      node_type> map_type;

  typedef typename row_matrix_type::mag_type mag_type;

  //@}
  //! @name Implementation of Teuchos::Describable
  //@{

  //! A one-line description of this object.
  virtual std::string description () const;

  //! Print the object to the given output stream.
  virtual void
  describe (Teuchos::FancyOStream &out,
            const Teuchos::EVerbosityLevel verbLevel =
            Teuchos::Describable::verbLevel_default) const;

  //@}
  //! \name Constructor and destructor
  //@{

  /// \brief Constructor
  ///
  /// \param A [in] The sparse matrix to which to apply the local filter.
  ///
  /// This class will <i>not</i> modify the input matrix.
  explicit LocalFilter_kokkos (const Teuchos::RCP<const row_matrix_type>& A);

  //! Destructor
  virtual ~LocalFilter_kokkos();

  //@}
  //! \name Matrix Query Methods
  //@{

  //! Returns the Map that describes the row distribution in this matrix.
  virtual Teuchos::RCP<const map_type> getRowMap() const;

  //! The (locally filtered) matrix's graph.
  virtual Teuchos::RCP<const Tpetra::RowGraph<local_ordinal_type,global_ordinal_type,node_type> >
  getGraph () const;

  //! The number of rows owned on the calling process.
  virtual size_t getLocalNumRows() const;

  //! The number of columns in the (locally filtered) matrix.
  virtual size_t getLocalNumCols() const;

  //! Returns the local number of entries in this matrix.
  virtual size_t getLocalNumEntries() const;

  /// \brief The current number of entries on this node in the specified local row.
  ///
  /// \return <tt>Teuchos::OrdinalTraits<size_t>::invalid()</tt> if
  ///   the specified local row is not valid on this process,
  ///   otherwise the number of entries in that row on this process.
  virtual size_t getNumEntriesInLocalRow (local_ordinal_type localRow) const;

  //! The maximum number of entries across all rows/columns on this process.
  virtual size_t getLocalMaxNumRowEntries() const;

  mutable nonconst_local_inds_device_view_type localIndices_;
  mutable nonconst_values_device_view_type Values_;


  //@}
private:

  /// Kokkos Crs matrix
  //KokkosSparse::CrsMatrix<scalar_type, local_ordinal_type, execution_space> A_;
  kokkos_crs_matrix_type Ak_;

  size_t NumNonzeros_;
  size_t MaxNumEntries_;

};// class LocalFilter_kokkos

}// namespace Ifpack2

#endif /* IFPACK2_LOCALFILTER_KOKKOS_DECL_HPP */
