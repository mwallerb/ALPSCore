/* Common methods for accumulators
 *
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <cstddef>
#include <cassert>
#include <stdexcept>
#include <complex>

#include <vector>
#include <array>
#include <Eigen/Dense>

#include <alps/alea/core.hpp>
#include <alps/alea/util.hpp>
#include <alps/alea/internal/computed.hpp>

// Actual declarations

namespace alps { namespace alea {

template <typename T>
using verify_acc_type = typename std::enable_if<is_alea_acc<T>::value, T>::type;


/** Add scalar value to accumulator */
template <typename Acc>
verify_acc_type<Acc> &operator<<(Acc& acc, const typename Acc::value_type& v)
{
    return acc << internal::value_adapter<typename Acc::value_type>(v);
}

/** Add Eigen vector-valued expression to accumulator */
template <typename Acc, typename Derived>
verify_acc_type<Acc> &operator<<(Acc& acc, const Eigen::DenseBase<Derived>& v)
{
    return acc << internal::eigen_adapter<typename Acc::value_type, Derived>(v);
}

/** Add `std::vector` to accumulator */
template <typename Acc>
verify_acc_type<Acc> &operator<<(Acc& acc, const std::vector<typename Acc::value_type>& v)
{
    return acc << internal::vector_adapter<typename Acc::value_type>(v);
}

/** Add `std::array` to accumulator */
template <typename Acc, size_t N>
verify_acc_type<Acc> &operator<<(Acc& acc, const std::array<typename Acc::value_type, N>& v)
{
    return acc << internal::array_adapter<typename Acc::value_type, N>(v);
}

template <typename Acc, typename Op>
typename std::enable_if<is_alea_acc<Acc>::value && is_custom_addable<Op>::value, Acc>::type &
        operator<<(Acc &acc, const Op &op)
{
    return acc << internal::custom_adapter<typename Acc::value_type, Op>(op, acc.size());
}

}}
