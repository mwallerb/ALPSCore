/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/alea/util.hpp>

#include <Eigen/Dense>

// Forward declarations

namespace alps { namespace alea { namespace internal {
    template <typename T> class value_adapter;
    template <typename T> class vector_adapter;
    template <typename T, size_t N> class array_adapter;
    template <typename T, typename Derived> class eigen_adapter;
}}}

// Actual declarations

namespace alps { namespace alea { namespace internal {

/**
 * Interface for a computed result (a result computed on-the-fly).
 *
 * As a trivial example, here is a vector-valued estimator of size 2 that
 * always adds the vector [1.0, -1.0] to the buffer:
 *
 *     struct trivial_computed : public computed<double>
 *     {
 *         size_t size() const { return 2; }
 *         void add_to(view<T> out) { out[0] += 1.0; out[1] -= 1.0; }
 *     }
 *
 * If a `computed` is passed to an accumulator, the accumulator will call the
 * `add_to()` method zero or more times with different buffers.  This allows
 * to avoid temporaries, as the addend can be constructed in-place, which also
 * allows for sparse data.  Also, as there is usually a bin size > 1, adding is
 * the fundamental operation.
 *
 * \warning Do not use this class directly, as its interface is not stable.
*
 * See also: is_custom_addable<T>
 */
template <typename T>
struct computed
{
    typedef T value_type;

    /** Number of elements of the computed result */
    virtual size_t size() const = 0;

    /**
     * Add computed result data to the buffer in `out`.  If `in(i)` is the
     * `i`-th component of the estimator, do the equivalent of:
     *
     *     for (size_t i = 0; i != size(); ++i)
     *         out[i] += in(i);
     */
    virtual void add_to(view<T> out) const = 0;

    /** Returns a clone of the estimator (optional) */
    virtual computed *clone() { throw unsupported_operation(); }

    /** Destroy estimator */
    virtual ~computed() { }
};

/** Adapter class that maps scalars on the computed interface */
template <typename T>
class value_adapter
    : public computed<T>
{
public:
    typedef T value_type;

public:
    value_adapter(T in) : in_(in) { }

    size_t size() const { return 1; }

    void add_to(view<T> out) const
    {
        if (out.size() != 1)
            throw size_mismatch();
        out.data()[0] += in_;
    }

    ~value_adapter() { }

private:
    T in_;
};

/** Adapter class that maps `std::vector<T>` on the computed interface */
template <typename T>
class vector_adapter
    : public computed<T>
{
public:
    typedef T value_type;

public:
    vector_adapter(const std::vector<T> &in) : in_(in) { }

    size_t size() const { return in_.size(); }

    void add_to(view<T> out) const
    {
        if (out.size() != in_.size())
            throw size_mismatch();
        for (size_t i = 0; i != in_.size(); ++i)
            out.data()[i] += in_[i];
    }

    ~vector_adapter() { }

private:
    const std::vector<T> &in_;
};

/** Adapter class that maps `std::array<T,N>` on the computed interface */
template <typename T, size_t N>
class array_adapter
    : public computed<T>
{
public:
    typedef T value_type;

public:
    array_adapter(const std::array<T, N> &in) : in_(in) {}

    size_t size() const { return in_.size(); }

    void add_to(view<T> out) const
    {
        if (out.size() != in_.size())
            throw size_mismatch();
        for (size_t i = 0; i != in_.size(); ++i)
            out.data()[i] += in_[i];
    }

    ~array_adapter() {}

private:
    const std::array<T, N> &in_;
};

/** Adapter class that maps Eigen columns on the computed interface */
template <typename T, typename Derived>
class eigen_adapter
    : public computed<T>
{
public:
    typedef T value_type;

public:
    eigen_adapter(const Eigen::DenseBase<Derived> &in)
        : in_(in)
    {
        EIGEN_STATIC_ASSERT_VECTOR_ONLY(Eigen::DenseBase<Derived>);
        static_assert(std::is_same<T, typename Derived::Scalar>::value,
                      "Type mismatch -- use explicit cast");
    }

    size_t size() const { return in_.size(); }

    void add_to(view<T> out) const
    {
        if (out.size() != (size_t)in_.rows())
            throw size_mismatch();

        typename eigen<T>::col_map out_map(out.data(), out.size());
        out_map += in_;
    }

    ~eigen_adapter() { }

private:
    const Eigen::DenseBase<Derived> &in_;
};

}}}  // namespace alps::alea::internal

// FIXME: remove!
namespace alps { namespace alea {
template <typename Derived>
internal::eigen_adapter<typename Derived::Scalar, Derived>
        make_adapter(const Eigen::DenseBase<Derived> &in)
{
    return internal::eigen_adapter<typename Derived::Scalar, Derived>(in);
}
}}
