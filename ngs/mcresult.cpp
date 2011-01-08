/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
 *                           Matthias Troyer <troyer@comp-phys.org>                *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <alps/ngs/mcresult.hpp>
#include <alps/ngs/mcresult_impl_base.ipp>
#include <alps/ngs/mcresult_impl_derived.ipp>

#include <alps/alea/observable.h>
#include <alps/alea/abstractsimpleobservable.h>

#include <iostream>

namespace alps {

    mcresult::mcresult()
        : impl_(NULL) 
    {}

    mcresult::mcresult(Observable const * obs) {
        if (dynamic_cast<AbstractSimpleObservable<double> const *>(obs) != NULL)
            impl_ = new detail::mcresult_impl_derived<detail::mcresult_impl_base, double>(
                dynamic_cast<AbstractSimpleObservable<double> const &>(*obs)
            );
        else if (dynamic_cast<AbstractSimpleObservable<std::valarray<double> > const *>(obs) != NULL)
            impl_ = new detail::mcresult_impl_derived<detail::mcresult_impl_base, std::vector<double> >(
                dynamic_cast<AbstractSimpleObservable<std::valarray<double> > const &>(*obs)
            );
        else
            throw std::runtime_error("unknown observable type");
        ref_cnt_[impl_] = 1;
    }

    mcresult::mcresult(mcresult const & rhs) {
        ++ref_cnt_[impl_ = rhs.impl_];
    }

    mcresult::~mcresult() {
        if (impl_ && !--ref_cnt_[impl_])
            delete impl_;
    }

    mcresult & mcresult::operator=(mcresult rhs) {
        if (impl_ && !--ref_cnt_[impl_])
            delete impl_;
        ++ref_cnt_[impl_ = rhs.impl_];
        return *this;
    }

    #define ALPS_MCRESULT_TPL_IMPL(T)                                                                  \
        template<> std::vector< T > const & mcresult::bins< T >() const { return impl_->bins< T >(); }            \
        template<> T const & mcresult::mean< T >() const { return impl_->mean< T >(); }                           \
        template<> T const & mcresult::error< T >() const { return impl_->error< T >(); }                         \
        template<> T const & mcresult::variance< T >() const { return impl_->error< T >(); }                      \
        template<> T const & mcresult::tau< T >() const { return impl_->error< T >(); }                           \
        template<> T const & mcresult::covariance< T >() const { return impl_->error< T >(); }
    ALPS_MCRESULT_TPL_IMPL(double)
    ALPS_MCRESULT_TPL_IMPL(std::vector<double>)
    #undef ALPS_MCRESULT_TPL_IMPL

    bool mcresult::can_rebin() const {
        return impl_->can_rebin();
    }

    bool mcresult::jackknife_valid() const {
        return impl_->jackknife_valid();
    }

    uint64_t mcresult::count() const {
        return impl_->count();
    }

    uint64_t mcresult::bin_size() const {
        return impl_->bin_size();
    }

    uint64_t mcresult::max_bin_number() const {
        return impl_->max_bin_number();
    }

    std::size_t mcresult::bin_number() const {
        return impl_->bin_number();
    }

    bool mcresult::has_variance() const {
        return impl_->has_variance();
    }

    bool mcresult::has_tau() const {
        return impl_->has_tau();
    }

    void mcresult::set_bin_size(uint64_t binsize) {
        impl_->set_bin_size(binsize);
    }

    void mcresult::set_bin_number(uint64_t bin_number) {
        impl_->set_bin_number(bin_number);
    }

    void mcresult::serialize(hdf5::iarchive & ar) {
        impl_->serialize(ar);
    }

    void mcresult::serialize(hdf5::oarchive & ar) const {
        impl_->serialize(ar);
    }

    void mcresult::output(std::ostream & os) const {
        impl_->output(os);
    }

    #ifdef ALPS_HAVE_MPI
        mcresult mcresult::reduce(boost::mpi::communicator const & communicator, std::size_t binnumber) {
            mcresult lhs;
            detail::mcresult_impl_base * impl = impl_->reduce(communicator, binnumber);
            if (communicator.rank() == 0)
                ref_cnt_[lhs.impl_ = impl] = 1;
            return lhs;
        }
    #endif

    bool mcresult::operator== (mcresult const & rhs) const {  \
        return impl_->operator== (rhs.impl_);                 \
    }
    bool mcresult::operator!= (mcresult const & rhs) const {  \
        return impl_->operator!= (rhs.impl_);                 \
    }


    mcresult & mcresult::operator+=(mcresult const & rhs) {
        impl_->operator+=(rhs.impl_);
        return *this;
    }
    mcresult & mcresult::operator-=(mcresult const & rhs) {
        impl_->operator-=(rhs.impl_);
        return *this;
    }
    mcresult & mcresult::operator*=(mcresult const & rhs) {
        impl_->operator*=(rhs.impl_);
        return *this;
    }
    mcresult & mcresult::operator/=(mcresult const & rhs) {
        impl_->operator/=(rhs.impl_);
        return *this;
    }
    /*
    template <typename T> mcresult & mcresult::operator+=(T const & rhs);
    template <typename T> mcresult & mcresult::operator-=(T const & rhs);
    template <typename T> mcresult & mcresult::operator*=(T const & rhs);
    template <typename T> mcresult & mcresult::operator/=(T const & rhs);
	*/
    mcresult & mcresult::operator+() {
        impl_->operator-();
        return *this;
    }
    mcresult & mcresult::operator-() {
        impl_->operator-();
        return *this;
    }

    std::map<detail::mcresult_impl_base *, std::size_t> mcresult::ref_cnt_;

    std::ostream & operator<<(std::ostream & os, mcresult const & result) {
        result.output(os);
        return os;
    }

    #define ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(FUN_NAME)                 \
        mcresult FUN_NAME (mcresult rhs) {                               \
            mcresult lhs;                                                \
            lhs.ref_cnt_[lhs.impl_ = rhs.impl_-> FUN_NAME ()] = 1;       \
            return lhs;                                                  \
        }
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(sin)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(cos)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(tan)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(sinh)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(cosh)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(tanh)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(asin)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(acos)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(atan)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(asinh)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(acosh)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(atanh)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(abs)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(sq)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(cb)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(sqrt)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(cbrt)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(exp)
    ALPS_NGS_MCRESULT_FREE_UNITARY_FUN(log)
    #undef ALPS_NGS_MCRESULT_FREE_UNITARY_FUN

    mcresult pow(mcresult rhs, double exponent) {
        mcresult lhs;
        lhs.ref_cnt_[lhs.impl_ = rhs.impl_->pow(exponent)] = 1;
        return lhs;
    }
/*
    #define ALPS_NGS_MCRESULT_OPERATOR(OPERATOR)                                                       \
        mcresult operator ## OPERATOR(mcresult lhs, mcresult  const & rhs);                            \
        template <typename T> mcresult operator ## OPERATOR(mcresult lhs, T const & rhs);              \
        template <typename T> mcresult operator ## OPERATOR(T const & lhs, mcresult const & rhs);      \
    ALPS_NGS_MCRESULT_OPERATOR(+)
    ALPS_NGS_MCRESULT_OPERATOR(-)
    ALPS_NGS_MCRESULT_OPERATOR(*)
    ALPS_NGS_MCRESULT_OPERATOR(/)
    #undef ALPS_NGS_MCRESULT_OPERATOR
*/
}