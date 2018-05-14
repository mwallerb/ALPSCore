#include <alps/alea.hpp>
#include <alps/testing/near.hpp>

#include <Eigen/Dense>

#include "gtest/gtest.h"

struct Ones {
    template <typename Derived>
    friend void operator+=(Eigen::MatrixBase<Derived> &store, const Ones &)
    {
        store.array() += 1.0;
    }
};

// Register type
namespace alps { namespace alea {
    template <> struct is_custom_addable<Ones> : std::true_type { };
}}

template <typename Acc>
class custom_case
    : public ::testing::Test
{
public:
    typedef Acc acc_type;
    typedef typename alps::alea::traits<Acc>::store_type  store_type;
    typedef typename alps::alea::traits<Acc>::value_type value_type;
    typedef typename alps::alea::traits<Acc>::result_type result_type;

    void test_add()
    {
        Acc acc(2);
        for (size_t i = 0; i != 1000; ++i)
            acc << Ones();

        result_type res = acc.finalize();
        ALPS_EXPECT_NEAR(res.mean()[0], value_type(1.0), 1e-10);
        ALPS_EXPECT_NEAR(res.mean()[1], value_type(1.0), 1e-10);
    }
};

typedef ::testing::Types<
      alps::alea::mean_acc<double>
    , alps::alea::mean_acc<std::complex<double> >
    , alps::alea::var_acc<double>
    , alps::alea::var_acc<std::complex<double> >
    , alps::alea::var_acc<std::complex<double>, alps::alea::elliptic_var >
    , alps::alea::cov_acc<double>
    , alps::alea::cov_acc<std::complex<double> >
    , alps::alea::cov_acc<std::complex<double>, alps::alea::elliptic_var >
    > has_mean;

TYPED_TEST_CASE(custom_case, has_mean);

TYPED_TEST(custom_case, test_add) { this->test_add(); }

