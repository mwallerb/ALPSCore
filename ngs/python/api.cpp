/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
 *                              Matthias Troyer <troyer@comp-phys.org>             *
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

#define PY_ARRAY_UNIQUE_SYMBOL pyngsapi_PyArrayHandle

#include <alps/ngs/api.hpp>
#include <alps/ngs/hdf5.hpp>
#include <alps/ngs/base.hpp>
#include <alps/ngs/params.hpp>
#include <alps/ngs/mcresults.hpp>

namespace alps {
	namespace detail {

        void save_results_export(mcresults const & res, params const & par, alps::hdf5::archive & ar, std::string const & path) {
            ar
                << make_pvp("/parameters", par)
            ;
            if (res.size())
                ar
                    << make_pvp(path, res)
                ;
        }
    }
}

BOOST_PYTHON_MODULE(pyngsapi_c) {

    boost::python::def("collectResults", &alps::collect_results<alps::base>);

    boost::python::def("saveResults", &alps::detail::save_results_export);

}