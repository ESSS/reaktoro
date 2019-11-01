// Reaktoro is a unified framework for modeling chemically reactive systems.
//
// Copyright (C) 2014-2018 Allan Leal
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this library. If not, see <http://www.gnu.org/licenses/>.

#include "MathUtils.hpp"

// Eigen includes
#include <Reaktoro/deps/eigen3/Eigen/QR>

// Reaktoro includes
#include <Reaktoro/Common/Exception.hpp>

namespace Reaktoro {

auto linearlyIndependentCols(MatrixConstRef A) -> Indices
{
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
    const unsigned rank = qr.rank();
    Eigen::VectorXi I = qr.colsPermutation().indices().segment(0, rank);
    std::sort(I.data(), I.data() + rank);
    Indices indices(I.data(), I.data() + rank);
    return indices;
}

auto linearlyIndependentRows(MatrixConstRef A) -> Indices
{
    const Matrix At = A.transpose();
    return linearlyIndependentCols(At);
}

auto linearlyIndependentCols(MatrixConstRef A, MatrixRef B) -> Indices
{
    Indices indices = linearlyIndependentCols(A);
    Matrix C(A.rows(), indices.size());
    for(unsigned i = 0; i < indices.size(); ++i)
        C.col(i) = A.col(indices[i]);
    B.noalias() = C;
    return indices;
}

auto linearlyIndependentRows(MatrixConstRef A, MatrixRef B) -> Indices
{
    Indices indices = linearlyIndependentRows(A);
    Matrix C(indices.size(), A.cols());
    for(unsigned i = 0; i < indices.size(); ++i)
        C.row(i) = A.row(indices[i]);
    B.noalias() = C;
    return indices;
}

auto inverseShermanMorrison(MatrixConstRef invA, VectorConstRef D) -> Matrix
{
    Matrix invM = invA;
    for(unsigned i = 0; i < D.rows(); ++i)
        invM = invM - (D[i]/(1 + D[i]*invM(i, i)))*invM.col(i)*invM.row(i);
    return invM;
}

/// Return the numerator and denominator of the rational number closest to `x`.
/// This methods expects `0 <= x <= 1`.
/// @param x The number for which the closest rational number is sought.
/// @param maxden The maximum denominator that the rational number can have.
auto farey(double x, unsigned maxden) -> std::tuple<long, long>
{
    long a = 0, b = 1;
    long c = 1, d = 1;
    while(b <= maxden && d <= maxden)
    {
        double mediant = double(a+c)/(b+d);
        if(x == mediant) {
            if(b + d <= maxden) return std::make_tuple(a+c, b+d);
            if(d > b) return std::make_tuple(c, d);
            return std::make_tuple(a, b);
        }
        if(x > mediant) {
            a = a+c;
            b = b+d;
        }
        else {
            c = a+c;
            d = b+d;
        }
    }

    return (b > maxden) ? std::make_tuple(c, d) : std::make_tuple(a, b);
}

auto rationalize(double x, unsigned maxden) -> std::tuple<long, long>
{
    long a, b, sign = (x >= 0) ? +1 : -1;
    if(std::abs(x) > 1.0) {
        std::tie(a, b) = farey(1.0/std::abs(x), maxden);
        return std::make_tuple(sign*b, a);
    }
    else {
        std::tie(a, b) = farey(std::abs(x), maxden);
        return std::make_tuple(sign*a, b);
    }
}

auto cleanRationalNumbers(double* vals, long size, long maxden) -> void
{
    long num, den;
    for(long i = 0; i < size; ++i)
    {
        std::tie(num, den) = rationalize(vals[i], maxden);
        vals[i] = static_cast<double>(num)/den;
    }
}

auto cleanRationalNumbers(MatrixRef A, long maxden) -> void
{
    cleanRationalNumbers(A.data(), A.size(), maxden);
}

template<typename VectorTypeX, typename VectorTypeY>
auto dot3p_(const VectorTypeX& x, const VectorTypeY& y, double s) -> double
{
   double shi = double(float(s));
   double slo = s - shi;
   for(int k = 0; k < x.size(); ++k)
   {
      double xhi = double(float(x[k]));
      double xlo = x[k] - xhi;
      double yhi = double(float(y[k]));
      double ylo = y[k] - yhi;
      double tmp = xhi*yhi;
      double zhi = double(float(tmp));
      double zlo = tmp - zhi + xhi*ylo + xlo*yhi + xlo*ylo;

      tmp = shi + zhi;
      double del = tmp - shi - zhi;
      shi = double(float(tmp));
      slo = tmp - shi + slo + zlo - del;
   }

   s = shi + slo;
   return s;
}

auto dot3p(VectorConstRef x, VectorConstRef y, double s) -> double
{
    return dot3p_(x, y, s);
}

auto residual3p(MatrixConstRef A, VectorConstRef x, VectorConstRef b) -> Vector
{
    const auto m = A.rows();
    Vector r = zeros(m);
    for(int k = 0; k < m; ++k)
        r[k] = dot3p(A.row(k), x, -b[k]);
    return r;
}

// This is temporarily here (will not be merged)
// To check that openlibm is indeed being linked
// (Implementation is in ChemicalSolver.cpp)
extern "C" int isopenlibm(void);
auto check_libm() -> std::vector<double>
{
    auto r = std::vector<double>(12, -1.0);

    r[0] = isopenlibm() - 1.0;

    // Values in comments are returned by the standard libraries in the respective
    // platform. If the platform is ommitted, it's the same as returned by openlibm

    // Windows: -0.34431837261747222
    r[1] = tanh(-0.3589835151970974) - (-0.34431837261747228);
    r[2] = std::tanh(-0.3589835151970974) - (-0.34431837261747228);

    // Linux: 1.0552633797823418e+43
    // NOTE: only `exp` tries to use `Reaktoro::exp`, might be a problem
    r[3] = ::exp(99.06494938359764) - 1.0552633797823417e+43;
    r[4] = std::exp(99.06494938359764) - 1.0552633797823417e+43;

    auto x = 0.70444454416678126;

    // Linux: 0.64761068800896837
    r[5] = sin(x) - 0.64761068800896848;
    r[6] = std::sin(x) - 0.64761068800896848;

    // Windows: 0.84991470526022272
    r[7] = tan(x) - 0.84991470526022261;
    r[8] = std::tan(x) - 0.84991470526022261;

    // Windows: 1.2585529728058984
    r[9] = cosh(x) - 1.2585529728058982;
    r[10] = std::cosh(x) - 1.2585529728058982;

    // Note: `floor` is being used from ucrtbase in Windows, due to a duplicated symbol
    // linker issue, but it doesn't seem problematic
    auto f = floor(1e15 + 0.1);
    r[11] = f - 1e15;

    return r;
}

} // namespace Reaktoro
