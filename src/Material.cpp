#include "Material.hpp"
#include <cmath>
#include <stdexcept>

FEMSim::NeoHookeanMaterial::NeoHookeanMaterial(
    double E, double nu) 
{
    mu_ = E / (2 * (1 + nu));
    lambda_ = (E * nu) / ((1 + nu) * (1 - 2*nu));
}

double FEMSim::NeoHookeanMaterial::energyDensity(Eigen::Matrix3d& F) const{
    double J = F.determinant();

    if (J <= 0) {
        throw std::runtime_error("ERROR: J <= 0");
    }

    // tr(F^T F) = ||F||_F^2
    double Ic = F.squaredNorm();

    double logJ = std::log(J);

    return 
        0.5 * (mu_ * (Ic - 3.0)) - (mu_ * logJ) + 0.5 * (lambda_ * logJ * logJ);
}

Eigen::Matrix3d FEMSim::NeoHookeanMaterial::firstPiolaStressTensor(Eigen::Matrix3d& F) {
    double J = F.determinant();

    if (J <= 0) {
        throw std::runtime_error("ERROR: J <= 0");
    }

    double logJ = std::log(J);

    Eigen::Matrix3d FinvT = F.inverse().transpose();

    return 
        (mu_ * (F - FinvT)) + (lambda_ * logJ * FinvT);
}

