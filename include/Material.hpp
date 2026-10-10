#pragma once

#include <Eigen/Dense>

namespace FEMSim {
    class NeoHookeanMaterial{
        public:
            /*
            \E is Young's modulus
            \nu is the Poisson ratio
            */
            NeoHookeanMaterial(
                const double E,
                const double nu
            );

            /*
            Returns the NeoHookean energy density
            */
            double energyDensity(
                const Eigen::Matrix3d& F
            ) const;

            /*
            Returns the first Piola-Kirchoff stress tensor
            */
            Eigen::Matrix3d firstPiolaStressTensor(
                const Eigen::Matrix3d& F
            ) const;

        private:
            double mu_;
            double lambda_;
    };
}