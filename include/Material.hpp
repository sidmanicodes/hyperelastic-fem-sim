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
                double E,
                double nu
            );

            /*
            Returns the NeoHookean energy density
            */
            double energyDensity(
                Eigen::Matrix3d& F
            ) const;

            /*
            Returns the first Piola-Kirchoff stress tensor
            */
            Eigen::Matrix3d firstPiolaStressTensor(
                Eigen::Matrix3d& F
            );

        private:
            double mu_;
            double lambda_;
    };
}