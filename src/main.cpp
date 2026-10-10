#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <array>
#include <Eigen/Dense>
#include <polyscope/polyscope.h>
#include <polyscope/volume_mesh.h>
#include <igl/readMESH.h>
#include "Material.hpp"

double pi = M_PI;

Eigen::Matrix3d computeDeformationGradient(
    int e,
    const Eigen::MatrixXd& x,
    const Eigen::MatrixXi& T,
    const std::vector<Eigen::MatrixXd>& DmInv
) 
{
    // Get vertex indices
    int i_0 = T(e, 0);
    int i_1 = T(e, 1);
    int i_2 = T(e, 2);
    int i_3 = T(e, 3);

    // Compute difference vectors
    Eigen::Vector3d dx_0 = x.row(i_1) - x.row(i_0);
    Eigen::Vector3d dx_1 = x.row(i_2) - x.row(i_0);
    Eigen::Vector3d dx_2 = x.row(i_3) - x.row(i_0);
    
    // Form Ds
    Eigen::Matrix3d Ds_e;
    Ds_e.col(0) = dx_0;
    Ds_e.col(1) = dx_1;
    Ds_e.col(2) = dx_2;

    // Return deformation gradient
    return Ds_e * DmInv[e];;
}

void addTetraForces(
    int e,
    const Eigen::MatrixXd& x,
    const Eigen::MatrixXi& T,
    const std::vector<std::array<Eigen::Vector3d, 4>>& gradN,
    const std::vector<double>& initialVols,
    const Eigen::Matrix3d F,
    const FEMSim::NeoHookeanMaterial& material,
    Eigen::MatrixXd& forces
)
{
    // Get vertex indices
    int i_0 = T(e, 0);
    int i_1 = T(e, 1);
    int i_2 = T(e, 2);
    int i_3 = T(e, 3);

    // Get the gradients of the shape functions
    std::array<Eigen::Vector3d, 4> gradN_e = gradN[e];

    double V = initialVols[e];

    Eigen::Matrix3d P = material.firstPiolaStressTensor(F);

    forces.row(i_0) += (-V * P * gradN_e[0]).transpose();
    forces.row(i_1) += (-V * P * gradN_e[1]).transpose();
    forces.row(i_2) += (-V * P * gradN_e[2]).transpose();
    forces.row(i_3) += (-V * P * gradN_e[3]).transpose();
}

int main() {
    // Initialize viewer
    polyscope::init();

    // Read in vertex locations, tetra indices, and faces
    Eigen::MatrixXd X;
    Eigen::MatrixXi T;
    Eigen::MatrixXi F;

    igl::readMESH("../data/cube86.mesh", X, T, F);

    /*
    Compute D_m^{-1} for each tetra (since these are in the material config, 
    they won't change; so we compute them before computing each deformation gradient)
    */
    std::vector<Eigen::MatrixXd> DmInv(T.rows());

    /*
    We also want to use each DmInv to calculate the
    gradients of the shape functions for FEM

    Each element is a vector of 4 eigen vectors (gradN0, ..., gradN3)
    */
    std::vector<std::array<Eigen::Vector3d, 4>> gradN(T.rows());

    /*
    The initial volume of each tetra is needed in order
    to compute the internal forces. This volume is given by:
    vol = |det(Dm)| / 6.0
    */
   std::vector<double> initialVols(T.rows());

    for (int e = 0; e < T.rows(); ++e) {
        // Get tetra indices
        int i_0 = T(e, 0);
        int i_1 = T(e, 1);
        int i_2 = T(e, 2);
        int i_3 = T(e, 3);

        // Compute dX column vectors
        Eigen::Vector3d dX_0 = X.row(i_1) - X.row(i_0);
        Eigen::Vector3d dX_1 = X.row(i_2) - X.row(i_0);
        Eigen::Vector3d dX_2 = X.row(i_3) - X.row(i_0);

        Eigen::Matrix3d Dm;
        Dm.col(0) = dX_0;
        Dm.col(1) = dX_1;
        Dm.col(2) = dX_2;

        // Store the material config volume of the element
        initialVols[e] = std::abs(Dm.determinant()) / 6.0;

        // Store Dm inverse for the eth tetrahedron
        DmInv[e] = Dm.inverse();

        // Compute gradient of eth shape function
        Eigen::Matrix3d gradN_e = DmInv[e].transpose();
        gradN[e][1] = gradN_e.col(0);
        gradN[e][2] = gradN_e.col(1);
        gradN[e][3] = gradN_e.col(2);

        /*
        N0 + ... + N3 = 1 <=> gradN0 + ... + gradN3 = 0
                          <=> gradN0 = -gradN1 - gradN2 - gradN3
        */
        gradN[e][0] = -(gradN[e][1] + gradN[e][2] + gradN[e][3]);
    }

    // For now, spatial config is equal to material config
    Eigen::MatrixXd x = X;

    double ymin = X.col(1).minCoeff();
    double ymax = X.col(1).maxCoeff();

    for (int i = 0; i < x.rows(); ++i) {

        double alpha =
            (X(i, 1) - ymin) /
            (ymax - ymin);

        x(i, 0) += 0.5 * alpha * alpha;
    }

    /*
    Create a NeoHookean material with Young's modulus 1000 Pa
    and Poisson ratio 0.3 (hyperelastic)
    */
    FEMSim::NeoHookeanMaterial material(1000.0, 0.3);

    double theta = pi/4;
    Eigen::Matrix3d rotateX;
    rotateX << 1.0, 0.0, 0.0,
               0.0, std::cos(theta), -std::sin(theta),
               0.0, std::sin(theta), std::cos(theta);

    // Eigen::Matrix3d shearTransform;
    // shearTransform <<   1.0, 1.5, 0.0,
    //                     0.0, 1.5, 0.0,
    //                     0.0, 1.5, 1.0;

    // x = (X * rotateX.transpose());
    // x = (x * shearTransform.transpose());

    // Compute deformation gradients (since x = X rn, F should be the identity)
    // Also compute the NeoHookean energies and the first Piola-Kirchoff stress tensors
    std::vector<double> J(T.rows());
    std::vector<double> energy(T.rows());
    std::vector<Eigen::Matrix3d> P(T.rows());
    Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(x.rows(), 3);
    for (int e = 0; e < T.rows(); ++e) {
        Eigen::Matrix3d Fe = computeDeformationGradient(e, x, T, DmInv);
        addTetraForces(e, x, T, gradN, initialVols, Fe, material, forces);
        
        J[e] = Fe.determinant();
        energy[e] = material.energyDensity(Fe);
        P[e] = material.firstPiolaStressTensor(Fe);
    }

    // Register mesh in viewer
    auto* mesh = polyscope::registerTetMesh("mesh", x, T);
    mesh->addCellScalarQuantity("J = det(F)", J)->setEnabled(true);
    mesh->addCellScalarQuantity("Strain energy density", energy)->setEnabled(true);
    mesh->setTransparency(0.75);

    auto* forceQ = mesh->addVertexVectorQuantity("Internal forces", forces);
    forceQ->setEnabled(true);
    forceQ->setVectorLengthScale(0.15);
    forceQ->setVectorRadius(0.004);

    // Display
    polyscope::show();
    return 0;
}
