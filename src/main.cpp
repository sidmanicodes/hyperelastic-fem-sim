#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <polyscope/polyscope.h>
#include <polyscope/volume_mesh.h>
#include <igl/readMESH.h>

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

        DmInv[e] = Dm.inverse();
    }

    // For now, spatial config is equal to material config
    Eigen::MatrixXd x = X;

    double theta = pi/4;
    Eigen::Matrix3d rotateX;
    rotateX << 1.0, 0.0, 0.0,
               0.0, std::cos(theta), -std::sin(theta),
               0.0, std::sin(theta), std::cos(theta);

    Eigen::Matrix3d shearTransform;
    shearTransform <<   1.0, 1.5, 0.0,
                        0.0, 1.5, 0.0,
                        0.0, 1.5, 1.0;

    x = (X * rotateX.transpose());
    x = (x * shearTransform.transpose());

    // Compute deformation gradient (since x = X rn, F should be the identity)
    std::vector<double> J(T.rows());
    for (int e = 0; e < T.rows(); ++e) {
        Eigen::Matrix3d Fe = computeDeformationGradient(e, x, T, DmInv);
        
        J[e] = Fe.determinant();
    }

    // Register mesh in viewer
    auto* mesh = polyscope::registerTetMesh("mesh", x, T);
    mesh->addCellScalarQuantity("J = det(F)", J)->setEnabled(true);

    // Display
    polyscope::show();
    return 0;
}
