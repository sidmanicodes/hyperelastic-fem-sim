#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <array>
#include <Eigen/Dense>
#include <polyscope/polyscope.h>
#include <polyscope/volume_mesh.h>
#include <polyscope/options.h>
#include <polyscope/view.h>
#include <igl/readMESH.h>
#include "Material.hpp"

const double pi = M_PI;                 //  π
const double g = 9.81;                  //  Gravitational constant
const double eps = 1e-6;                //  Used for approximating the directional derivative of the elastic force
const double beta = 1e-3;               //  Constant of proportionality for Rayleigh damping
const double mu_static = 0.6;           //  Static coeff of friction for wood (https://en.wikipedia.org/wiki/Friction#Approximate_coefficients_of_friction)
const double mu_kinetic = 0.4;          //  Kinetic coeff of friction for wood (https://en.wikipedia.org/wiki/Friction#Approximate_coefficients_of_friction)
const double restitution = 0.01;        //  Restitution vector for normal velocity after ground contact (assuming horizontal plane)
const double youngsModulus = 500.0;     //  Young's modulus of NeoHookean material
const double poissonRatio = 0.3;        //  Poisson ratio of NeoHookean material
const double height = 3.0;              //  Height from which object is dropped

void addGravityForce(int i, Eigen::VectorXd& mass, Eigen::MatrixXd& forces) {
    Eigen::Vector3d gravity(0.0, mass(i) * -g, 0.0);

    forces.row(i) += gravity.transpose();
}

void symplecticEuler(
    int i, 
    double dt, 
    Eigen::MatrixXd& x,
    Eigen::MatrixXd& X,
    Eigen::MatrixXd& v,
    Eigen::MatrixXd& forces,
    Eigen::VectorXd& mass
)
{    
    // Compute acceleration of ith vertex
    Eigen::Vector3d a = forces.row(i) / mass(i);

    // Update velocity
    v.row(i) += dt * a;

    // Update position
    x.row(i) += dt * v.row(i);

    // Account for ground collision
    if (x(i, 1) < 0.0) {
        x(i, 1) = 0.0;

        // Bounce vertex back up at a fraction of its velocity            
        if (v(i, 1) < 0.0) {
            v(i, 1) *= -restitution;
        }
    }
}

void addElasticForce(
    int e,
    const Eigen::MatrixXd& x,
    const Eigen::MatrixXi& T,
    const std::vector<std::array<Eigen::Vector3d, 4>>& gradN,
    const std::vector<double>& initialVols,
    const Eigen::Matrix3d F,
    const FEMSim::NeoHookeanMaterial& material,
    const Eigen::Matrix3d& P,
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

    forces.row(i_0) += (-V * P * gradN_e[0]).transpose();
    forces.row(i_1) += (-V * P * gradN_e[1]).transpose();
    forces.row(i_2) += (-V * P * gradN_e[2]).transpose();
    forces.row(i_3) += (-V * P * gradN_e[3]).transpose();
}

Eigen::Matrix3d computeDeformationGradient(
    int e,
    const Eigen::MatrixXd& x,
    const Eigen::MatrixXi& T,
    const std::vector<Eigen::Matrix3d>& DmInv
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

void applyFriction(
    int i,
    double dt,
    const Eigen::VectorXd& mass,
    const Eigen::MatrixXd& x,
    Eigen::MatrixXd& v,
    Eigen::MatrixXd& forces
)
{
    // Form tangential velocity vector and get its norm
    Eigen::Vector3d vt(v(i, 0), 0.0, v(i, 2));
    double vtNorm = vt.norm();
    
    /*
    We only want to apply friction upon contact (i.e. the vertex must
    be within eps of the ground)
    */
    if (std::abs(x(i, 1)) < eps) {
        /*
        Keeping with our assumption that the ground is level, the normal
        component of velocity is just the y-component
        */
        double vnNorm = v(i, 1);
        double JnNorm;

        /*
        If the normal component of velocity is pointing down and is sufficiently
        large in magnitude, the object hasn't stopped bouncing; thus the normal
        impulse we must apply to go from v to -e*v (where e is the restitution factor)
        can be derived to be the first equation

        Otherwise, the normal impulse is simply the integral of the normal force (which
        is equal to a vector of opposite magnitude of the sum of net forces in the 
        y-direction)
        */
        if (vnNorm < -eps) {
            JnNorm = mass(i) * (1 + restitution) * (-vnNorm);
        }
        else {
            JnNorm = std::max(0.0, -forces(i, 1)) * dt;
        }
        
        /*
        The impulse required to stop any tangential velocity (i.e. go from
        vt -> 0)
        */
        Eigen::Vector3d Jstop = -mass(i) * vt;

        /*
        If the impulse required to stop any tangential motion is less than
        the static friction impulse, we can safely set the tangential velocity to 0

        Otherwise, we generate an impulse with the magnitude of the kinetic friction 
        force in the direction of the tangent velocity. Since impulse is equivalently
        characterized as the change in momentum, dividing by the vertex mass give the
        resultant change in velocity. Adding this to the current velocity gives us our
        new velocity
        */
        if (Jstop.norm() <= mu_static * JnNorm) {
            v(i, 0) = 0.0;
            v(i, 2) = 0.0;
        }
        else {
            Eigen::Vector3d Jkin = -mu_kinetic * JnNorm * (vt / vtNorm);
            v.row(i) += (Jkin / mass(i)).transpose();
        }
    }
}

void step(
    double dt,
    Eigen::MatrixXd& x,
    Eigen::MatrixXd& X,
    Eigen::MatrixXd& v,
    Eigen::VectorXd& mass,
    const Eigen::MatrixXi& T,
    const std::vector<std::array<Eigen::Vector3d, 4>>& gradN,
    const std::vector<Eigen::Matrix3d>& DmInv,
    const FEMSim::NeoHookeanMaterial& material,
    const std::vector<double>& initialVols,
    Eigen::MatrixXd& netForces,
    std::vector<double>& J,
    std::vector<double>& energy,
    std::vector<Eigen::Matrix3d>& P
) 
{
    // Recompute forces for each substep, retaining them for visualization.
    netForces.setZero();
    Eigen::MatrixXd elasticForce = Eigen::MatrixXd::Zero(x.rows(), 3);
    
    // Perturb the spatial vector; this will be used to approximate
    // the Rayleigh damping
    Eigen::MatrixXd xPerturbed = x + eps * v;
    Eigen::MatrixXd elasticForcePerturbed = Eigen::MatrixXd::Zero(x.rows(), 3);

    for (int e = 0; e < T.rows(); ++e) {
        Eigen::Matrix3d Fe = computeDeformationGradient(e, x, T, DmInv);
        Eigen::Matrix3d FePerturbed = computeDeformationGradient(e, xPerturbed, T, DmInv);
        J[e] = Fe.determinant();
        energy[e] = material.energyDensity(Fe);
        P[e] = material.firstPiolaStressTensor(Fe);
        Eigen::Matrix3d P_Perturbed = material.firstPiolaStressTensor(FePerturbed);
        
        // Compute elastic forces for the original spatial vector and the perturbed one
        addElasticForce(e, x, T, gradN, initialVols, Fe, material, P[e], elasticForce);
        addElasticForce(e, xPerturbed, T, gradN, initialVols, FePerturbed, material, P_Perturbed, elasticForcePerturbed);
    }

    // Compute Rayleigh damping
    Eigen::MatrixXd rayleighDamping = beta * ((elasticForcePerturbed - elasticForce) / eps);

    // Sum up net forces
    netForces += elasticForce + rayleighDamping;

    for (int i = 0; i < x.rows(); ++i) {
        // Add gravitational forces to each vertex
        addGravityForce(i, mass, netForces);

        // Run Symplectic Euler
        symplecticEuler(i, dt, x, X, v, netForces, mass);

        // Apply friction to reduce tangential speed of the object
        applyFriction(i, dt, mass, x, v, netForces);
    }
}

int main() {
    // Initialize viewer
    polyscope::init();

    polyscope::options::groundPlaneHeightMode =
    polyscope::GroundPlaneHeightMode::Manual;
    polyscope::options::groundPlaneHeight = 0.0f;

    // Read in vertex locations, tetra indices, and faces
    Eigen::MatrixXd X;
    Eigen::MatrixXi T;
    Eigen::MatrixXi F;

    igl::readMESH("../data/cube86.mesh", X, T, F);

    // Scale reference config to have coordinates be in [-1, 1]
    X /= std::abs(X.maxCoeff());

    /*
    Compute D_m^{-1} for each tetra (since these are in the material config, 
    they won't change; so we compute them before computing each deformation gradient)
    */
    std::vector<Eigen::Matrix3d> DmInv(T.rows());

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

    /*
    In order to calculate the velocities of each mesh vertex, we need
    to solve for the acceleration, which is accomplished by a_i = M^{-1}f_i,
    where M is the mass matrix. For simplicity, we use the lumped mass matrix
    method, which is a block diagonal matrix (stored as a vector for efficiency)
    where the ith block is given by m_iI, where I is the 3x3 identity and m_i
    is the mass of the ith vertex
    */
    Eigen::VectorXd mass = Eigen::VectorXd::Zero(X.rows());
    double initial_density = 1.0;

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

        /*
        Use the volume to compute the mass of each vertex
        The mass of vertex i is the sum of contributions from
        each tetrahedron that shares the vertex, with the contribution
        being mass(tetra_e) / 4.0
        */
        double tet_mass = initialVols[e] * initial_density;
        for (int a = 0; a < 4; ++a) {
            int i = T(e, a);    // get the ith vertex number
            mass[i] += tet_mass / 4.0;
        }

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

    // Fix top of mesh to prevent falling through space
    double yMax = X.col(1).maxCoeff();

    // For now, spatial config is equal to material config
    Eigen::MatrixXd x = X;

    x.col(1).array() += height;

    /*
    Create a NeoHookean material with Young's modulus 1000 Pa
    and Poisson ratio 0.3 (hyperelastic)
    */
    FEMSim::NeoHookeanMaterial material(youngsModulus, poissonRatio);

    // Compute deformation gradients (since x = X rn, F should be the identity)
    // Also compute the NeoHookean energies and the first Piola-Kirchoff stress tensors
    std::vector<double> J(T.rows());
    std::vector<double> energy(T.rows());
    std::vector<Eigen::Matrix3d> P(T.rows());
    Eigen::MatrixXd forces = Eigen::MatrixXd::Zero(x.rows(), 3);
    Eigen::MatrixXd v = Eigen::MatrixXd::Zero(x.rows(), 3);
    
    // Register mesh in viewer
    auto* mesh = polyscope::registerTetMesh("mesh", x, T);

    // Frame the cube and the ground at y = 0 together.
    polyscope::view::lookAt(
        glm::vec3(6.0f, 4.0f, 8.0f),
        glm::vec3(0.5f, 2.0f, 0.5f)
    );

    constexpr double dt = 1e-4;
    constexpr int substepsPerFrame = (1 / dt) / 60.0;

    // Run animation in polyscope
    polyscope::state::userCallback = [&]() {
        for (int k = 0; k < substepsPerFrame; ++k) {
            step(
                dt,
                x,
                X,
                v,
                mass,
                T,
                gradN,
                DmInv,
                material,
                initialVols,
                forces,
                J,
                energy,
                P 
            );
        }
        mesh->updateVertexPositions(x);
        // mesh->addCellScalarQuantity("J = det(F)", J)->setEnabled(true);
        // mesh->addCellScalarQuantity("Strain energy density", energy)->setEnabled(true);
        // mesh->setTransparency(0.75);
    
        // auto* forceQ = mesh->addVertexVectorQuantity("Total forces", forces);
        // forceQ->setEnabled(true);
        // forceQ->setVectorLengthScale(0.15);
        // forceQ->setVectorRadius(0.004);
    };

    // Display
    polyscope::show();
    return 0;
}
