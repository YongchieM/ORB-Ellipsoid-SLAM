#ifndef OBJECTINITIALIZER_HPP
#define OBJECTINITIALIZER_HPP

#include "g2o_Object.h"
#include <opencv2/opencv.hpp>

using namespace Eigen;
using namespace g2o;

extern MatrixXd coeffs;
extern double centre_z;

Matrix<double,9,1> initializeWithCuboid(g2o::SE3Quat campose_cw, Eigen::Vector4d& bbox_value, Eigen::VectorXd& cube_para, Eigen::Matrix3d& calib);
Matrix<double,9,1> initializeWithCentre(Vector3d& obj_size, double height, g2o::SE3Quat campose_cw, Eigen::Vector4d& bbox_value, Eigen::Matrix3d& calib);

//
MatrixXd fromDetectionsToLines(VectorXd &detection_mat);
Matrix3Xd generateProjectionMatrix(const SE3Quat& campose_cw, const Matrix3d& calib);
MatrixXd getVectorFromPlanesHomo(MatrixXd &planes);



#endif
