/**
* This file is part of CubeSLAM
*
* Copyright (C) 2018  Shichao Yang (Carnegie Mellon Univ)
*/
#pragma once

#include "Thirdparty/g2o/g2o/core/base_multi_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/detect_3d_cuboid/include/matrix_utils.h"
// #include "detect_3d_cuboid/matrix_utils.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <math.h>
#include <algorithm> // std::swap

typedef Eigen::Matrix<double, 3, 3> Matrix3d;
typedef Eigen::Matrix<double, 4, 4> Matrix4d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;

typedef Eigen::Matrix<double, 2, 1> Vector2d;
typedef Eigen::Matrix<double, 3, 1> Vector3d;
typedef Eigen::Matrix<double, 4, 1> Vector4d;
typedef Eigen::Matrix<double, 5, 1> Vector5d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 7, 1> Vector7d;
typedef Eigen::Matrix<double, 8, 1> Vector8d;
typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 10, 1> Vector10d;


namespace g2o
{

using namespace Eigen;

// vehicle planar velocity  2Dof   [linear_velocity, steer_angle]
class VelocityPlanarVelocity : public BaseVertex<2, Vector2d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VelocityPlanarVelocity(){};

    virtual void setToOriginImpl()
    {
        _estimate.fill(0.);
    }

    virtual bool read(std::istream &is) { return true; };
    virtual bool write(std::ostream &os) const { return os.good(); };

    virtual void oplusImpl(const double *update)
    {
        Eigen::Map<const Vector2d> v(update);
        _estimate += v;
    }
};

// a local point in object frame. want it to lie inside cuboid. only point    object dimension is fixed. basically project the point onto surface
class UnaryLocalPoint : public BaseUnaryEdge<3, Vector3d, VertexSBAPointXYZ>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    UnaryLocalPoint(){};

    virtual bool read(std::istream &is) { return true; };
    virtual bool write(std::ostream &os) const { return os.good(); };

    void computeError();

    Vector3d objectscale;			 // fixed object dimensions
    double max_outside_margin_ratio; // truncate the error if point is too far from object
};

/*****************************************************/
// ellipsoid-slam
class ellipsoid
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SE3Quat pose; //in world coordinate
    Vector3d scale;

    Vector9d vec_minimal; //x,y,z,roll,pitch,yaw,a,b,c

    int miLabel;
    int miInstanceID;

    ellipsoid()
    {
        pose = SE3Quat();
        scale.setZero();
    }

    // xyz roll pitch yaw half_scale
    inline void fromMinimalVector(const Vector9d &v)
    {
        Eigen::Quaterniond posequat = zyx_euler_to_quat(v(3), v(4), v(5));
        pose = SE3Quat(posequat, v.head<3>());
        scale = v.tail<3>();
    }

    // xyz quaternion, half_scale
    inline void fromVector(const Vector10d &v)
    {
        pose.fromVector(v.head<7>());
        scale = v.tail<3>();
    }

    inline const Vector3d &translation() const { return pose.translation(); }
    inline void setTranslation(const Vector3d &t_) { pose.setTranslation(t_); }
    inline void setRotation(const Quaterniond &r_) { pose.setRotation(r_); }
    inline void setRotation(const Matrix3d &R) { pose.setRotation(Quaterniond(R)); }
    inline void setRotation(const Eigen::Vector3d &rpy)
    {
        Eigen::Matrix3d r = euler_zyx_to_rot(rpy(0),rpy(1),rpy(2));
        pose.setRotation(Quaterniond(r));
    }

    inline void setScale(const Vector3d &scale_) { scale = scale_; }


    ellipsoid(const g2o::ellipsoid &e)
    {
        pose = e.pose;
        scale = e.scale;
        vec_minimal = e.vec_minimal;
        updateValueFrom(e);
    }

    const ellipsoid& operator=(const g2o::ellipsoid &e)
    {
        pose = e.pose;
        scale = e.scale;
        vec_minimal = e.vec_minimal;

        updateValueFrom(e);
        return e;
    }


    // apply update to current cuboid. exponential map
    ellipsoid exp_update(const Vector9d &update) // apply update to current cuboid
    {
        ellipsoid res;
        res.pose = this->pose * SE3Quat::exp(update.head<6>()); // NOTE bug before. switch position
        res.scale = this->scale + update.tail<3>();
        return res;
    }

    // actual error between two cuboids.
    Vector9d ellipsoid_log_error(const ellipsoid &newone) const
    {
        Vector9d res;
        SE3Quat pose_diff = newone.pose.inverse() * this->pose;
        res.head<6>() = pose_diff.log(); //treat as se3 log error. could also just use yaw error
        res.tail<3>() = this->scale - newone.scale;
        return res;
    }


    // function called by g2o.
    Vector9d min_log_error(const ellipsoid &newone, bool print_details = false) const
    {
        bool whether_rotate_ellipsoid = true; // whether rotate ellipsoid to find smallest error
        if (!whether_rotate_ellipsoid)
            return ellipsoid_log_error(newone);

        // NOTE rotating ellipsoid... since we cannot determine the front face consistenly, different front faces indicate different yaw, scale representation.
        // need to rotate all 360 degrees (global ellipsoid might be quite different from local ellipsoid)
            // this requires the sequential object insertion. In this case, object yaw practically should not change much. If we observe a jump, we can use code
        // here to adjust the yaw.
        Vector4d rotate_errors_norm;
        Vector4d rotate_angles(-1, 0, 1, 2); // rotate -90 0 90 180
        Eigen::Matrix<double, 9, 4> rotate_errors;
        for (int i = 0; i < rotate_errors_norm.rows(); i++)
        {
            ellipsoid rotated_ellipsoid = newone.rotate_ellipsoid(rotate_angles(i) * M_PI / 2.0); // rotate new ellipsoids
            Vector9d ellipsoid_error = this->ellipsoid_log_error(rotated_ellipsoid);
            rotate_errors_norm(i) = ellipsoid_error.norm();
            rotate_errors.col(i) = ellipsoid_error;
        }
        int min_label;
        rotate_errors_norm.minCoeff(&min_label);
        if (print_details)
            if (min_label != 1)
                std::cout << "Rotate ellipsoid   " << min_label << std::endl;
        return rotate_errors.col(min_label);
    }

    // change front face by rotate along current body z axis. another way of representing ellipsoid. representing same ellipsoid (IOU always 1)
    ellipsoid rotate_ellipsoid(double yaw_angle) const // to deal with different front surface of cuboids
    {
        ellipsoid res;
        SE3Quat rot(Eigen::Quaterniond(cos(yaw_angle * 0.5), 0, 0, sin(yaw_angle * 0.5)), Vector3d(0, 0, 0)); // change yaw to rotation.
        res.pose = this->pose * rot;
        res.scale = this->scale;
        if ((yaw_angle == M_PI / 2.0) || (yaw_angle == -M_PI / 2.0) || (yaw_angle == 3 * M_PI / 2.0))
            std::swap(res.scale(0), res.scale(1));
        return res;
    }


    // transform a local ellipsoid to global ellipsoid  Twc is camera pose. from camera to world
    ellipsoid transform_from(const SE3Quat &Twc) const
    {
        ellipsoid res;
        res.pose = Twc * this->pose;
        res.scale = this->scale;
        return res;
    }

    // transform a global ellispoid to local ellipsoid  Twc is camera pose. from camera to world
    ellipsoid transform_to(const SE3Quat &Twc) const
    {
        ellipsoid res;
        res.pose = Twc.inverse() * this->pose;
        res.scale = this->scale;
        return res;
    }

    // xyz roll pitch yaw half_scale
    inline Vector9d toMinimalVector() const
    {
        Vector9d v;
        v.head<6>() = pose.toXYZPRYVector();
        v.tail<3>() = scale;
        return v;
    }

    // xyz quaternion, half_scale
    inline Vector10d toVector() const
    {
        Vector10d v;
        v.head<7>() = pose.toVector();
        v.tail<3>() = scale;
        return v;
    }

    Matrix4d similarityTransform() const
    {
        Matrix4d res = pose.to_homogeneous_matrix();
        Matrix3d scale_mat = scale.asDiagonal();
        res.topLeftCorner<3, 3>() = res.topLeftCorner<3, 3>() * scale_mat;
        return res;
    }

    // this*inv(other)
    ellipsoid timesInverse(const ellipsoid &other) const
    {
        Matrix4d current_homomat = similarityTransform();
        Matrix4d other_homomat = other.similarityTransform();
        Matrix4d result_homomat = current_homomat * other_homomat.inverse(); // [RS, t]
        Matrix3d result_rot = pose.rotation().toRotationMatrix() * other.pose.rotation().toRotationMatrix().inverse();

        ellipsoid res;
        res.setTranslation(result_homomat.col(3).head<3>());
        res.setScale(scale.array() / other.scale.array());
        res.setRotation(result_rot);

        return res;
    }

    //get the projected point of ellipsoid center on image plane
    Vector2d projectCenterIntoImagePoint(const SE3Quat& campose_cw, const Matrix3d& Kalib)
    {
        Eigen::Matrix3Xd  P = generateProjectionMatrix(campose_cw, Kalib);

        Vector3d center_pos = pose.translation();
        Vector4d center_homo = real_to_homo_coord<double>(center_pos);
        Vector3d u_homo = P * center_homo;
        Vector2d u = homo_to_real_coord_vec<double>(u_homo);

        return u;
    }

    //get the projected ellipse
    Vector5d projectOntoImageEllipse(const SE3Quat& campose_cw, const Matrix3d& Kalib) const
    {
        Matrix4d Q_star = generateQuadric();
        Eigen::Matrix3Xd  P = generateProjectionMatrix(campose_cw, Kalib);
        Matrix3d C_star = P * Q_star * P.transpose();
        Matrix3d C = C_star.inverse();
        C = C / C(2,2); // normalize

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(C);    // ascending sort by default
        Eigen::VectorXd eigens = es.eigenvalues();

        // If it is an ellipse, the sign of eigen values must be :  1 1 -1
        // Ref book : Multiple View Geometry in Computer Vision
        int num_pos = int(eigens(0)>0) +int(eigens(1)>0) +int(eigens(2)>0);
        int num_neg = int(eigens(0)<0) +int(eigens(1)<0) +int(eigens(2)<0);

        // matrix to equation coefficients: ax^2+bxy+cy^2+dx+ey+f=0
        double a = C(0,0);
        double b = C(0,1)*2;
        double c = C(1,1);
        double d = C(0,2)*2;
        double e = C(2,1)*2;
        double f = C(2,2);

        // get x_c, y_c, theta, axis1, axis2 from coefficients
        double delta = c*c - 4.0*a*b;
        double k = (a*f-e*e/4.0) - pow((2*a*e-c*d),2)/(4*(4*a*b-c*c));
        double theta = 1/2.0*atan2(b,(a-c));
        double x_c = (b*e-2*c*d)/(4*a*c-b*b);
        double y_c = (b*d-2*a*e)/(4*a*c-b*b);
        double a_2 =  2*(a* x_c*x_c+ c * y_c*y_c+ b *x_c*y_c -1) / (a + c + sqrt((a-c)*(a-c)+b*b));
        double b_2 =  2*(a*x_c*x_c+c*y_c*y_c+b*x_c*y_c -1) / (a + c - sqrt((a-c)*(a-c)+b*b));

        double axis1= sqrt(a_2);
        double axis2= sqrt(b_2);

        Vector5d output;
        output << x_c, y_c, theta, axis1, axis2;

        //std::cout<<x_c<<" "<<y_c<<" "<<theta<<" "<<axis1<<" "<<axis2<<std::endl;

        return output;
    }

    //get bbx from ellipse in image plane
    Vector4d getBoundingBoxFromEllipse(Vector5d& ellipse) const
    {
        double a = ellipse[3];
        double b = ellipse[4];
        double theta = ellipse[2];
        double x = ellipse[0];
        double y = ellipse[1];

        double cos_theta_2 = cos(theta)*cos(theta);
        double sin_theta_2 = 1- cos_theta_2;

        //std::cout<<cos_theta_2<<" "<<sin_theta_2<<std::endl;

        double x_limit = sqrt(a*a*cos_theta_2+b*b*sin_theta_2);
        double y_limit = sqrt(a*a*sin_theta_2+b*b*cos_theta_2);

        Vector4d output;
        //output[0] = x-x_limit; // left up
        //output[1] = y-y_limit;
        //output[2] = x+x_limit; // right down
        //output[3] = y+y_limit;

        output[0] = x-x_limit;
        output[1] = y-y_limit;
        output[2] = 2*x_limit;
        output[3] = 2*y_limit;
        return output;
    }

    Vector4d getBoundingBoxFromProjection(const SE3Quat& campose_cw, const Matrix3d& Kalib) const
    {
        Vector5d ellipse = projectOntoImageEllipse(campose_cw, Kalib);
        Vector4d bbox = getBoundingBoxFromEllipse(ellipse);
        return bbox; //topleft, width, height
    }

    //get projection matrix
    Eigen::Matrix3Xd generateProjectionMatrix(const SE3Quat& campose_cw, const Matrix3d& Kalib) const
    {
        Eigen::Matrix3Xd identity_lefttop;
        identity_lefttop.resize(3, 4);
        identity_lefttop.col(3)=Vector3d(0,0,0);
        identity_lefttop.topLeftCorner<3,3>() = Matrix3d::Identity(3,3);

        Eigen::Matrix3Xd proj_mat = Kalib * identity_lefttop;
        proj_mat = proj_mat * campose_cw.to_homogeneous_matrix();

        return proj_mat;
    }
    //get Q*
    Matrix4d generateQuadric() const
    {
        Vector4d axisVec;
        axisVec << 1/(scale[0]*scale[0]), 1/(scale[1]*scale[1]), 1/(scale[2]*scale[2]), -1;
        Matrix4d Q_c = axisVec.asDiagonal();
        Matrix4d Q_c_star = Q_c.inverse();
        Matrix4d Q_pose_matrix = pose.to_homogeneous_matrix();   // Twm  model in world,  world to model
        Matrix4d Q_c_star_trans = Q_pose_matrix * Q_c_star * Q_pose_matrix.transpose();

        return Q_c_star_trans;
    }


    /**************************/
    //for 3d bbox of ellipsoid
    // 8 corners 3*8 matrix, each col is x y z of a corner
    Matrix3Xd compute3D_BoxCorner() const
    {
        Matrix3Xd corners_body;
        corners_body.resize(3, 8);
        corners_body << 1, 1, -1, -1, 1, 1, -1, -1,
                1, -1, -1, 1, 1, -1, -1, 1,
                -1, -1, -1, -1, 1, 1, 1, 1;
        Matrix3Xd corners_world = homo_to_real_coord<double>(similarityTransform() * real_to_homo_coord<double>(corners_body));
        return corners_world;
    }

    // project corners onto image to get 8 points.  cam pose: world to cam
    Matrix2Xd projectOntoImage(const SE3Quat &campose_cw, const Matrix3d &Kalib) const
    {
        Matrix3Xd corners_3d_world = compute3D_BoxCorner();
        Matrix2Xd corner_2d = homo_to_real_coord<double>(Kalib * homo_to_real_coord<double>(campose_cw.to_homogeneous_matrix() * real_to_homo_coord<double>(corners_3d_world)));
        return corner_2d;
    }

    // get rectangles after projection  [topleft, bottomright]
    Vector4d projectOntoImageRect(const SE3Quat &campose_cw, const Matrix3d &Kalib) const
    {
        Matrix3Xd corners_3d_world = compute3D_BoxCorner();
        Matrix2Xd corner_2d = homo_to_real_coord<double>(Kalib * homo_to_real_coord<double>(campose_cw.to_homogeneous_matrix() * real_to_homo_coord<double>(corners_3d_world)));
        Vector2d bottomright = corner_2d.rowwise().maxCoeff(); // x y
        Vector2d topleft = corner_2d.rowwise().minCoeff();
        return Vector4d(topleft(0), topleft(1), bottomright(0), bottomright(1));
    }

    // get rectangles after projection  [center, width, height]
    Vector4d projectOntoImageBbox(const SE3Quat &campose_cw, const Matrix3d &Kalib) const
    {
        Vector4d rect_project = projectOntoImageRect(campose_cw, Kalib); // top_left, bottom_right  x1 y1 x2 y2
        Vector2d rect_center = (rect_project.tail<2>() + rect_project.head<2>()) / 2;
        Vector2d widthheight = rect_project.tail<2>() - rect_project.head<2>();
        return Vector4d(rect_center(0), rect_center(1), widthheight(0), widthheight(1));
    }

    // compute point surface error on the object
    Vector3d point_boundary_error(const Vector3d &point, const double max_outside_margin_ratio, double point_scale = 1) const;
    double point_ellipsoid_error(const Vector3d &point, const double max_outside_margin_ratio, double point_scale = 1) const;

    void setColor(const Vector3d& color, double alpha=1.0);
    Vector3d getColor();
    Vector4d getColorWithAlpha();
    bool isColorSet();

    bool checkObservability(const SE3Quat& campose_cw);

private:
    bool mbColor;
    Vector4d mvColor;

    void updateValueFrom(const g2o::ellipsoid& e)
    {
        this->miLabel = e.miLabel;
        this->mbColor = e.mbColor;
        this->mvColor = e.mvColor;
        this->miInstanceID = e.miInstanceID;
    }
};

class VertexEllipsoid : public BaseVertex<9, ellipsoid> // NOTE  this vertex stores object pose to world
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexEllipsoid()
    {
        fixedscale.setZero();
        whether_fixrollpitch = false;
        whether_fixrotation = false;
        whether_fixheight = false;
    };

    virtual void setToOriginImpl()
    {
        _estimate = ellipsoid();
        if (fixedscale(0) > 0)
            _estimate.scale = fixedscale;
    }

    virtual void oplusImpl(const double *update_);

    virtual bool read(std::istream &is) { return true; };
    virtual bool write(std::ostream &os) const { return os.good(); };

    // some mode parameters. a more efficient way is to create separate vertex
    Vector3d fixedscale;	   // if want to fix scale, set it to be true value
    bool whether_fixrollpitch; // for ground object, only update yaw
    bool whether_fixrotation;  // don't update any rotation
    bool whether_fixheight;	// object height is fixed
};

class VertexEllipsoidFixScale : public BaseVertex<6, ellipsoid> // less variables. should be faster  fixed scale should be set
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexEllipsoidFixScale()
    {
        fixedscale.setZero();
        whether_fixrollpitch = false;
        whether_fixrotation = false;
        whether_fixheight = false;
    };

    virtual void setToOriginImpl()
    {
        _estimate = ellipsoid();
        if (fixedscale(0) > 0)
            _estimate.scale = fixedscale;
    }

    virtual void oplusImpl(const double *update_);

    virtual bool read(std::istream &is) { return true; };
    virtual bool write(std::ostream &os) const { return os.good(); };

    Vector3d fixedscale;
    bool whether_fixrollpitch;
    bool whether_fixrotation;
    bool whether_fixheight;
};


// camera -object 2D projection error, rectangle difference, could also change to iou
class EdgeSE3EllipsoidProj : public BaseBinaryEdge<4, Vector4d, VertexSE3Expmap, VertexEllipsoid>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE3EllipsoidProj(){};

    virtual bool read(std::istream &is) { return true; };
    virtual bool write(std::ostream &os) const { return os.good(); };

    void computeError();
    double get_error_norm();
    Matrix3d Kalib;
};

// camera -fixscale_object 2D projection error, rectangle, could also change to iou
class EdgeSE3EllipsoidFixScaleProj : public BaseBinaryEdge<4, Vector4d, VertexSE3Expmap, VertexEllipsoidFixScale>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE3EllipsoidFixScaleProj(){};

    virtual bool read(std::istream &is) { return true; };
    virtual bool write(std::ostream &os) const { return os.good(); };

    void computeError();
    double get_error_norm();
    Matrix3d Kalib;
};

// camera -object 3D error
class EdgeSE3Ellipsoid : public BaseBinaryEdge<9, ellipsoid, VertexSE3Expmap, VertexEllipsoid>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE3Ellipsoid(){};

    virtual bool read(std::istream &is) { return true; };
    virtual bool write(std::ostream &os) const { return os.good(); };

    void computeError()
    {
        const VertexSE3Expmap *SE3Vertex = static_cast<const VertexSE3Expmap *>(_vertices[0]); //  world to camera pose
        const VertexEllipsoid *ellipsoidVertex = static_cast<const VertexEllipsoid *>(_vertices[1]);	//  object pose to world

        SE3Quat cam_pose_Twc = SE3Vertex->estimate().inverse();
        ellipsoid global_ellipsoid = ellipsoidVertex->estimate();
        ellipsoid esti_global_ellipsoid = _measurement.transform_from(cam_pose_Twc);
        _error = global_ellipsoid.min_log_error(esti_global_ellipsoid);
    }

    double get_error_norm() //for debug
    {
        const VertexSE3Expmap *SE3Vertex = static_cast<const VertexSE3Expmap *>(_vertices[0]);
        const VertexEllipsoid *ellipsoidVertex = static_cast<const VertexEllipsoid *>(_vertices[1]);

        SE3Quat cam_pose_Twc = SE3Vertex->estimate().inverse();
        ellipsoid global_ellipsoid = ellipsoidVertex->estimate();
        ellipsoid esti_global_ellipsoid = _measurement.transform_from(cam_pose_Twc);
        return global_ellipsoid.min_log_error(esti_global_ellipsoid).norm();
    }
};

// camera-ellipsoid 3d error, only object    camera fixed and provided
class EdgeSE3EllipsoidOnlyObject : public BaseUnaryEdge<9, ellipsoid, VertexEllipsoid>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3EllipsoidOnlyObject() {}

    virtual bool read(std::istream &is) { return true; };
    virtual bool write(std::ostream &os) const { return os.good(); };

    void computeError()
    {
        const VertexEllipsoid *ellipsoidVertex = static_cast<const VertexEllipsoid *>(_vertices[0]); //  object pose to world

        ellipsoid global_ellipsoid = ellipsoidVertex->estimate();
        ellipsoid esti_global_ellipsoid = _measurement.transform_from(cam_pose_Twc);
        _error = global_ellipsoid.min_log_error(esti_global_ellipsoid);
    }

    double get_error_norm()
    {
        const VertexEllipsoid *ellipsoidVertex = static_cast<const VertexEllipsoid *>(_vertices[0]);

        ellipsoid global_ellipsoid = ellipsoidVertex->estimate();
        ellipsoid esti_global_ellipsoid = _measurement.transform_from(cam_pose_Twc);
        return global_ellipsoid.min_log_error(esti_global_ellipsoid).norm();
    }

    SE3Quat cam_pose_Twc; // provided fixed camera pose
};


// object point surface error, both will optimize
class EdgePointEllipsoid : public BaseBinaryEdge<3, Vector3d, VertexSBAPointXYZ, VertexEllipsoid>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgePointEllipsoid(){};

    virtual bool read(std::istream &is) { return true; };
    virtual bool write(std::ostream &os) const { return os.good(); };

    void computeError();

    double max_outside_margin_ratio; // truncate the error if point is too far from object
};

// object point surface error, both will optimize.   object has fixed size.
class EdgePointEllipsoidFixScale : public BaseBinaryEdge<3, Vector3d, VertexSBAPointXYZ, VertexEllipsoidFixScale>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgePointEllipsoidFixScale(){};

    virtual bool read(std::istream &is) { return true; };
    virtual bool write(std::ostream &os) const { return os.good(); };

    void computeError();

    double max_outside_margin_ratio; // truncate the error if point is too far from object
};

// one object connected with all fixed points. only optimize object.  want object to contain points
class EdgePointEllipsoidOnlyObject : public BaseUnaryEdge<3, Vector3d, VertexEllipsoid>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgePointEllipsoidOnlyObject()
    {
        print_details = false;
        prior_object_half_size.setZero();
    };

    virtual bool read(std::istream &is) { return true; };
    virtual bool write(std::ostream &os) const { return os.good(); };

    bool print_details;

    void computeError();
    Vector3d computeError_debug();

    std::vector<Vector3d> object_points; // all the fixed points.
    double max_outside_margin_ratio;	 // truncate the error if point is too far from object
    Vector3d prior_object_half_size;	 // give a prior object size, otherwise a huge cuboid can always contain all points
};

// one object connected with all fixed points   similar as above
class EdgePointEllipsoidOnlyObjectFixScale : public BaseUnaryEdge<3, Vector3d, VertexEllipsoidFixScale>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgePointEllipsoidOnlyObjectFixScale()
    {
        print_details = false;
        prior_object_half_size.setZero();
    };

    virtual bool read(std::istream &is) { return true; };
    virtual bool write(std::ostream &os) const { return os.good(); };

    bool print_details;

    void computeError();
    Vector3d computeError_debug();

    std::vector<Vector3d> object_points;
    double max_outside_margin_ratio;
    Vector3d prior_object_half_size;
};
/*
// object point surface error, both will optimize
class EdgePointEllipsoid : public BaseBinaryEdge<1, double, VertexSBAPointXYZ, VertexEllipsoid>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgePointEllipsoid(){};

    virtual bool read(std::istream &is) { return true; };
    virtual bool write(std::ostream &os) const { return os.good(); };

    void computeError();

    double max_outside_margin_ratio; // truncate the error if point is too far from object
};

// object point surface error, both will optimize.   object has fixed size.
class EdgePointEllipsoidFixScale : public BaseBinaryEdge<1, double, VertexSBAPointXYZ, VertexEllipsoidFixScale>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgePointEllipsoidFixScale(){};

    virtual bool read(std::istream &is) { return true; };
    virtual bool write(std::ostream &os) const { return os.good(); };

    void computeError();

    double max_outside_margin_ratio; // truncate the error if point is too far from object
};

// one object connected with all fixed points. only optimize object.  want object to contain points
class EdgePointEllipsoidOnlyObject : public BaseUnaryEdge<1, double, VertexEllipsoid>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgePointEllipsoidOnlyObject()
    {
        print_details = false;
        prior_object_half_size.setZero();
    };

    virtual bool read(std::istream &is) { return true; };
    virtual bool write(std::ostream &os) const { return os.good(); };

    bool print_details;

    void computeError();
    double computeError_debug();

    std::vector<Vector3d> object_points; // all the fixed points.
    double max_outside_margin_ratio;	 // truncate the error if point is too far from object
    Vector3d prior_object_half_size;	 // give a prior object size, otherwise a huge cuboid can always contain all points
};

// one object connected with all fixed points   similar as above
class EdgePointEllipsoidOnlyObjectFixScale : public BaseUnaryEdge<1, double, VertexEllipsoidFixScale>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgePointEllipsoidOnlyObjectFixScale()
    {
        print_details = false;
        prior_object_half_size.setZero();
    };

    virtual bool read(std::istream &is) { return true; };
    virtual bool write(std::ostream &os) const { return os.good(); };

    bool print_details;

    void computeError();
    double computeError_debug();

    std::vector<Vector3d> object_points;
    double max_outside_margin_ratio;
    Vector3d prior_object_half_size;
};
*/

// dynamic point attached to ellipsoid, then project to camera to minimize reprojection error.
class EdgeDynamicPointEllipsoidCamera : public BaseMultiEdge<2, Vector2d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeDynamicPointEllipsoidCamera() { resize(3); };

    virtual bool read(std::istream &is) { return true; };
    virtual bool write(std::ostream &os) const { return os.good(); };

    void computeError();
    Vector2d computeError_debug();
    virtual void linearizeOplus(); // override linearizeOplus to compute jacobians

    double get_error_norm(bool print_details = false);

    Matrix3d Kalib;
};

// dynamic object motion constraints.
class EdgeObjectMotionEllipsoid : public BaseMultiEdge<3, Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeObjectMotionEllipsoid() { resize(3); };

    virtual bool read(std::istream &is) { return true; };
    virtual bool write(std::ostream &os) const { return os.good(); };

    void computeError();
    Vector3d computeError_debug();

    double delta_t; // to_time - from_time   positive   velocity needs time
    double get_error_norm(bool print_details = false);
};


} // namespace g2o
