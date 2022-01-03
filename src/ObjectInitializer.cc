#include "ObjectInitializer.h"

MatrixXd coeffs = MatrixXd::Zero(4,13);
double centre_z = 0;

//2d iou
float bbOverlap(const cv::Rect2d& box1,const cv::Rect2d& box2)
{
  if (box1.x > box2.x+box2.width) { return 0.0; }
  if (box1.y > box2.y+box2.height) { return 0.0; }
  if (box1.x+box1.width < box2.x) { return 0.0; }
  if (box1.y+box1.height < box2.y) { return 0.0; }
  float colInt =  std::min(box1.x+box1.width,box2.x+box2.width) - std::max(box1.x, box2.x);
  float rowInt =  std::min(box1.y+box1.height,box2.y+box2.height) - std::max(box1.y,box2.y);
  float intersection = colInt * rowInt;
  float area1 = box1.width*box1.height;
  float area2 = box2.width*box2.height;
  return intersection / (area1 + area2 - intersection);
}

Matrix3Xd generateProjectionMatrix(const SE3Quat& campose_cw, const Matrix3d& calib)
{
    Matrix3Xd identity_lefttop;
    identity_lefttop.resize(3, 4);
    identity_lefttop.col(3)=Vector3d(0,0,0);
    identity_lefttop.topLeftCorner<3,3>() = Matrix3d::Identity(3,3);

    Matrix3Xd proj_mat = calib * identity_lefttop;

    proj_mat = proj_mat * campose_cw.to_homogeneous_matrix();

    return proj_mat;
}

//tl,br
MatrixXd fromDetectionsToLines(VectorXd& detections)
{
    bool flag_openFilter = false;        // filter those lines lying on the image boundary
    double x1 = detections(0);
    double y1 = detections(1);
    double x2 = detections(2);
    double y2 = detections(3);

    Vector3d line1 (1, 0, -x1);
    Vector3d line2 (0, 1, -y1);
    Vector3d line3 (1, 0, -x2);
    Vector3d line4 (0, 1, -y2);

    // those lying on the image boundary have been marked -1
    MatrixXd line_selected(3, 0);
    MatrixXd line_selected_none(3, 0);
    if( !flag_openFilter || ( x1>0 && x1<640-1 ))
    {
        line_selected.conservativeResize(3, line_selected.cols()+1);
        line_selected.col(line_selected.cols()-1) = line1;
    }
    if( !flag_openFilter || (y1>0 && y1<480-1 ))
    {
        line_selected.conservativeResize(3, line_selected.cols()+1);
        line_selected.col(line_selected.cols()-1) = line2;
    }
    if( !flag_openFilter || (x2>0 && x2<640-1 ))
    {
        line_selected.conservativeResize(3, line_selected.cols()+1);
        line_selected.col(line_selected.cols()-1) = line3;
    }
    if( !flag_openFilter || (y2>0 && y2<480-1 ))
    {
        line_selected.conservativeResize(3, line_selected.cols()+1);
        line_selected.col(line_selected.cols()-1) = line4;
    }

    return line_selected;
}

MatrixXd getVectorFromPlanesHomo(MatrixXd &planes) {
    int cols = planes.cols();

    MatrixXd planes_vector(10,0);

    for(int i=0;i<cols;i++)
    {
        VectorXd p = planes.col(i);
        Eigen::Matrix<double,10,1> v;

        v << p(0)*p(0),2*p(0)*p(1),2*p(0)*p(2),2*p(0)*p(3),p(1)*p(1),2*p(1)*p(2),2*p(1)*p(3),p(2)*p(2),2*p(2)*p(3),p(3)*p(3);

        planes_vector.conservativeResize(planes_vector.rows(), planes_vector.cols()+1);
        planes_vector.col(planes_vector.cols()-1) = v;
    }
    return planes_vector;
}


Matrix<double,9,1> initializeWithCuboid(SE3Quat campose_cw, Vector4d &bbox_value, Eigen::VectorXd &cube_para, Eigen::Matrix3d& calib)
{
    VectorXd bbox(4);
    bbox << bbox_value(0)-bbox_value(2)/2.0,bbox_value(1)-bbox_value(3)/2.0,bbox_value(0)+bbox_value(2)/2.0,bbox_value(1)+bbox_value(3)/2.0;
    Matrix<double,9,1> para = cube_para.head(9);

    if(para.isZero())
    {
        return para;
    }

    //Matrix3d calib;
    //calib << 481.2, 0, 319.5,
            //0, 480.0, 239.5,
            //0,    0,     1;

    double t1,t2,t3;
    double r,p,y;
    double s1,s2,s3;
    t1 = para(0); t2 = para(1); t3 = para(2);
    r = para(3); p = para(4); y = para(5);
    s1 = para(6); s2 = para(7); s3 = para(8);

    Matrix3d rot = euler_zyx_to_rot(r,p,y);

    g2o::ellipsoid temp;
    temp.fromMinimalVector(para);
    Matrix4d quadric = temp.generateQuadric();

    MatrixXd lines = fromDetectionsToLines(bbox); //tl,br
    MatrixXd P = generateProjectionMatrix(campose_cw,calib);

    MatrixXd planes = P.transpose()*lines;

    MatrixXd planes_normalized;
    planes_normalized.resize(planes.rows(),planes.cols());
    for(int i=0; i<planes.cols(); i++)
    {
        Vector4d plane = planes.col(i);
        planes_normalized.col(i) = plane/plane.head(3).norm();
    }

    MatrixXd left;
    Vector4d right;
    left.resize(4,3);
    MatrixXd plane_vectors = getVectorFromPlanesHomo(planes_normalized);
    Matrix<double,10,1> tl_vec;
    tl_vec << t1*t1, t1*t2, t1*t3, t1, t2*t2, t2*t3, t2, t3*t3, t3, 1;
    //Vector3d s_(s1*s1,s2*s2,s3*s3);
    Vector3d s_(s1,s2,s3);
    Vector3d s2_(s_(0)*s_(0),s_(1)*s_(1),s_(2)*s_(2));
    Vector4d left_;
    for(int i=0; i<plane_vectors.cols(); i++)
    {
        Matrix<double,10,1> v = plane_vectors.col(i);
        double a = rot(0,0)*rot(0,0)*v(0)+rot(0,0)*rot(1,0)*v(1)+rot(0,0)*rot(2,0)*v(2)+rot(1,0)*rot(1,0)*v(4)+rot(1,0)*rot(2,0)*v(5)+rot(2,0)*rot(2,0)*v(7);
        double b = rot(0,1)*rot(0,1)*v(0)+rot(0,1)*rot(1,1)*v(1)+rot(0,1)*rot(2,1)*v(2)+rot(1,1)*rot(1,1)*v(4)+rot(1,1)*rot(2,1)*v(5)+rot(2,1)*rot(2,1)*v(7);
        double c = rot(0,2)*rot(0,2)*v(0)+rot(0,2)*rot(1,2)*v(1)+rot(0,2)*rot(2,2)*v(2)+rot(1,2)*rot(1,2)*v(4)+rot(1,2)*rot(2,2)*v(5)+rot(2,2)*rot(2,2)*v(7);
        Vector3d abc(a,b,c);
        left.row(i) = abc;
        right(i) = v.transpose()*tl_vec;

        //
        left_(i) = (double)(abc.transpose()*s2_);
    }

    double numerator = left_.transpose()*right;
    double denominator = left_.transpose()*left_;
    double scale = numerator/denominator;

    Matrix<double,9,1> para_scaled = para;
    para_scaled.tail<3>() = std::sqrt(scale)*s_;

    return para_scaled;
}

Matrix<double,9,1> initializeWithCentre(Vector3d& obj_size, double height, g2o::SE3Quat campose_cw, Eigen::Vector4d& bbox_value, Eigen::Matrix3d& calib)
{
    Eigen::VectorXd pose = campose_cw.inverse().toVector();

    VectorXd bbox(4);
    bbox << bbox_value(0)-bbox_value(2)/2.0,bbox_value(1)-bbox_value(3)/2.0,bbox_value(0)+bbox_value(2)/2.0,bbox_value(1)+bbox_value(3)/2.0;

    //Matrix3d calib;
    //calib << 481.2, 0, 319.5,
            //0, 480.0, 239.5,
            //0,    0,     1;

    centre_z = height;

    MatrixXd lines = fromDetectionsToLines(bbox);

    // get projection matrix
    MatrixXd P = generateProjectionMatrix(campose_cw, calib);

    //back-poject the centre point
    MatrixXd P_inv = P.transpose()*((P*P.transpose()).inverse());
    //std::cout<<"P inv"<<std::endl;
    //std::cout<<P_inv<<std::endl;
    Vector3d pt_im;
    //pt_im << bbs(0,0)+bbs(0,2)/2.0,bbs(0,1)+bbs(0,3)/2.0,1.0;
    pt_im << bbox_value(0),bbox_value(1),1.0;
    Vector4d pt = P_inv*pt_im;
    pt /= pt(3);
    //std::cout<<pt.transpose()<<std::endl;
    Vector4d Cam;
    Cam << pose(0),pose(1),pose(2),1;
    double u = (centre_z-Cam(2))/(pt(2)-Cam(2));
    Vector3d position(0,0,centre_z);
    position(0) = Cam(0)+u*(pt(0)-Cam(0));
    position(1) = Cam(1)+u*(pt(1)-Cam(1));
    std::cout<<"position: "<<position.transpose()<<std::endl;

    MatrixXd planes = P.transpose()*lines;

    MatrixXd planes_normalized;
    planes_normalized.resize(planes.rows(),planes.cols());
    for(int i=0; i<planes.cols(); i++)
    {
        Vector4d plane = planes.col(i);
        planes_normalized.col(i) = plane/plane.head(3).norm();
    }
    //std::cout<<"planes: "<<planes_normalized<<std::endl;

    MatrixXd plane_vectors = getVectorFromPlanesHomo(planes_normalized);

    //generate equation paras
    double ax1=obj_size(0), ax2=obj_size(1), ax3=obj_size(2);
    coeffs.resize(4,13);
    for(int i=0; i<plane_vectors.cols(); i++)
    {
        Matrix<double,10,1> v = plane_vectors.col(i);
        coeffs(i,0) = v(0)*ax3*ax3+v(7)*ax1*ax1;
        coeffs(i,1) = v(0)*ax1*ax1+v(7)*ax3*ax3;
        coeffs(i,2) = v(2)*(ax3*ax3-ax1*ax1);
        coeffs(i,3) = -v(0);
        coeffs(i,4) = -v(4);
        coeffs(i,5) = -v(7);
        coeffs(i,6) = -v(1);
        coeffs(i,7) = -v(2);
        coeffs(i,8) = -v(5);
        coeffs(i,9) = -v(3);
        coeffs(i,10) = -v(6);
        coeffs(i,11) = -v(8);
        coeffs(i,12) = v(4)*ax2*ax2-v(9);
    }
    //std::cout<<"coeffs..."<<std::endl;
    //std::cout<<coeffs<<std::endl;

    //find best rotation
    double max_iou=0;
    double rot_final=0;
    for(double yaw=-3.14159; yaw<=3.14159; yaw+=0.1)
    {
        double dis=0;
        for(int i=0; i<4; i++)
        {
            dis += std::fabs(coeffs(i,0)*sin(yaw)*sin(yaw)+coeffs(i,1)*cos(yaw)*cos(yaw)+coeffs(i,2)*cos(yaw)*sin(yaw)+coeffs(i,3)*position(0)*position(0)+coeffs(i,4)*position(1)*position(1)+coeffs(i,5)*position(2)*position(2)+
                   coeffs(i,6)*position(0)*position(1)+coeffs(i,7)*position(0)*position(2)+coeffs(i,8)*position(1)*position(2)+coeffs(i,9)*position(0)+coeffs(i,10)*position(1)+coeffs(i,11)*position(2)+coeffs(i,12));
        }

        //std::cout<<"rot: "<<yaw<<" dis: "<<dis<<std::endl;
        Eigen::Matrix<double,9,1> vec_temp;
        vec_temp << position(0),position(1),centre_z,0,0,yaw,ax1,ax2,ax3;
        g2o::ellipsoid e_temp;
        e_temp.fromMinimalVector(vec_temp);
        Vector4d bbox_temp = e_temp.getBoundingBoxFromProjection(campose_cw,calib);
        cv::Rect2d rect1(bbox_value(0)-bbox_value(2)/2.0,bbox_value(1)-bbox_value(3)/2.0,bbox_value(2),bbox_value(3));
        cv::Rect2d rect2(bbox_temp(0)-bbox_temp(2)/2.0,bbox_temp(1)-bbox_temp(3)/2.0,bbox_temp(2),bbox_temp(3));
        //std::cout<<"bbox temp: "<<bbox_temp.transpose()<<std::endl;
        double err = std::fabs(bbox_value(0)-bbox_temp(0))+std::fabs(bbox_value(1)-bbox_temp(1))+std::fabs(bbox_value(2)-bbox_temp(2))+std::fabs(bbox_value(3)-bbox_temp(3));
        //std::cout<<"error: "<<err1<<std::endl;
        float bb_iou = bbOverlap(rect1,rect2);
        //std::cout<<"bb_iou: "<<bb_iou<<std::endl;

        if(bb_iou > max_iou)
        {
            max_iou = bb_iou;
            rot_final = yaw;
        }
    }
    //std::cout<<"rot: "<<rot_final<<std::endl;

    Eigen::Matrix<double,9,1> min_vec;
    min_vec << position(0),position(1),centre_z,0,0,rot_final,ax1,ax2,ax3;

    return min_vec;
}

Matrix<double,9,1> initializeWithDimension(cv::Mat &all_lines_mat, Vector3d &obj_size, double height, SE3Quat campose_cw, Vector4d &bbox_value, Matrix3d &calib)
{
    Eigen::VectorXd pose = campose_cw.inverse().toVector();

    Eigen::VectorXd cam_pose_Twc = pose;
    Matrix4d transToWorld;
    transToWorld.setIdentity();
    transToWorld.block(0,0,3,3) = Quaterniond(cam_pose_Twc(6),cam_pose_Twc(3),cam_pose_Twc(4),cam_pose_Twc(5)).toRotationMatrix();
    transToWorld.col(3).head(3) = Eigen::Vector3d(cam_pose_Twc(0),cam_pose_Twc(1),cam_pose_Twc(2));
    std::cout<<"transToWorld: \n"<<transToWorld<<std::endl;
    Eigen::Vector3d orientation;
    rot_to_euler_zyx<double>(transToWorld.block(0,0,3,3),orientation(0),orientation(1),orientation(2));
    std::cout<<"camera orientation: "<<orientation.transpose()<<std::endl;
    Matrix3d rotationToWorld = transToWorld.topLeftCorner<3,3>();
    Matrix3d invR = rotationToWorld.inverse();
    Matrix3d KinvR = calib*invR;

    //edge detection
    //line_lbd_detect line_lbd_obj;
    //line_lbd_obj.use_LSD = true;
    //line_lbd_obj.line_length_thres = 15;  // remove short edges
    //cv::Mat all_lines_mat;
    //line_lbd_obj.detect_filter_lines(im, all_lines_mat);
    Eigen::MatrixXd all_lines_raw(all_lines_mat.rows,4);
    for (int rr=0;rr<all_lines_mat.rows;rr++)
        for (int cc=0;cc<4;cc++)
            all_lines_raw(rr,cc) = all_lines_mat.at<float>(rr,cc);

    align_left_right_edges(all_lines_raw); // this should be guaranteed when detecting edges

    int left_x_raw = bbox_value(0)-bbox_value(2)/2.0;
    int top_y_raw = bbox_value(1)-bbox_value(3)/2.0;
    int obj_width_raw = bbox_value(2);
    int obj_height_raw = bbox_value(3);
    int right_x_raw = left_x_raw + obj_width_raw;
    int down_y_raw = top_y_raw + obj_height_raw;

    Vector2d lefttop_expan = Vector2d(left_x_raw-10, top_y_raw-10);
    Vector2d rightbottom_expan = Vector2d(right_x_raw+10, down_y_raw+10);

    // find edges inside the object bounding box
    MatrixXd all_lines_inside_object(all_lines_raw.rows(), all_lines_raw.cols()); // first allocate a large matrix, then only use the toprows to avoid copy, alloc
    int inside_obj_edge_num = 0;
    for (int edge_id = 0; edge_id < all_lines_raw.rows(); edge_id++)
        if (check_inside_box(all_lines_raw.row(edge_id).head<2>(), lefttop_expan, rightbottom_expan))
            if (check_inside_box(all_lines_raw.row(edge_id).tail<2>(), lefttop_expan, rightbottom_expan))
            {
                all_lines_inside_object.row(inside_obj_edge_num) = all_lines_raw.row(edge_id);
                inside_obj_edge_num++;
            }

    // merge edges and remove short lines, after finding object edges.  edge merge in small regions should be faster than all.
    double pre_merge_dist_thre = 20;
    double pre_merge_angle_thre = 5;
    double edge_length_threshold = 30;
    MatrixXd all_lines_merge_inobj;
    merge_break_lines(all_lines_inside_object.topRows(inside_obj_edge_num), all_lines_merge_inobj, pre_merge_dist_thre,
                      pre_merge_angle_thre, edge_length_threshold);

    // compute edge angels and middle points
    VectorXd lines_inobj_angles(all_lines_merge_inobj.rows());
    MatrixXd edge_mid_pts(all_lines_merge_inobj.rows(), 2);
    for (int i = 0; i < all_lines_merge_inobj.rows(); i++)
    {
        lines_inobj_angles(i) = std::atan2(all_lines_merge_inobj(i, 3) - all_lines_merge_inobj(i, 1), all_lines_merge_inobj(i, 2) - all_lines_merge_inobj(i, 0)); // [-pi/2 -pi/2]
        edge_mid_pts.row(i).head<2>() = (all_lines_merge_inobj.row(i).head<2>() + all_lines_merge_inobj.row(i).tail<2>()) / 2;
    }

    VectorXd bbox(4);
    bbox << bbox_value(0)-bbox_value(2)/2.0,bbox_value(1)-bbox_value(3)/2.0,bbox_value(0)+bbox_value(2)/2.0,bbox_value(1)+bbox_value(3)/2.0;

    //Matrix3d calib;
    //calib << 481.2, 0, 319.5,
            //0, 480.0, 239.5,
            //0,    0,     1;

    centre_z = height;

    MatrixXd lines = fromDetectionsToLines(bbox);

    // get projection matrix
    MatrixXd P = generateProjectionMatrix(campose_cw, calib);

    //back-poject the centre point
    MatrixXd P_inv = P.transpose()*((P*P.transpose()).inverse());
    //std::cout<<"P inv"<<std::endl;
    //std::cout<<P_inv<<std::endl;
    Vector3d pt_im;
    //pt_im << bbs(0,0)+bbs(0,2)/2.0,bbs(0,1)+bbs(0,3)/2.0,1.0;
    pt_im << bbox_value(0),bbox_value(1),1.0;
    Vector4d pt = P_inv*pt_im;
    pt /= pt(3);
    //std::cout<<pt.transpose()<<std::endl;
    Vector4d Cam;
    Cam << pose(0),pose(1),pose(2),1;
    double u = (centre_z-Cam(2))/(pt(2)-Cam(2));
    Vector3d position(0,0,centre_z);
    position(0) = Cam(0)+u*(pt(0)-Cam(0));
    position(1) = Cam(1)+u*(pt(1)-Cam(1));
    std::cout<<"position: "<<position.transpose()<<std::endl;

    MatrixXd planes = P.transpose()*lines;

    MatrixXd planes_normalized;
    planes_normalized.resize(planes.rows(),planes.cols());
    for(int i=0; i<planes.cols(); i++)
    {
        Vector4d plane = planes.col(i);
        planes_normalized.col(i) = plane/plane.head(3).norm();
    }
    //std::cout<<"planes: "<<planes_normalized<<std::endl;

    MatrixXd plane_vectors = getVectorFromPlanesHomo(planes_normalized);

    //generate equation paras
    double ax1=obj_size(0), ax2=obj_size(1), ax3=obj_size(2);
    coeffs.resize(4,13);
    for(int i=0; i<plane_vectors.cols(); i++)
    {
        Matrix<double,10,1> v = plane_vectors.col(i);
        coeffs(i,0) = v(0)*ax3*ax3+v(7)*ax1*ax1;
        coeffs(i,1) = v(0)*ax1*ax1+v(7)*ax3*ax3;
        coeffs(i,2) = v(2)*(ax3*ax3-ax1*ax1);
        coeffs(i,3) = -v(0);
        coeffs(i,4) = -v(4);
        coeffs(i,5) = -v(7);
        coeffs(i,6) = -v(1);
        coeffs(i,7) = -v(2);
        coeffs(i,8) = -v(5);
        coeffs(i,9) = -v(3);
        coeffs(i,10) = -v(6);
        coeffs(i,11) = -v(8);
        coeffs(i,12) = v(4)*ax2*ax2-v(9);
    }
    //std::cout<<"coeffs..."<<std::endl;
    //std::cout<<coeffs<<std::endl;

    //find best rotation
    double max_iou=0;
    double min_angle_error = 100;
    double rot_final1=0,rot_final2=0,rot_final=0;
    for(double yaw=-3.14159; yaw<=3.14159; yaw+=0.1)
    {
        double dis=0;
        for(int i=0; i<4; i++)
        {
            dis += std::fabs(coeffs(i,0)*sin(yaw)*sin(yaw)+coeffs(i,1)*cos(yaw)*cos(yaw)+coeffs(i,2)*cos(yaw)*sin(yaw)+coeffs(i,3)*position(0)*position(0)+coeffs(i,4)*position(1)*position(1)+coeffs(i,5)*position(2)*position(2)+
                   coeffs(i,6)*position(0)*position(1)+coeffs(i,7)*position(0)*position(2)+coeffs(i,8)*position(1)*position(2)+coeffs(i,9)*position(0)+coeffs(i,10)*position(1)+coeffs(i,11)*position(2)+coeffs(i,12));
        }

        //std::cout<<"rot: "<<yaw<<" dis: "<<dis<<std::endl;
        Eigen::Matrix<double,9,1> vec_temp;
        vec_temp << position(0),position(1),centre_z,0,0,yaw,ax1,ax2,ax3;
        g2o::ellipsoid e_temp;
        e_temp.fromMinimalVector(vec_temp);
        Vector4d bbox_temp = e_temp.getBoundingBoxFromProjection(campose_cw,calib);
        cv::Rect2d rect1(bbox_value(0)-bbox_value(2)/2.0,bbox_value(1)-bbox_value(3)/2.0,bbox_value(2),bbox_value(3));
        cv::Rect2d rect2(bbox_temp(0)-bbox_temp(2)/2.0,bbox_temp(1)-bbox_temp(3)/2.0,bbox_temp(2),bbox_temp(3));
        //std::cout<<"bbox temp: "<<bbox_temp.transpose()<<std::endl;
        double err = std::fabs(bbox_value(0)-bbox_temp(0))+std::fabs(bbox_value(1)-bbox_temp(1))+std::fabs(bbox_value(2)-bbox_temp(2))+std::fabs(bbox_value(3)-bbox_temp(3));
        //std::cout<<"error: "<<err1<<std::endl;
        float bb_iou = bbOverlap(rect1,rect2);
        //std::cout<<"bb_iou: "<<bb_iou<<std::endl;

        MatrixXd corners_2d = e_temp.projectOntoImage(campose_cw,calib);
        MatrixXi vps_box_edge_pt_ids(3,4);
        vps_box_edge_pt_ids << 1,4,2,3,
                             1,2,3,4,
                             1,5,2,6;

        vps_box_edge_pt_ids.array() -= 1;

        Vector2d vp_1, vp_2, vp_3;
        getVanishingPoints(KinvR, yaw, vp_1, vp_2, vp_3); // for object x y z  axis
        MatrixXd all_vps(3, 2);
        all_vps.row(0) = vp_1;
        all_vps.row(1) = vp_2;
        all_vps.row(2) = vp_3;
        //std::cout<<"obj_yaw_esti  "<<obj_yaw_esti<<"  "<<obj_yaw_id<<std::endl;
        MatrixXd all_vp_bound_edge_angles = VP_support_edge_infos(all_vps, edge_mid_pts, lines_inobj_angles,
                                                                  Vector2d(15, 10));
        int invalid_edges=0;
        bool few_edges = false;
        for(int r=0; r<all_vp_bound_edge_angles.rows(); r++)
        {
            Eigen::VectorXd vp_bound_edge_angle = all_vp_bound_edge_angles.row(r);
            if(std::isnan(vp_bound_edge_angle(0))||std::isnan(vp_bound_edge_angle(1)))
                invalid_edges++;
            if(invalid_edges>1)
                few_edges = true;
        }

        double total_angle_diff = box_edge_alignment_angle_error(all_vp_bound_edge_angles, vps_box_edge_pt_ids, corners_2d);
        //std::cout<<"angle_error: "<<total_angle_diff<<std::endl;

        if(few_edges || total_angle_diff>1.0)
        {
            if(bb_iou > max_iou)
            {
                max_iou = bb_iou;
                rot_final1 = yaw;
            }
        }
        else
        {
            if(total_angle_diff < min_angle_error)
            {
                min_angle_error = total_angle_diff;
                rot_final2 = yaw;
            }
        }

    }
    if(!rot_final2)
        rot_final = rot_final1;
    else
        rot_final = rot_final2;
    //std::cout<<"rot: "<<rot_final<<std::endl;

    Eigen::Matrix<double,9,1> min_vec;
    min_vec << position(0),position(1),centre_z,0,0,rot_final,ax1,ax2,ax3;

    return min_vec;
}
