// std c
#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>

// opencv
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

// // ros
// #include <ros/ros.h>
// #include <ros/package.h>

#include <deque>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

// ours
#include "detect_3d_cuboid.h"
#include "object_3d_util.h"
#include "tictoc_profiler/profiler.hpp"
#include "line_lbd/line_lbd_allclass.h"
#include "line_lbd/line_descriptor.hpp"

using namespace std;
using namespace Eigen;


void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}

void read_yaml(const string &path_to_yaml, Eigen::Matrix3d & Kalib, float& depth_scale)
{
    // string strSettingPath = path_to_dataset + "/ICL.yaml";
    cv::FileStorage fSettings(path_to_yaml, cv::FileStorage::READ);

    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;
    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    // Load camera parameters from settings file

    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

    Kalib<< fx,  0,  cx,
            0,  fy,  cy,
            0,  0,   1;
    depth_scale = fSettings["DepthMapFactor"];

}

double box3d_iou(const cuboid *sample, const cuboid *ground)
{
    // get 2d area in the top
    // append outer and inners() https://www.boost.org/doc/libs/1_65_1/libs/geometry/doc/html/geometry/reference/models/model_polygon.html
    typedef boost::geometry::model::point<double, 2, boost::geometry::cs::cartesian> point_t;
    typedef boost::geometry::model::polygon<point_t> polygon_t;
    polygon_t sam_poly, gt_poly;
    Eigen::Matrix<double, 3, 8> sam_corner_3d = sample->box_corners_3d_world;
    Eigen::Matrix<double, 3, 8> gt_corner_3d = ground->box_corners_3d_world;
    for (size_t i = 0; i < 4; i++)
    {
      point_t sam_top_points(sam_corner_3d(0,i),sam_corner_3d(1,i));
      point_t gt_top_points(gt_corner_3d(0,i),gt_corner_3d(1,i));
      boost::geometry::append(sam_poly.outer(), sam_top_points);
      boost::geometry::append(gt_poly.outer(), gt_top_points);
      if (i == 3) // add start point to make a closed form
      {
        boost::geometry::append(sam_poly.outer(), point_t(sam_corner_3d(0,0),sam_corner_3d(1,0)));
        boost::geometry::append(gt_poly.outer(), point_t(gt_corner_3d(0,0),gt_corner_3d(1,0)));
      }
    }
    std::vector<polygon_t> inter_poly;
    boost::geometry::intersection(sam_poly, gt_poly, inter_poly);
    double inter_area = inter_poly.empty() ? 0 : boost::geometry::area(inter_poly.front());
    double union_area = boost::geometry::area(sam_poly) + boost::geometry::area(gt_poly) - inter_area;// boost::geometry::union_(poly1, poly2, un);
    double iou_2d = inter_area / union_area;
    // std::cout << "iou2d: " << iou_2d << std::endl;
    double h_up = min(sam_corner_3d(2,4),gt_corner_3d(2,4));
    double h_down = max(sam_corner_3d(2,0),gt_corner_3d(2,0));
    double inter_vol = inter_area * max(0.0, h_up - h_down);
    double sam_vol = sample->scale(0)*2 * sample->scale(1)*2 * sample->scale(2)*2;
    double gt_vol = ground->scale(0)*2 * ground->scale(1)*2 * ground->scale(2)*2;
    double iou_3d = inter_vol / (sam_vol + gt_vol - inter_vol);
    // std::cout << "iou3d: " << iou_3d <<  std::endl;
    return iou_3d;
}

double center_distance(const cuboid *sample_obj, const cuboid *ground_obj)
{
    Eigen::Vector3d sample_3d_center = sample_obj->pos;    // only need pos, rotY, and scale
    Eigen::Vector3d ground_3d_center = ground_obj->pos;    // only need pos, rotY, and scale
		double x_err = sample_3d_center(0) - ground_3d_center(0);
		double y_err = sample_3d_center(1) - ground_3d_center(1);
		double z_err = sample_3d_center(2) - ground_3d_center(2);
		double dis_err = sqrt(x_err*x_err + y_err*y_err + z_err*z_err);
    return dis_err;
}

int main(int argc, char **argv)
{
    // ros::init(argc, argv, "detect_3d_cuboid");
    // ros::NodeHandle nh;
    // ca::Profiler::enable();
    string path_to_dataset = argv[1];
    string base_folder = path_to_dataset;

    bool whether_plot_final_images = true; // after iou or center selection
    bool whether_plot_proposal_images = false;
    bool whether_plot_2dbbox = false;
    bool whether_use_init_cam = true; // when use init cam, should sample camera roll and pitch
    bool whether_use_3d_IoU = false;
    bool whether_use_center_distance = false;
    bool whether_save_final_cuboids_txt = true;

    ofstream online_stream_cube;
    ofstream online_stream_camera;
    if (whether_save_final_cuboids_txt)// save all cube from all frame
    {
        string savepath_cube = base_folder + "/online_cubes.txt";
        string savepath_camera = base_folder + "/online_camera.txt";
        online_stream_cube.open(savepath_cube.c_str());
        online_stream_camera.open(savepath_camera.c_str());
    }

    // Load camera parameters from settings file
    std::string strSettingPath = path_to_dataset+"/TUM3.yaml";
    Eigen::Matrix3d Kalib;
    float depth_map_scaling;
    read_yaml(strSettingPath, Kalib, depth_map_scaling);
    std::cout<<"Kalib: "<< Kalib<<std::endl;

    // std::string truth_camera_pose = base_folder+"/pop_cam_poses_saved.txt";// data: time, x, y, z, qx, qy, qz, qw
    std::string truth_camera_pose = base_folder+"/truth_cam_poses.txt";// data: time, x, y, z, qx, qy, qz, qw
    Eigen::MatrixXd truth_frame_poses(100,8);
    if (!read_all_number_txt(truth_camera_pose,truth_frame_poses))
	return -1;

    std::string truth_cuboid_file = base_folder+"/cuboid_list.txt";// data: xyz, yaw, whl
    Eigen::MatrixXd truth_cuboid_list(6,7);
    if (!read_all_number_txt(truth_cuboid_file, truth_cuboid_list))
	return -1;


    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = base_folder+"/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);
    vector<string> vstrBboxFilenames;
    string strFile_yolo = base_folder+"/yolov3_bbox.txt";
    LoadImages(strFile_yolo, vstrBboxFilenames, vTimestamps);

    int total_frame_number = truth_frame_poses.rows();

    detect_3d_cuboid detect_cuboid_obj;
    detect_cuboid_obj.whether_plot_detail_images = false;
    detect_cuboid_obj.whether_plot_final_images = false;
    detect_cuboid_obj.print_details = false; // false  true
    detect_cuboid_obj.set_calibration(Kalib);
    detect_cuboid_obj.whether_sample_bbox_height = false;
    detect_cuboid_obj.whether_sample_cam_roll_pitch = false;
    detect_cuboid_obj.nominal_skew_ratio = 2;
    detect_cuboid_obj.max_cuboid_num = 100;


    bool has_detected_cuboid = false;
    Matrix<double,4,4> transToWolrd_init;
    // total_frame_number = 1;
    for (int frame_index = 0; frame_index < total_frame_number; frame_index++)
    {
        // ca::Profiler::tictoc("iou_prediction");
        // std::clock_t begin1 = clock();
        // ca::Profiler::tictoc("iou_prediction");
        // std::clock_t begin2 = clock();
        // std::cout<<"iou_prediction time: "<< double(begin2-begin1) / CLOCKS_PER_SEC<<std::endl;
        // frame_index = 411;// 250,450;
        char frame_index_c[256];
        sprintf(frame_index_c, "%04d", frame_index); // format into 4 digit
        std::cout << "frame_index: " << frame_index << std::endl;

        //load image
        cv::Mat rgb_img = cv::imread(base_folder+"/"+vstrImageFilenames[frame_index], 1);
        //read cleaned yolo 2d object detection
        Eigen::MatrixXd raw_2d_objs(10,5);  // 2d rect [x1 y1 width height], and prob
        raw_2d_objs.setZero();
        if (!read_all_number_txt(base_folder+"/"+vstrBboxFilenames[frame_index], raw_2d_objs))
        return -1;
        // std::cout << "raw_2d_objs: " << raw_2d_objs << std::endl;

        raw_2d_objs.leftCols<2>().array() -=1;   // change matlab coordinate to c++, minus 1
        if(!raw_2d_objs.isZero())
        {
        if (whether_plot_2dbbox)
        {
            cv::Mat output_img = rgb_img.clone();
            for (size_t box_id = 0; box_id < raw_2d_objs.rows(); box_id++)
            {
                Eigen::MatrixXd raw_2d_objs_edge;
                raw_2d_objs_edge.resize(4,4);
                raw_2d_objs_edge << raw_2d_objs(box_id,0), raw_2d_objs(box_id,1), raw_2d_objs(box_id,0)+raw_2d_objs(box_id,2), raw_2d_objs(box_id,1),
                                    raw_2d_objs(box_id,0), raw_2d_objs(box_id,1), raw_2d_objs(box_id,0), raw_2d_objs(box_id,1)+raw_2d_objs(box_id,3),
                                    raw_2d_objs(box_id,0), raw_2d_objs(box_id,1)+raw_2d_objs(box_id,3), raw_2d_objs(box_id,0)+raw_2d_objs(box_id,2), raw_2d_objs(box_id,1)+raw_2d_objs(box_id,3),
                                    raw_2d_objs(box_id,0)+raw_2d_objs(box_id,2), raw_2d_objs(box_id,1), raw_2d_objs(box_id,0)+raw_2d_objs(box_id,2), raw_2d_objs(box_id,1)+raw_2d_objs(box_id,3);

                plot_image_with_edges(output_img, output_img, raw_2d_objs_edge, cv::Scalar(255, 0, 0));
            }
            cv::imshow("2d bounding box", output_img);
            cv::waitKey(0);
        }

	      //edge detection
        line_lbd_detect line_lbd_obj;
        line_lbd_obj.use_LSD = true;
        line_lbd_obj.line_length_thres = 15;  // remove short edges
        cv::Mat all_lines_mat;
        line_lbd_obj.detect_filter_lines(rgb_img, all_lines_mat);
        Eigen::MatrixXd all_lines_raw(all_lines_mat.rows,4);
        for (int rr=0;rr<all_lines_mat.rows;rr++)
            for (int cc=0;cc<4;cc++)
                all_lines_raw(rr,cc) = all_lines_mat.at<float>(rr,cc);
        // std::cout << "all_lines_raw: " << all_lines_raw << std::endl;

        // only first truth pose is used. to directly visually compare with truth pose. also provide good roll/pitch
        Eigen::MatrixXd cam_pose_Twc = truth_frame_poses.row(frame_index).tail<7>(); // xyz, q1234
        // Eigen::MatrixXd cam_pose_Twc = truth_frame_poses.row(0).tail<7>(); // xyz, q1234
        std::cout << "cam_pose_Twc: \n" << cam_pose_Twc << std::endl;
        Matrix<double,4,4> transToWolrd;
        transToWolrd.setIdentity();
        transToWolrd.block(0,0,3,3) = Quaterniond(cam_pose_Twc(6),cam_pose_Twc(3),cam_pose_Twc(4),cam_pose_Twc(5)).toRotationMatrix();
        transToWolrd.col(3).head(3) = Eigen::Vector3d(cam_pose_Twc(0), cam_pose_Twc(1), cam_pose_Twc(2));
        std::cout << "transToWolrd: \n" << transToWolrd << std::endl;
        Eigen::Vector3d orientation;
        rot_to_euler_zyx<double>(transToWolrd.block(0,0,3,3), orientation(0), orientation(1), orientation(2));
        std::cout << "camera orientation: " << orientation.transpose() << std::endl;

        if(frame_index == 0)
            transToWolrd_init = transToWolrd;
        if(whether_use_init_cam)
            detect_cuboid_obj.whether_sample_cam_roll_pitch = (frame_index!=0); // first frame doesn't need to sample cam pose. could also sample. doesn't matter much

        std::vector<ObjectSet> frames_cuboids;
        if(whether_use_init_cam)
            detect_cuboid_obj.detect_cuboid(rgb_img, transToWolrd_init, raw_2d_objs, all_lines_raw, frames_cuboids);
        else
            detect_cuboid_obj.detect_cuboid(rgb_img, transToWolrd, raw_2d_objs, all_lines_raw, frames_cuboids);

        // in cabinet dataset, only one object, we always use frames_cuboids[0]
        std::cout << "proposal size: " << frames_cuboids[0].size() << std::endl;

        has_detected_cuboid = frames_cuboids.size()>0 && frames_cuboids[0].size()>0;
        int cube_final_id = 0;
        double max_3d_iou = -100.0;
        cuboid *ground_obj = new cuboid();
        Eigen::Matrix<double, 1, 7> cube_ground_truth = truth_cuboid_list.row(0);
        ground_obj->pos = cube_ground_truth.block(0,0,1,3).transpose();
        ground_obj->rotY = cube_ground_truth(0,3);
        ground_obj->scale = cube_ground_truth.block(0,4,1,3).transpose();
        ground_obj->box_corners_3d_world = compute3D_BoxCorner(*ground_obj);    // only need pos, rotY, and scale

        for (int cuboid_id = 0; cuboid_id < frames_cuboids[0].size(); cuboid_id++)
        {
            cuboid *detected_cube = frames_cuboids[0][cuboid_id];
            if (whether_use_center_distance)
            {
                detected_cube->skew_ratio = - center_distance(detected_cube, ground_obj);
                if(detected_cube->skew_ratio > max_3d_iou)
                {
                    cube_final_id = cuboid_id;
                    max_3d_iou = detected_cube->skew_ratio;
                }
            } //if (whether_use_center_distance)

            else if(whether_use_3d_IoU)
            {
                // std::cout << "cuboid "<< cuboid_id<< " proposal: " << proposal_id << " pos: " << proposal_cube->pos.transpose()
                //         << " rot_y:" << proposal_cube->rotY << " scale: " <<proposal_cube->scale.transpose() << std::endl;
                detected_cube->skew_ratio = box3d_iou(detected_cube, ground_obj);
                if(detected_cube->skew_ratio > max_3d_iou)
                {
                    cube_final_id = cuboid_id;
                    max_3d_iou = detected_cube->skew_ratio;
                }
            }


            if(whether_plot_proposal_images&&frames_cuboids[0].size()>0)
            {
                std::cout << "cuboid_id "<< cuboid_id << "pos: " << detected_cube->pos.transpose() << "error: " << detected_cube->edge_distance_error
                            << " " << detected_cube->edge_angle_error << " " << detected_cube->normalized_error << std::endl;
                cv::Mat draw_cuboid_img = rgb_img.clone();
                plot_image_with_cuboid(draw_cuboid_img, detected_cube);
                cv::imshow("image every proposal", draw_cuboid_img);
                cv::waitKey(0);
            }

        } // loop cuboid_id

        // plot every frame
        if (whether_plot_final_images && has_detected_cuboid)
        {
            cv::Mat draw_cuboid_img = rgb_img.clone();
            cuboid *final_cube = frames_cuboids[0][cube_final_id];
            plot_image_with_cuboid(draw_cuboid_img, final_cube);
            cv::imshow("image every frame", draw_cuboid_img);
            cv::waitKey(0);
        }

        if (whether_save_final_cuboids_txt && has_detected_cuboid)
        {
            // save final cuboid information // frame_index, xyz, rpy, whl
            cuboid *final_cube = frames_cuboids[0][cube_final_id];
            Matrix<double,4,4> cube_matrix;
            cube_matrix.setIdentity();
            cube_matrix.block(0,0,3,3) = euler_zyx_to_rot(0.0, 0.0, final_cube->rotY);
            cube_matrix.col(3).head(3) = final_cube->pos;
            Eigen::Vector3d orientation_tmp;
            rot_to_euler_zyx<double>(cube_matrix.block(0,0,3,3), orientation_tmp(0), orientation_tmp(1), orientation_tmp(2));
            online_stream_cube << frame_index << " "  << final_cube->pos(0) << " " << final_cube->pos(1)
                << " " << final_cube->pos(2) << " " << orientation_tmp(0) << " " << orientation_tmp(1)
                << " " << orientation_tmp(2) << " " << final_cube->scale(0) << " " << final_cube->scale(1) 
                << " " << final_cube->scale(2)  << " " << "\n";

            // save final camera information // frame_index, xyz, qxyzw
            Vector3d new_camera_eulers =  detect_cuboid_obj.cam_pose_raw.euler_angle;
            new_camera_eulers(0) += final_cube->camera_roll_delta; new_camera_eulers(1) += final_cube->camera_pitch_delta;
            // Matrix3d rotation_new = euler_zyx_to_rot<double>(new_camera_eulers(0),new_camera_eulers(1),new_camera_eulers(2));
            Vector3d trans = transToWolrd.col(3).head<3>();
            Eigen::Quaterniond qwxyz = zyx_euler_to_quat<double>(new_camera_eulers(0),new_camera_eulers(1),new_camera_eulers(2));
            if (whether_use_init_cam) // calculate 3d iou in global pose, change to global
            {
                trans = transToWolrd_init.col(3).head<3>();
                // Eigen::Matrix3d mat_temp = transToWolrd_init.block(0,0,3,3);
                // qwxyz = Quaterniond(mat_temp);
            }
            online_stream_camera << frame_index << " "  << trans(0) << " " << trans(1) << " " << trans(2)
                << " " << qwxyz.x() << " " << qwxyz.y()  << " " << qwxyz.z() << " " << qwxyz.w() 
                << " " << "\n";
        }

        }   //if(!raw_2d_objs.isZero()) if no detection
        else // when 2d box is null
            has_detected_cuboid = false;

    } // loop frame_index

    if(whether_save_final_cuboids_txt)
    {
        online_stream_cube.close();
        online_stream_camera.close();
    }

    // ca::Profiler::print_aggregated(std::cout);
    return 0;
}

