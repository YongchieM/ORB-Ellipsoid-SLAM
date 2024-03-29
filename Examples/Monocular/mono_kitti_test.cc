/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>

#include<opencv2/core/core.hpp>

#include"System.h"
#include "Parameters.h"

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    /*
    if(argc != 2)
    {
        cerr << endl << "Usage: ./mono_kitti path_to_sequence" << endl;
        return 1;
    }*/

    string base_folder = "/data/master_thesis_data/kitti/sequence_07";

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    //LoadImages(string(argv[1]), vstrImageFilenames, vTimestamps);
    LoadImages(base_folder, vstrImageFilenames, vTimestamps);
    cout << "\033[1;31m change print color examples \033[0m "<< endl; // https://stackoverflow.com/questions/2616906/how-do-i-output-coloured-text-to-a-linux-terminal

    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    // ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);
    // ORB_SLAM2::base_data_folder = argv[1];
    ORB_SLAM2::base_data_folder = base_folder;
    ORB_SLAM2::scene_unique_id = ORB_SLAM2::kitti;
    ORB_SLAM2::associate_point_with_object = true;
    ORB_SLAM2::bundle_object_opti = true;
    ORB_SLAM2::build_worldframe_on_ground = true;
    ORB_SLAM2::camera_object_BA_weight = 0.2;
    ORB_SLAM2::enable_ground_height_scale = true;
    ORB_SLAM2::mono_firstframe_truth_depth_init = true;
    //
    ORB_SLAM2::reset_orientation = false;
    ORB_SLAM2::loop_close = false;

    //string data_folder = argv[1];
    string data_folder = base_folder;
    string strVocFile = data_folder + "/ORBvoc.txt";
    string strSettingsFile = data_folder + "/KITTI.yaml";
    ORB_SLAM2::System SLAM(strVocFile, strSettingsFile, ORB_SLAM2::System::MONOCULAR, true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;
    // Main loop
    // nImages = 60;
    cv::Mat im;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        std::cout << "------------------------frame index: "  << ni  << "-------------------" << std::endl;
        im = cv::imread(vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im,tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");    

    char bStop;

    cerr << "Please type 'x', if you want to shutdown windows." << endl;

    while (bStop != 'x'){
        bStop = getchar();
    }

    // Stop all threads
    SLAM.Shutdown();

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_2/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}
