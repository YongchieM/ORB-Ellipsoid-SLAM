#include <iostream>
#include "g2o_Object.h"

using namespace std;

int main()
{
    Eigen::Matrix<double,9,1> min_vec;
    min_vec << 0,0,0,0,0,0,1,2,3;
    g2o::ellipsoid e;
    e.fromMinimalVector(min_vec);
    Eigen::Matrix4d Q = e.generateQuadric();
    Eigen::Matrix4d Q_prim = Q.inverse();


    Eigen::Vector4d x;
    x << 0,2,0,1;
    double dis = x.transpose()*Q_prim*x;
    cout<<dis<<endl;

    return 0;
}
