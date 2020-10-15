#ifndef _KALMAN_H
#define _KALMAN_H

#include <vector>
#include <map>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <string>

class Kalman
{
    cv::Mat motion_mat;
    cv::Mat update_mat;
    const int ndim = 4;
    float dt;
    double std_weight_position;
    double std_weight_velocity;

public:
    Kalman();
    Kalman &operator=(const Kalman &kalman);
    void initiate(cv::Mat measurement, cv::Mat &mean, cv::Mat &covariance);
    void predict(cv::Mat &mean, cv::Mat &covariance);
    void project(cv::Mat mean, cv::Mat covariance, cv::Mat &projected_mean, cv::Mat &projected_cov);
    void update(cv::Mat &mean, cv::Mat &covariance, cv::Mat measurement);
};

#endif /*_KALMAN_FILTER_H*/