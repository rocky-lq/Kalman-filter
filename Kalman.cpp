#include "Kalman.h"
#include <iostream>
#include <cstring>

using namespace std;
using namespace cv;

Kalman::Kalman()
{
    dt = 1.0;
    motion_mat = Mat::eye(2 * ndim, 2 * ndim, CV_64F);
    for (int i = 0; i < ndim; i++)
        motion_mat.at<double>(i, ndim + i) = dt;
    update_mat = Mat::eye(ndim, 2 * ndim, CV_64F);
    std_weight_position = 1. / 20;
    std_weight_velocity = 1. / 160;
}

void Kalman::initiate(Mat measurement, Mat &mean, Mat &covariance)
{
    Mat mean_pos = measurement;
    Mat mean_vel = Mat::zeros(mean_pos.size(), CV_64F);
    hconcat(mean_pos, mean_vel, mean);
    Mat_<double> std = (Mat_<double>(1, 8) << 2 * std_weight_position * measurement.at<double>(0, 3),
                        2 * std_weight_position * measurement.at<double>(0, 3),
                        1e-2,
                        2 * std_weight_position * measurement.at<double>(0, 3),
                        10 * std_weight_velocity * measurement.at<double>(0, 3),
                        10 * std_weight_velocity * measurement.at<double>(0, 3),
                        1e-5,
                        10 * std_weight_velocity * measurement.at<double>(0, 3));

    for (int i = 0; i < covariance.cols; i++)
        covariance.at<double>(i, i) = std.at<double>(0, i) * std.at<double>(0, i);
}

void Kalman::predict(Mat &mean, Mat &covariance)
{

    Mat_<double> std_pos = (Mat_<double>(1, 4) << std_weight_position * mean.at<double>(0, 3),
                            std_weight_position * mean.at<double>(0, 3),
                            1e-2,
                            std_weight_position * mean.at<double>(0, 3));
    Mat_<double> std_vel = (Mat_<double>(1, 4) << std_weight_velocity * mean.at<double>(0, 3),
                            std_weight_velocity * mean.at<double>(0, 3),
                            1e-5,
                            std_weight_velocity * mean.at<double>(0, 3));

    Mat motion_cov = Mat::zeros(2 * ndim, 2 * ndim, CV_64F);
    for (int i = 0; i < ndim; i++)
    {
        motion_cov.at<double>(i, i) = std_pos.at<double>(0, i) * std_pos.at<double>(0, i);
        motion_cov.at<double>(i + ndim, i + ndim) = std_vel.at<double>(0, i) * std_vel.at<double>(0, i);
    }
    mean = mean * motion_mat.t();
    covariance = motion_mat * covariance * motion_mat.t() + motion_cov;
}

void Kalman::project(Mat mean, Mat covariance, Mat &projected_mean, Mat &projected_cov)
{
    Mat_<double> std = (Mat_<double>(1, 4) << std_weight_position * mean.at<double>(0, 3),
                        std_weight_position * mean.at<double>(0, 3),
                        1e-1,
                        std_weight_position * mean.at<double>(0, 3));
    Mat innovation_cov = Mat::zeros(ndim, ndim, CV_64F);
    for (int i = 0; i < ndim; i++)
        innovation_cov.at<double>(i, i) = std.at<double>(0, i) * std.at<double>(0, i);

    mean = (update_mat * mean.t()).t();
    covariance = update_mat * covariance * update_mat.t();
    projected_mean = mean;
    projected_cov = covariance + innovation_cov;
}

void Kalman::update(Mat &mean, Mat &covariance, Mat measurement)
{
    Mat projected_mean = Mat::zeros(1, ndim, CV_64F);
    Mat projected_cov = Mat::zeros(ndim, ndim, CV_64F);

    project(mean, covariance, projected_mean, projected_cov);
    // projected_cov * kalman_gain = b;
    Mat b = (covariance * update_mat.t()).t();
    Mat kalman_gain;
    solve(projected_cov, b, kalman_gain, DECOMP_CHOLESKY);
    kalman_gain = kalman_gain.t();
    Mat innovation = measurement - projected_mean;
    mean = mean + innovation * kalman_gain.t();
    covariance = covariance - kalman_gain * projected_cov * kalman_gain.t();
}

Kalman &Kalman::operator=(const Kalman &kalman)
{
    this->motion_mat = kalman.motion_mat;
    this->update_mat = kalman.motion_mat;
    this->dt = kalman.dt;
    this->std_weight_position = kalman.std_weight_position;
    this->std_weight_velocity = kalman.std_weight_velocity;
    return *this;
}