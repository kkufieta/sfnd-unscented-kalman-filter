#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // initial state vector
  x_ = VectorXd::Zero(n_x_);

  // create matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.7;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);
  P_(3, 3) = std_laspx_ * std_laspy_;
  P_(4, 4) = std_laspx_ * std_laspy_;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values
   */

  // Initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // Define spreading parameter
  lambda_ = 3 - n_aug_;

  // set measurement dimension, radar can measure r, phi, and r_dot
  nrad_z_ = 3;

  // set measurement dimension, laser can measure px, py
  nlas_z_ = 2;
}

UKF::~UKF() {}

// Generate augmented sigma points
void UKF::GenerateAugmentedSigmaPoints(MatrixXd &Xsig_aug) {
  // Process noise covariance matrix
  MatrixXd Q = MatrixXd(2, 2);
  Q << std_a_ * std_a_, 0, 0, std_yawdd_ * std_yawdd_;

  // create augmented mean vector
  VectorXd x_aug = VectorXd::Zero(7);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(7, 7);

  // create augmented mean state
  x_aug.head(n_x_) = x_;

  // create augmented covariance matrix
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(2, 2) = Q;

  // create square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++) {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug.col(i + n_aug_ + 1) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }
}

// Predict sigma points by using the process model
void UKF::SigmaPointPrediction(MatrixXd &Xsig_aug, double delta_t) {

  // create matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);
  // predict sigma points
  VectorXd x(n_aug_);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x = Xsig_aug.col(i);
    float v = x[2], psi = x[3], dpsi = x[4], nu_a = x[5], nu_dpsi = x[6];
    Xsig_pred_.col(i) = x.head(n_x_);
    if (fabs(dpsi) > 0.001) {
      Xsig_pred_.col(i)[0] += v / dpsi * (sin(psi + dpsi * delta_t) - sin(psi));
      Xsig_pred_.col(i)[1] +=
          v / dpsi * (-cos(psi + dpsi * delta_t) + cos(psi));
      Xsig_pred_.col(i)[3] += dpsi * delta_t;
    } else {
      Xsig_pred_.col(i)[0] += v * cos(psi) * delta_t;
      Xsig_pred_.col(i)[1] += v * sin(psi) * delta_t;
    }
    // add noise
    Xsig_pred_.col(i)[0] += 1 / 2. * pow(delta_t, 2) * cos(psi) * nu_a;
    Xsig_pred_.col(i)[1] += 1 / 2. * pow(delta_t, 2) * sin(psi) * nu_a;
    Xsig_pred_.col(i)[2] += delta_t * nu_a;
    Xsig_pred_.col(i)[3] += 1 / 2. * pow(delta_t, 2) * nu_dpsi;
    Xsig_pred_.col(i)[4] += delta_t * nu_dpsi;
  }
}

void UKF::PredictMeanAndCovariance() {

  // create vector for predicted state
  x_ = VectorXd::Zero(n_x_);
  // create covariance matrix for prediction
  P_ = MatrixXd::Zero(n_x_, n_x_);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    float w = i == 0 ? lambda_ / (lambda_ + n_aug_ * 1.0)
                     : 1 / (2 * (lambda_ + n_aug_ * 1.0));
    // predict state mean
    x_ += w * Xsig_pred_.col(i);
  }

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    float w = i == 0 ? lambda_ / (lambda_ + n_aug_ * 1.0)
                     : 1 / (2 * (lambda_ + n_aug_ * 1.0));
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;
    // predict state covariance matrix
    P_ += w * x_diff * x_diff.transpose();
  }
}

void UKF::PredictLidarMeasurement(VectorXd &z_pred, MatrixXd &S,
                                  MatrixXd &Zsig) {

  MatrixXd R = MatrixXd(nlas_z_, nlas_z_);
  R << std_laspx_ * std_laspx_, 0, 0, std_laspy_ * std_laspy_;

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    MatrixXd &x = Xsig_pred_;
    double px = x.col(i)[0], py = x.col(i)[1];
    Zsig.col(i) << px, py;
  }

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // set weight
    double w = i == 0 ? lambda_ / (lambda_ + n_aug_) : 0.5 / (lambda_ + n_aug_);
    // calculate mean predicted measurement
    z_pred += w * Zsig.col(i);
  }

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // set weight
    double w = i == 0 ? lambda_ / (lambda_ + n_aug_) : 0.5 / (lambda_ + n_aug_);
    // calculate innovation covariance matrix S
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += w * z_diff * z_diff.transpose();
  }
  S += R;
}

void UKF::PredictRadarMeasurement(VectorXd &z_pred, MatrixXd &S,
                                  MatrixXd &Zsig) {

  MatrixXd R = MatrixXd(nrad_z_, nrad_z_);
  R << std_radr_ * std_radr_, 0, 0, 0, std_radphi_ * std_radphi_, 0, 0, 0,
      std_radrd_ * std_radrd_;

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    MatrixXd &x = Xsig_pred_;
    double px = x.col(i)[0], py = x.col(i)[1], v = x.col(i)[2];
    double psi = x.col(i)[3], dpsi = x.col(i)[4];
    double rho = sqrt(px * px + py * py);
    double phi, d_rho;
    if (abs(px) < 1e-6) {
      phi = py >= 0 ? M_PI / 2 : -M_PI / 2;
    } else {
      phi = atan(py / px);
    }
    if (abs(rho) < 1e-6) {
      d_rho = 0;
    } else {
      d_rho = (px * cos(psi) + py * sin(psi)) * v / rho;
    }
    if (px < 0) {
      phi += M_PI;
    }
    Zsig.col(i) << rho, phi, d_rho;
  }

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // set weight
    double w = i == 0 ? lambda_ / (lambda_ + n_aug_) : 0.5 / (lambda_ + n_aug_);
    // calculate mean predicted measurement
    z_pred += w * Zsig.col(i);
  }

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // set weight
    double w = i == 0 ? lambda_ / (lambda_ + n_aug_) : 0.5 / (lambda_ + n_aug_);
    // calculate innovation covariance matrix S
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff[1] > M_PI)
      z_diff[1] -= 2. * M_PI;
    while (z_diff[1] < -M_PI)
      z_diff[1] += 2. * M_PI;
    S += w * z_diff * z_diff.transpose();
  }
  S += R;
}

void UKF::UpdateState(MatrixXd &Zsig, VectorXd &z_pred, MatrixXd &S,
                      VectorXd &z, MeasurementPackage::SensorType sensor_type) {

  // set measurement dimension
  int n_z_;
  if (sensor_type == MeasurementPackage::RADAR) {
    // radar can measure r, phi, and r_dot
    n_z_ = nrad_z_;
  } else {
    // laser can measure px, py
    n_z_ = nlas_z_;
  }

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_);

  // calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    // set weight
    double w = i == 0 ? lambda_ / (lambda_ + n_aug_) : 0.5 / (lambda_ + n_aug_);

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff[3] > M_PI)
      x_diff[3] -= 2. * M_PI;
    while (x_diff[3] < -M_PI)
      x_diff[3] += 2. * M_PI;

    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    if (sensor_type == MeasurementPackage::RADAR) {
      // angle normalization
      while (z_diff[1] > M_PI)
        z_diff[1] -= 2. * M_PI;
      while (z_diff[1] < -M_PI)
        z_diff[1] += 2. * M_PI;
    }
    Tc += w * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z - z_pred;
  if (sensor_type == MeasurementPackage::RADAR) {
    // angle normalization
    while (z_diff[1] > M_PI)
      z_diff[1] -= 2. * M_PI;
    while (z_diff[1] < -M_PI)
      z_diff[1] += 2. * M_PI;
  }

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) { // Initialize the state with the first measurement
    time_us_ = meas_package.timestamp_;

    // state vector x_: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and
    // rad
    double px, py, v = 0;
    if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // set measurement dimension, radar can measure r, phi, and r_dot
      double r = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double r_dot = meas_package.raw_measurements_[2];
      px = r * cos(phi);
      py = r * sin(phi);
      if (abs(px) > 1e-6) {
        v = r_dot * sqrt(1 + pow((py / px), 2));
      }
      x_ << px, py, v, 0, 0;
      is_initialized_ = true;
    } else if (use_laser_ &&
               meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // set measurement dimension, laser can measure px, py
      px = meas_package.raw_measurements_[0];
      py = meas_package.raw_measurements_[1];
      x_ << px, py, v, 0, 0;
      is_initialized_ = true;
    }
  } else { // Make prediction and update with new measurement
    double delta_t = (meas_package.timestamp_ - time_us_) / 1e6;

    if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {

      double r = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double r_dot = meas_package.raw_measurements_[2];
      Prediction(delta_t);
      UpdateRadar(meas_package);
    } else if (use_laser_ &&
               meas_package.sensor_type_ == MeasurementPackage::LASER) {
      Prediction(delta_t);
      UpdateLidar(meas_package);
    }
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * Estimate the object's location.
   * Modify the state vector, x_. Predict sigma points, the state,
   * and the state covariance matrix.
   */
  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  GenerateAugmentedSigmaPoints(Xsig_aug);
  SigmaPointPrediction(Xsig_aug, delta_t);
  // Update state and covariance matrix by predicting it
  // using the process model
  PredictMeanAndCovariance();
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * Use lidar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  time_us_ = meas_package.timestamp_;

  // Update state using lidar measurements
  // mean predicted measurement: lidar
  VectorXd z_pred_las = VectorXd::Zero(nlas_z_);
  // measurement covariance matrix S
  MatrixXd S_las = MatrixXd::Zero(nlas_z_, nlas_z_);
  // create matrix for sigma points in measurement space
  MatrixXd Zsig_las = MatrixXd(nlas_z_, 2 * n_aug_ + 1);
  // predict lidar measurements
  PredictLidarMeasurement(z_pred_las, S_las, Zsig_las);

  UpdateState(Zsig_las, z_pred_las, S_las, meas_package.raw_measurements_,
              MeasurementPackage::LASER);
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * Use radar data to update the belief
   * about the object's position. Modify the state vector, x_, and
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  time_us_ = meas_package.timestamp_;
  // Update state using radar measurements
  // mean predicted measurement: radar
  VectorXd z_pred_rad = VectorXd::Zero(nrad_z_);
  // measurement covariance matrix S
  MatrixXd S_rad = MatrixXd::Zero(nrad_z_, nrad_z_);
  // create matrix for sigma points in measurement space
  MatrixXd Zsig_rad = MatrixXd(nrad_z_, 2 * n_aug_ + 1);
  // predict radar measurements
  PredictRadarMeasurement(z_pred_rad, S_rad, Zsig_rad);

  UpdateState(Zsig_rad, z_pred_rad, S_rad, meas_package.raw_measurements_,
              MeasurementPackage::RADAR);
}