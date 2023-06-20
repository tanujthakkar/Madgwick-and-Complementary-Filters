#! /usr/bin/env python3

import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('TkAgg')
from scipy import io
import time

from rotplot import rotplot

np.set_printoptions(threshold=np.inf, linewidth=150)

def load_data(imu_data, imu_params, gt_data=None):
    """Load data from files.

    Args:
        imu_data: path to imu data
        imu_params: path to imu parameters
        gt_data: path to ground truth data
    
    Returns:
        imu_data: imu data
        imu_params: imu parameters
        gt_data: ground truth data
    """

    imu_data = io.loadmat(imu_data)
    imu_data = np.vstack((imu_data['ts'], imu_data['vals'])).T

    imu_params = io.loadmat(imu_params)['IMUParams']

    if gt_data is not None:
        gt_data = io.loadmat(gt_data)
        ts = gt_data['ts'].reshape(-1, 1)
        rots = gt_data['rots']
        euler = np.array([rot_mat_to_euler(rots[:,:,i]) for i in range(rots.shape[2])])
        gt_data = np.hstack((ts, euler))

    return imu_data, imu_params, gt_data

def preprocess_data(imu_data, imu_params, gyro_bias_n=200):
    """Preprocess imu data.

    Args:
        imu_data: imu data
        imu_params: imu parameters
    
    Returns:
        imu_data: imu data
    """

    imu_data[:, 0] -= imu_data[0, 0]

    # convert to SI units
    imu_data[:,1] = 9.81*(imu_data[:,1]*imu_params[0,0] + imu_params[1,0])
    imu_data[:,2] = 9.81*(imu_data[:,2]*imu_params[0,1] + imu_params[1,1])
    imu_data[:,3] = 9.81*(imu_data[:,3]*imu_params[0,2] + imu_params[1,2])

    gyro_bias = calculate_gyro_bias(imu_data, gyro_bias_n)
    imu_data[:,4:7] = (3300/1023) * (np.pi/180) * 0.3 * (imu_data[:,4:7] - gyro_bias)

    return imu_data

def calculate_gyro_bias(imu_data, n):
    """Calculate gyro bias.

    Args:
        imu_data: imu data
        n: number of initial samples to use
    
    Returns:
        gyro_bias: gyro bias
    """

    gyro_bias = np.mean(imu_data[:n,4:7], axis=0)

    return gyro_bias

def rot_mat_to_euler(R):
    """Convert rotation matrix to Z-Y-X euler angles.

    Args:
        R: rotation matrix
    
    Returns:
        euler: euler angles
    """

    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2,1] , R[2,2])
        pitch = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
        yaw = np.arctan2(R[1,0], R[0,0])
    else:
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = 0

    euler = np.array([roll, pitch, yaw])

    return euler

def euler_to_rot_mat(euler):
    """Convert euler angles to rotation matrix.

    Args:
        euler: euler angles
    
    Returns:
        R: rotation matrix
    """

    roll, pitch, yaw = euler

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])

    R = np.dot(np.dot(Rz, Ry), Rx)

    return R

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--imu_data', help='path to imu data', required=True)
    parser.add_argument('--imu_params', help='path to imu parameters', required=True)
    parser.add_argument('--gyro_bias_n', help='number of initial samples to use for gyro bias estimation')
    parser.add_argument('--gt_data', help='path to ground truth data')
    parser.add_argument('--output', help='path to output file')
    parser.add_argument('--plot', help='plot results', action='store_true')

    args = parser.parse_args()
    IMU_DATA = args.imu_data
    IMU_PARAMS = args.imu_params
    OUTPUT = args.output
    GYRO_BIAS_N = args.gyro_bias_n
    GT_DATA = args.gt_data
    PLOT = args.plot

    imu_data, imu_params, gt_data = load_data(IMU_DATA, IMU_PARAMS, GT_DATA)  # load data

    imu_data = preprocess_data(imu_data, imu_params)  # convert to SI units

    if PLOT:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plt.ion()
        plt.show()
        for i, s in enumerate(gt_data):
            ax.clear()
            # Print timestamp on plot
            ax.text2D(0.05, 0.95, "t = %.3f" % (s[0] - gt_data[0][0]), transform=ax.transAxes)
            rot_mat = euler_to_rot_mat(gt_data[i][1:4])
            rotplot(rot_mat, ax)
            plt.draw()
            plt.pause(0.001)

if __name__ == '__main__':
    main()