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

    # imu_data[:, 0] -= imu_data[0, 0]

    # convert to SI units
    imu_data[:,1] = 9.81*(imu_data[:,1]*imu_params[0,0] + imu_params[1,0])
    imu_data[:,2] = 9.81*(imu_data[:,2]*imu_params[0,1] + imu_params[1,1])
    imu_data[:,3] = 9.81*(imu_data[:,3]*imu_params[0,2] + imu_params[1,2])

    gyro_bias = calculate_gyro_bias(imu_data, gyro_bias_n)
    w = (3300/1023) * (np.pi/180) * 0.3 * (imu_data[:,4:7] - gyro_bias)

    # Change order to roll, pitch, yaw
    imu_data[:,4] = w[:,1]
    imu_data[:,5] = w[:,2]
    imu_data[:,6] = w[:,0]

    # Fix gyro bias order
    gyro_bias = gyro_bias[[1,2,0]]

    return imu_data, gyro_bias

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

def gyroscopic(imu_data):
    t = 0  # initial time
    w_t = [0, 0, 0]  # attitude estimate at time t
    w = np.zeros((0, 4))  # attitude estimates

    for i in range(1, len(imu_data)):
        dt = imu_data[i,0] - imu_data[i-1,0]
        w_t += (imu_data[i,4:7]) * dt
        w = np.insert(w, w.shape[0], np.hstack((imu_data[i,0], w_t)), axis=0)
        t += dt

    return w

def accelerometer(imu_data):
    t = 0  # initial time
    w_t = [0, 0, 0]  # attitude estimate at time t
    w = np.zeros((0, 4))  # attitude estimates

    for i in range(1, len(imu_data)):
        dt = imu_data[i,0] - imu_data[i-1,0]
        roll = np.arctan2(imu_data[i,2], np.sqrt(imu_data[i,1]**2 + imu_data[i,3]**2))
        pitch = np.arctan2(-imu_data[i,1], np.sqrt(imu_data[i,2]**2 + imu_data[i,3]**2))
        yaw = np.arctan2(np.sqrt(imu_data[i,1]**2 + imu_data[i,2]**2), imu_data[i,3])
        w_t = np.array([roll, pitch, yaw])
        w = np.insert(w, w.shape[0], np.hstack((imu_data[i,0], w_t)), axis=0)
        t += dt

    return w

def complementary_filter(imu_data, alpha=0.2, beta=0.8, gamma=0.6):
    t = 0  # initial time
    w_t_a = np.array([0, 0, 0], dtype=np.float64)  # accel attitude estimate at time t
    w_t_g = np.array([0, 0, 0], dtype=np.float64)  # gyro attitude estimate at time t
    w_t = np.array([0, 0, 0])  # attitude estimate at time t
    w = np.zeros((0, 4))  # attitude estimates

    for i in range(1, len(imu_data)):
        dt = imu_data[i,0] - imu_data[i-1,0]
        roll = np.arctan2(imu_data[i,2], np.sqrt(imu_data[i,1]**2 + imu_data[i,3]**2))
        pitch = np.arctan2(-imu_data[i,1], np.sqrt(imu_data[i,2]**2 + imu_data[i,3]**2))
        yaw = np.arctan2(np.sqrt(imu_data[i,1]**2 + imu_data[i,2]**2), imu_data[i,3])
        w_a = np.array([roll, pitch, yaw])
        w_t_a = (1 - alpha) * w_a  + alpha * w_t_a  # low pass filter

        w_g = (imu_data[i,4:7]) * dt
        w_t_g += (1 - beta) * w_t_g + (1 - beta) * (w_g - w_t_g)  # high pass filter
        
        # Weighted sum of gyro and accelerometer
        w_t = ((1 - gamma) * (w_t + w_t_g)) + (gamma * w_t_a)
        w = np.insert(w, w.shape[0], np.hstack((imu_data[i,0], w_t)), axis=0)
        t += dt

    return w

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--imu_data', help='path to imu data', required=True)
    parser.add_argument('--imu_params', help='path to imu parameters', required=True)
    parser.add_argument('--gyro_bias_n', help='number of initial samples to use for gyro bias estimation')
    parser.add_argument('--gt_data', help='path to ground truth data')
    parser.add_argument('--output', help='path to output file')
    parser.add_argument('--plot', help='plot results', action='store_true')
    parser.add_argument('--visualize', help='visualize results', action='store_true')

    args = parser.parse_args()
    IMU_DATA = args.imu_data
    IMU_PARAMS = args.imu_params
    OUTPUT = args.output
    GYRO_BIAS_N = args.gyro_bias_n
    GT_DATA = args.gt_data
    PLOT = args.plot
    VISUALIZE = args.visualize

    imu_data, imu_params, gt_data = load_data(IMU_DATA, IMU_PARAMS, GT_DATA)  # load data

    imu_data, gyro_bias = preprocess_data(imu_data, imu_params)  # convert to SI units

    w_gyro = gyroscopic(imu_data)  # estimate attitude using gyroscopic model
    w_accel = accelerometer(imu_data)  # estimate attitude using accelerometer model
    w_complementary = complementary_filter(imu_data)  # estimate attitude using complementary filter

    if PLOT:
        fig = plt.figure()

        ax_roll = fig.add_subplot(311)
        ax_roll.plot(w_gyro[:,0], w_gyro[:,1], label='Gyro')
        ax_roll.plot(w_accel[:,0], w_accel[:,1], label='Accel')
        ax_roll.plot(w_complementary[:,0], w_complementary[:,1], label='Complementary')
        ax_roll.plot(gt_data[:,0], gt_data[:,1], label='Vicon')
        ax_roll.set_title('Roll')
        plt.legend()

        ax_pitch = fig.add_subplot(312)
        ax_pitch.plot(w_gyro[:,0], w_gyro[:,2], label='Gyro')
        ax_pitch.plot(w_accel[:,0], w_accel[:,2], label='Accel')
        ax_pitch.plot(w_complementary[:,0], w_complementary[:,2], label='Complementary')
        ax_pitch.plot(gt_data[:,0], gt_data[:,2], label='Vicon')
        ax_pitch.set_title('Pitch')

        ax_yaw = fig.add_subplot(313)
        ax_yaw.plot(w_gyro[:,0], w_gyro[:,3], label='Gyro')
        ax_yaw.plot(w_accel[:,0], w_accel[:,3], label='Accel')
        ax_yaw.plot(w_complementary[:,0], w_complementary[:,3], label='Complementary')
        ax_yaw.plot(gt_data[:,0], gt_data[:,3], label='Vicon')
        ax_yaw.set_title('Yaw')

        plt.show()


    if VISUALIZE:
        fig = plt.figure()
        ax_gyro = fig.add_subplot(141, projection='3d')
        ax_accel = fig.add_subplot(142, projection='3d')
        ax_comp = fig.add_subplot(143, projection='3d')
        ax_gt = fig.add_subplot(144, projection='3d')

        plt.ion()
        plt.show()
        for i in range(0, len(gt_data), 10):
            ax_gyro.clear()
            ax_accel.clear()
            ax_comp.clear()
            ax_gt.clear()

            # Print timestamp on plot
            ax_gt.text2D(0.05, 0.95, "t = %.3f" % (gt_data[i][0] - gt_data[0][0]), transform=ax_gyro.transAxes)
            rot_mat_gyro = euler_to_rot_mat(w_gyro[i][1:4])
            rotplot(rot_mat_gyro, ax_gyro)
            ax_gyro.set_title("Gyroscopic")

            rot_mat_accel = euler_to_rot_mat(w_accel[i][1:4])
            rotplot(rot_mat_accel, ax_accel)
            ax_accel.set_title("Accelerometer")

            rot_mat_comp = euler_to_rot_mat(w_complementary[i][1:4])
            rotplot(rot_mat_comp, ax_comp)
            ax_comp.set_title("Complementary")

            rot_mat_gt = euler_to_rot_mat(gt_data[i][1:4])
            rotplot(rot_mat_gt, ax_gt)
            ax_gt.set_title("Vicon")

            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

            plt.draw()
            plt.pause(0.001)

if __name__ == '__main__':
    main()