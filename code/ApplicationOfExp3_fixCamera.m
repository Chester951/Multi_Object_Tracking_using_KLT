clear all; close all; clc;
% this program is used to analysis the fix camera video in exp3
% i call two object
% 1. MultiKLT - compute the structure vibration
%
%

% Video path
path_video = 'D:\3_NTU_Research\9_myGraduate\B_ResearchData\EXP2_FRAME_SI\TEST3\videos';
videoName = 'fix_camera.mov';

% D600 instrinsic
load("D:\3_NTU_Research\9_myGraduate\B_ResearchData\EXP2_FRAME_SI\calib\d600\cameraParams.mat")
intrinsics = cameraParams.Intrinsics;

% Call the MultiKLT object
% some input params
n_object = 2; % tracking 4 object
marker_length = 50; % marker size is 50mm

KLT = MultiKLT(fullfile(path_video, videoName), n_object, marker_length, intrinsics);
KLT.process();
KLT.save()

