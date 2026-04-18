clc;
clear;
close all;

% Sample Data: [Attendance Assignment Test Output]
% Output: 1 = Poor, 2 = Average, 3 = Good
data = [
40 45 50 1;
50 55 60 1;
60 60 65 2;
70 65 70 2;
80 75 80 3;
90 85 90 3;
85 80 85 3;
55 50 55 1;
65 60 60 2;
75 70 75 2;
];

input = data(:,1:3);
output = data(:,4);

% Combine for training
trainingData = [input output];

% Generate initial FIS using grid partition
fis = genfis1(trainingData,3,'gbellmf');

% Train using ANFIS (Neural Network Learning)
[trainedFis, trainError] = anfis(trainingData, fis, 50);

% Plot error
figure;
plot(trainError);
title('Training Error');

% Test the system
testInput = [85 80 90];
result = evalfis(trainedFis, testInput);

disp(['Predicted Performance Level: ', num2str(result)]);
