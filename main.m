% Main Script for Cloud Drift Optimization Algorithm (CDO)
%
% This script initializes the parameters, invokes the Cloud Drift Optimization (CDO) algorithm,
% and visualizes the results. The CDO algorithm is a nature-inspired metaheuristic designed for
% solving optimization problems efficiently.
%
% Author: Mohammad Alibabaei Shahraki
% Date: 02/23/2025
% Description: Implementation of the Cloud Drift Optimization Algorithm for global optimization.

%% Clear Environment and Close Figures
clear all; % Clears all variables from the workspace.
close all; % Closes all open figure windows.
clc;        % Clears the Command Window.

%% Parameter Initialization
N = 1000; % Number of search agents (population size). This determines the diversity of solutions.
Function_name = 'F1'; % Name of the benchmark function to test (e.g., F1-F13).
T = 200; % Maximum number of iterations. Controls the stopping criterion of the algorithm.
dimSize = 100; % Dimensionality of the problem. Defines the number of decision variables.

%% Load Benchmark Function Details
% Retrieve the lower bound (lb), upper bound (ub), dimension (dim), and objective function (fobj)
% corresponding to the selected benchmark function.
[lb, ub, dim, fobj] = Get_Functions_CDO(Function_name, dimSize);

%% Execute the CDO Algorithm
% Call the CDO function with the initialized parameters.
% Outputs:
% - Best_fitness: The optimal value of the objective function.
% - Best_position: The solution vector corresponding to the best fitness.
% - Convergence_curve: A vector storing the best fitness values over iterations.
[Best_fitness, Best_position, Convergence_curve] = CDO(N, T, lb, ub, dim, fobj);

%% Display Results
% Print the best solution and its corresponding fitness value in the Command Window.
disp(['The best location of CDO is: ', num2str(Best_position)]);
disp(['The best fitness of CDO is: ', num2str(Best_fitness)]);

%% Plot Convergence Curve
% Visualize the convergence behavior of the CDO algorithm.
figure; % Create a new figure window.
semilogy(Convergence_curve, 'r', 'LineWidth', 2); % Plot the convergence curve using a logarithmic scale.
xlabel('Iterations'); % Label for the x-axis.
ylabel('Best Fitness Value'); % Label for the y-axis.
title('Convergence Curve of Cloud Drift Optimization Algorithm'); % Title of the plot.
legend('CDO', 'Location', 'NorthEast'); % Add