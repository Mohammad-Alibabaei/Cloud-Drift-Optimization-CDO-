# Cloud Drift Optimization (CDO) Algorithm

## Overview
The Cloud Drift Optimization (CDO) algorithm is a metaheuristic optimization algorithm inspired by the movement of clouds in the atmosphere. This algorithm is designed to solve complex optimization problems.

## Features
- Inspired by natural phenomena.
- Suitable for continuous optimization problems.
- Easy to implement and customize.

## How to Use
1. Download the code from this repository.
2. Run the `main.m` file in MATLAB.
3. Define your objective function and pass it to the algorithm.

## Example
```matlab
% Example usage of CDO
[Best_fitness, Best_position, Convergence_curve] = CDO(30, 100, -10, 10, 10, @sphere);
