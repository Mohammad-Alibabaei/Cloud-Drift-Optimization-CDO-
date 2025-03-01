% Enhanced Cloud Drift Optimization (CDO) Algorithm
%
% This function implements the CDO algorithm for solving optimization problems.
% It balances exploration and exploitation to find the global optimum.
%
% Syntax:
% [Best_fitness, Best_position, Convergence_curve] = CDO(N, Max_iter, lb, ub, dim, fobj)
%
% Inputs:
% N - Population size
% Max_iter - Maximum number of iterations
% lb - Lower bounds of search space
% ub - Upper bounds of search space
% dim - Problem dimensionality
% fobj - Objective function handle
%
% Outputs:
% Best_fitness - Best fitness value found
% Best_position - Best solution position found
% Convergence_curve - Convergence curve over iterations
%
% Key Features:
% - Dynamic weight adjustment based on fitness rank
% - Random reinitialization to avoid premature convergence
% - Fine-tuning in final iterations for precision
% - Early stopping if target precision is reached

function [Best_fitness, Best_position, Convergence_curve] = CDO(N, Max_iter, lb, ub, dim, fobj)
    disp('Enhanced Cloud Drift Optimization (CDO) is optimizing your problem');
    
    % Initialization
    Best_position = zeros(1, dim);
    Best_fitness = inf;
    AllFitness = inf * ones(N, 1);
    weight = ones(N, dim);
    X = initialization(N, dim, ub, lb);
    Convergence_curve = zeros(1, Max_iter);
    it = 1;
    search_history = X;

    % Algorithm parameters
    z = 0.005;  % Initial probability of random movement
    StoppingThreshold = 1e-300;  % Lower stopping threshold for higher precision

    while it <= Max_iter
        % Evaluate objective function for each solution
        for i = 1:N
            X(i, :) = min(max(X(i, :), lb), ub);  % Keep solutions within bounds
            AllFitness(i) = fobj(X(i, :));
        end
        
        % Sort solutions based on fitness values
        [SmellOrder, SmellIndex] = sort(AllFitness);
        bestFitness = SmellOrder(1);
        worstFitness = SmellOrder(N);
        S = bestFitness - worstFitness + eps;
        
        % Update weights dynamically
        for i = 1:N
            for j = 1:dim
                if i <= (N / 2)
                    weight(SmellIndex(i), j) = 1 + (0.3 + 0.7 * rand()) * log10((bestFitness - SmellOrder(i)) / S + 1);
                else
                    weight(SmellIndex(i), j) = 1 - (0.3 + 0.7 * rand()) * log10((SmellOrder(i) - bestFitness) / S + 1);
                end
            end
        end
        
        % Update the best solution found so far
        if bestFitness < Best_fitness
            Best_position = X(SmellIndex(1), :);
            Best_fitness = bestFitness;
        end
        
        % Stop early if precision target is reached
        if Best_fitness < StoppingThreshold
            disp(['Converged at iteration ', num2str(it)]);
            break;
        end
        
        % Adjust control parameters dynamically
        a = atanh(-it / Max_iter + 1);
        b = 1 - it / Max_iter;
        z = 0.002 + 0.003 * (1 - it / Max_iter);  % Reduce random jumps over iterations

        % Update particle positions
        for i = 1:N
            if rand < z
                % Random reinitialization of some solutions
              X(i, :) = min(max((ub - lb) .* rand(1, dim) + lb, lb), ub);
            else
                p = tanh(abs(AllFitness(i) - Best_fitness));
                vb = unifrnd(-0.2 * a, 0.2 * a, 1, dim);
                vc = unifrnd(-0.2 * b, 0.2 * b, 1, dim);
                
                for j = 1:dim
                    r = rand();
                    A = randi([1, N]);
                    B = randi([1, N]);
                    
                    if r < p
                        % Exploitation phase
                        X(i, j) = Best_position(j) + 0.8 * vb(j) * (weight(i, j) * X(A, j) - X(B, j));
                    else
                        % Exploration phase
                        X(i, j) = vc(j) * X(i, j);
                    end
                    
                    % Fine-tuning in the final iterations
                    if it > 0.9 * Max_iter
                         X(i, j) = X(i, j) * (1 - 1e-12 * randn());
                    end
                end
            end
            X(i, :) = min(max(X(i, :), lb), ub);
        end
        
        % Store convergence data
        Convergence_curve(it) = Best_fitness;
        it = it + 1;
    end
    
end
