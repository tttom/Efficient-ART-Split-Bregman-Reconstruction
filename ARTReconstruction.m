% [x, errors, xNorms] = ARTReconstruction(A, b, nbIterations, x0, relaxationParameter)
%
% Algebraic reconstruction technique, also known as Kaczmarz method.
%
% Inputs:
% 
% A                      = System matrix or Jacobian matrix, nr x nc
%                        = d, data, nr x 1
% nbIterations           = number of iterations
% relaxationParameter    = relaxation parameter (between 0 and 1). Choose 0.1 to
% remove noise and 0.9 to get a closer fit of the data
% x0                     = initial guess of the solution image, nc x 1. Initialize
% as zeros(nc,1) 
%
% Outputs:
%
% x      = solution image, nc x 1
% errors = Optional output vector indicating the solution error per iteration.
% xNorms = Optional output vector indicating the norm of the solution at each iteration
%
% Original code downloaded from the repository
% https://github.com/HGGM-LIM/Efficient-ART-Split-Bregman-Reconstruction
%
% If you use this code, please cite Chamorro-Servent et al. Use of Split
% Bregman denoising for iterative reconstruction in fluorescence diffuse
% optical tomography. J Biomed Opt, 18(7):076016, 2013.
% http://dx.doi.org/10.1117/1.JBO.18.7.076016
%
% Juan FPJ Abascal, Judit Chamorro-Servent, Juan Aguirre
% Departamento de Bioingenieria e Ingenieria Aeroespacial
% Universidad Carlos III de Madrid, Madrid, Spain
% juanabascal78@gmail.com, juchamser@gmail.com, desco@hggm.es
%
%
% Modified 2017-04-22 by Tom Vettenburg for efficiency (3-4 fold)
% Single threaded performance may even be better for this algorithm.
% (start Matlab with matlab.exe -singleCompThread)
% https://github.com/tttom/Efficient-ART-Split-Bregman-Reconstruction
%
function [x, errors, xNorms] = ARTReconstruction(A, b, nbIterations, x0, relaxationParameter)
    if nargin<1 || isempty(A),
        nbEqn = 1000;
        nbVar = nbEqn;
        A = randn(nbEqn, nbVar);
        clear nbEqn nbVar
    end
    [nbEquations, nbVariables] = size(A);
    if nargin<2 || isempty(b),
        b = A*ones(nbVariables, 1);
    end
    if nargin<3 || isempty(nbIterations),
        nbIterations = 100;
    end
    if nargin<4 || isempty(x0),
        x0 = zeros(nbVariables, 1, 'double');
    end
    if nargin<4 || isempty(relaxationParameter),
        relaxationParameter = 1;
    end
    
    wantPerformanceFigures = nargout~=1;
    
    % Initialization.
    if wantPerformanceFigures,
        errors = zeros(1, nbIterations);
        xNorms = zeros(1, nbIterations);
    end
    
    % replace with transpose for efficiency
    A = A.';
    
    % precalculate the equation sqd weights
    equationSqdNorms = sum(abs(A).^2,1);
    
    Anorm = A*spdiags(relaxationParameter./equationSqdNorms.', 0, nbEquations, nbEquations);
    
    % do the Kaczmarz iteration
    x = x0;
    for itIdx = 1:nbIterations,
        for eqnIdx = 1:nbEquations,
            x = x + (b(eqnIdx)-A(:, eqnIdx)'*x)*Anorm(:,eqnIdx);
        end
        if wantPerformanceFigures,
            errors(itIdx) = norm(b-A.'*x);
            xNorms(itIdx) = norm(x);
        end
    end
    
    clear A;
    
    if nargout == 0 && wantPerformanceFigures,
        close all;
        fig = figure();
        axs(1) = subplot(1,2,1);
        semilogy([1:nbIterations], errors); title('error');
        xlabel('iteration'); ylabel('error');
        axs(2) = subplot(1,2,2);
        semilogy([1:nbIterations], xNorms); title('xNorm');
        xlabel('iteration'); ylabel('xNorm');
        
        linkaxes(axs, 'x');
        
        clear x;
    end
end