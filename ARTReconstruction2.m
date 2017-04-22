% [x,errors,xNorms] = ARTReconstruction2(A, b, nbIterations, x0, relaxationParameter)
%
% Algebraic Reconstruction Technique or Kaczmarz iteration
%
function [x, errors, xNorms] = ARTReconstruction2(A, b, nbIterations, x0, relaxationParameter)
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