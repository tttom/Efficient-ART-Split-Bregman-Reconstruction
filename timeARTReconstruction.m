%
%
%
function timeARTReconstruction()
    s = RandStream('mt19937ar', 'Seed', 1);
    
    nbTests = 10;
    
    % Prepare some test data
    nbEquations = 1000;
    nbVariables = 1000;
    A = s.randn(nbEquations, nbVariables);
    x = randn(nbVariables, 1);
    b = A*x;
    nbIterations = 100;
    x0 = zeros(nbVariables, 1, 'double');
    relaxationParameter = 1;

    calcError = @(xRec) norm(A*xRec - b)./norm(b);

    
    previousTimes = zeros(1, nbTests);
    newTimes = zeros(1, nbTests);
    previousErrors = zeros(1, nbTests);
    newErrors = zeros(1, nbTests);
    for idx = 1:nbTests,
        tic();
        xReconstructed = ARTReconstruction(A, b, relaxationParameter, nbIterations, x0);
        previousTimes(idx) = toc();
        previousErrors(idx) = calcError(xReconstructed);
        clear xReconstructed;

        tic();
        xReconstructed = ARTReconstruction2(A, b, nbIterations, x0, relaxationParameter);
        newTimes(idx) = toc();
        newErrors(idx) = calcError(xReconstructed);
        clear xReconstructed;
    end
    
    disp(sprintf('Shortest run times previous implementation: %0.3fs, and new implementation:  %0.3fs.', [min(previousTimes) min(newTimes)]));
    disp(sprintf('Relative error previous implementation: %d, and new implementation:  %d.', [max(previousErrors) max(newErrors)]));
end