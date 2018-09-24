function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Randomly initialising some values for C and sigma
C_tried = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_tried = [0.1; 0.3; 0.5; 1; 1.5; 2; 3; 4];

% Declaring error matrix which will be 8*8 in this case
error = zeros(length(C_tried), length(sigma_tried));

% Training model using SVM and calculating errors and storing them
for i = 1:length(C_tried),
   
   for j = 1:length(sigma_tried),

      model= svmTrain(X, y, C_tried(i), @(x1, x2) gaussianKernel(x1, x2, sigma_tried(j)));

      predictions = svmPredict(model, Xval);
      error_temp = mean(double(predictions ~= yval));
      error(j,i) = error_temp;
      
   end

end

%error
% Calculating the minimum error
min_value = min( min(error) );
%min_value

% Getting the index values of the min error and then getting C and sigma
for  i = 1:length(C_tried),
   for j = 1:length(sigma_tried);

      if (error(j,i) == min_value)
         %i
         %j
         C = C_tried(i);
         sigma = sigma_tried(j);
         break;
      endif

   end
end


% =========================================================================

end
