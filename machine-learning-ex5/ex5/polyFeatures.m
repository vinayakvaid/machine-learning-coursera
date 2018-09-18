function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 

% First column of X_poly will be same as of X's first column
X_poly(:,1) = X(:,1);
for i = 2:p,
   
   % Creating polynomial features such that 2nd column features are square of first column, third column are cube of first column and so on
   X_poly (:,i) = X(:,1).^ i;

end



% =========================================================================

end
