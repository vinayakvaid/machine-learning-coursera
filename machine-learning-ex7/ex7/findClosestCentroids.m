function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

norm_distances = zeros(size(X,1),K);

for i = 1:size(X,1),
   
  for j = 1:K,
     
     % we are calculating the distance of each point from the     	centroids and storing the distances in the rows of a 	%%temporary matrix

     distance = norm ( X(i,:) - centroids(j,:) );
     distance = distance * distance;
     %distance = sum(power( (X(i,:) - centroids(j,:) ) , 2));
     norm_distances(i,j) = distance;

  end

end

% here we are taking the distances and finding the min distance to perform centroid assignment step
for i = 1:size(norm_distances,1),

  row_under_invest = norm_distances(i,:);
  [val, ind] = min(row_under_invest);
  idx(i,1) = ind;  

end




% =============================================================

end

