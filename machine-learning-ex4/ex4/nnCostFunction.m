function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1) X];

%Now, implement forward propagation, similar to ex3
z1 = sigmoid(Theta1 * X');
a2 = [ones(1, size(z1, 2)); z1];
a3 = sigmoid(Theta2 * a2);
h = a3;

% Now we tranform the y result vector into a matrix where 1s in the
% columns map to the corresponding values of y
yMatrix = zeros(num_labels, m);

for i=1:num_labels,
    yMatrix(i,:) = (y==i);
end


% Now that we have y as a 10x5000 matrix instead of a 5000x1 vector,
% we can use it to calculate our cost as compared to h (which is a3)

% Note that for this vectorized implementation, y(i)k is given as
% yMatrix and h is given as h(thetha)(x(i))k

J = (sum( sum( -1*yMatrix.*log(h) - (1 - yMatrix).*log(1-h) ) ))/m;


% Calculating regularisation term
% Since Theta1 and Theta2 were containing columns from bias units, so we would have to first remove them

Theta1_reg = Theta1;
Theta2_reg = Theta2;

% Removing first column due to bias units
Theta1_reg(:,1) = [];
Theta2_reg(:,1) = [];

temp_thetas = [Theta1_reg(:) ; Theta2_reg(:)];

temp_thetas = temp_thetas.^ 2;
temp_reg_term = (lambda / (2*m) ) * sum(temp_thetas);

% Adding regularisation term to cost function
J = J + temp_reg_term;



% Now we will implement backpropagation algorithm to calculate gradients


% Initialising capital_delta2
capital_delta1 = zeros(size(Theta1));
capital_delta2 = zeros(size(Theta2));

for i= 1:m,
	
% performing feed forward propagation to calculate activations of all units in network

  a_one = X(i,:);
  z_two = Theta1 * a_one' ;
  
  a_two = sigmoid(z_two);
  a_two = [1;a_two];
  
  z_three = Theta2 * a_two;
  a_three = sigmoid(z_three);
  temp_hypothesis = a_three;


% calculating delta3
  delta3 = temp_hypothesis - yMatrix(:,i);
  %delta3 = temp_hypothesis - temp_y(i,:)';

% calculating delta2, also we will add bias term to z_two
  z_two = [1 ; z_two];
  delta2 = (Theta2' * delta3) .* sigmoidGradient(z_two);

% calculating capital_delta1 and capital_delta2 and also, we will be removing delta2_0 since we are not considering regularistion here
  delta2 = delta2(2:end);
  capital_delta1 = capital_delta1 + ( delta2 * a_one );
  capital_delta2 = capital_delta2 + ( delta3 * a_two' );
  

end

% calculating unregularized gradient by dividing the accumulated gradients by m 
Theta1_grad = capital_delta1/m; 
Theta2_grad = capital_delta2/m;



% Now we will add regularization term to the gradients
for i= 1:size(Theta1_grad,1),
   for j = 2:size(Theta1_grad,2),
      
      Theta1_grad(i,j) = Theta1_grad(i,j) + (
(lambda/m)* Theta1(i,j) );      

   end
end 

for i= 1:size(Theta2_grad,1),
   for j = 2:size(Theta2_grad,2),
      
      Theta2_grad(i,j) = Theta2_grad(i,j) + (
(lambda/m)* Theta2(i,j) );      

   end
end 


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
