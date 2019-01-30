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


X = [ones(m, 1) X]; %5000*401 it adds the bias column 
z1=X*Theta1'; % %(25*401)x(401*5000) = (25*5000) matir
a2=sigmoid(z1); %5000*25 the output of the first hidden layer

a2 = [ones(m,1) a2]; %5000*26 it adds the bias column
z2=a2*Theta2'; %5000*10
a3=sigmoid(z2); %5000*10 the output layer 

y_new = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);

%Compute cost function
J = (1/m) * sum ( sum ( ( (-y_new) .* log(a3) - (1-y_new) .* log(1-a3) ) ) );

T1 = Theta1(:,2:end);
T2 = Theta2(:,2:end);

% add regularization
Reg = (lambda / (2*m)) * ( sum ( sum (T1.^2) ) + sum ( sum (T2.^2) ) );

J = J + Reg;

% -------------------------------------------------------------
% Backpropagation

for t=1:m

    % Step 1
	a1 = X(t,:); % X already have a bias Line 44 (1*401)
    a1 = a1'; % (401*1)
	z2 = Theta1 * a1; % (25*401)*(401*1)
	a2 = sigmoid(z2); % (25*1)
    
    a2 = [1 ; a2]; % adding a bias (26*1)
	z3 = Theta2 * a2; % (10*26)*(26*1)
	a3 = sigmoid(z3); % final activation layer a3 == h(theta) (10*1)
    
    % Step 2
	delta_3 = a3 - y_new(t,:)'; % (10*1)
	
    z2=[1; z2]; % bias (26*1)
    % Step 3
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2); % ((26*10)*(10*1))=(26*1)

    % Step 4
	delta_2 = delta_2(2:end); % skipping sigma2(0) (25*1)

	Theta2_grad = Theta2_grad + delta_3 * a2'; % (10*1)*(1*26)=(10*26)
	Theta1_grad = Theta1_grad + delta_2 * a1'; % (25*1)*(1*401)=(25*401)
    
end;

% Step 5
Theta2_grad = (1/m) * Theta2_grad; % (10*26)
Theta1_grad = (1/m) * Theta1_grad; % (25*401)

% -------------------------------------------------------------
%Regularizing Backpropagation
%The bias parameter is not regularized so we exclude it 
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda/m)*Theta1(:,2:end));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda/m)*Theta2(:,2:end));
% -------------------------------------------------------------
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
