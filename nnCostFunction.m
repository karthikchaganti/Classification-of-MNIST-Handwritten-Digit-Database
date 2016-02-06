function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   train_X, train_Y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(train_X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

train_X = [ones(m, 1) train_X];
p = zeros(size(train_X, 1), 1);

activations2 = [ones(m,1) sigmoid(train_X*Theta1')];
activations3 = sigmoid(activations2*Theta2');

for i = 1:m
	y_binary = [1:num_labels] == train_Y(i);
	J = J + sum(-y_binary.*log(activations3(i,:)) - (1-y_binary).*log(1-activations3(i,:)))/m;
end

for i = 1:m
	y_binary = [1:num_labels] == train_Y(i);
	delta3 = (activations3(i,:)-y_binary)';
	delta2 = Theta2'*delta3.*(activations2(i,:)'.*(1-activations2(i,:)'));
	Theta2_grad = Theta2_grad + delta3*activations2(i,:);
	trimRowOne = delta2*train_X(i,:);
	Theta1_grad = Theta1_grad + trimRowOne([2:end],:);
end

Theta2_grad = Theta2_grad/m;
Theta1_grad = Theta1_grad/m;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

J = J + lambda/(2*m)*(sum(sum(Theta1(:,[2:end]).^2)) + sum(sum(Theta2(:,[2:end]).^2)));



Theta2_grad = Theta2_grad + lambda/m * [zeros(size(Theta2,1),1) Theta2(:,[2:end])];
Theta1_grad = Theta1_grad + lambda/m * [zeros(size(Theta1,1),1) Theta1(:,[2:end])];
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = 1.0 ./ (1.0 + exp(-z));
end
