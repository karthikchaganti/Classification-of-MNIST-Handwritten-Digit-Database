function [accuracy_nn,pred] = NeuralNetwork_main(train_X,train_Y)

input_layer_size  = 785;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
                          

m = size(train_X, 1);

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

options = optimset('MaxIter', 2);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p,input_layer_size,hidden_layer_size,num_labels, train_X, train_Y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

             
pred = predict(Theta1, Theta2, train_X);
pred=pred-1;

accuracy_nn =mean(double(pred == train_Y)) * 100 ;
function W1 = randInitializeWeights(L_in, L_out)
W1 = zeros(L_out, 1 + L_in);
epsilon_init = 0.12;
W1 = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
end





end
             