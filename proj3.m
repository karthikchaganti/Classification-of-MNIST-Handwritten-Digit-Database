function root = myfunction()

%%%%%%%%%% CLASSIFICATION OF HANDWRITTEN DIGITS - PROJECT 3 - CSE 574 %%%%%%%%%%
clc;
clear;
UBitname = ['k' 'c' 'h' 'a' 'g' 'a' 'n' 't'];
personNumber = ['5' '0' '1' '6' '9' '4' '4' '1'];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load data.mat;
%train_data_images = train_X;
%train_data_labels = train_Y;
% Call Logistic Regression Function to calculate the accuracy
[accuracy_logistic Wlr train_data_images train_data_labels]= LogisticReg();
[accuracy_nn pred] = NeuralNetwork_main(train_data_images,train_data_labels);
Wlr=Wlr';
blr = Wlr(785,:);
Wlr = Wlr(1:784,:);
h='sigmoid';
% Call Neural Network Function to calculate the accuracy

% Save proj3.mat
save proj3.mat;
end


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
function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[dummy, p] = max(h2, [], 2);

% =========================================================================


end
function [accuracy, optimal_theta,train_data_images,train_data_labels] = LogisticReg()

% Parameters
k = 10;
lambda = 1;

% Load data
train_data_images = loadMNISTImages('train-images.idx3-ubyte');
train_data_labels = loadMNISTLabels('train-labels.idx1-ubyte');
train_data_images = train_data_images';
row_images_size = size(train_data_images, 1);
column_images_size= size(train_data_images, 2);
length_labels = length(train_data_labels);

%Create a zeros matrix of theta
theta = zeros(k,column_images_size+1);

%Append a cloumn of ones to data
train_data_images = [ones(row_images_size,1) train_data_images];

%Call fmincg function to get the optimal theta
for c = 0:k-1
    init_theta = zeros(column_images_size + 1, 1);
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    [theta] = fmincg (@(t)(lrCostFunction(t, train_data_images, (train_data_labels == c), lambda)), init_theta, options);
if (c==0)
	history_theta=theta';
else
	history_theta=[history_theta; theta'];
end

end

optimal_theta=history_theta;

p = zeros(size(train_data_images, 1), 1);
z4=train_data_images*optimal_theta';
h4=sigmoid(z4);
[pval, p]=max(h4,[],2); 
p=p-1;
accuracy = mean(double(p == train_data_labels)) * 100;



% Function LoadImages
function images = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
%images = permute(images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;
end

%Function LoadLabels
function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

end

%Function LR_CostFunction
function [J, grad] = lrCostFunction(theta_pass, X, y, lambda)
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta_pass));
sig = sigmoid(X * theta_pass);
cost = -y .* log(sig) - (1 - y) .* log(1 - sig);
thetaNoZero = [ [ 0 ]; theta_pass([2:length(theta_pass)]) ];
J = (1 / m) * sum(cost) + (lambda / (2 * m)) * sum(thetaNoZero .^ 2);
grad = (1 / m) .* (X' * (sig - y)) + (lambda / m) * thetaNoZero;
grad = grad(:);
end

function get_sigmoid=sigmoid(a)
get_sigmoid = 1.0 ./ (1.0 + exp(-a));
end

end
