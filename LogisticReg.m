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
