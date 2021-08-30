function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%l-nhuhoc
h1=X*theta;h=sigmoid(h1);
j1=-y.*log(h);j2=-(1-y).*log(1-h);
j12=1/m*(j1+j2);
%% Compute the cost of a particular choice of theta
J=sum(j12);
%% Compute the partial derivatives and set grad
hy=h-y;
k=length(theta);
for j=1:k
  grad(j)=1/m*sum( hy.*X(:,j) );
endfor







% =============================================================

end
