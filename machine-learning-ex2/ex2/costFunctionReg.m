function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
%%l-nhuhoc
k=length(theta);
h1=X*theta; h=sigmoid(h1);
j1=-y.*log(h);
j2=-(1-y).*log(1-h);
j3=lambda/(2*m)*sum(theta(2:k,:).^2);
J=1/m*sum(j1+j2)+j3;
%% compute the partial derivatives and set grad
k=length(theta);
for j=1:k
  if j==1
    grad(1)=1/m*sum( (h-y).*X(:,j) );
  else
    grad(j)=1/m*sum( (h-y).*X(:,j) ) +lambda/m*theta(j);
  endif
endfor




% =============================================================

end
