function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time. 



EPSILON = 10^(-4);
n = size(theta);

%Simply applying the formula for gradient checking for each individual theta 

for i = 1:n
    
  plus = theta
  plus(i) = plus(i) + EPSILON
  minus = theta 
  minus(i) = minus(i) - EPSILON
  numgrad(i) = ( J(plus) - J(minus))/(2 * EPSILON)

end



%% ---------------------------------------------------------------
end
