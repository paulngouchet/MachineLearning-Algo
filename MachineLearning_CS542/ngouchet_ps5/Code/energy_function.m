function [ result ] = energy_function( xi, n1, n2, n3, n4, yi, h, beta, eta )
%ENERGY_FUNCTION Summary of this function goes here
%   Detailed explanation goes here
    
result = xi * (h - (beta * (n1 + n2 + n3 + n4)) - (eta * yi));

end