% Computation without any loop
% Computation just using LIC already known as being the answer


figures = load( 'detroit.mat' );
  
 split = figures.data(:, 9:10);

 
 
 HOM = figures.data(:,10);
 FTP = figures.data(:,1);
 WE = figures.data(:,9);
 LIC = figures.data(:,4);
 static_vector = [1;1;1;1;1;1;1;1;1;1;1;1;1];
 static_matrix = [static_vector, FTP, WE];
 
 new_matrix = [static_matrix, LIC];
 
 beta = (((new_matrix')*new_matrix)^(-1))*(new_matrix')*HOM
 
 y_hat = new_matrix * beta ;
 
 diff = y_hat - HOM;
 
 diff_square = diff.^2;
 
 sum_error = sum(diff_square);
 
 least_square_error = sum_error/(2*13)
 
 % y = -58.1244 + 0.1847*FTP + 0.1068*WE + 0.0165*LIC
 
 
 
 
 
 
 
 
 