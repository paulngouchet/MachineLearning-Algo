% Computation without any loop
% Computation just using LIC already known as being the answer


figures = load( 'detroit.mat' );
  
 split = figures.data(:, 9:10);
 
 fprintf('the number is %d' , split(1,2));
 
 
 HOM = figures.data(:,10);
 FTP = figures.data(:,1);
 WE = figures.data(:,9);
 LIC = figures.data(:,4);
 static_vector = [1;1;1;1;1;1;1;1;1;1;1;1;1];
 static_matrix = [static_vector, FTP, WE];
 
 array_errors = [] ;
 
 
 i = 2 
 while(i < 9)
     
     temp_matrix = figures.data(:,i);
         
     new_matrix = [static_matrix, temp_matrix];
     
     beta = (((new_matrix')*new_matrix)^(-1))*(new_matrix')*HOM;
     
     y_hat = new_matrix * beta ;
     
     diff = y_hat - HOM;
     
     diff_square = diff.^2;
     
     sum_error = sum(diff_square);
     
     least_square_error = sum_error/(2*13);
     
     array_errors = [array_errors; least_square_error];
     
     i = i + 1 ;
 end
 
result = array_errors 

plot(result,'-')

% after visualizing the result, the smallest value correspond to LIC which will be then be chosen as the third variable 
 
 