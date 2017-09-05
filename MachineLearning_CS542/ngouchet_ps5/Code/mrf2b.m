close all
image = imread('Lena-noise.png');
image = int16(image);




y = image;
dimension = size(image);
xDimension = dimension(2);
yDimension = dimension(1);

combination = @(x, N) (1 + mod(x-1, N));

d = 1.0;
s = 1.0;
step = 1;

flag=1;    
count = 0;
while (flag && count < 1000) 
    count = count + 1;
    flag=0;    
    for i=1:xDimension
        for j=1:yDimension
  
            xi = image(j,i);
            yi = y(j,i);
            a = image(combination(j - 1, yDimension), i);
            b = image(j, combination(i + 1, xDimension));
            c = image(combination(j+1,yDimension), i);
            d = image(j, combination(i - 1, xDimension));
            
            no_change = (-d * abs(xi - yi)) - (s * (abs(xi - a) + abs(xi - b) + abs(xi - c) + abs(xi - d)));
            xi = min(255, xi + step);  
            pos_change = (-d * abs(xi - yi)) - (s * (abs(xi - a) + abs(xi - b) + abs(xi - c) + abs(xi - d)));
            xi = image(j,i);
            xi = max(0, xi - step);
            neg_change = (-d * abs(xi - yi)) - (s * (abs(xi - a) + abs(xi - b) + abs(xi - c) + abs(xi - d)));
            xi = image(j,i);
            
            if pos_change > no_change
                flag = 1;
                image(j,i) = min(255, xi + step);
            end
            if neg_change > no_change
                flag = 1;
                image(j,i) = max(0, xi - step);
            end
         
%      
            
        end
    end
    
end





solution = imread('Lena.png');

% Calculate Accuracy
%accuracy = 1 - (sum(sum(abs(image - solution ))) / (xDimension * yDimension) )

% Display
imshow(uint8(image));
%imsave(uint8(image))


% Show original
%figure();
%imshow(uint8(y));