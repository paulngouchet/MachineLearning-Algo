clear all
img = imread('Bayes-noise.png');
image = img(:,:,1);
image = int8(image);
grayscale = transform_grayscale(image);
y = grayscale;
sz = size(grayscale);
xDimensinon = sz(2);
yDimensinon = sz(1);
count = 0;
h = -.01;
beta = 15;
eta = 10;
flip=1;    


while (flip) 
    count = count + 1;
    flip=0;    
    for i=2:xDimensinon - 1
        for j=2:yDimensinon - 1
    
            no_flip_energy = energy_function(grayscale(j,i ), grayscale( j, i+1 ), grayscale( j, i-1), grayscale( j+1, i ), grayscale( j-1, i ), y(j,i), h, beta, eta );
            flip_energy = energy_function(-1*grayscale(j,i), grayscale( j, i+1), grayscale( j, i-1), grayscale( j+1, i ), grayscale( j-1, i ), y(j,i), h, beta, eta );
            
            if flip_energy < no_flip_energy
                grayscale(j, i) = -1 * grayscale(j,i);
                flip = 1;
            end
        end
    end    
end

correct = imread('Bayes.png');
correct_need = int8(correct(:,:,1));
correct_binary = transform_grayscale(correct_need);
accuracy = correctness(correct_binary, grayscale);
grayscale = uint8(grayscale);
imshow(255 * grayscale);
figure();
imshow(uint8(y) * 255);
fprintf('Accuracy: %.2f \n', accuracy)

