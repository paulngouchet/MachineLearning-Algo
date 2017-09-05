function acc = correctness(noise, correct)
[row, col] = size(noise);
total = row*col;
same = 0;
for i = 1:row
    for j = 1:col
        if correct(i,j) == noise(i,j)
            same = same + 1;
        end
    end
end
acc = 100*same/total;
end
           