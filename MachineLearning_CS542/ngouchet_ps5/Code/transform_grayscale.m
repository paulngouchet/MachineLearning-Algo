function mat = transform_grayscale(first_row)
[row, col] = size(first_row);
for i = 1:row
    for j = 1:col
        if first_row(i,j) < 127
            first_row(i,j) = -1;
        else
            first_row(i,j) = 1;
        end
    end
end
mat = first_row;
end