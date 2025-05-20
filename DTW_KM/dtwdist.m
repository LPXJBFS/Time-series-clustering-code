function d = dtwdist(Xi, Xj)
    [m, ~] = size(Xj);
    d = zeros(m, 1);
    for j = 1:m
        d(j) = dtw(Xi, Xj(j, :));
    end
end


