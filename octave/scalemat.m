function S = scalemat(sx, sy, sz)
    % scalemat creates a scale matrix with given scale values
    if (nargin != 3)
        usage("scalemat(sx, sy, sz)");
    end
    S = [sx, 0, 0
         0, sy, 0
         0, 0, sz];
end

