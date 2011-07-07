function R = rotmat(n, alpha)
    % rotmat computes a rotation matrix that rotates
    % around the vector nx, ny, nz by angle alpha
    if (nargin != 2)
        usage("rotmat(n, alpha)\n   with n being a 3 element vector");
    end
    n = n / norm(n); % make norm vector
    c = cos(alpha);
    s = sin(alpha);
    ac = 1 - c;
    xx = c + n(1)^2*ac;
    xy = n(1)*n(2)*ac-n(3)*s;
    xz = n(1)*n(3)*ac+n(2)*s;
    yx = n(2)*n(1)*ac+n(3)*s;
    yy = c+n(2)^2*ac;
    yz = n(2)*n(3)*ac-n(1)*s;
    zx = n(3)*n(1)*ac-n(2)*s;
    zy = n(3)*n(2)*ac+n(1)*s;
    zz = c+n(3)^2*ac;
    R = [xx, xy, xz
         yx, yy, yz
         zx, zy, zz];
end

