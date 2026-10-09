function write_xyzuvw(filename, X, Y, Ux, Uy, Uz)
%WRITE_XYZUVW Write a displacement grid in the layout read by deformation_util.read_disp_file.
%
%   write_xyzuvw(filename, X, Y, Ux, Uy, Uz)
%
%   X, Y [km] from meshgrid (rows follow y, columns follow x); Ux, Uy, Uz [m].
%   Line 1 holds the array size as "nx <rows> ny <cols> nz 1"; read_disp_file
%   reshapes the columns with order='F', so X(:) column-major order reproduces
%   the meshgrid layout (X[0,1] - X[0,0] = dx) on the Python side.

fid = fopen(filename, 'wt');
if fid < 0
    error('write_xyzuvw:open', 'Cannot open %s for writing', filename);
end
fprintf(fid, 'nx %d ny %d nz %d\n', size(X, 1), size(X, 2), 1);
fprintf(fid, 'x_km y_km z_km ux_m uy_m uz_m\n');
fprintf(fid, '%.4f %.4f %.4f %.6e %.6e %.6e\n', ...
        [X(:), Y(:), zeros(numel(X), 1), Ux(:), Uy(:), Uz(:)]');
fclose(fid);
end
