function [x_km, y_km] = ll2xy_axial(lon, lat, ini_lon, ini_lat)
%LL2XY_AXIAL Local east/north [km] in the frame used by scripts/projection.py::ll2xy.
%
%   [x_km, y_km] = ll2xy_axial(lon, lat, ini_lon, ini_lat)
%
%   projection.ll2xy projects with pyproj UTM zone 9 (WGS84) and subtracts the
%   projected origin (ini_lon, ini_lat) = (-130.1, 45.9). This reproduces that
%   with the Snyder (1987) transverse-Mercator series, so the MATLAB grids and
%   station positions line up with the Python side without a Mapping Toolbox.

[x, y] = utm9(lon, lat);
[x0, y0] = utm9(ini_lon, ini_lat);
x_km = (x - x0) / 1000;
y_km = (y - y0) / 1000;
end


function [x, y] = utm9(lon, lat)
a = 6378137;
f = 1 / 298.257223563;
k0 = 0.9996;
lon0 = -129 * pi / 180;  % UTM zone 9 central meridian

e2 = f * (2 - f);
ep2 = e2 / (1 - e2);
phi = lat * pi / 180;
lam = lon * pi / 180;

N = a ./ sqrt(1 - e2 * sin(phi).^2);
T = tan(phi).^2;
C = ep2 * cos(phi).^2;
A = (lam - lon0) .* cos(phi);
M = a * ((1 - e2/4 - 3*e2^2/64 - 5*e2^3/256) * phi ...
         - (3*e2/8 + 3*e2^2/32 + 45*e2^3/1024) * sin(2*phi) ...
         + (15*e2^2/256 + 45*e2^3/1024) * sin(4*phi) ...
         - (35*e2^3/3072) * sin(6*phi));

x = k0 * N .* (A + (1 - T + C) .* A.^3 / 6 ...
               + (5 - 18*T + T.^2 + 72*C - 58*ep2) .* A.^5 / 120) + 500000;
y = k0 * (M + N .* tan(phi) .* (A.^2 / 2 + (5 - T + 9*C + 4*C.^2) .* A.^4 / 24 ...
          + (61 - 58*T + T.^2 + 600*C - 330*ep2) .* A.^6 / 720));
end
