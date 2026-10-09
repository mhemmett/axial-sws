function [ux, uy, uz] = mogi_disp(x, y, x0, y0, depth, dV, nu)
%MOGI_DISP Surface displacement of a point pressure source in an elastic half-space.
%
%   [ux, uy, uz] = mogi_disp(x, y, x0, y0, depth, dV, nu)
%
%   Mogi (1958) point source, written in terms of the cavity volume change dV so
%   no shear modulus is needed:
%       u = (1 - nu) * dV / pi * [x - x0, y - y0, depth] / R^3
%
%   x, y, x0, y0, depth  [m]   (depth positive down)
%   dV                   [m^3] (positive = inflation)
%   nu                   Poisson's ratio
%   ux, uy, uz           [m]   (east, north, up)

dx = x - x0;
dy = y - y0;
R3 = (dx.^2 + dy.^2 + depth^2).^1.5;
C = (1 - nu) * dV / pi;

ux = C * dx ./ R3;
uy = C * dy ./ R3;
uz = C * depth ./ R3;
end
