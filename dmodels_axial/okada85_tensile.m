function [uE, uN, uZ] = okada85_tensile(e, n, depth, strike, dip, L, W, opening, nu)
%OKADA85_TENSILE Surface displacement of a rectangular tensile dislocation (dike/sill).
%
%   [uE, uN, uZ] = okada85_tensile(e, n, depth, strike, dip, L, W, opening, nu)
%
%   Okada (1985) closed-form surface solution, opening (U3) component only, in the
%   centroid convention of Beauducel's okada85.m (the version shipped with DMODELS):
%
%   e, n     [m]   observation east/north, relative to the fault CENTROID
%   depth    [m]   depth of the fault centroid (positive down)
%   strike   [deg] clockwise from north; dip is to the right of strike
%   dip      [deg] from horizontal (90 = vertical dike, 0 = horizontal sill)
%   L, W     [m]   along-strike length, down-dip width
%   opening  [m]   tensile opening (positive = opening)
%   nu             Poisson's ratio
%
%   Observation points lying exactly on the surface projection of the dislocation
%   edges are singular; keep the dike top below the surface or off-grid.

strike = strike * pi / 180;
dip = dip * pi / 180;

% Okada's origin is the bottom corner of the fault
d = depth + sin(dip) * W / 2;
ec = e + cos(strike) * cos(dip) * W / 2;
nc = n - sin(strike) * cos(dip) * W / 2;
x = cos(strike) * nc + sin(strike) * ec + L / 2;
y = sin(strike) * nc - cos(strike) * ec + cos(dip) * W;

p = y * cos(dip) + d * sin(dip);
q = y * sin(dip) - d * cos(dip);

ux = opening / (2 * pi) * chinnery(@ux_tf, x, p, L, W, q, dip, nu);
uy = opening / (2 * pi) * chinnery(@uy_tf, x, p, L, W, q, dip, nu);
uz = opening / (2 * pi) * chinnery(@uz_tf, x, p, L, W, q, dip, nu);

uE = sin(strike) * ux - cos(strike) * uy;
uN = cos(strike) * ux + sin(strike) * uy;
uZ = uz;
end


function u = chinnery(f, x, p, L, W, q, dip, nu)
% Chinnery's notation f(xi, eta)||
u = f(x, p, q, dip, nu) - f(x, p - W, q, dip, nu) ...
    - f(x - L, p, q, dip, nu) + f(x - L, p - W, q, dip, nu);
end


function u = ux_tf(xi, eta, q, dip, nu)
R = sqrt(xi.^2 + eta.^2 + q.^2);
u = q.^2 ./ (R .* (R + eta)) - I3(eta, q, dip, nu, R) * sin(dip)^2;
end


function u = uy_tf(xi, eta, q, dip, nu)
R = sqrt(xi.^2 + eta.^2 + q.^2);
db = eta * sin(dip) - q * cos(dip);
u = -db .* q ./ (R .* (R + xi)) ...
    - sin(dip) * (xi .* q ./ (R .* (R + eta)) - atan(xi .* eta ./ (q .* R))) ...
    - I1(xi, eta, q, dip, nu, R) * sin(dip)^2;
end


function u = uz_tf(xi, eta, q, dip, nu)
R = sqrt(xi.^2 + eta.^2 + q.^2);
yb = eta * cos(dip) + q * sin(dip);
u = yb .* q ./ (R .* (R + xi)) ...
    + cos(dip) * (xi .* q ./ (R .* (R + eta)) - atan(xi .* eta ./ (q .* R))) ...
    - I5(xi, eta, q, dip, nu, R) * sin(dip)^2;
end


function tf = is_vertical(dip)
tf = abs(cos(dip)) < 1e-10;
end


function I = I1(xi, eta, q, dip, nu, R)
db = eta * sin(dip) - q * cos(dip);
if is_vertical(dip)
    I = -(1 - 2 * nu) / 2 * xi .* q ./ (R + db).^2;
else
    I = (1 - 2 * nu) * (-xi ./ (cos(dip) * (R + db))) - tan(dip) * I5(xi, eta, q, dip, nu, R);
end
end


function I = I3(eta, q, dip, nu, R)
yb = eta * cos(dip) + q * sin(dip);
db = eta * sin(dip) - q * cos(dip);
if is_vertical(dip)
    I = (1 - 2 * nu) / 2 * (eta ./ (R + db) + yb .* q ./ (R + db).^2 - log(R + eta));
else
    I = (1 - 2 * nu) * (yb ./ (cos(dip) * (R + db)) - log(R + eta)) ...
        + tan(dip) * I4(db, eta, q, dip, nu, R);
end
end


function I = I4(db, eta, q, dip, nu, R)
if is_vertical(dip)
    I = -(1 - 2 * nu) * q ./ (R + db);
else
    I = (1 - 2 * nu) / cos(dip) * (log(R + db) - sin(dip) * log(R + eta));
end
end


function I = I5(xi, eta, q, dip, nu, R)
db = eta * sin(dip) - q * cos(dip);
if is_vertical(dip)
    I = -(1 - 2 * nu) * xi * sin(dip) ./ (R + db);
else
    X = sqrt(xi.^2 + q.^2);
    I = (1 - 2 * nu) * 2 / cos(dip) ...
        * atan((eta .* (X + q * cos(dip)) + X .* (R + X) * sin(dip)) ...
               ./ (xi .* (R + X) * cos(dip)));
    I(xi == 0) = 0;
end
end
