function [ux, uy, uz] = yang_disp(x, y, x0, y0, depth, a, b, strike, plunge, P_over_mu, nu)
%YANG_DISP Surface displacement of a pressurized prolate spheroid (Yang et al., 1988).
%
%   [ux, uy, uz] = yang_disp(x, y, x0, y0, depth, a, b, strike, plunge, P_over_mu, nu)
%
%   The dMODELS yang.m algorithm (Battaglia et al., 2013), as ported in uafgeotools/vmod
%   (vmod/source/yang.py), evaluated at the free surface (z = 0) only.
%
%   x, y, x0, y0, depth [m]   centre position, depth positive down
%   a, b                [m]   semi-major and semi-minor axes, a > b
%   strike              [deg] azimuth of the major axis, clockwise from north
%   plunge              [deg] angle of the major axis from horizontal (90 = vertical)
%                             (dMODELS theta/phi). The axis plunges TOWARD the strike
%                             azimuth (its deep end points that way), and the uplift
%                             peak sits on that side, like the hanging wall of a
%                             dipping dike. Checked against a cutde boundary-element
%                             model of the same cavity.
%   P_over_mu                 excess pressure / shear modulus (displacement is linear in it)
%   nu                        Poisson's ratio
%
%   Valid for a * sin(plunge) < depth (the spheroid must stay below the surface).

if a <= b
    error('yang_disp:shape', 'Need a prolate spheroid (a > b)');
end
if a * sin(plunge * pi / 180) >= depth
    error('yang_disp:depth', 'Spheroid reaches the surface');
end
% The closed form is singular for exactly horizontal or vertical axes
plunge = min(max(plunge, 0.1), 89.9);

mu = 1;
P = P_over_mu;
lambda = 2 * mu * nu / (1 - 2 * nu);
phi = strike * pi / 180;
theta = plunge * pi / 180;
c = sqrt(a^2 - b^2);

coeffs = [1 / (16 * mu * (1 - nu)), 3 - 4 * nu, 4 * (1 - nu) * (1 - 2 * nu)];
sph = spheroid(a, b, c, lambda, mu, nu, P);

cosp = cos(phi);
sinp = sin(phi);
xn = x - x0;
yn = y - y0;
xp = xn * cosp - yn * sinp;
yp = yn * cosp + xn * sinp;

[Up1, Up2, Up3] = yang_kernel(sph, c, depth, xp, yp, nu, theta, coeffs);
[Um1, Um2, Um3] = yang_kernel(sph, -c, depth, xp, yp, nu, theta, coeffs);
U1r = -Up1 + Um1;
U2r = -Up2 + Um2;

ux = U1r * cosp + U2r * sinp;
uy = -U1r * sinp + U2r * cosp;
uz = Up3 - Um3;
end


function sph = spheroid(a, b, c, lambda, mu, nu, P)
L1 = log((a - c) / (a + c));
iia = 2 / a / c^2 + L1 / c^3;
iiaa = 2 / 3 / a^3 / c^2 + 2 / a / c^4 + L1 / c^5;
coef1 = -2 * pi * a * b^2;
Ia = coef1 * iia;
Iaa = coef1 * iiaa;

u = 8 * pi * (1 - nu);
Q = 3 / u;
R = (1 - 2 * nu) / u;

a11 = 2 * R * (Ia - 4 * pi);
a12 = -2 * R * (Ia + 4 * pi);
a21 = Q * a^2 * Iaa + R * Ia - 1;
a22 = -(Q * a^2 * Iaa + Ia * (2 * R - Q));

coef2 = 3 * lambda + 2 * mu;
w = 1 / (a11 * a22 - a12 * a21);
e11 = (3 * a22 - a12) * P * w / coef2;
e22 = (a11 - 3 * a21) * P * w / coef2;

sph.a = a;
sph.b = b;
sph.c = c;
sph.Pdila = 2 * mu * (e11 - e22);
Pstar = lambda * e11 + 2 * (lambda + mu) * e22;
sph.a1 = -2 * b^2 * sph.Pdila;
sph.b1 = 3 * b^2 * sph.Pdila / c^2 + 2 * (1 - 2 * nu) * Pstar;
end


function [u1, u2, u3] = yang_kernel(sph, xi, z0, x, y, nu, theta, coeffs)
% Yang et al. (1988) displacement for one end (xi = +-c) at the surface, z = 0
a = sph.a;
b = sph.b;
c = sph.c;
Pdila = sph.Pdila;
a1 = sph.a1;
b1 = sph.b1;
sinth = sin(theta);
costh = cos(theta);
nu4 = coeffs(2);
nu1 = 1 - nu;
coeff = a * b^2 / c^3 * coeffs(1);

xi2 = xi * costh;
xi3 = xi * sinth;
x2 = y;
x3 = -z0;
xbar3 = z0;
y1 = x;
y2 = x2 - xi2;
y3 = x3 - xi3;
ybar3 = xbar3 + xi3;
r2 = x2 * sinth - x3 * costh;
q2 = x2 * sinth + xbar3 * costh;
r3 = x2 * costh + x3 * sinth;
q3 = -x2 * costh + xbar3 * sinth;
rbar3 = r3 - xi;
qbar3 = q3 + xi;
R1 = sqrt(y1.^2 + y2.^2 + y3.^2);
R2 = sqrt(y1.^2 + y2.^2 + ybar3.^2);
C0 = z0 / sinth;

betatop = costh * q2 + (1 + sinth) * (R2 + qbar3);
betabottom = costh * y1;
atnbeta = pi / 2 * sign(betatop);
nz = abs(betabottom) ~= 0;
atnbeta(nz) = atan(betatop(nz) ./ betabottom(nz));

Rr = R1 + rbar3;
Rq = R2 + qbar3;
Ry = R2 + ybar3;
lRr = log(Rr);
lRq = log(Rq);
lRy = log(Ry);

A1star = a1 ./ (R1 .* Rr) + b1 * (lRr + (r3 + xi) ./ Rr);
Abar1star = -a1 ./ (R2 .* Rq) - b1 * (lRq + (q3 - xi) ./ Rq);
A1 = xi ./ R1 + lRr;
Abar1 = xi ./ R2 - lRq;
A2 = R1 - r3 .* lRr;
Abar2 = R2 - q3 .* lRq;
A3 = xi * rbar3 ./ R1 + R1;
Abar3 = xi * qbar3 ./ R2 - R2;

B = xi * (xi + C0) ./ R2 - Abar2 - C0 * lRq;
Bstar = a1 ./ R1 + 2 * b1 * A2 + nu4 * (a1 ./ R2 + 2 * b1 * Abar2);

ff1 = xi * y1 ./ Ry ...
      + 3 / costh^2 * (y1 .* lRy * sinth - y1 .* lRq + 2 * q2 .* atnbeta) ...
      + 2 * y1 .* lRq - 4 * xbar3 * atnbeta / costh;
ff2 = xi * y2 ./ Ry ...
      + 3 / costh^2 * (q2 .* lRq * sinth - q2 .* lRy + 2 * y1 .* atnbeta * sinth ...
                       + costh * (R2 - ybar3)) ...
      - 2 * costh * Abar2 + 2 / costh * (xbar3 * lRy - q3 .* lRq);
ff3 = (q2 .* lRq - q2 .* lRy * sinth + 2 * y1 .* atnbeta) / costh ...
      + 2 * sinth * Abar2 + q3 .* lRy - xi;

% At z = 0 the F, F* terms of Yang et al. vanish
u1 = coeff * (A1star + nu4 * Abar1star) .* y1;
u2 = coeff * (sinth * (A1star .* r2 + nu4 * Abar1star .* q2) + costh * Bstar);
u3 = coeff * (-costh * (A1star .* r2 + nu4 * Abar1star .* q2) + sinth * Bstar);

u1 = u1 + 2 * coeff * Pdila * ((A1 + nu4 * Abar1) .* y1 - coeffs(3) * ff1);
u2 = u2 + 2 * coeff * Pdila * (sinth * (A1 .* r2 + nu4 * Abar1 .* q2) - coeffs(3) * ff2 ...
                               + 4 * nu1 * costh * (A2 + Abar2) ...
                               + costh * (A3 - nu4 * Abar3));
u3 = u3 + 2 * coeff * Pdila * (costh * (-A1 .* r2 + nu4 * Abar1 .* q2) + coeffs(3) * ff3 ...
                               + 4 * nu1 * sinth * (A2 + Abar2) ...
                               + sinth * (A3 + nu4 * Abar3 - 2 * nu4 * B));
end
