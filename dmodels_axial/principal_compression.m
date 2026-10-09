function [az_deg, diff_Pa] = principal_compression(exx, eyy, exy, mu, nu)
%PRINCIPAL_COMPRESSION Azimuth of the most compressive horizontal stress axis.
%
%   [az_deg, diff_Pa] = principal_compression(exx, eyy, exy, mu, nu)
%
%   Same 2-D stress as scripts/deformation_util.py::strain2stress / compute_sigma1_2d:
%   sigma = lambda * tr(eps) * I + 2 * mu * eps (tension positive). The most
%   compressive axis is the eigenvector of the SMALLER eigenvalue.
%
%   az_deg   axis azimuth, clockwise from north, in [0, 180)
%   diff_Pa  sigma_1 - sigma_2 (largest minus smallest eigenvalue), >= 0

lambda = 2 * nu * mu / (1 - 2 * nu);
tr = exx + eyy;
sxx = lambda * tr + 2 * mu * exx;
syy = lambda * tr + 2 * mu * eyy;
sxy = 2 * mu * exy;

% Most tensile axis at theta (CCW from east); most compressive is theta + 90
theta = 0.5 * atan2(2 * sxy, sxx - syy);
az_deg = mod(-theta * 180 / pi, 180);
diff_Pa = sqrt((sxx - syy).^2 + 4 * sxy.^2);
end
