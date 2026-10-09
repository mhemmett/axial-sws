function c = axial_disp_components(x_km, y_km, p)
%AXIAL_DISP_COMPONENTS Unit-amplitude displacement fields of each source [m].
%
%   c = axial_disp_components(x_km, y_km, p)
%
%   Everything is linear, so each scenario is a weighted sum of these:
%     c.infl  inflation sources with 1 m^3 total volume change (split by p.inflation.weight)
%     c.dike  the 2015 dike at its full p.dike.opening_m
%     c.ext   uniaxial extension of unit strain along p.extension.azimuth_deg
%   Each has fields ux, uy, uz (east, north, up), the same size as x_km.

nu = p.elastic.nu;
xm = x_km * 1000;
ym = y_km * 1000;

% Inflation: Mogi spheres
w = p.inflation.weight / sum(p.inflation.weight);
c.infl = zero_field(xm);
for k = 1:numel(w)
    [ux, uy, uz] = mogi_disp(xm, ym, p.inflation.x_km(k) * 1000, ...
                             p.inflation.y_km(k) * 1000, ...
                             p.inflation.depth_km(k) * 1000, w(k), nu);
    c.infl = add_field(c.infl, ux, uy, uz);
end

% Dike: Okada rectangle, trace defined like get_fault_coord (math angle)
d = p.dike;
a = d.angle_math_deg * pi / 180;
centre = d.start_km + 0.5 * d.length_km * [cos(a), sin(a)];
strike = mod(90 - d.angle_math_deg, 360);
width_m = (d.bottom_km - d.top_km) * 1000 / sin(d.dip_deg * pi / 180);
depth_centroid_m = 0.5 * (d.top_km + d.bottom_km) * 1000;
[ux, uy, uz] = okada85_tensile(xm - centre(1) * 1000, ym - centre(2) * 1000, ...
                               depth_centroid_m, strike, d.dip_deg, ...
                               d.length_km * 1000, width_m, d.opening_m, nu);
c.dike = struct('ux', ux, 'uy', uy, 'uz', uz);

% Extension: u = n (n . r), unit strain; the reference point only adds a
% rigid translation, so the caldera centre keeps the numbers small
b = p.extension.azimuth_deg * pi / 180;
n = [sin(b), cos(b)];
along = (xm - 8000) * n(1) + (ym - 5500) * n(2);
c.ext = struct('ux', n(1) * along, 'uy', n(2) * along, 'uz', zeros(size(xm)));
end


function f = zero_field(x)
f = struct('ux', zeros(size(x)), 'uy', zeros(size(x)), 'uz', zeros(size(x)));
end


function f = add_field(f, ux, uy, uz)
f.ux = f.ux + ux;
f.uy = f.uy + uy;
f.uz = f.uz + uz;
end
