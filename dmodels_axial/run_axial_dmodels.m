function run_axial_dmodels(model)
%RUN_AXIAL_DMODELS Rebuild the Axial DMODELS-style deformation grids for 2015 and 2026.
%
%   Two pre-eruption states (parameters in axial_dmodels_params.m):
%     pre_2015  inflation since 2011 + tectonic extension (no dike)
%     pre_2026  inflation since 2015 + the 2015 dike + tectonic extension
%   each written for every extension rate in p.extension.rates_per_yr.
%
%   model picks the inflation source: 'two_sphere' (default, Kidiwela Mogi pair),
%   'yang' (Baillard's single prolate spheroid) or 'yang_reversed' (its axis flipped).
%
%   Self-contained (Mogi or Yang 1988 + Okada 1985 + uniform extension); no DMODELS
%   install needed.
%   Outputs:
%     p.out_dir (gitignored Axial_Deformation/remake_2015_2026/<model>/):
%     def_<scenario>_rate<r>.xyzuvw   grids for scripts/deformation_util.read_disp_file
%     p.summary_dir (tracked dmodels_axial/outputs/<model>/):
%     manifest.csv                    source amplitudes behind every grid
%     station_predictions.csv         most-compressive azimuth + sigma1-sigma2 per station
%     station_components.csv          the same for each source on its own
%     maps_<scenario>.png, station_azimuth_vs_extension.png
%
%   Run from this directory:  run_axial_dmodels  or  run_axial_dmodels('yang')

here = fileparts(mfilename('fullpath'));
addpath(here);
if nargin < 1
    model = 'two_sphere';
end
p = axial_dmodels_params(model);
if ~exist(p.out_dir, 'dir')
    mkdir(p.out_dir);
end
if ~exist(p.summary_dir, 'dir')
    mkdir(p.summary_dir);
end
mu = p.elastic.mu_Pa;
nu = p.elastic.nu;

% ---------------------------------------------------------------- geometry
[X, Y] = meshgrid(p.grid.x_km, p.grid.y_km);
[sx, sy] = ll2xy_axial(p.stations.lon, p.stations.lat, p.origin.lon, p.origin.lat);
[rx, ry] = ll2xy_axial(p.uplift_ref.lon, p.uplift_ref.lat, p.origin.lon, p.origin.lat);
n_sta = numel(sx);

% Five-point stencil at each station: centre, +x, -x, +y, -y
h_km = p.fd_step_m / 1000;
SX = [sx; sx + h_km; sx - h_km; sx; sx];
SY = [sy; sy; sy; sy + h_km; sy - h_km];

G = axial_disp_components(X, Y, p);
S = axial_disp_components(SX, SY, p);
R = axial_disp_components(rx, ry, p);

% Dike surface trace, for the maps
a = p.dike.angle_math_deg * pi / 180;
dike_xy = [p.dike.start_km; p.dike.start_km + p.dike.length_km * [cos(a), sin(a)]];

% ---------------------------------------------------------------- components alone
fid = fopen(fullfile(p.summary_dir, 'station_components.csv'), 'wt');
% Strains are linear in each source, so any scenario mix can be rebuilt from these rows
fprintf(fid, ['component,station,x_km,y_km,shmax_az_deg,stress_diff_MPa_per_unit,', ...
              'exx_per_unit,eyy_per_unit,exy_per_unit\n']);
names = {'infl', 'dike', 'ext'};
if strcmp(p.inflation.model, 'yang')
    units = {'per unit P/mu', 'at full opening', 'per unit strain'};
else
    units = {'per m^3', 'at full opening', 'per unit strain'};
end
for c = 1:numel(names)
    [az, df, e] = station_stress(S.(names{c}), n_sta, p.fd_step_m, mu, nu);
    for k = 1:n_sta
        fprintf(fid, '%s,%s,%.4f,%.4f,%.2f,%.6e,%.8e,%.8e,%.8e\n', names{c}, ...
                p.stations.name{k}, sx(k), sy(k), az(k), df(k) / 1e6, e(k, :));
    end
    pairs = [p.stations.name; num2cell(az)];
    fprintf('%-5s (%s): %s\n', names{c}, units{c}, sprintf('%s %5.1f  ', pairs{:}));
end
fclose(fid);

% ---------------------------------------------------------------- scenarios
fman = fopen(fullfile(p.summary_dir, 'manifest.csv'), 'wt');
fprintf(fman, ['file,scenario,rate_per_yr,years,ext_strain,ext_azimuth_deg,', ...
               'infl_model,infl_scale,uplift_ref_m,dike_opening_m\n']);
fsta = fopen(fullfile(p.summary_dir, 'station_predictions.csv'), 'wt');
fprintf(fsta, ['scenario,rate_per_yr,ext_strain,station,x_km,y_km,uz_m,', ...
               'shmax_az_deg,stress_diff_MPa\n']);

rates = p.extension.rates_per_yr;
n_rate = numel(rates);
n_scen = numel(p.scenarios);
AZ = zeros(n_scen, n_rate, n_sta);
map_rates = rates(unique([1, ceil(n_rate / 2), n_rate]));

for s = 1:n_scen
    sc = p.scenarios(s);
    dV = sc.uplift_m / R.infl.uz;
    dike_w = double(sc.include_dike);

    fig = figure('Visible', 'off', 'Position', [100, 100, 420 * numel(map_rates), 420]);
    i_map = 0;

    for r = 1:n_rate
        strain = rates(r) * sc.years;
        w = [dV, dike_w, strain];

        U = combine(G, w);
        fname = sprintf('def_%s_rate%.0e.xyzuvw', sc.name, rates(r));
        write_xyzuvw(fullfile(p.out_dir, fname), X, Y, U.ux, U.uy, U.uz);
        fprintf(fman, '%s,%s,%.3e,%.2f,%.4e,%.1f,%s,%.6e,%.3f,%.2f\n', fname, sc.name, ...
                rates(r), sc.years, strain, p.extension.azimuth_deg, model, dV, ...
                dV * R.infl.uz, dike_w * p.dike.opening_m);

        US = combine(S, w);
        [az, df] = station_stress(US, n_sta, p.fd_step_m, mu, nu);
        AZ(s, r, :) = az;
        for k = 1:n_sta
            fprintf(fsta, '%s,%.3e,%.4e,%s,%.4f,%.4f,%.4f,%.2f,%.4f\n', sc.name, rates(r), ...
                    strain, p.stations.name{k}, sx(k), sy(k), US.uz(1, k), az(k), df(k) / 1e6);
        end

        if any(rates(r) == map_rates)
            i_map = i_map + 1;
            subplot(1, numel(map_rates), i_map);
            plot_map(X, Y, U, mu, nu, sx, sy, p.stations.name, ...
                     dike_xy, sc.include_dike);
            title(sprintf('%s (%s), extension %.0e /yr (strain %.1e)', ...
                          strrep(sc.name, '_', ' '), strrep(model, '_', ' '), ...
                          rates(r), strain));
        end
    end
    print(fig, fullfile(p.summary_dir, sprintf('maps_%s.png', sc.name)), '-dpng', '-r150');
    close(fig);
end
fclose(fman);
fclose(fsta);

% ---------------------------------------------------------------- azimuth vs extension
fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 650]);
x_rate = rates;
x_rate(x_rate == 0) = rates(2) / 3;  % put the no-extension case on the log axis
styles = {'-o', '--s'};
for k = 1:n_sta
    subplot(2, 3, k);
    hold on;
    for s = 1:n_scen
        az = squeeze(AZ(s, :, k))';
        % Break the line where the axial azimuth wraps through 0/180
        xr = x_rate(:);
        cut = find(abs(diff(az)) > 90);
        for c = numel(cut):-1:1
            az = [az(1:cut(c)); NaN; az(cut(c) + 1:end)];
            xr = [xr(1:cut(c)); NaN; xr(cut(c) + 1:end)];
        end
        semilogx(xr, az, styles{s}, 'LineWidth', 1.2);
    end
    set(gca, 'XScale', 'log');
    ylim([0, 180]);
    yticks(0:30:180);
    grid on;
    title(p.stations.name{k});
    xlabel('extension rate (/yr); leftmost point = none');
    ylabel('\sigma_{Hmax} azimuth (deg)');
    if k == 1
        legend(strrep({p.scenarios.name}, '_', ' '), 'Location', 'best');
    end
end
print(fig, fullfile(p.summary_dir, 'station_azimuth_vs_extension.png'), '-dpng', '-r150');
close(fig);

fprintf('Wrote %d grids to %s, tables and figures to %s\n', n_scen * n_rate, p.out_dir, p.summary_dir);
end


% =================================================================== helpers
function U = combine(C, w)
U.ux = w(1) * C.infl.ux + w(2) * C.dike.ux + w(3) * C.ext.ux;
U.uy = w(1) * C.infl.uy + w(2) * C.dike.uy + w(3) * C.ext.uy;
U.uz = w(1) * C.infl.uz + w(2) * C.dike.uz + w(3) * C.ext.uz;
end


function [az, df, strain] = station_stress(U, n, h_m, mu, nu)
% U fields are 5 x n: rows are the stencil [centre; +x; -x; +y; -y]
% strain is n x 3: [exx, eyy, exy]
blk = @(f, i) f(i, 1:n);
dux_dx = (blk(U.ux, 2) - blk(U.ux, 3)) / (2 * h_m);
duy_dx = (blk(U.uy, 2) - blk(U.uy, 3)) / (2 * h_m);
dux_dy = (blk(U.ux, 4) - blk(U.ux, 5)) / (2 * h_m);
duy_dy = (blk(U.uy, 4) - blk(U.uy, 5)) / (2 * h_m);
exy = 0.5 * (dux_dy + duy_dx);
[az, df] = principal_compression(dux_dx, duy_dy, exy, mu, nu);
az = az(:)';
df = df(:)';
strain = [dux_dx(:), duy_dy(:), exy(:)];
end


function plot_map(X, Y, U, mu, nu, sx, sy, names, dike_xy, show_dike)
dx = (X(1, 2) - X(1, 1)) * 1000;
dy = (Y(2, 1) - Y(1, 1)) * 1000;
[dux_dx, dux_dy] = gradient(U.ux, dx, dy);
[duy_dx, duy_dy] = gradient(U.uy, dx, dy);
[az, ~] = principal_compression(dux_dx, duy_dy, 0.5 * (dux_dy + duy_dx), mu, nu);

contourf(X, Y, U.uz, 20, 'LineStyle', 'none');
if exist('parula', 'file') || exist('parula', 'builtin')
    colormap(parula);
else
    colormap(viridis);  % Octave has no parula
end
cb = colorbar;
ylabel(cb, 'uplift (m)');
hold on;

% sigma_Hmax ticks every 0.5 km, fixed length (axial, so drawn both ways)
step = max(1, round(0.5 / (X(1, 2) - X(1, 1))));
ii = 1:step:size(X, 1);
jj = 1:step:size(X, 2);
xs = X(ii, jj);
ys = Y(ii, jj);
a = az(ii, jj) * pi / 180;
L = 0.18;
plot([xs(:) - L * sin(a(:)), xs(:) + L * sin(a(:))]', ...
     [ys(:) - L * cos(a(:)), ys(:) + L * cos(a(:))]', 'k-', 'LineWidth', 0.6);

if show_dike
    plot(dike_xy(:, 1), dike_xy(:, 2), 'r-', 'LineWidth', 2.5);
end
plot(sx, sy, 'w^', 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', 'MarkerSize', 7);
text(sx + 0.15, sy, names, 'FontSize', 7, 'Color', 'w');
axis equal;
xlim([5, 11]);
ylim([1, 8]);
xlabel('x (km E of 130.1W)');
ylabel('y (km N of 45.9N)');
end
