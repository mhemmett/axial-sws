function p = axial_dmodels_params()
%AXIAL_DMODELS_PARAMS Every physical and numerical input to run_axial_dmodels.m.
%
%   Two pre-eruption states of the caldera, each relative to the deflated state
%   right after the previous eruption:
%
%     pre_2015  inflation since the 2011 eruption + tectonic extension, no dike
%     pre_2026  inflation since the 2015 eruption + the dike emplaced in 2015
%               + tectonic extension
%
%   Values marked ASSUMED are placeholders with a stated provenance, not fitted
%   results; check them before treating the grids as final.

% ---------------------------------------------------------------- frame / grid
% Same local frame as scripts/projection.py::ll2xy (UTM zone 9 minus this origin)
p.origin.lon = -130.1;
p.origin.lat = 45.9;

% Covers the six caldera stations with margin; km, z = 0 (seafloor) slice only
p.grid.x_km = 2:0.1:14;
p.grid.y_km = 0:0.1:11;

% Elastic constants. Displacements depend only on nu; mu sets the stress scale
p.elastic.nu = 0.25;
p.elastic.mu_Pa = 30e9;

% Half-width [m] of the finite-difference stencil for stresses at the stations
p.fd_step_m = 10;

% ---------------------------------------------------------------- stations
% data/stations_axial.llz (lon, lat); the six production stations
p.stations.name = {'AXAS1', 'AXAS2', 'AXCC1', 'AXEC1', 'AXEC2', 'AXEC3'};
p.stations.lon = [-129.999289, -130.0141, -130.0089, -129.9797, -129.9738, -129.9785];
p.stations.lat = [45.93356, 45.93377, 45.95468, 45.94958, 45.93967, 45.93607];

% Uplift calibration point: the Central Caldera BOTPT shares the AXCC1 node
p.uplift_ref.lon = -130.0089;
p.uplift_ref.lat = 45.95468;

% ---------------------------------------------------------------- inflation
% Kidiwela two-sphere Mogi model (scripts/mogi_stress_model.py::SPHERES), local km.
% Only the geometry and the relative strength (dP * R^3) are used; the total
% volume is rescaled per scenario to hit the uplift target at uplift_ref.
p.inflation.x_km = [7.57, 7.53];
p.inflation.y_km = [4.55, 6.60];
p.inflation.depth_km = [3.33, 1.25];
p.inflation.weight = [0.43^3, 0.20^3];

% ---------------------------------------------------------------- 2015 dike
% ASSUMED: surface trace of Baillard's 'dike_syn_1' (scripts/deformation_analysis.py:
% start (8.2, 5.2) km, math angle 89 deg, 4 km), opening from baillard_simple_model.py.
% Depth range ASSUMED: from just below the seafloor to the AMC roof.
p.dike.start_km = [8.2, 5.2];
p.dike.angle_math_deg = 89;      % CCW from east, as in get_fault_coord
p.dike.length_km = 4.0;
p.dike.top_km = 0.1;
p.dike.bottom_km = 2.0;
p.dike.dip_deg = 90;
p.dike.opening_m = 2.0;

% ---------------------------------------------------------------- extension
% Uniaxial horizontal extension accumulated since the previous eruption:
% strain = rate * years. ASSUMED azimuth: ridge-normal, with the Juan de Fuca
% ridge striking ~N20E at Axial. (The atan2 fits in scripts/ used a fixed
% regional compression of 170 deg, i.e. extension at 80 deg - try that too.)
p.extension.azimuth_deg = 110;
p.extension.rates_per_yr = [0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4];

% ---------------------------------------------------------------- scenarios
% uplift_m: Central Caldera uplift since the previous post-eruption minimum.
%   2015: ASSUMED equal to the 2015 co-eruptive deflation (~2.4 m, README / site),
%         i.e. the caldera had recovered the 2011 deflation by April 2015.
%   2026: total re-inflation since the 2015-05-02 minimum (~2.6 m, site BOTPT figure).
% years: since the previous eruption (2011-04-06 -> 2015-04-24; 2015-04-24 -> 2026-05).
p.scenarios(1).name = 'pre_2015';
p.scenarios(1).uplift_m = 2.4;
p.scenarios(1).years = 4.05;
p.scenarios(1).include_dike = false;

p.scenarios(2).name = 'pre_2026';
p.scenarios(2).uplift_m = 2.6;
p.scenarios(2).years = 11.05;
p.scenarios(2).include_dike = true;

% ---------------------------------------------------------------- output
here = fileparts(mfilename('fullpath'));
% Grids are data (gitignored); tables and figures go to the tracked outputs/ folder
p.out_dir = fullfile(here, '..', 'Axial_Deformation', 'remake_2015_2026');
p.summary_dir = fullfile(here, 'outputs', 'two_sphere');
end
