clear all
close all

%%%%
% Script made to compute the deformation on Axial Seamount resulting from both
% the expension of a dike and inflation/deflation of the spheroid of Nononer
% All distances unit are convert to meters
% Warning: Yangdisp and okada92 do not use the same units for the shear modulus

% FAULT PARAMETERS
% fault     a string that define the kind of fault: strike, dip or tensile
% xi        x start
% yi        y start
% xf        x finish
% yf        y finish
% zt        top
%           (positive downward and defined as depth below the reference surface)
% zb        bottom; zb > zt
%           (positive downward and defined as depth below the reference surface)
% U         fault slip
%           strike slip fault: U > 0 right lateral strike slip
%           dip slip fault   : U > 0 reverse slip
%           tensile fault    : U > 0 tensile opening fault
% delta     dip angle from horizontal reference surface (90� = vertical fault)
%           delta can be between 0� and 90� but must be different from zero!
%
% CRUST PARAMETERS                                                                  
% mu        shear modulus
% nu        Poisson's ratio
%
% GEODETIC BENCHMARKS
% x,y       benchmark location (must be COLUMN vectors)
% z         depth of internal deformation (z=0 is the free surface)

scenario=9;

%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Define Parameters %%%
%%%%%%%%%%%%%%%%%%%%%%%%%

if scenario==1
    %%% 2 dikes + spheroid (deflation)
    dike_centers=[[8.08 5.45];[8.08 5.45]]; % km
    dike_angles=[85;120]; % CCW from East
    dike_lengths=[[0 4];[1.5 0]]; % in km, to the left and to the right
    dike_tops=[0.0;0]; % km
    dike_bots=[2,2];
    dike_deltas=[90,90];
    dike_Us=[1,1];
    dike_faults={'tensile','tensile'};
    sph_P = -0.1; %0.1; % in GPa (shoud be around mu *0.1)
    output_file='def_2_dike_1_sph.xyzuvw';
elseif scenario==2
    %%% 1 dike + no shperoid
    dike_centers=[[8.2 5.2];[8.2 5.2];[8.75 3.79];[8. 5.2]]; % km
    dike_angles=[-62,-62,118,85]; % CCW from East
    dike_lengths=[[0 1.5];[0 1.5];[0 1.5];[0 4]]; % in km, to the left and to the right
    dike_tops=[0.1,0.1,1,0]; % km
    dike_bots=[1.0,1,2,2];
    dike_deltas=[80,80,69,90];
    dike_Us=[-0,0,0,1];
    dike_faults={'dip','tensile','dip','tensile'};
    sph_P = -0;%-0.25; %0.1; % in GPa (shoud be around mu *0.1)
    output_file='def_2.xyzuvw';  
elseif scenario==3
    %%% 2 dikes + no spheroid
    dike_centers=[[8.08 5.45];[8.08 5.45]]; % km
    dike_angles=[85;120]; % CCW from East
    dike_lengths=[[0 4];[1.5 0]]; % in km, to the left and to the right
    dike_tops=[0.0;0]; % km
    sph_P = 0.0; %0.1; % in GPa (shoud be around mu *0.1)
    output_file='def_2_dike_0_sph.xyzuvw';
elseif scenario==4
    %%% 0 dikes + spheroid
    dike_centers=[]; % km
    dike_angles=[85;120]; % CCW from East
    dike_lengths=[[0 4];[1.5 0]]; % in km, to the left and to the right
    dike_tops=[0.0;0]; % km
    sph_P = -0.13; %0.1; % in GPa (shoud be around mu *0.1)   
    output_file='def_7.xyzuvw';
elseif scenario==5
    %%% 3 dikes + no shperoid
    dike_centers=[[8.4 5.4];[9 3];[8. 5.2]]; % km
    dike_angles=[-62,118,85]; % CCW from East
    dike_lengths=[[0 1.5];[0 2];[0 4]]; % in km, to the left and to the right
    dike_tops=[0.0,1,0]; % km
    dike_bots=[1.0,2,2];
    dike_deltas=[80,69,90];
    dike_Us=[-1.2,-1,1];
    dike_faults={'dip','dip','tensile'};
    sph_P = -0.05; %0.1; % in GPa (shoud be around mu *0.1)
    output_file='def_3_dike_1_sph_nnt.xyzuvw';
elseif scenario==6
    %%% 3 dikes + no shperoid
    dike_centers=[[8.4 5.4];[9 3];[8. 5.2]]; % km
    dike_angles=[-62,118,85]; % CCW from East
    dike_lengths=[[0 1.5];[0 2];[0 4]]; % in km, to the left and to the right
    dike_tops=[0.0,1,0]; % km
    dike_bots=[1.0,2,2];
    dike_deltas=[80,69,90];
    dike_Us=[1,-1,1];
    dike_faults={'tensile','dip','tensile'};
    sph_P = -0.08; %0.1; % in GPa (shoud be around mu *0.1)
    output_file='def_3_dike_1_sph_tnt.xyzuvw';
 elseif scenario==7
    %%% 3 dikes + no shperoid
    dike_centers=[[8.2 5.2];[8.2 5.2];[8.75 3.79];[8.2 5.2]]; % km
    dike_angles=[-62,-62,118,89]; % CCW from East
    dike_lengths=[[0 1.5];[0 1.5];[0 1.5];[0 4]]; % in km, to the left and to the right
    dike_tops=[0.1,0.1,1,0]; % km
    dike_bots=[1.0,1,2,2];
    dike_deltas=[80,80,69,90];
    dike_Us=[-0.2,1,1,1];
    dike_faults={'dip','tensile','dip','tensile'};
    sph_P = -0.13;%-0.115;%-0.25; %0.1; % in GPa (shoud be around mu *0.1)
    output_file='def_6.xyzuvw';  
  elseif scenario==8  
    %%% 3 dikes + Inflation
 
    dike_centers=[[8.2 5.2];[8.2 5.2];[8.75 3.79];[9.05 4.4];[9.05 4.4]]; % km
    dike_angles=[-62,-62,118,97,-99]; % CCW from East
    dike_lengths=[[0 1.5];[0 1.5];[0 1.5];[0,3.7];[0,5]]; % in km, to the left and to the right
    dike_tops=[0.1,0.1,1, 0,0]; % km
    dike_bots=[1.0,1,2, 2,2];
    dike_deltas=[80,80,69, 90,90];
    dike_Us=[+0.2,0, -1, 1.2,1];
    dike_faults={'dip','tensile','dip', 'tensile','tensile'};
    sph_P = +0.1;%-0.115;%-0.25; %0.1; % in GPa (shoud be around mu *0.1)
    output_file='def_pre_2.xyzuvw';  
   
   elseif scenario==9  
    %%% 3 dikes + Inflation
    % names [FI1,FI1,FO1,D2,D3,FI2,FO2]
    dike_centers=[[8.2 5.2];[8.2 5.2];[8.75 3.79];[9.05 4.4];[9.05 4.4];[7.12,4.88];[6.92 2.6]]; % km
    dike_angles=[-62,-62,118,97,-99,-68,112]; % CCW from East
    dike_lengths=[[0 1.5];[0 1.5];[0 1.5];[0,3.7];[0,5];[0,2];[0,2]]; % in km, to the left and to the right
    dike_tops=[0.1,0.1,1, 0,0 ,0.1,0]; % km
    dike_bots=[1.0,1,2, 2,2, 1,0.3];
    dike_deltas=[80,80,69, 90,90,45,45];
    dike_Us=[+0.2,0, -1, 1.2,1, -0,0];
    dike_faults={'dip','tensile','dip', 'tensile','tensile', 'dip','dip'};
    sph_P = +0.08;%-0.115;%-0.25; %0.1; % in GPa (shoud be around mu *0.1)
    output_file='def_pre_2.xyzuvw';  
   
    
end


%%% Station cells

sta_coord=containers.Map;
sta_coord('AXCC1')=[7.14,5.98,-2.45];
sta_coord('AXEC2')=[9.84,4.28,-1.06];
sta_coord('AXID1')=[9.49,2.73,-1.38];
sta_coord('AXAS1')=[7.85,3.62,-2.00];



%%% General
%%% Dike Geometry


[num_dikes,~]=size(dike_centers);

%%% Spheroid Geometry

sph_x0 = 8.84; %8.84; % km
sph_y0 = 5.38;%4;%5.38; % km
sph_z0 = 3.81; % km
sph_a = 2.2; % semi major axis km
sph_b = 0.38; % semi minor axis km
sph_theta = 77; % from horizontal (90° means vertical)
sph_phi = 286; % CW from North


% sph_x0 = 8.84; %8.84; % km
% sph_y0 = 5.38;%4;%5.38; % km
% sph_z0 = 1.0; % km
% sph_a = 2; % semi major axis km
% sph_b = 0.2; % semi minor axis km
% sph_theta = 1; % from horizontal (90° means vertical)
% sph_phi = 1; % CW from North



%%% Physical parameters

%lambda = 1; % in GPa
mu = 1; % in GPa
nu = 0.25;
lambda = (2*nu*mu)/(1-2*nu);

%%% Grid parameters

x=4:0.05:11; % km 
y=1:0.05:10; % km
z=0:0.5:1; % km

%%%%%%%%%%%%%%%
%%% Prepare %%%
%%%%%%%%%%%%%%%

%%% Grid
x = x(:)*1000;
y = y(:)*1000;
z = z(:)*1000;
[X,Y,Z]=meshgrid(x,y,z);
x=X(:);
y=Y(:);
z=Z(:);


tot_Ux = zeros(size(x));
tot_Uy = zeros(size(x));
tot_Uz = zeros(size(x));

%%% Dike deformation

for k_dike=1:num_dikes
   %%% Assign
   dike_center=dike_centers(k_dike,:);
   dike_angle=dike_angles(k_dike);
   dike_length=dike_lengths(k_dike,:);
   dike_top=dike_tops(k_dike);
   dike_bot=dike_bots(k_dike);
   dike_U=dike_Us(k_dike);
   dike_delta=dike_deltas(k_dike);
   dike_fault=dike_faults{k_dike};
   
   %%% Compute
   
   dike_end=dike_center+dike_length(2)*[cosd(dike_angle) sind(dike_angle)];
    dike_start=dike_center-dike_length(1)*[cosd(dike_angle) sind(dike_angle)];
    dike_xi=dike_start(1)*1000;
    dike_yi=dike_start(2)*1000;
    dike_xf=dike_end(1)*1000;
    dike_yf=dike_end(2)*1000;
    dike_zt=dike_top*1000;
    dike_zb=dike_bot*1000;

    [dike_Ux, dike_Uy,dike_Uz] = okada92(dike_fault,...
        dike_xi,dike_yi,dike_xf,dike_yf,dike_zt,dike_zb,...
        dike_U,dike_delta,mu,nu,x,y,z);
    
    tot_Ux=tot_Ux+dike_Ux;
    tot_Uy=tot_Uy+dike_Uy;
    tot_Uz=tot_Uz+dike_Uz;

end

sph_x0 = sph_x0 *1000; 
sph_y0 = sph_y0 *1000; 
sph_z0 = sph_z0 *1000; 
sph_a = sph_a *1000; 
sph_b = sph_b *1000; 
sph_theta = sph_theta*pi/180;
sph_phi = sph_phi*pi/180;
sph_lambda=lambda*1e9; % in Pa
sph_mu=mu*1e9;
sph_P=sph_P*1e9;

%%% Spheroid deformation

[sph_Ux, sph_Uy,sph_Uz] = yangdisp(sph_x0,sph_y0,sph_z0,sph_a,sph_b,...
sph_lambda,sph_mu,nu,sph_P,sph_theta,sph_phi,x,y,z);

tot_Ux=tot_Ux+sph_Ux;
tot_Uy=tot_Uy+sph_Uy;
tot_Uz=tot_Uz+sph_Uz;

%%% Print to file

data=[x/1000,y/1000,z/1000,tot_Ux,tot_Uy,tot_Uz];
[nx,ny,nz]=size(X);
fic = fopen(output_file,'w');
fprintf(fic,'nx %i ny %i nz %i\n',nx,ny,nz); % header
fprintf(fic,'%7s %7s %7s %7s %7s %7s\n','X','Y','Z','Ux','Uy','Uz');
fprintf(fic,'%7.3f %7.3f %7.3f %7.3f %7.3f %7.3f\n',transpose(data));
fclose(fic);

%%%%%%%%%%%%
%%% Plot %%%
%%%%%%%%%%%%
% 
% sph_Ux=reshape(sph_Ux,size(X));
% sph_Uy=reshape(sph_Uy,size(X));
% sph_Uz=reshape(sph_Uz,size(X));
% 
tot_Ux=reshape(tot_Ux,size(X));
tot_Uy=reshape(tot_Uy,size(X));
tot_Uz=reshape(tot_Uz,size(X));

tot_Uh=sqrt(tot_Uy.^2+tot_Ux.^2);

close all

figure()
[c,h]=contourf(X(:,:,1),Y(:,:,1),tot_Uh(:,:,1)); clabel(c,h);
axis('equal')
stations = keys(sta_coord) ;
vals = values(sta_coord) ;
 for i = 1:length(sta_coord)
    station=stations{i};
    val=vals{i};
    t = num2cell(val);
    [xs,ys,dzs] = deal(t{:});
    xs=xs*1000;
    ys=ys*1000;
    hold on
    plot(xs,ys,'ok','MarkerFaceColor','k')
 end

figure()
[c,h]=contourf(X(:,:,1),Y(:,:,1),tot_Uz(:,:,1),20); clabel(c,h);
axis('equal')
stations = keys(sta_coord) ;
vals = values(sta_coord) ;
 for i = 1:length(sta_coord)
    station=stations{i};
    val=vals{i};
    t = num2cell(val);
    [xs,ys,dzs] = deal(t{:});
    xs=xs*1000;
    ys=ys*1000;
    hold on
    plot(xs,ys,'ok','MarkerFaceColor','k')
 end




% [Ux Uy Uz dwdx dwdy eea gamma1 gamma2]=okada92(fault,xi,yi,xf,yf,zt,zb,U,delta,mu,nu,x,y,z);
% 
% 
% Uh=sqrt(Uy.^2+Ux.^2);


