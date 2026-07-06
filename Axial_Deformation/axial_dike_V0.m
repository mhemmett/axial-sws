clear all
close all

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

%%% Define Parameters

lambda = 3.3885e10;
mu = 1; % In Gigapascal
nu = 0.25; 
xi=0;yi=0;
xf=0;yf=5; % In km
zt=0;zb=4;
fault='tensile';
x=-2:0.01:2;
y=-2:0.01:2;
z=0:1:2;
U=1;
delta=90;

%%% Process
x = x(:)*1000;
y = y(:)*1000;
z = z(:)*1000;
[X,Y,Z]=meshgrid(x,y,z);
x=X(:);
y=Y(:);
z=Z(:);

xf=xf*1000;
xi=xi*1000;
yf=yf*1000;
xf=xf*1000;
zt=zt*1000;
zb=zb*1000;

[Ux Uy Uz dwdx dwdy eea gamma1 gamma2]=okada92(fault,xi,yi,xf,yf,zt,zb,U,delta,mu,nu,x,y,z);

Ux=reshape(Ux,size(X));
Uy=reshape(Uy,size(X));
Uz=reshape(Uz,size(X));

Uh=sqrt(Uy.^2+Ux.^2);

[c,h]=contourf(X(:,:,1),Y(:,:,1),Uh(:,:,1),20); clabel(c,h);
axis('equal')
