clc;
clear;
close all;

%% ---------------- PARAMETERS ----------------
a = 1;
b = 1;

lx1 = 6.36;
ly1 = lx1;
dx1 = 5.3e-3;
dy1 = dx1;

%% ---------------- GRID ----------------
[m1,n1] = meshgrid(-0.5*lx1:dx1:0.5*lx1, ...
                   +0.5*ly1:-dy1:-0.5*ly1);

%% ---------------- T & R ----------------
T = zeros(500);
R = zeros(500);

for x = 1:500
    for y = 1:500
        T(x,y) = 2*pi*0.02*(x-251);
        R(x,y) = 2*pi*0.02*(y-251);
    end
end

%% ---------------- CREATE DIRECTORIES (ONCE) ----------------
basePath = '/Users/jahnavigupta/Downloads/pattern_360';

dir0 = fullfile(basePath,'pattern_cir_0');
dir2 = fullfile(basePath,'pattern_cir_2pibi3');
dir4 = fullfile(basePath,'pattern_cir_4pibi3');

if ~exist(dir0,'dir'), mkdir(dir0); end
if ~exist(dir2,'dir'), mkdir(dir2); end
if ~exist(dir4,'dir'), mkdir(dir4); end

%% ---------------- MAIN LOOP ----------------
for l = 101:10:350
    for p = 101:10:360

        fprintf('Iteration: %d, sub: %d\n', l, p);

        %% Phase-shifted patterns
        p_0        = a + b*cos( m1(l,p)*T + n1(l,p)*R );
        p_2pibi3   = a + b*cos( m1(l,p)*T + n1(l,p)*R + 2*pi/3 );
        p_4pibi3   = a + b*cos( m1(l,p)*T + n1(l,p)*R + 4*pi/3 );

        %% Normalize
        I_0        = double(p_0)/2;
        I_2pibi3  = double(p_2pibi3)/2;
        I_4pibi3  = double(p_4pibi3)/2;

        %% Embed into DMD-sized frame (768 × 1280)
        V_0 = zeros(768,1280);
        V_2 = zeros(768,1280);
        V_4 = zeros(768,1280);

        V_0(384-249:384+250,640-249:640+250) = I_0;
        V_2(384-249:384+250,640-249:640+250) = I_2pibi3;
        V_4(384-249:384+250,640-249:640+250) = I_4pibi3;

        %% Convert to uint8
        V_0u = uint8(V_0 * 255);
        V_2u = uint8(V_2 * 255);
        V_4u = uint8(V_4 * 255);

        %% Save images
        imwrite(V_0u, fullfile(dir0, ...
            ['pat0_kx_',num2str(l),'_ky_',num2str(p),'.bmp']));

        imwrite(V_2u, fullfile(dir2, ...
            ['pat2_kx_',num2str(l),'_ky_',num2str(p),'.bmp']));

        imwrite(V_4u, fullfile(dir4, ...
            ['pat4_kx_',num2str(l),'_ky_',num2str(p),'.bmp']));
    end
end
