clc; clear;

%% ================= LOAD OBJECT =================
N = 256;
ih0 = imread('smile.jpg');
ih0 = ih0(:,:,1);
ih0 = imresize(ih0,[N N]);

obj = single(im2double(ih0));
[Nx, Ny] = size(obj);

M = 5000;

%% ================= PRECOMPUTE SPECKLE GRID =================
dx = 20e-3;
lx = 20;

[m,n] = meshgrid(-0.5*lx:dx:0.5*lx, +0.5*lx:-dx:-0.5*lx);

Rho_m = 0.0020;
A_g2 = exp(-(m.^2+n.^2)/Rho_m^2);
H2 = fft2(single(A_g2));

clear A_g2;

%% ================= CREATE MATFILE (ON DISK) =================
matObj = matfile('GI_data(speckle)_5000_256.mat','Writable',true);

% Allocate ON DISK (not RAM)
matObj.patterns = zeros(Nx,Ny,M,'single');
matObj.measurements = zeros(M,1,'single');

%% ================= INITIALIZE =================
measurements = zeros(M,1,'single');
GI = zeros(Nx,Ny,'single');

rng(1);   % reproducibility

%% ================= PASS 1: GENERATE + SAVE =================
for k = 1:M
    
    % Random field
    C_a2 = fft2(single(rand(size(m)) - 0.5));
    
    % Correlated field
    C_d2 = ifft2(C_a2 .* real(H2));
    
    % Phase screen
    C2 = exp(1i * 10*pi * (real(C_d2)) / max(real(C_d2(:))));
    C2 = C2(1:Nx,1:Ny);
    
    % Speckle intensity
    E_speckle = fft2(C2);
    I_speckle = abs(E_speckle).^2;
    
    I_speckle = I_speckle / mean(I_speckle(:));
    
    % Measurement
    measurements(k) = sum(I_speckle .* obj, 'all');
    
    % ===== SAVE DIRECTLY TO DISK =====
    matObj.patterns(:,:,k) = I_speckle;
    matObj.measurements(k,1) = measurements(k);
end

B_mean = mean(measurements);

%% ================= PASS 2: RECONSTRUCTION =================
rng(1);   % regenerate same patterns

for k = 1:M
    
    C_a2 = fft2(single(rand(size(m)) - 0.5));
    C_d2 = ifft2(C_a2 .* real(H2));
    
    C2 = exp(1i * 10*pi * (real(C_d2)) / max(real(C_d2(:))));
    C2 = C2(1:Nx,1:Ny);
    
    E_speckle = fft2(C2);
    I_speckle = abs(E_speckle).^2;
    
    I_speckle = I_speckle / mean(I_speckle(:));
    
    GI = GI + (measurements(k)/B_mean - 1) .* I_speckle;
end

GI = GI / M;

%% ================= DISPLAY =================
figure;
% subplot(1,2,1);
imagesc(obj); axis image off;
title('Original Object');
figure;
% subplot(1,2,2);
imagesc(GI); axis image off;
%title('Speckle Ghost Image');

%% ================= SAVE FINAL GI =================
%save('GI_reconstruction_1000_128.mat','GI','-v7.3');

