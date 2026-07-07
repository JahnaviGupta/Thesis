clear;
clc;
close all;
% Add l1magic path
path(path,'./l1magic/Optimization');
%% ================= PARAMETERS =================
N = 64;
%% ================= OBJECT =================
img = imread("cro_64.tiff");
obj = im2double(img(:,:,1));
obj = imresize(obj,[N N]);
%% ================= SAMPLING PARAMETERS =================
mode = 'square';
low_percent  = 0.05;
rand_percent = 0.05; 
%% ================= FOURIER ORDERING =================
switch lower(mode)
    case 'square'
        order = [];
        mask = false(N,N);
        r = 1;
        while size(order,1) < N*N
            temp = false(N,N);
            temp(1:r,1:r) = 1;
            newpix = temp & ~mask;
            [ii,jj] = find(newpix);
            order = [order; ii jj];
            mask = temp;
            r = min(r+1,N);
        end
end
%% ================= FOURIER COORDINATES =================
[U,V] = meshgrid(-N/2:N/2-1,-N/2:N/2-1);
%% ================= SAMPLING =================
total_coeff = N*N;
low_num = round(low_percent * total_coeff);
remain_part = order(low_num+1:end,:);
rand_num = round(rand_percent * size(remain_part,1));
rng('shuffle');
idx = randperm(size(remain_part,1), rand_num);
final_order = [order(1:low_num,:); remain_part(idx,:)];
M = size(final_order,1);
fprintf('Sampling ratio = %.2f %%\n',100*M/(N*N));
%% ================= SPATIAL GRID =================

[X,Y] = meshgrid(0:N-1,0:N-1);

%% ================= STORAGE =================

b = zeros(M,1);

A = zeros(M,N*N);

%% ================= CONSTANTS =================

a = 1;

b_amp = 1;

%% ================= MEASUREMENT LOOP =================

for k = 1:M

    % fprintf('Measurement %d / %d\n',k,M);

    r = final_order(k,1);

    c = final_order(k,2);

    %% Fourier frequencies

    u = U(r,c);

    v = V(r,c);

    %% normalized frequencies

    Fx = u / N;

    Fy = v / N;

    %% ================= FOURIER PATTERNS =================

    p_0 = a + b_amp .* ...
        cos(2*pi*(Fx.*X + Fy.*Y) + 0);

    p_pi = a + b_amp .* ...
        cos(2*pi*(Fx.*X + Fy.*Y) + pi);

    p_pibi2 = a + b_amp .* ...
        cos(2*pi*(Fx.*X + Fy.*Y) + (pi/2));

    p_3pibi2 = a + b_amp .* ...
        cos(2*pi*(Fx.*X + Fy.*Y) + (3*pi/2));

    %% ================= BUCKET MEASUREMENTS =================

    K_0 = obj .* p_0;

    Sum_0 = sum(K_0(:));

    I_0 = Sum_0 .* conj(Sum_0);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    K_pi = obj .* p_pi;

    Sum_pi = sum(K_pi(:));

    I_pi = Sum_pi .* conj(Sum_pi);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    K_pibi2 = obj .* p_pibi2;

    Sum_pibi2 = sum(K_pibi2(:));

    I_pibi2 = Sum_pibi2 .* conj(Sum_pibi2);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    K_3pibi2 = obj .* p_3pibi2;

    Sum_3pibi2 = sum(K_3pibi2(:));

    I_3pibi2 = Sum_3pibi2 .* conj(Sum_3pibi2);

    %% ================= COMPLEX FOURIER COEFFICIENT =================

    C = ((I_0 - I_pi) + ...
    1i*(I_pibi2 - I_3pibi2)) / (4*a*b_amp);

    b(k) = C;

    %% ================= SENSING MATRIX =================

    phi = 2*pi*(Fx.*X + Fy.*Y);

    A(k,:) = exp(-1i*phi(:)).';

end
% %% ================= OPTIONAL NOISE =================
% noise_level = 1e-5;
% b = b + noise_level * ...
%     (randn(size(b)) + 1i*randn(size(b)));
%% ==========================================================
%% REAL AUGMENTED SYSTEM
%% ==========================================================

A2 = [real(A); imag(A)];

b2 = [real(b); imag(b)];

%% ==========================================================
%% REMOVE ZERO ROWS
%% ==========================================================

nz = vecnorm(A2,2,2) > 1e-12;

A2 = A2(nz,:);

b2 = b2(nz);

%% ==========================================================
%% GLOBAL NORMALIZATION ONLY
%% ==========================================================

A2 = double(A2);

A2 = A2 ./ sqrt(N*N);

b2 = b2 ./ sqrt(N*N);
%% ==========================================================
%% INVERSE RECONSTRUCTION
%% ==========================================================
x_inv = pinv(A2,1e-4) * b2;
x_inv_img = reshape(x_inv,[N N]);
%% ==========================================================
%% TV RECONSTRUCTION
%% ==========================================================
x0 = A2' * b2;
%% ================= INITIAL FEASIBLE POINT =================
epsilon = 0.005 * norm(b2);
% x0 = zeros(size(A2,2),1);
% 
% res = norm(A2*x0 - b2);
% 
% if res >= epsilon
% 
%     x0 = A2' * ((A2*A2' + 1e-6*eye(size(A2,1))) \ b2);
% 
% end
% 
% A2 = double(A2);

A2 = A2 ./ max(abs(A2(:)));
x_tv = tvqc_logbarrier(x0, A2, [], b2, epsilon, 1e-3);
x_tv_img = reshape(x_tv,[N N]);
% xp = l1qc_logbarrier(x0, A2, [], b2, epsilon, 1e-3);
% x_l1_img = reshape(xp, [N N]);
%% ==========================================================
%% DIRECT FOURIER RECONSTRUCTION
%% ==========================================================
Fspec = zeros(N,N);
for k = 1:M
    r = final_order(k,1);
    c = final_order(k,2);
    Fspec(r,c) = b(k);
end
%% inverse FFT reconstruction
direct_rec = abs(ifft2(ifftshift(Fspec)));
%% ==========================================================
%% RESULTS
%% ==========================================================
figure;
subplot(1,3,1);
imagesc(obj);
axis image off;
colormap gray;
title('Original');
subplot(1,3,2);
imagesc(log(1+abs(Fspec)));
axis image off;
colormap gray;
title('Fourier Spectrum');
subplot(1,3,3);
imagesc(direct_rec);
axis image off;
colormap gray;
title('Direct Reconstruction');
%% ==========================================================
%% INVERSE + TV COMPARISON
%% ==========================================================
%% ==========================================================
%% DISPLAY NORMALIZED IMAGES
%% ==========================================================

orig_disp = mat2gray(obj);

inv_disp = mat2gray(abs(x_inv_img));

tv_disp = mat2gray(abs(x_tv_img));
% l1_disp = mat2gray(abs(x_l1_img));

figure;

subplot(1,3,1);
imagesc(orig_disp); colormap gray; axis image;
title('Original');

subplot(1,3,2);
imagesc(inv_disp); colormap gray; axis image;
title('Inverse');

subplot(1,3,3);
imagesc(tv_disp); colormap gray; axis image;
title('TV');
% subplot(1,3,3);
% imagesc(l1_disp); colormap gray; axis image;
% title('L1');
%% ==========================================================
%% NORMALIZATION
%% ==========================================================
orig      = mat2gray(obj);
inv_rec = mat2gray(abs(x_inv_img));
tv_rec = mat2gray(abs(x_tv_img));
% l1_rec = mat2gray(abs(x_l1_img));
direct_rec = mat2gray(direct_rec);
%% ==========================================================
%% METRICS
%% ==========================================================

mse_inv = mean((orig(:) - inv_rec(:)).^2);

mse_tv = mean((orig(:) - tv_rec(:)).^2);
% mse_l1 = mean((orig(:) - l1_rec(:)).^2);

psnr_inv = 10 * log10(1 / mse_inv);

psnr_tv = 10 * log10(1 / mse_tv);
% psnr_l1 = 10 * log10(1 / mse_l1);

%% ==========================================================
%% FINAL METRICS
%% ==========================================================
fprintf('\n');
fprintf('Sampling ratio      = %.2f %%\n',100*M/(N*N));
fprintf('Low frequency used  = %.2f %%\n',low_percent*100);
fprintf('Random used         = %.2f %%\n',rand_percent*100);
fprintf('PSNR Inverse        = %.2f dB\n',psnr_inv);
% fprintf('PSNR L1             = %.2f dB\n',psnr_l1);
fprintf('PSNR TV             = %.2f dB\n',psnr_tv);