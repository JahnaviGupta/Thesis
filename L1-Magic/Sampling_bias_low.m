clear; clc;

% Add l1magic
path(path, './l1magic/Optimization');

%% ================= PARAMETERS =================
N = 64;

img = imread('cro_64.tiff');
obj = im2double(img(:,:,1));
obj = imresize(obj, [N N]);

x_true = obj(:);

mode = 'square';

low_percent  = 0.30;
rand_percent = 0.00;

%% ================= ORDERING =================
switch lower(mode)
    case 'square'
        order = [];
        mask = false(N,N);
        r = 1;
        while size(order,1) < N*N
            temp = false(N,N);
            temp(1:r,1:r)=1;
            newpix = temp & ~mask;
            [ii,jj] = find(newpix);
            order = [order; ii jj];
            mask = temp;
            r = min(r+1,N);
        end
end

%% ================= SAMPLING (PURE LOW FREQUENCY) =================

total_coeff = N*N;

M = round(low_percent * total_coeff);

% take ONLY first M coefficients (low frequency region)
final_order = order(1:M, :);

fprintf('Sampling ratio = %.2f %%\n',100*M/(N*N));

%% ================= BUILD A AND b =================
A = zeros(M, N*N);
b = zeros(M,1);

for k = 1:M
    
    i = final_order(k,1);
    j = final_order(k,2);
    
    delta = zeros(N,N);
    delta(i,j)=1;
    
    % Hadamard basis (ORTHONORMAL)
    H = ifwht(ifwht(delta')') / N;
 
    % Differential patterns
    Hp = (1 + H)/2;
    Hm = (1 - H)/2;
    % Bucket signals
    Ip = sum(sum(Hp .* obj));
    Im = sum(sum(Hm .* obj));
    h_sp = (abs(Ip)).^2;
    h_sm = (abs(Im)).^2;
    
    % Linear measurement
    b(k) = h_sp-h_sm;
    
    A(k,:) = H(:)';
end

%% ================= ADD NOISE =================
b = b + 0.01 * randn(size(b));

%% ================= INVERSE =================
x_inv = pinv(A) * b;
x_inv_img = reshape(x_inv, [N N]);

%% ================= TV RECONSTRUCTION =================
x0 = A' * b;

epsilon = 0.005 * norm(b);

% x_tv = tvqc_logbarrier(x0, A, [], b, epsilon, 1e-3);
% x_tv_img = reshape(x_tv, [N N]);
xp = l1qc_logbarrier(x0, A, [], b, epsilon, 1e-3);
x_l1_img = reshape(xp, [N N]);

%% ================= DIRECT HADAMARD (PARTIAL) =================
C = zeros(N,N);

for k = 1:M
    i = final_order(k,1);
    j = final_order(k,2);
    C(i,j) = b(k);
end

temp = ifwht(C);
rec  = ifwht(temp')' / N;
rec  = real(rec);

%% ================= HADAMARD SPECTRUM VIEW =================
% figure;
% 
% subplot(1,3,1);
% imagesc(obj);
% axis image off;
% colormap gray;
% title('Original');
% 
% subplot(1,3,2);
% imagesc(log(abs(C)+1));
% axis image off;
% colormap gray;
% title('Measured Spectrum');
% 
% subplot(1,3,3);
% imagesc(rec);
% axis image off;
% colormap gray;
% title('Hadamard Reconstruction');
%% ================= DISPLAY =================
figure;

subplot(1,3,1);
imagesc(obj); colormap gray; axis image;
title('Original');

subplot(1,3,2);
imagesc(x_inv_img); colormap gray; axis image;
title('Inverse');

% subplot(1,3,3);
% imagesc(x_tv_img); colormap gray; axis image;
% title('TV');
subplot(1,3,3);
imagesc(x_l1_img); colormap gray; axis image;
title('L1');

% ================= NORMALIZATION FOR PSNR =================
obj_n = (obj - min(obj(:))) / (max(obj(:)) - min(obj(:)));

% x_tv_n = (x_tv_img - min(x_tv_img(:))) / ...
%          (max(x_tv_img(:)) - min(x_tv_img(:)));
x_l1_n = (x_l1_img - min(x_l1_img(:))) / ...
         (max(x_l1_img(:)) - min(x_l1_img(:)));

x_inv_n = (x_inv_img - min(x_inv_img(:))) / ...
          (max(x_inv_img(:)) - min(x_inv_img(:)));

rec_n = (rec - min(rec(:))) / ...
        (max(rec(:)) - min(rec(:)));

mse_inv = mean((obj_n(:) - x_inv_n(:)).^2);
% mse_tv  = mean((obj_n(:) - x_tv_n(:)).^2);
mse_l1  = mean((obj_n(:) - x_l1_n(:)).^2);

psnr_inv = 10 * log10(1 / mse_inv);
% psnr_tv  = 10 * log10(1 / mse_tv);
psnr_l1  = 10 * log10(1 / mse_l1);

fprintf('\n');
fprintf('Sampling ratio      = %.2f %%\n',100*M/(N*N));
fprintf('Low frequency used  = %.2f %%\n',low_percent*100);
fprintf('Random used         = %.2f %%\n',rand_percent*100);
fprintf('PSNR Inverse        = %.2f dB\n',psnr_inv);
fprintf('PSNR L1             = %.2f dB\n',psnr_l1);
% fprintf('PSNR TV             = %.2f dB\n',psnr_tv);