clear; clc;

%% ================= PARAMETERS =================
N = 32;
num_patterns = N * N;

%% ================= LOAD OBJECT =================
% data = load('boats.mat');
% 
% ih2 = data.boats;
ih2 = imread('smile.jpg');
ih2 = im2double(ih2);
ih2 = imresize(ih2,[N N]);

%% ================= CREATE MAT FILE =================
matObj = matfile('DHSI_dataset_32.mat','Writable',true);

% Allocate directly on disk
matObj.patterns     = zeros(N,N,num_patterns,'single');
matObj.measurements = zeros(num_patterns,1,'single');

%% ================= STORAGE =================
h_sp = zeros(N,N,'single');
h_sm = zeros(N,N,'single');

%% ================= DIFFERENTIAL HSI =================
idx = 1;

for i = 1:N
    
    % fprintf('Iteration: %d\n',i);

    for j = 1:N
        
        % fprintf('Sub: %d\n',j);

        %% Delta basis
        p_0 = zeros(N,N,'single');
        p_0(i,j) = 1;

        %% Hadamard basis
        e = ifwht(ifwht(p_0')') ;

        H = single(e);

        %% Positive & Negative patterns
        Hp = 0.5 * (1 + H);
        Hm = 0.5 * (1 - H);

        %% Bucket measurements
        e_p = Hp .* ih2;
        e_m = Hm .* ih2;

        ce_p = sum(e_p(:));
        ce_m = sum(e_m(:));

        I_p = ce_p .* conj(ce_p);
        I_m = ce_m .* conj(ce_m);

        %% Differential measurement
        C = I_p - I_m;

        %% Save ON DISK
        matObj.patterns(:,:,idx) = H;
        matObj.measurements(idx,1) = single(C);

        %% Optional storage for direct reconstruction
        h_sp(i,j) = I_p;
        h_sm(i,j) = I_m;

        idx = idx + 1;
    end
end

%% ================= DIRECT RECONSTRUCTION =================
C_xy = h_sp - h_sm;

d  = ifwht(C_xy);
hd = ifwht(d');

recon = hd';

%% ================= DISPLAY =================
figure;

subplot(1,2,1);
imagesc(ih2);
axis image off;
colormap gray;
title('Original Object');

subplot(1,2,2);
imagesc(recon);
axis image off;
colormap parula;
colorbar;
title('Differential HSI Reconstruction');