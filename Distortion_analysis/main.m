% This is the distortion analysis presented in the LICONN publication[1]. 
% The code was adapted from previously published methods by the Chen group [2] with the relevant code: 
% https://github.com/Yujie-S/Click-ExM_data_process_and_example.git. 
% For distortion analysis details, see LICONN [1] Methods.

% References:
% [1] Mojtaba R. Tavakoli, Julia Lyudchik et al, Light-microscopy based dense connectomic reconstruction of mammalian brain tissue, bioRxiv 2024.03.01.582884; doi: https://doi.org/10.1101/2024.03.01.582884
% [2] Sun, De., Fan, X., Shi, Y. et al, Click-ExM enables expansion microscopy for all biomolecules. Nat Methods 18, 107–113 (2021). https://doi.org/10.1038/s41592-020-01005-2

close all;
clear all;

%% loading data, setting parameters
mkdir outputs; %creating a directory with the processed data
str = './outputs';
%loading aligned pre/post-expandsion images
pre = im2double(imread('pre.tif')); % pre-expansion image
post = im2double(imread('post.tif')); % post-expansion image
% parameters to adjust
pre_pix = 0.15; % pixel size of the pre-expansion image in micrometers
post_pix = 0.15; %pixel size of the post-expansion image in micrometers
exp_factor = 17.3; %expansion factor
periodicity = 60; %step in the mesh for the distortion field (in pixels)
scale = 1.5; %scale for the arrows in the distortion field
samples = 200000; % number of pairs of points sampled from the ROI for the measurement error calculation
bin_size = 6; % binning of the measurement length in pixels for the measurement error plot
crop_reg = [304, 66, 966, 1982]; % dimensions and location of the crop rectangle in pixels: [xmin, ymin, width, height]

%% pre-processing step
% selecting ROI
pre_crop = imcrop(pre, crop_reg);
post_crop = imcrop(post, crop_reg);
% 2D Gaussian smoothing
pre_gauss = imgaussfilt(pre_crop, 1); 
post_gauss = imgaussfilt(post_crop, 6); 
moving = post_gauss; 
fixed = pre_gauss;
% thresholding the pre- and post- expanded images
fixed_mask = imbinarize(fixed, 0.0028);
moving_mask = imbinarize(moving, 0.00188);
% combining masks of pre- and post- expanded images
comb = or(fixed_mask,moving_mask);
% filtering the main (the biggest) structure in the mask
cc4 = bwconncomp(comb,4);
L4 = labelmatrix(cc4);
s = regionprops(L4,'Area');
v = cell2mat(struct2cell(s));
x = find(v==max(v));
mask_main = L4 == x;
% dilating the mask
mask = imdilate(mask_main, strel('disk', 20)); 
% Inspecting the achieved mask
mask_fig = figure;
imshowpair(mask,fixed);
close(mask_fig);

%% transfomation via imregdemons.m
reg_nr = demons(moving, fixed);
Dis_x = reg_nr.DisplacementField(:,:,1);
Dis_y = reg_nr.DisplacementField(:,:,2);
Dis(:,:,1) = Dis_x;
Dis(:,:,2) = Dis_y;
%masking the distortion field
Dis_x_mask = Dis_x.*mask;
Dis_y_mask = Dis_y.*mask;
% generating a mesh for the distortion field
[fixed_size_x, fixed_size_y] = size(fixed);
x = 1:periodicity:fixed_size_x;    
y = 1:periodicity:fixed_size_y;
[mesh_x,mesh_y] = meshgrid(y,x);
dis_u_mesh = Dis_x_mask(x,y);
dis_v_mesh = Dis_y_mask(x,y);

%% visualization of the non-rigid registration results
nr_fig = figure;
set(gcf,'Position', [0 0 1500 800]);
subplot(1,3,1);
imshowpair(moving,fixed);
title({['Prior non-rigid registration (nr)']; ['']; ['green: post'];
    ['magenta: pre']});
subplot(1,3,2);
imshowpair(moving,reg_nr.RegisteredImage);     % non-rigid registration result
title({['After non-rigid registration']; ['']; ['green: post (before nr registration)'];
    ['magenta: post (after nr registration)']});
subplot(1,3,3);
imshowpair(moving,reg_nr.RegisteredImage);     % for distortion field plot
title({['Distortion field']; ['']; ['green: post  (before nr registration)'];
    ['magenta: post (after nr registration)']});
hold on;
quiver(mesh_x,mesh_y,-scale*dis_u_mesh, -scale*dis_v_mesh, "off", 'Color', [1 1 0.99],'LineWidth',1);     % plot distortion field
saveas(nr_fig, strcat(str,'/analysis_overview.png'), 'png'); 
close(nr_fig);


nr_fig2 = figure;
imshowpair(moving,reg_nr.RegisteredImage); 
hold on;
quiver(mesh_x,mesh_y,-scale*dis_u_mesh, -scale*dis_v_mesh, "off", 'Color', [1 1 0.99],'LineWidth',1);
saveas(nr_fig2, strcat(str,'/post- transformed & post- processed & distortion field.png'), 'png');
close(nr_fig2);

nr_fig4 = figure;
imshowpair(moving,fixed); 
saveas(nr_fig4, strcat(str,'/pre- & post-processed overlay.png'), 'png');
close(nr_fig4);

nr_fig3 = figure;
imshowpair(moving,fixed); 
hold on;
quiver(mesh_x,mesh_y,-scale*dis_u_mesh, -scale*dis_v_mesh, "off", 'Color', [1 1 0.99],'LineWidth',1); 
saveas(nr_fig3, strcat(str,'/pre- & post- processed & distortion field.png'), 'png'); 
close(nr_fig3);

%% Error calculation across different measurement lengths

[l, error] = error_calculation(Dis,mask,samples);
[l_binned, error_mean, error_std]=error_plot(l,error,bin_size);

% converting the measurement errors and measurement lengths to pre-expansion units
l_binned_u = l_binned.* post_pix ./ exp_factor;     
error_mean_u = error_mean .* post_pix ./ exp_factor;
error_std_u = error_std .* post_pix ./ exp_factor;
error_minus_std = error_mean_u - error_std_u;
error_plus_std = error_mean_u + error_std_u;

% smoothing the error plot with a median filter
error_minus_std_smooth = medfilt1(error_minus_std,30);
error_plus_std_smooth = medfilt1(error_plus_std,30);
error_smooth = medfilt1(error_mean_u,30);

mes_length = l_binned_u(1:length(error_mean_u)-30);
mes_error_mstd = error_minus_std_smooth(1:length(error_mean_u)-30);
mes_error_pstd = error_plus_std_smooth(1:length(error_mean_u)-30);
mes_error = error_smooth(1:length(error_mean_u)-30);

% plotting measurement error curves
error_plot = figure;
hold on;
plot(mes_length, mes_error_mstd, 'b--');
plot(mes_length, mes_error_pstd, 'b--');
plot(mes_length, mes_error, 'k-');
xlabel('measurement length (μm)');
ylabel('measurement error (μm)');

%% Saving the processed data
saveas(error_plot, strcat(str, '/error_plot.png'), 'png');
csvwrite(strcat(str, '/measurement_length.csv'), mes_length');
csvwrite(strcat(str, '/error_mean-std.csv'), mes_error_mstd);
csvwrite(strcat(str, '/error_mean+std.csv'), mes_error_pstd);
csvwrite(strcat(str, '/error_mean.csv'), mes_error);
imwrite(im2uint16(moving),strcat(str, '/post_processed.tif'));
imwrite(im2uint16(fixed),strcat(str, '/pre_processed.tif'));
imwrite(im2uint16(reg_nr.RegisteredImage),strcat(str, '/post_transformed_via_imregdemons.tif'));
imwrite(im2uint8(mask),strcat(str, '/mask.tif'));
close all;