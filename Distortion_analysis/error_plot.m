function [bins, error_mean, error_std] = error_plot(d, error, size_bins)
% This function calculates the mean and standard deviation of the
% measurement error acoss different measurement length

% Input:
%   - d: the measurement length - distances between pairs of points (in pixels)
%   - error: the measurement error
%   - size_bins: the bin size for the measurement length
% Output:
%   - bins: the array with different measurement lengths
%   - error_mean: the array with average measurement errors for each measurement length
%   - error_std: the array with standart deviations of measurement errors for each measurement length

%   The code was adapted from previously published methods by the Chen group [1] with the relevant code: 
% https://github.com/Yujie-S/Click-ExM_data_process_and_example.git
%   [1] Sun, De., Fan, X., Shi, Y. et al, Click-ExM enables expansion microscopy for all biomolecules. Nat Methods 18, 107–113 (2021). https://doi.org/10.1038/s41592-020-01005-2

    bins = 0:size_bins:max(d);
    s = size(bins);
    n_bins = s(2)-1;
    bin_indx = discretize(d, bins);
    
    error_mean = zeros(n_bins,1);
    error_std = zeros(n_bins,1);
    
    for i = 1:1:n_bins-1
        pos_i = i==bin_indx;
        error_i = error(pos_i);
        error_mean(i) = mean(error_i);
        error_std(i) = std(error_i,1);
    end

end