function [d, error] = error_calculation(Dis, mask, N_rnd_pairs)
% error_calculation computes the measurement error
% 
% Input:
%   - Dis: a distortion field of size (height x width x 2), which stores the horizontal and vertical projections of the distortion field.
%   - mask (optional): a binary mask of size (height x width) that
%           represents the area used to select pairs of points for the measurement error calculation. 
%           Default is a mask of ones.
%   - N_rnd_pairs (optional): the number of randomly sampled pairs of points used to
%   compute the measurement error. Default is 100000.
%
% Output:
%   - d: the array with distances (measurement lengths) between pairs of points (in pixels)
%   - error: the measurement error
    
    U = Dis(:,:, 1);
    V = Dis(:,:, 2);

    img_shape = size(Dis);
    img_shape = img_shape(1:2);

    if nargin < 2 || isempty(mask)
        mask = ones(img_shape, 'logical');
    end

    [pts_x, pts_y] = find(mask);
    pts = [pts_x, pts_y];

    Up = U(mask);
    Vp = V(mask);

    offset = [Up, Vp];

    if nargin < 3 || isempty(N_rnd_pairs)
        N_rnd_pairs = 100000;
    end

    d = zeros(N_rnd_pairs, 1);
    error = zeros(N_rnd_pairs, 1);

    idx = randi(length(pts), N_rnd_pairs, 2);

    pt1 = pts(idx(:,1), :);
    pt2 = pts(idx(:,2), :);

    d = sqrt(sum((pt1 - pt2).^2, 2));

    offset1 = offset(idx(:,1), :);
    offset2 = offset(idx(:,2), :);
    
    error = sqrt(sum((offset2 - offset1).^2,2));
end
