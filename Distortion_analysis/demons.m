function [moving_nr] = demons(moving,fixed)
%non-rigid registration via imregdemons.m (https://de.mathworks.com/help/images/ref/imregdemons.html)

%Inputs:
%   - moving: moving image for non-rigid registration
%   - fixed: fixed image for non-rigid registration
%Outputs:
%   - moving_nr: the array with the distortion field and the registered
%   image via imregdemons.m

%   The code was adapted from previously published methods by the Chen group [1] with the relevant code: 
% https://github.com/Yujie-S/Click-ExM_data_process_and_example.git
%   [1] Sun, De., Fan, X., Shi, Y. et al, Click-ExM enables expansion microscopy for all biomolecules. Nat Methods 18, 107–113 (2021). https://doi.org/10.1038/s41592-020-01005-2

% Normalizing fixed image
finiteIdx = isfinite(fixed(:));
fixed(isnan(fixed)) = 0;
fixed(fixed==Inf) = 1;
fixed(fixed==-Inf) = 0;
fixedmin = min(fixed(:));
fixedmax = max(fixed(:));

if isequal(fixedmax,fixedmin)
    fixed = 0*fixed;
else
    fixed(finiteIdx) = (fixed(finiteIdx) - fixedmin) ./ (fixedmax - fixedmin);
end

% Normalizing moving image
finiteIdx2 = isfinite(moving(:));
moving(isnan(moving)) = 0;
moving(moving==Inf) = 1;
moving(moving==-Inf) = 0;
movingmin = min(moving(:));
movingmax = max(moving(:));
if isequal(movingmax,movingmin)
    moving = 0*moving;
else
    moving(finiteIdx2) = (moving(finiteIdx2) - movingmin) ./ (movingmax - movingmin);
end

% Non-rigid registration
[moving_nr.DisplacementField,moving_nr.RegisteredImage] = imregdemons(moving,fixed,[300,200,150,100,50],'AccumulatedFieldSmoothing',2.0,'PyramidLevels',5);

end

