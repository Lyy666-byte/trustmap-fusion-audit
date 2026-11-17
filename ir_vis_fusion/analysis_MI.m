function MI = analysis_MI(A,B,F)
%ANALYSIS_MI  Mutual information MI(A,F) + MI(B,F), in bits.
%   A, B, F: uint8 grayscale images in [0,255] of identical size.
%
%   BUG HISTORY (both found and fixed 2026-09):
%   (1) The original code called MutualInformation(), a routine written for
%       COLUMN VECTORS of samples. Passing a 2-D image took its size(X,2)>1
%       branch, in which JointEntropy() evaluates entropy(cumsum(...)) on a
%       monotone double vector; MATLAB's entropy() pushes that through
%       im2uint8, saturating every value to a single bin and returning 0.
%       The estimator therefore degenerated to  0 + entropy(F) - 0  for each
%       source, i.e. MI == 2*EN for every image, carrying no cross-modal
%       information whatsoever.
%   (2) The joint-histogram routine shipped alongside it
%       (mutual_information_images) swapped the two marginals: it used
%       sum(JPDF,1) - the marginal of the SECOND image - indexed by the FIRST
%       image's intensity, and vice versa. For identical inputs the joint
%       histogram is symmetric so MI(F,F) still came out right, but for two
%       different images the denominator could be zero and the sum returned
%       Inf.
%
%   This implementation is the standard 256x256 joint-histogram plug-in
%   estimator, vectorised. Sanity property: MI(X,X) == H(X).

MI = mi_pair(A,F) + mi_pair(B,F);
end

% -------------------------------------------------------------------------
function I = mi_pair(X,Y)
X = double(X(:)) + 1;          % intensities 1..256
Y = double(Y(:)) + 1;
J = accumarray([X Y], 1, [256 256]);
J = J / sum(J(:));             % joint pdf

px = sum(J,2);                 % marginal of X, indexed by row  i
py = sum(J,1);                 % marginal of Y, indexed by col  j

[ii,jj] = find(J > 0);
p  = J(sub2ind([256 256], ii, jj));
den = px(ii) .* py(jj)';       % py is 1x256 -> py(jj)' is a column
I  = sum( p .* log2( p ./ den ) );
end
