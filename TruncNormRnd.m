function r = TruncNormRnd(mu,sigma,l,u)
%RANDOMTRUNCATEDNORMAL Summary of this function goes here
%   Detailed explanation goes here
t=truncate(makedist('normal',mu,sigma),l,u);

r=random(t,1);

end

