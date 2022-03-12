function [ y, gm ] = rbf_Boot( ncl, dim, xinput, Ci, s, wj )
%                       RBF NEURAL NETWORK MODEL
%   Detailed explanation goes here
y = 0;
gm = zeros(1,ncl);
%..........................................................................
for i=1:ncl
    for r=1:dim
        S3(i,r) = -((xinput(r)- Ci(i,r)).^2/s(i).^2);
    end
        S2(1,i) = exp(sum(S3(i,:)));% hasta aqui ok
       % Rm(iter,i)=S2(1,i);  % R es para LMS
 end
     S=sum(S2(1,:));

 for i=1:ncl
     gm(1,i)=S2(1,i)/S;                                 % normalized output
  %  if Iter==1
  %      H(iter,i) = gm(1,i); % Para LMS
  %   end
 end

 for i=1:ncl
     y = y + wj(i)*gm(1,i);
 end
%..........................................................................
end

