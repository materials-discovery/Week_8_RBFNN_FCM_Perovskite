%--------------------------------------------------------------------------
%    GRADIENT DESCENT APPLIED TO Radial Basis Function neural NetworK
%                            and Fuzzy C-Means
%               this code is to model Perovskite Properties
%--------------------------------------------------------------------------
clear;
clc;
% read the data sets you have in the excel files
input_data = csvread('dataset.csv'); % inputs to the RBF neural Network
% NORMALISE THE INPUT (ORIGINAL) DATA SET
X = normalisation_1( input_data, 1 );
% READ THE OUTPUT WHICH IS THE TARGET (NOTE: REMEMBER THIS IS SUPERVISED LEARNING)
target = csvread('Tc_temperature.csv');
%target = normalisation_1( target, 2 );
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%--------------------------------------------------------------------------
%                       CROSS-VALIDATION DATA SPLIT
%--------------------------------------------------------------------------
samples4training = round(0.8*size(X,1));
dimension = size(X,2);
samples4testing = size(X,1) - (samples4training);
px = randperm(size(X,1));
% Variables for training
data4training = zeros(samples4training,dimension);
label4training = zeros(samples4training,1);
% Variables for testing
data4testing = zeros(samples4testing,dimension);
label4testing = zeros(samples4testing,1);
%..........................................................................
%                                Data 4 Training
%..........................................................................
% for training (data for)
index = 0;

   for k = 1:samples4training
         index = index + 1;
         data4training(index,:) = X(px(k),1:end);
         label4training(index,1) = target(px(k),1);
   end
%           Creating Testing Data
index = 0;
%..........................................................................
% this is the loopr testing
    for k = samples4training + 1:samples4training + samples4testing 
        index = index + 1;
                data4testing(index,:) = X(px(k),1:end);
                label4testing(index,1) = target(px(k),1);
    end
    
%..........................................................................
%  Number of Clusters
ncl = 25;
%                                - Fuzzy C-Means -
[ci,U] = fcm(data4training,ncl);
% Initialization of RBFNN parameters
s = 0.5*ones(1,ncl); % s is sigma 
% Output weights wj
wi = 177*ones(1,ncl);
% Extract number of Patterns or Samples and dimensions
[P dim] = size(data4training); 
%..........................................................................
% Parameters for training he neural network
eta_n = 0.00008;  % Learning RATE
%--------------------------------------------------------------------------
max_iterations = 3000; 
rmse4test = zeros(max_iterations,1);
%..........................................................................
%  Creating labels for yd (target output)
%..........................................................................
% Create variables used for the Learning Training 
 inc_cm = zeros(1,ncl); % increment/update for 
 inc_dm = zeros(1,ncl);
 inc_Cm = zeros(ncl,dim);
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%%                        - MAIN TRAINING LOOP -
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for Iter=1:max_iterations     % Loop for the iterations
    Pi(Iter) = 0;      
    rmse(Iter) = 0;
%--------------------------------------------------------------------------    
for iter=1:P  % Loop for presenting all the samples
    y(iter,1)=0.0;
%-------------------------------------------------------------------------
 [ y(iter), gm ] = rbf_Boot( ncl, dim, data4training(iter,:), ci, s, wi );
 if y(iter) < 0 % Limiter
 y(iter) = 0;
 end
%--------------------------------------------------------------------------
                          % Error Calculation 
%--------------------------------------------------------------------------
  e = (y(iter) - label4training(iter,1));
  error(iter) = e^2;
  rmse(Iter) = rmse(Iter) + e^2;
%**************************************************************************
if Iter 
Cm = ci; % centers
dm = s;  % Standard deviation 
cm = wi; % Output weigths
%--------------------------------------------------------------------------
%                   Computing PArtial derivatives
%                       - BEP On-line Update -
%--------------------------------------------------------------------------
%Cm used to update Centers
%dm used to update standard deviation of each Gaussian
%cm Used to update each output weight wj
for k = 1:ncl
    Sq(1,k) = 0;
    for k1=1:dim
        % Membership Centers Update (Cik)
        aux = eta_n*e*gm(1,k)*(wi(1,k)-y(iter,1)); 
        inc_Cm(k,k1) = ( -aux*(data4training(iter,k1) - ci(k,k1))*(1/dm(k)^2) );
        Cm(k,k1) = Cm(k,k1) + inc_Cm(k,k1); 
        aux1 = (data4training(iter,k1) - ci(k,k1))^2;
        Sq(k) = Sq(k) + (aux1/s(k)^3);   
    end
    % Width update (si standard deviation)
    inc_dm(k) = -aux*Sq(k);
    dm(k)= dm(k) + inc_dm(k); 
    % Consequent Update (Output weigth - wi)
    inc_cm(k) = -(eta_n*e*gm(1,k));
    cm(k)= cm(k) + inc_cm(k); 
    %......................................................................
end
%--------------------------------------------------------------------------
min_width = 0.5;  % min value for the standard deviation "si"
 for ik = 1:ncl
    if dm(ik)<min_width
              dm(ik) = min_width; %  As the width of each Gaussian must not be zero we use a lower limit of 0.1
    else    
    end    
 end
%--------------------------------------------------------------------------
     ci = Cm;
     s  = dm; % s is sigma or the width of each n-Gaussian Function (Activation function)
     wi = cm;
end
%--------------------------------------------------------------------------
end
%--------------------------------------------------------------------------
rmse(Iter) = sqrt(rmse(Iter)/P);
R2(Iter) = rsquare(label4training,y);
% yd  the target which is the temperature Tc in the article
% y is the output predicted by the Radial Basis Function NN
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Section for testing the trained neural network
for iter4test=1:samples4testing  % Loop for presenting all the samples
     ytest(iter4test,1) = 0.0;
%-------------------------------------------------------------------------
  [ ytest(iter4test), gm4test ] = rbf_Boot( ncl, dim, data4testing(iter4test,:), ci, s, wi );
      if ytest(iter4test) < 0 % Limiter
         ytest(iter4test) = 0;
      end
%--------------------------------------------------------------------------
%                           % Error Calculation 
%--------------------------------------------------------------------------
   error4test = (ytest(iter4test) - label4testing(iter4test,1));
   rmse4test(Iter) = rmse4test(Iter) + error4test.^2;
end % end for loop 
 rmse4test(Iter) = sqrt(rmse4test(Iter)/samples4testing);
%--------------------------------------------------------------------------
% PRINTING RESULTS
[Iter rmse(Iter) rmse4test(Iter)  eta_n R2(Iter)*1000 ] % THIS LINE IS USED TO PRINT IN THE COMMAND WINDOW VALUES YOU WANT TO PRINT
end % end of fisrt loop
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%% PLOTTING RESULTS Y VS LABEL
%..........................................................................
plot_type = 1; % if equal to 1 training results are plot, otherwise for testing
if plot_type == 2
   plot(label4training,y,'o'); % Output RBFNN
   hold on
   plot(label4training,label4training); % Target
   % This data set has 3 classes
else
    plot(label4testing,ytest,'o'); % Output RBFNN
    hold on
    plot(label4testing,label4testing); % Target
    % This data set has 3 classes

end