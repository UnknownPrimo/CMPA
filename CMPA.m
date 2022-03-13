%% ELEC 4700 
%% PA 8 - Diode Paramater Extraction

clf;

% Parameters to be extracted
Is = 0.01e-12; %Forward bias saturation current.
Ib = 0.1e-12; %Breakdown saturation current.
Vb = 1.3; %Breakdown volatage.
Gp = 0.1; %Parasitic parallel conductance.

%Task 1: 
%V vector from -1.95 to 0.7 volts with 200 steps.
V = linspace(-1.95,0.7,200);

% Current(I)expected physical behaviour
I = Is*(exp(1.2*V/0.025) - 1) + Gp*V - Ib*exp((-1.2/0.025)*(V+Vb));

% Current(I) vector
Ivec = zeros(200,1);
% Loop 
for j=1:length(I)
    Lowcurrent = I(j) - 0.20*I(j);
    Highcurrent = I(j) + 0.20*I(j);
    
    % I vector with 20% random variation in the current to represent 
    % experimental noise.
    Ivec(j) = (Highcurrent-Lowcurrent)*rand();
end
%plots
figure(1)
plot(V, I);
title('Current and Voltage extracted paramerter without noise');
xlabel('V (V)');
ylabel('I (mA)');
hold on;

figure(2);
semilogy(V, Ivec);
title('Current and Voltage extracted parameters with Noise');
xlabel('V (V)');
ylabel('I (mA)');
hold on;

% Task 2: 2. Polynomial fitting

% 4th order polynomial
poly4 = polyfit(V,I,4);
ply_y4 = polyval(poly4,V);

% plots
figure(1)
plot(V,ply_y4);
title('Current and Voltage extracted paramerter without noise');
xlabel('V (V)');
ylabel('I (mA)');
hold on;

figure(2);
semilogy(V, ply_y4);
title('Current and Voltage extracted parameters with Noise');
xlabel('V (V)');
ylabel('I (mA)');
hold on;

% 8th order polynomial
poly8 = polyfit(V,I,8);
ply_y8 = polyval(poly8,V);

% plots
figure(3)
plot(V, ply_y8);
title('Current and Voltage extracted paramerter without noise');
xlabel('V (V)');
ylabel('I (mA)');
hold on;

figure(4);
semilogy(V, ply_y8);
title('Current and Voltage extracted parameters with Noise');
xlabel('V (V)');
ylabel('I (mA)');
hold on;

% Task 3: Nonlinear curve fitting to a physical model using fit()

x = V;
% a)
fo = fittype('A.*(exp(1.2*x/25e-3)-1) + 0.2.*x - C*(exp(1.2*(-(x+1))/25e-3)-1)');

ff = fit(V',I',fo);

If = ff(x);

figure(5);
semilogy(x,If);
title('Current and Voltage with fit() function');
xlabel('V (V)');
ylabel('I (mA)');
hold on;

% b)
fo2 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1))/25e-3)-1)');

ff2 = fit(V',I',fo2);

If2 = ff2(x);

figure(6);
semilogy(x,If2);
title('Current and Voltage with fit() function');
xlabel('V (V)');
ylabel('I (mA)');
hold on;

% c)
fo3 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');

ff3 = fit(V',I',fo3);

If3 = ff3(x);

figure(7);
semilogy(x,If3);
title('Current and Voltage with fit() function');
xlabel('V (V)');
ylabel('I (mA)');
hold on;

% Task 4: Fitting using the Neural Net model.
inputs = V.';
targets = I.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs)
view(net)
Inn = outputs


