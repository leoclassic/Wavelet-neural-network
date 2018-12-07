function [] = WNN
clc
close all

X = (-4:.01:4)';
T = sin(X);
nHidden = 20;
nIter = 1000;
lr = 0.05;

model = initial(X,T,nHidden,nIter,lr,true);
[model,convergence] = train(model,X,T);

Y = feedforward(model,X);

figure
subplot(2,1,1)
plot(X,T,'b',X,Y,'r')
subplot(2,1,2)
semilogy(convergence)

mse(Y-T)
end

function model = initial(X,T,nHidden,nIter,lr,direct)
nInput = size(X,2);
nOutput = size(T,2);

model.nInput = nInput;
model.nOutput = nOutput;
model.nHidden = nHidden;
model.direct = direct;
model.translation = randn(nHidden,nInput);
model.dilation = rand(nHidden,nInput);
if direct
    model.DirectWeight = randn(nInput,nOutput);
end
model.OutputWeight = randn(nHidden,nOutput);
model.Bias = randn(1,nOutput);
model.nIter = nIter;
model.lr = lr;
end

function [model,convergence] = train(model,X,T)
convergence = inf(1,model.nIter);
for iter = 1:model.nIter
    [Y,Hidden] = feedforward(model,X);
    E = T-Y;
    convergence(iter) = E'*E;
    model = backward(model,X,Hidden,E);
    
    subplot(2,1,1)
    plot(X,T,'b',X,Y,'r')
    subplot(2,1,2)
    semilogy(convergence(1:iter))
    drawnow
end
close all
end

function output = wavelet(x)
output = (1-x.^2).*exp(-0.5*x.^2);
end

function output = dWavelet(x)
% Novel Neuronal Activation Functions for Feedforward Neural Networks
%
output = 2.*x.*(0.5*x.^2-0.5-1).*exp(-0.5*x.^2);
end

function output = wavelon(model,X)
output = zeros(size(X,1),model.nHidden);
for i = 1:model.nHidden
    z = (X-model.translation(i,:))./model.dilation(i,:);
    output(1,i) = prod(wavelet(z));
end
end

function [output,Hidden] = feedforward(model,X)
Hidden = zeros(size(X,1),model.nHidden);
for i = 1:size(X,1)
    Hidden(i,:) = wavelon(model,X(i,:));
end
if model.direct
    Direct = X*model.DirectWeight;
    output = Hidden*model.OutputWeight + repmat(model.Bias,size(X,1),1) + Direct;
else
    output = Hidden*model.OutputWeight + repmat(model.Bias,size(X,1),1);
end
end

function model = backward(model,X,Hidden,error)
% Initialization by a Novel Clustering for Wavelet Neural Network as Time Series Predictor
%
N = size(X,1);

deltaBias = error'*ones(N,1);
deltaOutputWeight = error'*Hidden;
deltaDirectWeight = error'*X;

model.Bias = model.Bias + model.lr.*(1/N).*deltaBias;
model.OutputWeight = model.OutputWeight + model.lr.*(1/N).*deltaOutputWeight';
if model.direct
    model.DirectWeight = model.DirectWeight + model.lr.*(1/N).*deltaDirectWeight';
end

for i = 1:model.nHidden
    for j = 1:model.nInput
        F = ones(size(X,1),1);
        for k = 1:model.nInput
            z = (X(:,k)-model.translation(i,k))./model.dilation(i,k);
            if j~=k
                F = F.*wavelet(z);
            else
                F = F.*dWavelet(z);
            end
        end
        deltaTranslation = (error*model.OutputWeight(i,:))'*F*(-1./model.dilation(i,j));
        model.translation(i,j) = model.translation(i,j) + model.lr.*(1/N).*deltaTranslation;
        
        deltaDilation = (error*model.OutputWeight(i,:).*F)'*(X-model.translation(i,j)).*(-1./(model.dilation(i,j).^2));
        model.dilation(i,j) = model.dilation(i,j) + model.lr.*(1/N).*deltaDilation;
    end
end
end
