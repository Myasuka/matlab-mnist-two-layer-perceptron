function [hiddenWeights, outputWeights, error] = trainStochasticSquaredErrorTwoLayerPerceptron(activationFunction, dActivationFunction, numberOfHiddenUnits, inputValues, targetValues, epochs, batchSize, learningRate)
% trainStochasticSquaredErrorTwoLayerPerceptron Creates a two-layer perceptron
% and trains it on the MNIST dataset.
%
% INPUT:
% activationFunction             : Activation function used in both layers.
% dActivationFunction            : Derivative of the activation
% function used in both layers.
% numberOfHiddenUnits            : Number of hidden units.
% inputValues                    : Input values for training (784 x 60000)
% targetValues                   : Target values for training (1 x 60000)
% epochs                         : Number of epochs to train.
% batchSize                      : Plot error after batchSize images.
% learningRate                   : Learning rate to apply.
%
% OUTPUT:
% hiddenWeights                  : Weights of the hidden layer.
% outputWeights                  : Weights of the output layer.
% 

    % The number of training vectors.
    trainingSetSize = size(inputValues, 2);
    
    % Input vector has 784 dimensions.
    inputDimensions = size(inputValues, 1);
    % We have to distinguish 10 digits.
    outputDimensions = size(targetValues, 1);
    
    % Initialize the weights for the hidden layer and the output layer.
    hiddenWeights = rand(numberOfHiddenUnits, inputDimensions);
    outputWeights = rand(outputDimensions, numberOfHiddenUnits);
    
    hiddenWeights = hiddenWeights./size(hiddenWeights, 2);
    outputWeights = outputWeights./size(outputWeights, 2);
    
    
    figure; hold on;

    for k = 1: epochs
       % Select which input vector to train on.
       n = floor(max(rand(1)*trainingSetSize - batchSize, 1));
            
       % Propagate the input vector through the network.
       inputVector = inputValues(:, n: n + batchSize ); 
       hiddenActualInput = hiddenWeights*inputVector ;
       hiddenOutputVector = activationFunction(hiddenActualInput) ;
       outputActualInput = outputWeights*hiddenOutputVector ; 
       outputVector = activationFunction(outputActualInput);
       targetVector = targetValues(:, n: n + batchSize);
            
       % Backpropagate the errors.
       outputDelta = dActivationFunction(outputActualInput).*(outputVector - targetVector); 
       hiddenDelta = dActivationFunction(hiddenActualInput).*(outputWeights'*outputDelta); 
            
       outputWeights = outputWeights - learningRate.*outputDelta*hiddenOutputVector' / batchSize;
       hiddenWeights = hiddenWeights - learningRate.*hiddenDelta*inputVector' / batchSize; 
       if mod(k, 100) == 0
          result = activationFunction(outputWeights*activationFunction(hiddenWeights*inputVector)) - targetVector;
          error = 0;
          for i = 1: size(result,2)
              error += norm(result(:,i), 2);
          end;
          error = error / batchSize;
          plot(k/100, error,'*');
        end;
     end;   
end