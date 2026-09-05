%% SPEECH DENOISING USING DEEP LEARNING NETWORKS
% This script denoises a noisy WAV file using a deep learning network
% following the MathWorks example

clear; close all; clc;

%% Load Audio Files
% Load your noisy audio file
[noisyAudio, fs] = audioread("1.wav");

% Check if clean audio exists for comparison
if isfile("1_clean.wav")
    [cleanAudio, fs_clean] = audioread("1_clean.wav");
else
    cleanAudio = []; % Will skip comparison if not available
end

% Visualize the audio signals
t = (1/fs)*(0:numel(noisyAudio)-1);
figure(1)
tiledlayout(2,1)

nexttile
plot(t,noisyAudio)
title("Original Noisy Audio")
xlabel("Time (s)")
grid on

if ~isempty(cleanAudio)
    nexttile
    plot(t(1:min(numel(cleanAudio),numel(noisyAudio))), ...
        cleanAudio(1:min(numel(cleanAudio),numel(noisyAudio))))
    title("Reference Clean Audio")
else
    nexttile
    plot(t,noisyAudio)
    title("Same Audio (No clean reference)")
end

%% System Parameters
windowLength = 256;
win = hamming(windowLength,"periodic");
overlap = round(0.75*windowLength);
fftLength = windowLength;
numFeatures = fftLength/2 + 1;  % 129 features
numSegments = 8;                % 8 consecutive STFT vectors

%% Downsample Audio to 8 kHz
% This reduces computational load
inputFs = fs;
outputFs = 8000;
if inputFs ~= outputFs
    src = dsp.SampleRateConverter(InputSampleRate=inputFs, ...
                                   OutputSampleRate=outputFs, ...
                                   Bandwidth=7920);
    decimationFactor = inputFs/outputFs;
    L = floor(numel(noisyAudio)/decimationFactor);
    noisyAudio = noisyAudio(1:decimationFactor*L);
    noisyAudio = src(noisyAudio);
    reset(src);
    fs = outputFs;
    
    if ~isempty(cleanAudio)
        L_clean = floor(numel(cleanAudio)/decimationFactor);
        cleanAudio = cleanAudio(1:decimationFactor*L_clean);
        cleanAudio = src(cleanAudio);
        reset(src);
    end
end

%% Generate STFT Features
% Compute magnitude STFT vectors from noisy audio
noisySTFT = stft(noisyAudio, Window=win, OverlapLength=overlap, ...
                 fftLength=fftLength);
noisyPhase = angle(noisySTFT(1:numFeatures,:));
noisySTFT = abs(noisySTFT(1:numFeatures,:));

% Create 8-segment predictor signals with 7-segment overlap
noisySTFT_padded = [noisySTFT(:,1:numSegments-1), noisySTFT];
predictors = zeros(numFeatures, numSegments, ...
                   size(noisySTFT_padded,2) - numSegments + 1);
for index = 1:(size(noisySTFT_padded,2) - numSegments + 1)
    predictors(:,:,index) = noisySTFT_padded(:,index:index + numSegments -1);
end

% If clean audio is available, generate target STFT
if ~isempty(cleanAudio)
    % Make sure both signals have same length for training
    minLen = min(numel(cleanAudio), numel(noisyAudio));
    cleanAudio = cleanAudio(1:minLen);
    noisyAudio = noisyAudio(1:minLen);
    
    cleanSTFT = stft(cleanAudio, Window=win, OverlapLength=overlap, ...
                     fftLength=fftLength);
    cleanSTFT = abs(cleanSTFT(1:numFeatures,:));
    targets = cleanSTFT;
    
    disp("Training data prepared with clean reference");
else
    % If no clean audio, use noisy STFT as targets (not ideal but workable)
    targets = noisySTFT;
    disp("Warning: No clean reference. Using noisy STFT as targets.");
end

%% Normalize Data
predictorsNorm = predictors;
noisyMean = mean(predictorsNorm(:));
noisyStd = std(predictorsNorm(:));
predictorsNorm(:) = (predictorsNorm(:) - noisyMean) / noisyStd;

targetsNorm = targets;
cleanMean = mean(targetsNorm(:));
cleanStd = std(targetsNorm(:));
targetsNorm(:) = (targetsNorm(:) - cleanMean) / cleanStd;

%% Reshape for Deep Learning Network
predictorsNorm = reshape(predictorsNorm, numFeatures, numSegments, 1, ...
                         size(predictorsNorm,3));
targetsNorm = reshape(targetsNorm, 1, 1, size(targetsNorm,1), ...
                      size(targetsNorm,2));

%% Split into Training and Validation Sets (99% train, 1% validation)
inds = randperm(size(predictorsNorm,4));
L = round(0.99*size(predictorsNorm,4));

trainPredictors = predictorsNorm(:,:,:,inds(1:L));
trainTargets = targetsNorm(:,:,:,inds(1:L));

validatePredictors = predictorsNorm(:,:,:,inds(L+1:end));
validateTargets = targetsNorm(:,:,:,inds(L+1:end));

disp("Data preparation complete!")
disp(sprintf("Training samples: %d", size(trainPredictors,4)))
disp(sprintf("Validation samples: %d", size(validatePredictors,4)))

%% SELECT NETWORK TYPE
% Choose between fully connected (simpler, slower) or convolutional (faster)
networkType = "fullyConnected";  % Options: "fullyConnected" or "convolutional"

%% BUILD NETWORK ARCHITECTURE
if strcmp(networkType, "fullyConnected")
    % Fully Connected Network
    layers = [
        imageInputLayer([numFeatures, numSegments])
        fullyConnectedLayer(1024)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(1024)
        batchNormalizationLayer
        reluLayer
        fullyConnectedLayer(numFeatures)
        ];
    disp("Fully Connected Network architecture created")
    
else
    % Convolutional Network (faster with fewer parameters)
    layers = [
        imageInputLayer([numFeatures, numSegments])
        convolution2dLayer([9 8], 18, Stride=[1 100], Padding="same")
        batchNormalizationLayer
        reluLayer
        repmat([
            convolution2dLayer([5 1], 30, Stride=[1 100], Padding="same")
            batchNormalizationLayer
            reluLayer
            convolution2dLayer([9 1], 8, Stride=[1 100], Padding="same")
            batchNormalizationLayer
            reluLayer
            convolution2dLayer([9 1], 18, Stride=[1 100], Padding="same")
            batchNormalizationLayer
            reluLayer
            ], 4, 1)
        convolution2dLayer([5 1], 30, Stride=[1 100], Padding="same")
        batchNormalizationLayer
        reluLayer
        convolution2dLayer([9 1], 8, Stride=[1 100], Padding="same")
        batchNormalizationLayer
        reluLayer
        convolution2dLayer([129 1], 1, Stride=[1 100], Padding="same")
        ];
    disp("Convolutional Network architecture created")
end

%% TRAINING OPTIONS
miniBatchSize = 128;
if strcmp(networkType, "fullyConnected")
    % For fully connected: squeeze targets
    options = trainingOptions("adam", ...
        MaxEpochs=5, ...
        InitialLearnRate=1e-5, ...
        MiniBatchSize=miniBatchSize, ...
        Shuffle="every-epoch", ...
        Plots="training-progress", ...
        Verbose=true, ...
        ValidationFrequency=floor(size(trainPredictors,4)/miniBatchSize), ...
        LearnRateSchedule="piecewise", ...
        LearnRateDropFactor=0.9, ...
        LearnRateDropPeriod=1, ...
        ValidationData={validatePredictors, squeeze(validateTargets)'});
    
    trainTargetsFC = squeeze(trainTargets)';
    
else
    % For convolutional: permute targets
    options = trainingOptions("adam", ...
        MaxEpochs=5, ...
        InitialLearnRate=1e-5, ...
        MiniBatchSize=miniBatchSize, ...
        Shuffle="every-epoch", ...
        Plots="training-progress", ...
        Verbose=true, ...
        ValidationFrequency=floor(size(trainPredictors,4)/miniBatchSize), ...
        LearnRateSchedule="piecewise", ...
        LearnRateDropFactor=0.9, ...
        LearnRateDropPeriod=1, ...
        ValidationData={validatePredictors, permute(validateTargets,[3 1 2 4])});
    
    trainTargetsFC = permute(trainTargets, [3 1 2 4]);
end

%% TRAIN THE NETWORK
disp(" ")
disp("========== TRAINING NETWORK ==========")
disp(sprintf("Network type: %s", networkType))
disp(sprintf("Epochs: 5"))
disp(sprintf("Mini-batch size: %d", miniBatchSize))
disp(sprintf("Training samples: %d", size(trainPredictors,4)))
disp(" ")

denoiseNet = trainnet(trainPredictors, trainTargetsFC, layers, "mse", options);

disp(" ")
disp("========== NETWORK SUMMARY ==========")
summary(denoiseNet);

%% TEST THE NETWORK ON YOUR AUDIO FILE
disp(" ")
disp("========== DENOISING AUDIO ==========")

% Regenerate predictors from the noisy audio (same steps as training prep)
testPredictors = zeros(numFeatures, numSegments, ...
                       size(noisySTFT_padded,2) - numSegments + 1);
for index = 1:(size(noisySTFT_padded,2) - numSegments + 1)
    testPredictors(:,:,index) = noisySTFT_padded(:,index:index + numSegments - 1);
end

% Normalize
testPredictors(:) = (testPredictors(:) - noisyMean) / noisyStd;
testPredictors = reshape(testPredictors, numFeatures, numSegments, 1, ...
                         size(testPredictors,3));

% Predict denoised STFT
if strcmp(networkType, "fullyConnected")
    STFTDenoised = predict(denoiseNet, testPredictors);
    STFTDenoised = STFTDenoised';
else
    STFTDenoised = predict(denoiseNet, testPredictors);
    STFTDenoised = squeeze(STFTDenoised)';
end

% Denormalize
STFTDenoised(:) = cleanStd * STFTDenoised(:) + cleanMean;

% Ensure proper dimensions
if size(STFTDenoised,1) > numFeatures
    STFTDenoised = STFTDenoised(1:numFeatures, :);
end

% Add phase information back
STFTDenoised = STFTDenoised .* exp(1j*noisyPhase);
STFTDenoised = [conj(STFTDenoised(end-1:-1:2,:)); STFTDenoised];

% Convert back to time domain
denoisedAudio = istft(STFTDenoised, Window=win, OverlapLength=overlap, ...
                      fftLength=fftLength, ConjugateSymmetric=true);

% Match length to original
minAudioLen = min(numel(denoisedAudio), numel(noisyAudio));
denoisedAudio = denoisedAudio(1:minAudioLen);
noisyAudio = noisyAudio(1:minAudioLen);

disp("Audio denoising complete!")

%% SAVE DENOISED AUDIO
outputFilename = "denoised_audio.wav";
audiowrite(outputFilename, denoisedAudio, fs);
disp(sprintf("Denoised audio saved as: %s", outputFilename))

%% VISUALIZE RESULTS
figure(2)
tiledlayout(2,2)

% Time domain
t_noisy = (1/fs)*(0:numel(noisyAudio)-1);
t_denoised = (1/fs)*(0:numel(denoisedAudio)-1);

nexttile
plot(t_noisy, noisyAudio)
title("Original Noisy Audio")
xlabel("Time (s)")
ylabel("Amplitude")
grid on

nexttile
plot(t_denoised, denoisedAudio)
title(sprintf("Denoised Audio (%s)", networkType))
xlabel("Time (s)")
ylabel("Amplitude")
grid on

% Frequency domain
nexttile
spectrogram(noisyAudio, win, overlap, fftLength, fs, 'yaxis')
title("Noisy Audio Spectrogram")
colorbar

nexttile
spectrogram(denoisedAudio, win, overlap, fftLength, fs, 'yaxis')
title(sprintf("Denoised Audio Spectrogram (%s)", networkType))
colorbar

%% AUDIO PLAYBACK OPTIONS (uncomment to use)
% disp(" ")
% disp("Press any key in the figure to hear the audio samples...")
% figure(3)
% sound(noisyAudio, fs);
% pause
% 
% sound(denoisedAudio, fs);
% pause

disp(" ")
disp("Script complete! Your denoised audio has been saved.")