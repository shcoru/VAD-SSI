clc; clear; close all;

%% =========================================================================
%  SPEECH DENOISING VIA DEEP LEARNING
%  Supports either:
%    1. a single noisy/clean pair in the workspace, or
%    2. a NOIZEUS-style dataset with many noisy variants per clean utterance.
%
%  For NOIZEUS, point datasetRoot at the folder that contains the corpus.
%  The script discovers clean/noisy pairs, trains on all pairs, and then
%  denoises one selected noisy file from the dataset.
% =========================================================================

%% -------------------------------------------------------------------------
%  1. DATASET CONFIGURATION
% -------------------------------------------------------------------------
datasetMode = "noizeus";          % "noizeus" or "single"
datasetRoot = fullfile(pwd, "NOIZEUS");

singleNoisyFile = fullfile(pwd, "1.wav");
singleCleanFile = fullfile(pwd, "1_clean.wav");

% Optional: restrict which noisy file is denoised after training.
% Leave empty to use the first discovered pair.
selectedNoisyPattern = "";

snrLevels = [-5, 0, 5, 10, 15];
targetFs = 8000;

%% -------------------------------------------------------------------------
%  2. DL PARAMETERS
% -------------------------------------------------------------------------
windowLength = 256;
win = hamming(windowLength, "periodic");
overlapLen = round(0.75 * windowLength);
fftLen = windowLength;
numFeatures = fftLen/2 + 1;
numSegments = 8;

miniBatch = 128;
nEpochs = 20;

%% -------------------------------------------------------------------------
%  3. DISCOVER TRAINING PAIRS
% -------------------------------------------------------------------------
pairs = resolveDatasetPairs(datasetMode, datasetRoot, singleNoisyFile, singleCleanFile);
if isempty(pairs)
    error("No noisy/clean speech pairs were found. Check datasetRoot and file layout.");
end

fprintf("Discovered %d noisy/clean pair(s).\n", numel(pairs));

inferenceIdx = chooseInferencePair(pairs, selectedNoisyPattern);
fprintf("Inference file: %s\n", pairs(inferenceIdx).noisyPath);

%% -------------------------------------------------------------------------
%  4. BUILD TRAINING SET FROM ALL PAIRS
% -------------------------------------------------------------------------
allPredictors = cell(1, numel(pairs));
allTargets = cell(1, numel(pairs));
inferenceData = struct();

for pairIdx = 1:numel(pairs)
    [noisyAudio, cleanAudio, fs] = loadAlignedAudio(pairs(pairIdx).noisyPath, ...
        pairs(pairIdx).cleanPath, targetFs);

    [pairPredictors, pairTargets, pairMeta] = buildTrainingExamples( ...
        noisyAudio, cleanAudio, fs, snrLevels, win, overlapLen, fftLen, numFeatures, numSegments);

    allPredictors{pairIdx} = pairPredictors;
    allTargets{pairIdx} = pairTargets;

    fprintf("Pair %d/%d: %s -> %d frames\n", ...
        pairIdx, numel(pairs), pairs(pairIdx).label, size(pairPredictors, 3));

    if pairIdx == inferenceIdx
        inferenceData.noisyAudio = noisyAudio;
        inferenceData.cleanAudio = cleanAudio;
        inferenceData.fs = fs;
        inferenceData.vadInfo = pairMeta.vadInfo;
        inferenceData.label = pairs(pairIdx).label;
    end
end

predictors = cat(3, allPredictors{:});
targets = cat(3, allTargets{:});

fprintf("Total training frames: %d\n", size(predictors, 3));

%% -------------------------------------------------------------------------
%  5. NORMALISE AND SPLIT
% -------------------------------------------------------------------------
noisyMean = mean(predictors(:));
noisyStd = std(predictors(:));
predictors = (predictors - noisyMean) / max(noisyStd, eps);

cleanMean = mean(targets(:));
cleanStd = std(targets(:));
targets = (targets - cleanMean) / max(cleanStd, eps);

N = size(predictors, 3);
predictors = reshape(predictors, numFeatures, numSegments, 1, N);
targets = reshape(targets, numFeatures, numSegments, 1, N);

shuffIdx = randperm(N);
splitAt = max(1, min(N - 1, round(0.99 * N)));

trainPred = predictors(:,:,:, shuffIdx(1:splitAt));
trainTgt = targets(:,:,:, shuffIdx(1:splitAt));
valPred = predictors(:,:,:, shuffIdx(splitAt+1:end));
valTgt = targets(:,:,:, shuffIdx(splitAt+1:end));

fprintf("Train: %d | Val: %d\n", size(trainPred, 4), size(valPred, 4));

%% -------------------------------------------------------------------------
%  6. BUILD NETWORK
% -------------------------------------------------------------------------
layers = [
    imageInputLayer([numFeatures, numSegments], Normalization="none")
    convolution2dLayer([9 8], 18, Stride=[1 1], Padding="same")
    batchNormalizationLayer
    reluLayer

    repmat([
        convolution2dLayer([5 1], 30, Stride=[1 1], Padding="same")
        batchNormalizationLayer
        reluLayer
        convolution2dLayer([9 1], 8, Stride=[1 1], Padding="same")
        batchNormalizationLayer
        reluLayer
        convolution2dLayer([9 1], 18, Stride=[1 1], Padding="same")
        batchNormalizationLayer
        reluLayer
    ], 4, 1)

    convolution2dLayer([5 1], 30, Stride=[1 1], Padding="same")
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([9 1], 8, Stride=[1 1], Padding="same")
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([129 1], 1, Stride=[1 1], Padding="same")
];

%% -------------------------------------------------------------------------
%  7. TRAIN
% -------------------------------------------------------------------------
validationFrequency = max(1, floor(size(trainPred, 4) / miniBatch));

opts = trainingOptions("adam", ...
    MaxEpochs=nEpochs, ...
    InitialLearnRate=5e-4, ...
    MiniBatchSize=miniBatch, ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Verbose=true, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.95, ...
    LearnRateDropPeriod=2, ...
    ValidationFrequency=validationFrequency, ...
    ValidationData={valPred, valTgt});

disp("===== TRAINING =====")
denoiseNet = trainnet(trainPred, trainTgt, layers, "mse", opts);
summary(denoiseNet);

%% -------------------------------------------------------------------------
%  8. DENOISE ONE SELECTED NOISY FILE
% -------------------------------------------------------------------------
disp("===== DENOISING SELECTED FILE =====")

noisyFull = stft(inferenceData.noisyAudio, Window=win, OverlapLength=overlapLen, fftLength=fftLen);
noisyPhase = angle(noisyFull(1:numFeatures, :));
noisyMagFull = abs(noisyFull(1:numFeatures, :));

noisyPadFull = [noisyMagFull(:, 1:numSegments-1), noisyMagFull];
nFrFull = size(noisyPadFull, 2) - numSegments + 1;
testPred = zeros(numFeatures, numSegments, nFrFull);
for frameIdx = 1:nFrFull
    testPred(:, :, frameIdx) = noisyPadFull(:, frameIdx:frameIdx+numSegments-1);
end

testPred = (testPred - noisyMean) / max(noisyStd, eps);
testPred = reshape(testPred, numFeatures, numSegments, 1, nFrFull);

stftOut = predict(denoiseNet, testPred);
stftOut = squeeze(stftOut(:, numSegments, 1, :));
stftOut = cleanStd * stftOut + cleanMean;
stftOut = stftOut';

nCols = min(size(stftOut, 1), size(noisyPhase, 2));
stftOut = stftOut(1:nCols, :)';
phaseAligned = noisyPhase(:, 1:nCols);

stftComplex = stftOut .* exp(1j * phaseAligned);
stftFull = [stftComplex; conj(stftComplex(end-1:-1:2, :))];
denoisedAudio = istft(stftFull, Window=win, OverlapLength=overlapLen, ...
    fftLength=fftLen, ConjugateSymmetric=true);

outputName = "denoised_" + matlab.lang.makeValidName(inferenceData.label) + ".wav";
audiowrite(outputName, denoisedAudio / (max(abs(denoisedAudio)) + eps), inferenceData.fs);
fprintf("Saved: %s\n", outputName);

%% -------------------------------------------------------------------------
%  9. PLOT RESULTS
% -------------------------------------------------------------------------
Lp = min([numel(denoisedAudio), numel(inferenceData.noisyAudio), numel(inferenceData.cleanAudio)]);
t = (0:Lp-1) / inferenceData.fs;

figure("Name", "Denoising Results")
tiledlayout(3, 2)

nexttile; plot(t, inferenceData.noisyAudio(1:Lp)); title("Noisy Input"); grid on; ylabel("Amp")
nexttile; spectrogram(inferenceData.noisyAudio(1:Lp), win, overlapLen, fftLen, inferenceData.fs, "yaxis"); title("Noisy Spectrogram")

nexttile; plot(t, inferenceData.cleanAudio(1:Lp)); title("Clean Reference"); grid on; ylabel("Amp")
nexttile; spectrogram(inferenceData.cleanAudio(1:Lp), win, overlapLen, fftLen, inferenceData.fs, "yaxis"); title("Clean Spectrogram")

nexttile; plot(t, denoisedAudio(1:Lp)); title("Denoised Output"); grid on; ylabel("Amp"); xlabel("Time (s)")
nexttile; spectrogram(denoisedAudio(1:Lp), win, overlapLen, fftLen, inferenceData.fs, "yaxis"); title("Denoised Spectrogram")

figure("Name", "VAD Noise Extraction")
plotVadOverlay(inferenceData.noisyAudio, inferenceData.fs, inferenceData.vadInfo);

%% -------------------------------------------------------------------------
%  LOCAL FUNCTIONS
% -------------------------------------------------------------------------
function pairs = resolveDatasetPairs(datasetMode, datasetRoot, singleNoisyFile, singleCleanFile)
if datasetMode == "single"
    if ~isfile(singleNoisyFile) || ~isfile(singleCleanFile)
        error("Single-file mode expects both %s and %s to exist.", singleNoisyFile, singleCleanFile);
    end

    pairs = struct( ...
        "noisyPath", string(singleNoisyFile), ...
        "cleanPath", string(singleCleanFile), ...
        "label", "single_pair");
    return
end

if ~isfolder(datasetRoot)
    error("NOIZEUS dataset folder not found: %s", datasetRoot);
end

wavFiles = dir(fullfile(datasetRoot, "**", "*.wav"));
if isempty(wavFiles)
    pairs = struct([]);
    return
end

allPaths = string(fullfile({wavFiles.folder}, {wavFiles.name}));
allFolders = lower(string({wavFiles.folder}));
allNames = string({wavFiles.name});
allStems = erase(allNames, ".wav");

folderNames = strings(size(allFolders));
for idx = 1:numel(allFolders)
    [~, folderNames(idx)] = fileparts(allFolders(idx));
end

isClean = contains(allFolders, "clean") | folderNames == "clean" | endsWith(lower(allStems), "_clean");
cleanPaths = allPaths(isClean);
cleanStems = erase(allStems(isClean), "_clean");

noisyPaths = allPaths(~isClean);
noisyNames = allNames(~isClean);
noisyStems = erase(noisyNames, ".wav");

pairList = struct("noisyPath", {}, "cleanPath", {}, "label", {});

for cleanIdx = 1:numel(cleanPaths)
    thisStem = cleanStems(cleanIdx);
    for noisyIdx = 1:numel(noisyPaths)
        if isStemMatch(thisStem, noisyStems(noisyIdx))
            pairList(end+1).noisyPath = noisyPaths(noisyIdx); %#ok<AGROW>
            pairList(end).cleanPath = cleanPaths(cleanIdx);
            pairList(end).label = composeLabel(noisyPaths(noisyIdx), cleanPaths(cleanIdx));
        end
    end
end

if isempty(pairList)
    error("No NOIZEUS-style noisy/clean filename matches were found under %s", datasetRoot);
end

pairs = pairList;
end

function tf = isStemMatch(cleanStem, noisyStem)
cleanStem = lower(string(cleanStem));
noisyStem = lower(string(noisyStem));

if noisyStem == cleanStem
    tf = true;
    return
end

pattern = "(^|[_-])" + regexptranslate("escape", cleanStem) + "($|[_-])";
tf = ~isempty(regexp(char(noisyStem), char(pattern), "once"));
end

function label = composeLabel(noisyPath, cleanPath)
[noisyFolder, noisyStem] = fileparts(noisyPath);
[~, cleanStem] = fileparts(cleanPath);
[~, noisyFolderName] = fileparts(noisyFolder);

label = string(noisyFolderName) + "_" + string(noisyStem);

if strlength(label) == 0
    label = string(cleanStem);
end
end

function inferenceIdx = chooseInferencePair(pairs, selectedNoisyPattern)
if strlength(selectedNoisyPattern) == 0
    inferenceIdx = 1;
    return
end

hits = find(contains(lower(string({pairs.noisyPath})), lower(selectedNoisyPattern)));
if isempty(hits)
    error("No noisy file matched selectedNoisyPattern: %s", selectedNoisyPattern);
end

inferenceIdx = hits(1);
end

function [noisyAudio, cleanAudio, fs] = loadAlignedAudio(noisyPath, cleanPath, targetFs)
[noisyAudio, fsNoisy] = audioread(noisyPath);
[cleanAudio, fsClean] = audioread(cleanPath);

noisyAudio = mean(noisyAudio, 2);
cleanAudio = mean(cleanAudio, 2);

if fsNoisy ~= fsClean
    error("Sample-rate mismatch between %s (%d Hz) and %s (%d Hz).", ...
        noisyPath, fsNoisy, cleanPath, fsClean);
end

fs = fsNoisy;
if fs ~= targetFs
    noisyAudio = resample(noisyAudio, targetFs, fs);
    cleanAudio = resample(cleanAudio, targetFs, fs);
    fs = targetFs;
end

L = min(numel(noisyAudio), numel(cleanAudio));
noisyAudio = noisyAudio(1:L);
cleanAudio = cleanAudio(1:L);
end

function [predictors, targets, meta] = buildTrainingExamples(noisyAudio, cleanAudio, fs, ...
    snrLevels, win, overlapLen, fftLen, numFeatures, numSegments)
L = min(numel(noisyAudio), numel(cleanAudio));
noisyAudio = noisyAudio(1:L);
cleanAudio = cleanAudio(1:L);

[noiseOnly, vadInfo] = extractNoiseSignal(noisyAudio, fs);
if numel(noiseOnly) < max(round(0.25 * L), 1)
    noiseOnly = noisyAudio - cleanAudio;
end

if isempty(noiseOnly)
    error("Unable to derive a usable noise-only signal for training augmentation.");
end

if numel(noiseOnly) < 2 * L
    noiseOnly = repmat(noiseOnly, ceil((2 * L) / numel(noiseOnly)), 1);
end
noiseOnly = noiseOnly(1:2 * L);

cleanStft = stft(cleanAudio, Window=win, OverlapLength=overlapLen, fftLength=fftLen);
cleanMag = abs(cleanStft(1:numFeatures, :));
cleanWindows = makeSegmentWindows(cleanMag, numSegments);

variantPredictors = cell(1, numel(snrLevels) + 1);
variantTargets = cell(1, numel(snrLevels) + 1);

pairedNoisyStft = stft(noisyAudio, Window=win, OverlapLength=overlapLen, fftLength=fftLen);
pairedNoisyMag = abs(pairedNoisyStft(1:numFeatures, :));
variantPredictors{1} = makeSegmentWindows(pairedNoisyMag, numSegments);
variantTargets{1} = cleanWindows;

cleanPow = sum(cleanAudio .^ 2) + eps;
for snrIdx = 1:numel(snrLevels)
    snrDb = snrLevels(snrIdx);
    maxStart = numel(noiseOnly) - L + 1;
    startIdx = randi(maxStart);
    noiseSlice = noiseOnly(startIdx:startIdx + L - 1);
    noisePow = sum(noiseSlice .^ 2) + eps;
    noiseScaled = noiseSlice * sqrt(cleanPow / noisePow / (10 ^ (snrDb / 10)));
    noisyTrain = cleanAudio + noiseScaled;

    noisyStft = stft(noisyTrain, Window=win, OverlapLength=overlapLen, fftLength=fftLen);
    noisyMag = abs(noisyStft(1:numFeatures, :));

    variantPredictors{snrIdx + 1} = makeSegmentWindows(noisyMag, numSegments);
    variantTargets{snrIdx + 1} = cleanWindows;
end

predictors = cat(3, variantPredictors{:});
targets = cat(3, variantTargets{:});
meta.vadInfo = vadInfo;
end

function windows = makeSegmentWindows(magnitudeStft, numSegments)
padded = [magnitudeStft(:, 1:numSegments-1), magnitudeStft];
nFrames = size(padded, 2) - numSegments + 1;
windows = zeros(size(magnitudeStft, 1), numSegments, nFrames);

for frameIdx = 1:nFrames
    windows(:, :, frameIdx) = padded(:, frameIdx:frameIdx + numSegments - 1);
end
end

function [noiseOnly, vadInfo] = extractNoiseSignal(noisyAudio, fs)
L = numel(noisyAudio);

vadFrame = 128;
vadHop = vadFrame / 2;
alphaSmooth = 0.95;
betaSmooth = 0.95;
alphaSpeech = 3.5;
alphaNoise = 2.2;
nInit = 10;

vadWin = hamming(vadFrame);
Nvad = floor((L - vadFrame) / vadHop) + 1;
Nfreq = floor(vadFrame / 2) + 1;
Xmag = zeros(Nfreq, Nvad);

for frameIdx = 1:Nvad
    idx = (frameIdx - 1) * vadHop + (1:vadFrame);
    frame = noisyAudio(idx) .* vadWin;
    Xk = fft(frame, vadFrame);
    Xmag(:, frameIdx) = abs(Xk(1:Nfreq));
end

vad = zeros(Nvad, 1);
noiseSpec = Xmag(:, 1) .^ 2;
muNoise = log(mean(noiseSpec) + eps);
sigmaNoise = std(log(noiseSpec + eps));

for frameIdx = 1:Nvad
    framePow = Xmag(:, frameIdx) .^ 2;
    logEnergy = log(mean(framePow) + eps);

    thrSpeech = muNoise + alphaSpeech * sigmaNoise;
    thrNoise = muNoise + alphaNoise * sigmaNoise;

    if frameIdx <= nInit
        vad(frameIdx) = 0;
    elseif logEnergy > thrSpeech
        vad(frameIdx) = 1;
    elseif logEnergy < thrNoise
        vad(frameIdx) = 0;
    elseif frameIdx > 1
        vad(frameIdx) = vad(frameIdx - 1);
    end

    if vad(frameIdx) == 0
        noiseSpec = alphaSmooth * noiseSpec + (1 - alphaSmooth) * framePow;
        muInst = log(mean(noiseSpec) + eps);
        muOld = muNoise;
        muNoise = betaSmooth * muNoise + (1 - betaSmooth) * muInst;
        sigmaNoise = sqrt(betaSmooth * sigmaNoise ^ 2 + (1 - betaSmooth) * (muInst - muOld) ^ 2);
        sigmaNoise = max(sigmaNoise, 1e-6);
    end
end

bufLen = (Nvad - 1) * vadHop + vadFrame;
noiseBuf = zeros(bufLen, 1);
winSum = zeros(bufLen, 1);

for frameIdx = 1:Nvad
    if vad(frameIdx) == 0
        idx = (frameIdx - 1) * vadHop + (1:vadFrame);
        noiseBuf(idx) = noiseBuf(idx) + noisyAudio(idx) .* vadWin .^ 2;
        winSum(idx) = winSum(idx) + vadWin .^ 2;
    end
end

noiseBuf = noiseBuf ./ max(winSum, 1e-8);
noiseOnly = noiseBuf(winSum > 0.1);

vadInfo.VAD = vad;
vadInfo.vadFrame = vadFrame;
vadInfo.vadHop = vadHop;
vadInfo.signalLength = L;
vadInfo.fs = fs;
end

function plotVadOverlay(noisyAudio, fs, vadInfo)
L = min(numel(noisyAudio), vadInfo.signalLength);
t = (0:L-1) / fs;
vadSig = zeros(L, 1);

for frameIdx = 1:numel(vadInfo.VAD)
    idx = (frameIdx - 1) * vadInfo.vadHop + (1:vadInfo.vadFrame);
    idx = idx(idx <= L);
    vadSig(idx) = vadSig(idx) + vadInfo.VAD(frameIdx);
end

vadSig = vadSig > 0;
plot(t, noisyAudio(1:L) / max(abs(noisyAudio(1:L)) + eps), "b"); hold on
plot(t, -0.3 * vadSig, "r", LineWidth=1.5);
legend("Noisy Audio (norm)", "VAD=1 (speech)");
grid on
xlabel("Time (s)")
title("VAD Decision - Red=speech, baseline=noise")
end