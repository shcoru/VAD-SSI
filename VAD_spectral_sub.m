% Spectral Subtraction with VAD-guided Noise Estimation
% Combines Boll (1979) spectral subtraction with ISPC VAD
% Sydney Corum + integration
% Uses VAD-detected noise-only frames for adaptive noise estimation

clear;clc; close all;
%% Audio Import
[x,  fs]     = audioread('1.wav');
%[x,fs_clean] = audioread('1_clean.wav');
[clean,fs_clean] = audioread('1_clean.wav');

%% Parameters
% Spectral subtraction parameters
gamma        = 2;      % Spectral power: 1 = magnitude, 2 = power
overlap_frac = 0.5;    % 50% overlap
win_dur_s    = 0.02;   % 20 ms window
beta_over    = 1.0;    % Over-subtraction factor
noise_floor  = 0.01;   % Spectral floor fraction
n_avg        = 3;      % Frames to average for smoothing

% VAD parameters
alpha  = 0.98;     % Noise spectrum update rate
beta   = 0.95;     % Statistical smoothing factor
alpha_s    = 3.5;      % Speech threshold multiplier
alpha_n    = 2.2;      % Noise threshold multiplier

% Minimum noise-only frames required before processing
min_noise_frames = 10;

windowlen = round(fs * win_dur_s);
SP        = round(windowlen * (1 - overlap_frac));

fprintf('Sample rate  : %d Hz\n', fs);
fprintf('Window       : %d samples (%.0f ms)\n', windowlen, win_dur_s*1000);
fprintf('Hop          : %d samples\n', SP);

%% Compute STFT of noisy signal
[X, XPhase, frames_x, N_frames] = compute_stft(x, windowlen, SP, gamma);

FreqResol      = size(X, 1);
numberOfFrames = size(X, 2);

fprintf('Frames       : %d\n', numberOfFrames);
fprintf('Freq bins    : %d\n', FreqResol);
% STFT  Buffer Data into the kth frame  in freqeuncy domain
% Take signal, chop into frames with index label

frame_length = round(0.02*fs); % Might need tuning
num_frames = floor(length(x) / frame_length);
x = x(1:num_frames * frame_length);
x_frames = reshape(x,frame_length,num_frames);
VAD = zeros(num_frames,1); % intialize masking

% Frequency Domain
% need to add windowing with overlap and overlap reconstruction
X = fft(x_frames);

% Noise Spectrum and Noise mean for first frame, first baseline
% Intial
% Noise Spectrum
% Power Domain
N= abs(X(:,1)).^2; % First noisy frame N(w) = X(w,1)
% First noisy mean
mu_N = log(mean(N) +eps) ; % Noise mean
sigma_N = std(log(N +eps)); % inital std

frame_spectrum_update= N;
frame_mean_update = mu_N;

% Updating Noise Estimation when NO speech is present
alpha_s = 3.5; % adjustment params
alpha_n = 2.2; 


frame_energy_log = zeros(1, num_frames);
speech_thres_log = zeros(1, num_frames);
noise_thres_log  = zeros(1, num_frames);


% Perhaps some work to adapt threshold experimentally
for k = 1:num_frames
    current_frame = abs(X(:,k)).^2;
    frame_mag = log(mean(current_frame) + eps);
    frame_energy_log(k) = frame_mag;
    speech_thres = mu_N + alpha_s*sigma_N;
    noise_thres =mu_N +alpha_n*sigma_N ;
    speech_thres_log(k) = speech_thres; % Log speech threshold
    noise_thres_log(k) = noise_thres;   % Log noise threshold
    fprintf('Frame %d: Energy=%.2f, SpeechThr=%.2f, NoiseThr=%.2f, VAD=%d\n',...
        k, frame_mag, speech_thres, noise_thres, VAD(k));
        if frame_mag > speech_thres
        VAD(k) = 1; % frame contains a speech
        elseif frame_mag < noise_thres
        VAD(k) = 0; % noise or silence frame
        
        else
            if k>1
            VAD(k) = VAD(k-1);
            end
        end
        if frame_mag < (mu_N + sigma_N) % If no speech present, update noise parametere
        N = alpha * N + (1 - alpha) * current_frame;
        
        % Eq 2.5: Instantaneous mean
        mu_inst = log(mean(N) +eps);
        
        % Eq 2.6: Smoothed mean
        mu_N = beta * mu_N + (1 - beta) * mu_inst;
        
        % Eq 2.7: Variance + std update
        sigma_N = sqrt(beta * sigma_N^2 + (1 - beta) * (mu_inst - mu_N)^2);

        end
end


figure;
plot(frame_energy_log, 'b', 'LineWidth', 1.5); hold on;
plot(speech_thres_log, 'r--', 'LineWidth', 1.5);
plot(noise_thres_log, 'g--', 'LineWidth', 1.5);

stairs(VAD * max(frame_energy_log), 'k', 'LineWidth', 2);

legend('Frame Energy', 'Speech Threshold', 'Noise Threshold', 'VAD');
xlabel('Frame Index');
ylabel('Energy');
title('VAD Threshold Visualization');
grid on;

% Expand VAD to match signal length
VAD_signal = zeros(length(x),1);
for k = 1:num_frames
    idx_start = (k-1)*frame_length + 1;
    idx_end   = k*frame_length;
    
    VAD_signal(idx_start:idx_end) = VAD(k);
end
clean_norm = clean / max(abs(clean));
VAD_scaled = VAD_signal * max(clean_norm); % scale VAD to match amplitude
figure;
plot(clean_norm, 'b'); hold on;
plot(VAD_scaled, 'r', 'LineWidth', 1.5);

legend('Clean Speech', 'VAD');
xlabel('Sample Index');
ylabel('Amplitude');
title('VAD Overlay on Clean Speech');
grid on;

%% Extra Functions


function [spec, phase, frames, T] = compute_stft(sig, winlen, hop, gam)
    win = hamming(winlen);
    sig = sig(:);
    % padding signal for wrapping
    n_frames = floor((length(sig) - winlen) / hop) + 1;
    pad_len  = max(0, (n_frames - 1) * hop + winlen - length(sig));
    sig      = [sig; zeros(pad_len, 1)];
    
    T      = n_frames;
    F      = floor(winlen/2) + 1;
    spec   = zeros(F, T);
    phase  = zeros(F, T);
    frames = zeros(winlen, T);
    
    for m = 1:T
        idx         = (m-1)*hop + 1 : (m-1)*hop + winlen;
        frame       = sig(idx) .* win;
        frames(:,m) = frame;
        X           = fft(frame, winlen);
        Xhalf       = X(1:F);
        spec(:,m)   = abs(Xhalf) .^ gam;
        phase(:,m)  = angle(Xhalf);
    end
end
