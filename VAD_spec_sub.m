% =========================================================================
% Spectral Subtraction with VAD-guided Noise Estimation
% Based on: Faneuff & Brown, ISPC 2003
%           Boll (1979) spectral subtraction
%           Johnston (1988) perceptual weighting (threshold floor only here)
%
% Authors: Sydney Corum + integration
% =========================================================================

clear; clc; close all;

%% -------------------------------------------------------------------------
%  1. AUDIO IMPORT
% -------------------------------------------------------------------------
%[x,  fs]       = audioread('1.wav');
[x,  fs]       = audioread('sp01_car_sn15.wav');
%[clean, fs_cl] = audioread('1_clean.wav');
[clean, fs_cl] = audioread('sp01.wav');

% Ensure mono column vectors
x     = mean(x,     2);
clean = mean(clean, 2);

if fs ~= fs_cl
    clean = resample(clean, fs, fs_cl);
end

% Resample to 8 kHz to match paper conditions
target_fs = 8000;
if fs ~= target_fs
    x     = resample(x,     target_fs, fs);
    clean = resample(clean, target_fs, fs);
    fs    = target_fs;
    fprintf('Resampled to %d Hz\n', fs);
end

%% -------------------------------------------------------------------------
%  2. PARAMETERS
% -------------------------------------------------------------------------
% --- Framing ---
win_dur_s    = 128 / 8000;   % 128 samples at 8 kHz (matches paper Section 5.2)
frame_length = round(fs * win_dur_s);   % = 128 samples
overlap_frac = 0.5;
hop          = round(frame_length * (1 - overlap_frac));   % 50% overlap hop

% --- VAD ---
alpha   = 0.95;   % Noise spectrum smoothing  (Eq. 2.4, Virag value)
beta    = 0.95;   % Mean/variance smoothing   (Eq. 2.6-2.7, Virag value)
alpha_s = 3.5;    % Speech threshold multiplier  (tune experimentally)
alpha_n = 2.2;    % Noise threshold multiplier

n_init  = 10;     % Frames assumed noise-only for initialisation (paper Section 2)

% --- Spectral Subtraction ---
over_sub    = 1.5;   % Over-subtraction factor (a in paper Fig 3-1)
noise_floor = 0.01;  % Spectral floor as fraction of noise estimate (b)

fprintf('Sample rate  : %d Hz\n', fs);
fprintf('Frame length : %d samples (%.1f ms)\n', frame_length, frame_length/fs*1000);
fprintf('Hop size     : %d samples\n', hop);

%% -------------------------------------------------------------------------
%  3. STFT
% -------------------------------------------------------------------------
[X_mag, X_phase, N_frames] = compute_stft(x, frame_length, hop);
% X_mag  : [F x T]  one-sided magnitude spectrum
% X_phase: [F x T]  one-sided phase spectrum
F = size(X_mag, 1);   % number of frequency bins

fprintf('Frames       : %d\n', N_frames);
fprintf('Freq bins    : %d\n', F);

%% -------------------------------------------------------------------------
%  4. VAD  (Section 2 of paper)
% -------------------------------------------------------------------------
% Work in LOG domain throughout for numerical stability.
% All energy/threshold quantities are log-power.

VAD = zeros(N_frames, 1);

% -- Initialise from first frame (Eq. 2.2) --
N_spec  = abs(X_mag(:, 1)).^2;          % noise power estimate [F x 1]
mu_N    = log(mean(N_spec) + eps);      % log-mean of noise power
sigma_N = std(log(N_spec  + eps));      % log-std  of noise power

% Diagnostic logs
frame_energy_log  = zeros(1, N_frames);
speech_thres_log  = zeros(1, N_frames);
noise_thres_log   = zeros(1, N_frames);

for k = 1:N_frames

    % --- Current frame power & log-energy ---
    frame_pow  = X_mag(:, k).^2;                   % [F x 1]
    frame_logE = log(mean(frame_pow) + eps);        % scalar log-energy

    % --- Thresholds (Eq. 2.8 - 2.9) ---
    speech_thres = mu_N + alpha_s * sigma_N;
    noise_thres  = mu_N + alpha_n * sigma_N;

    frame_energy_log(k) = frame_logE;
    speech_thres_log(k) = speech_thres;
    noise_thres_log(k)  = noise_thres;

    % --- VAD decision ---
    if k <= n_init
        % Paper: first 10 frames assumed noise (Section 2)
        VAD(k) = 0;
    elseif frame_logE > speech_thres
        VAD(k) = 1;
    elseif frame_logE < noise_thres
        VAD(k) = 0;
    else
        % Ambiguous: carry forward previous decision
        if k > 1
            VAD(k) = VAD(k-1);
        end
    end

    % --- Noise estimator update — only when NO speech (Eq. 2.4 - 2.7) ---
    if VAD(k) == 0
        % Eq. 2.4: update noise power spectrum
        N_spec = alpha * N_spec + (1 - alpha) * frame_pow;

        % Eq. 2.5: instantaneous log-mean of updated estimate
        mu_inst = log(mean(N_spec) + eps);

        % Cache old mean BEFORE updating (needed for variance, Eq. 2.7)
        mu_N_old = mu_N;

        % Eq. 2.6: smooth the mean
        mu_N = beta * mu_N + (1 - beta) * mu_inst;

        % Eq. 2.7: smooth the variance (use OLD mean to compute deviation)
        sigma_N = sqrt(beta * sigma_N^2 + (1 - beta) * (mu_inst - mu_N_old)^2);
        sigma_N = max(sigma_N, 1e-6);   % guard against collapse to zero
    end
end

fprintf('\nVAD: %d speech frames / %d total (%.0f%%)\n', ...
    sum(VAD), N_frames, 100*mean(VAD));

%% -------------------------------------------------------------------------
%  5. SPECTRAL SUBTRACTION  (Section 3.1 of paper)
%     S_hat(w,k) = max( |M|^2 - a*|N|^2 , b*|N|^2 )
%     Phase from noisy signal (Wang & Lim justification)
% -------------------------------------------------------------------------
S_mag = zeros(F, N_frames);

% Re-run noise estimator to get per-frame N_spec for subtraction
% (reset to same initial condition used in VAD pass)
N_spec_ss = abs(X_mag(:, 1)).^2;

for k = 1:N_frames
    frame_pow = X_mag(:, k).^2;

    % Subtraction with spectral floor
    sub   = frame_pow - over_sub * N_spec_ss;
    floor = noise_floor * N_spec_ss;
    S_mag(:, k) = sqrt(max(sub, floor));   % back to magnitude

    % Update noise estimate only on noise frames
    if VAD(k) == 0
        N_spec_ss = alpha * N_spec_ss + (1 - alpha) * frame_pow;
    end
end

%% -------------------------------------------------------------------------
%  6. RECONSTRUCT TIME-DOMAIN SIGNAL  (IFFT + overlap-add)
% -------------------------------------------------------------------------
x_enhanced = reconstruct_signal(S_mag, X_phase, frame_length, hop, length(x));

%% -------------------------------------------------------------------------
%  7. SPEECH QUALITY METRICS  (Section 4.3 / Table 4.1)
% -------------------------------------------------------------------------
% Align lengths
L = min([length(x_enhanced), length(clean), length(x)]);
x_enh  = x_enhanced(1:L);
x_cln  = clean(1:L);
x_nsy  = x(1:L);

metrics_noisy    = speech_quality_metrics(x_nsy,  x_cln, frame_length, hop);
metrics_enhanced = speech_quality_metrics(x_enh,  x_cln, frame_length, hop);

fprintf('\n--- Speech Quality Metrics (Table 4.1 style) ---\n');
fprintf('%-25s %10s %10s\n', 'Metric', 'Noisy', 'Enhanced');
fprintf('%-25s %10.2f %10.2f\n', 'SNR (dB)',          metrics_noisy.SNR,  metrics_enhanced.SNR);
fprintf('%-25s %10.2f %10.2f\n', 'Seg-SNR (dB)',      metrics_noisy.SSNR, metrics_enhanced.SSNR);

%% -------------------------------------------------------------------------
%  8. PLOTS
% -------------------------------------------------------------------------

% -- 8a. VAD threshold visualisation --
figure('Name','VAD Thresholds');
plot(frame_energy_log, 'b',  'LineWidth', 1.5); hold on;
plot(speech_thres_log, 'r--','LineWidth', 1.5);
plot(noise_thres_log,  'g--','LineWidth', 1.5);
stairs(VAD * max(frame_energy_log), 'k', 'LineWidth', 2);
legend('Frame Log-Energy','Speech Threshold','Noise Threshold','VAD (scaled)');
xlabel('Frame Index'); ylabel('Log Energy');
title('VAD Threshold Visualisation'); grid on;

% -- 8b. VAD overlay on clean speech --
VAD_sig = vad_to_signal(VAD, frame_length, hop, length(x));
t_plot = min([length(x), length(clean), length(VAD_sig)]);
t = (0:t_plot-1) / fs;
figure('Name','VAD on Clean Speech');
plot(t, clean(1:t_plot)/max(abs(clean(1:t_plot))+eps), 'b'); hold on;
plot(t, VAD_sig(1:t_plot) * 0.9, 'r', 'LineWidth', 1.2);
legend('Clean Speech (normalised)','VAD');
xlabel('Time (s)'); ylabel('Amplitude');
title('VAD Overlay on Clean Speech'); grid on;

% -- 8c. Waveform comparison --
figure('Name','Waveform Comparison');
subplot(3,1,1); plot(t, x(1:length(t)));     title('Noisy Input');   ylabel('Amp'); grid on;
subplot(3,1,2); plot(t, clean(1:length(t))); title('Clean Reference');ylabel('Amp'); grid on;
subplot(3,1,3); plot((0:L-1)/fs, x_enh);    title('Enhanced Output');ylabel('Amp'); xlabel('Time (s)'); grid on;

% -- 8d. Spectrogram comparison --
figure('Name','Spectrograms');
subplot(1,3,1); spectrogram(x,     hamming(frame_length), hop, frame_length, fs, 'yaxis'); title('Noisy');
subplot(1,3,2); spectrogram(clean, hamming(frame_length), hop, frame_length, fs, 'yaxis'); title('Clean');
subplot(1,3,3); spectrogram(x_enh, hamming(frame_length), hop, frame_length, fs, 'yaxis'); title('Enhanced');

% -- 8e. Save enhanced audio --
audiowrite('1_enhanced.wav', x_enh / max(abs(x_enh)+eps), fs);
fprintf('\nSaved: 1_enhanced.wav\n');

%% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function [mag, phase, T] = compute_stft(sig, win_len, hop)
% One-sided STFT with Hamming window.
% Returns mag [F x T] and phase [F x T].
    sig = sig(:);
    win = hamming(win_len);
    F   = floor(win_len/2) + 1;

    % Zero-pad so last frame is complete
    n_frames = floor((length(sig) - win_len) / hop) + 1;
    pad      = max(0, (n_frames-1)*hop + win_len - length(sig));
    sig      = [sig; zeros(pad,1)];
    T        = n_frames;

    mag   = zeros(F, T);
    phase = zeros(F, T);

    for m = 1:T
        idx       = (m-1)*hop + (1:win_len);
        frame     = sig(idx) .* win;
        Xf        = fft(frame, win_len);
        mag(:,m)  = abs(Xf(1:F));
        phase(:,m)= angle(Xf(1:F));
    end
end

% -------------------------------------------------------------------------
function sig = reconstruct_signal(mag, phase, win_len, hop, out_len)
% Overlap-add IFFT reconstruction from one-sided mag + phase.
    T   = size(mag, 2);
    win = hamming(win_len);

    % Reconstruct full (two-sided) spectrum
    sig_buf = zeros((T-1)*hop + win_len, 1);
    win_sum = zeros(size(sig_buf));

    for m = 1:T
        % Mirror one-sided spectrum to full
        Xhalf = mag(:,m) .* exp(1j * phase(:,m));
        if mod(win_len,2) == 0
            Xfull = [Xhalf; conj(Xhalf(end-1:-1:2))];
        else
            Xfull = [Xhalf; conj(Xhalf(end:-1:2))];
        end

        frame = real(ifft(Xfull, win_len));
        idx   = (m-1)*hop + (1:win_len);
        sig_buf(idx) = sig_buf(idx) + frame .* win;
        win_sum(idx) = win_sum(idx) + win.^2;
    end

    % Normalise by window overlap sum (avoids amplitude ripple)
    win_sum  = max(win_sum, 1e-8);
    sig_buf  = sig_buf ./ win_sum;
    sig      = sig_buf(1:min(out_len, end));
end

% -------------------------------------------------------------------------
function VAD_sig = vad_to_signal(VAD, win_len, hop, out_len)
% Expand per-frame VAD decisions to sample-level using overlap logic.
    T       = length(VAD);
    buf_len = (T-1)*hop + win_len;
    VAD_sig = zeros(buf_len, 1);
    cnt     = zeros(buf_len, 1);

    for m = 1:T
        idx           = (m-1)*hop + (1:win_len);
        VAD_sig(idx)  = VAD_sig(idx)  + VAD(m);
        cnt(idx)      = cnt(idx) + 1;
    end

    cnt     = max(cnt, 1);
    VAD_sig = VAD_sig ./ cnt >= 0.5;   % majority vote across overlapping frames
    VAD_sig = double(VAD_sig(1:min(out_len, end)));
    if length(VAD_sig) < out_len
        VAD_sig = [VAD_sig; zeros(out_len - length(VAD_sig), 1)];
    end
end

% -------------------------------------------------------------------------
function m = speech_quality_metrics(test, ref, win_len, hop)
% Compute SNR and Segmental SNR (Table 4.1).
    L   = min(length(test), length(ref));
    t   = test(1:L);
    r   = ref(1:L);
    err = r - t;

    % Global SNR
    m.SNR = 10*log10(sum(r.^2) / (sum(err.^2) + eps));

    % Segmental SNR (per frame, averaged, clipped to [-10, 35] dB)
    n_frames = floor((L - win_len) / hop) + 1;
    ssnr_buf = zeros(n_frames, 1);
    for k = 1:n_frames
        idx   = (k-1)*hop + (1:win_len);
        r_seg = r(idx);
        e_seg = err(idx);
        val   = 10*log10(sum(r_seg.^2) / (sum(e_seg.^2) + eps));
        ssnr_buf(k) = max(-10, min(35, val));
    end
    m.SSNR = mean(ssnr_buf);
end