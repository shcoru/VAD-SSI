% =========================================================================
% Spectral Subtraction with MCRA-guided Noise Estimation
% Based on: Faneuff & Brown, ISPC 2003 (spectral subtraction framework)
%           Boll (1979) spectral subtraction
%           Johnston (1988) perceptual weighting (threshold floor only here)
%           Cohen & Berdugo, "Speech enhancement for non-stationary noise
%             environments," Signal Processing 81 (2001) 2403-2418 (MCRA
%             noise PSD tracking — replaces the fixed-threshold VAD-gated
%             noise estimator formerly used here)
%
% Authors: Sydney Corum + integration
% =========================================================================

clear; clc; close all;

%% -------------------------------------------------------------------------
%  1. AUDIO IMPORT
% -------------------------------------------------------------------------
%[x,  fs]       = audioread('1.wav');
[x,  fs]       = audioread('sp01_car_sn5.wav');
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

% --- MCRA Noise PSD Tracking (Cohen & Berdugo, 2001) ---
freq_smooth_halfwidth = 1;    % bins each side for periodogram frequency smoothing
alpha_s_mcra  = 0.8;          % periodogram temporal smoothing        (Eq. 3)
alpha_d_mcra  = 0.95;         % base noise-PSD smoothing constant     (Eq. 9)
alpha_p_mcra  = 0.2;          % speech-presence probability smoothing (Eq. 7)
delta_mcra    = 5;            % Sr threshold for presence indicator   (tune 2-5 per noise type)
min_win_dur_s = 1.6;          % minima-search window duration (s) — minima tracking horizon
gamma_md      = 0.998;        % minima-tracker smoothing constant
beta_md       = 0.8;          % minima-tracker bias-compensation constant

% --- Broadband VAD (diagnostics + hangover only — noise tracking above ---
% --- no longer depends on a hard decision, so it can't be mis-fed back) ---
p_thresh   = 0.2;    % mean speech-presence probability above which frame = speech
                      % (NOTE: this averages a per-bin binary decision across
                      % all F bins, and speech rarely lights up every bin at
                      % once, so 0.2-0.3 is typically "active" here, not 0.5 —
                      % re-check against your own p_avg distribution if you
                      % change the frame size or noise type)
hang_dur_s = 0.15;   % hangover duration (s): hold "speech" through weak trailing frames

L_min       = max(1, round(min_win_dur_s / (hop / fs)));   % minima window, in frames
hang_frames = round(hang_dur_s / (hop / fs));              % hangover, in frames

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
%  4. NOISE PSD TRACKING + VAD  (MCRA — Cohen & Berdugo, 2001)
% -------------------------------------------------------------------------
% Noise PSD is tracked continuously per frequency bin via a smoothed,
% minima-tracked speech-presence probability — no hard VAD gate is needed
% to decide when to update it, so a wrong broadband decision can no longer
% stall the estimate or contaminate it with speech energy (the failure
% mode of the old fixed-threshold, VAD-gated estimator).
X_pow = X_mag.^2;   % [F x T] noisy periodogram

[lambda_d, p_avg] = mcra_noise_estimate(X_pow, freq_smooth_halfwidth, ...
    alpha_s_mcra, alpha_d_mcra, alpha_p_mcra, delta_mcra, L_min, gamma_md, beta_md);

% Broadband hard VAD — derived only for diagnostics/plots and for the
% hangover below; it plays no role in the noise estimate above.
VAD = zeros(N_frames, 1);
hang_ctr = 0;
for k = 1:N_frames
    if p_avg(k) > p_thresh
        VAD(k)   = 1;
        hang_ctr = hang_frames;
    elseif hang_ctr > 0
        VAD(k)   = 1;
        hang_ctr = hang_ctr - 1;
    else
        VAD(k) = 0;
    end
end

% Diagnostic logs
frame_energy_log = log(mean(X_pow,    1) + eps);   % broadband log-energy per frame
noise_floor_log  = log(mean(lambda_d, 1) + eps);   % tracked adaptive noise floor
speech_prob_log  = p_avg;                          % mean speech-presence probability

fprintf('\nVAD: %d speech frames / %d total (%.0f%%)\n', ...
    sum(VAD), N_frames, 100*mean(VAD));

%% -------------------------------------------------------------------------
%  5. SPECTRAL SUBTRACTION  (Section 3.1 of paper)
%     S_hat(w,k) = max( |M|^2 - a*lambda_d(w,k) , b*lambda_d(w,k) )
%     Noise PSD is the per-bin, per-frame MCRA estimate from Section 4.
%     Phase from noisy signal (Wang & Lim justification)
% -------------------------------------------------------------------------
sub   = X_pow - over_sub * lambda_d;
floor = noise_floor * lambda_d;
S_mag = sqrt(max(sub, floor));   % back to magnitude, [F x N_frames]

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

% -- 8a. Adaptive noise floor & VAD visualisation --
figure('Name','Noise Floor & VAD');
subplot(2,1,1);
plot(frame_energy_log, 'b',  'LineWidth', 1.5); hold on;
plot(noise_floor_log,  'g--','LineWidth', 1.5);
stairs(VAD * max(frame_energy_log), 'k', 'LineWidth', 1.5);
legend('Frame Log-Energy','Adaptive Noise Floor (MCRA)','VAD (scaled)');
xlabel('Frame Index'); ylabel('Log Energy');
title('Adaptive Noise Floor Tracking'); grid on;

subplot(2,1,2);
plot(speech_prob_log, 'm', 'LineWidth', 1.5); hold on;
plot([1 N_frames], [p_thresh p_thresh], 'k--');
xlabel('Frame Index'); ylabel('Speech-presence probability');
title('MCRA Mean Speech-Presence Probability'); grid on; ylim([0 1]);

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
function [lambda_d, p_avg] = mcra_noise_estimate(X_pow, hw, alpha_s, alpha_d, ...
    alpha_p, delta, L, gamma_md, beta_md)
% MCRA noise PSD tracking (Cohen & Berdugo, Signal Processing 81 (2001)
% 2403-2418). Tracks a per-bin noise PSD estimate lambda_d [F x T] from the
% noisy periodogram X_pow [F x T], using a minima-controlled, recursively
% smoothed speech-presence probability instead of a hard VAD gate. Also
% returns p_avg [1 x T], the frequency-averaged speech-presence probability
% (for a diagnostic broadband VAD only — it is not fed back into lambda_d).
    [F, T] = size(X_pow);

    win = hanning(2*hw + 1); win = win / sum(win);   % frequency smoothing window

    S        = zeros(F, T);
    S_min    = zeros(F, T);
    S_tmp    = zeros(F, T);
    p        = zeros(F, T);
    lambda_d = zeros(F, T);

    for l = 1:T
        Sf = conv(X_pow(:, l), win, 'same');   % Eq. 3: smooth over frequency

        if l == 1
            S(:, l)        = Sf;
            S_min(:, l)    = Sf;
            S_tmp(:, l)    = Sf;
            lambda_d(:, l) = X_pow(:, l);       % initialise from first frame
            continue;
        end

        S(:, l) = alpha_s * S(:, l-1) + (1 - alpha_s) * Sf;   % Eq. 3: smooth over time

        % Minima tracking, dual-buffer (Doblinger-style) — Eq. 4-5
        below = S(:, l) < S_min(:, l-1);
        Smin_new = zeros(F, 1);
        Stmp_new = zeros(F, 1);
        Smin_new(below) = S(below, l);
        Stmp_new(below) = S(below, l);
        Smin_new(~below) = gamma_md * S_min(~below, l-1) + ...
            ((1 - gamma_md) / (1 - beta_md)) * (S(~below, l) - beta_md * S(~below, l-1));
        Stmp_new(~below) = min(S_tmp(~below, l-1), S(~below, l));

        if mod(l, L) == 0
            % Periodic buffer swap: refresh the running minimum from the
            % window just completed instead of letting it drift forever.
            Smin_new = min(S_tmp(:, l-1), S(:, l));
            Stmp_new = S(:, l);
        end
        S_min(:, l) = Smin_new;
        S_tmp(:, l) = Stmp_new;

        % Speech-presence indicator + probability smoothing — Eq. 6-7
        Sr = S(:, l) ./ max(S_min(:, l), eps);
        I  = double(Sr > delta);
        p(:, l) = alpha_p * p(:, l-1) + (1 - alpha_p) * I;

        % Time-varying smoothing factor + noise PSD update — Eq. 8-9
        alpha_d_tilde  = alpha_d + (1 - alpha_d) * p(:, l);
        lambda_d(:, l) = alpha_d_tilde .* lambda_d(:, l-1) + (1 - alpha_d_tilde) .* X_pow(:, l);
    end

    p_avg = mean(p, 1);
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