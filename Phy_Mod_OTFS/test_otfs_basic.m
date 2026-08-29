clear;
clc;
close all;

% OTFS parameters
M = 16;              % delay bins
N = 8;               % Doppler bins
M_mod = 4;           % QPSK
SNR_dB = 10;

% Generate random QPSK symbols
bits = randi([0 1], M*N*log2(M_mod), 1);

x = qammod(bits, M_mod, ...
    'InputType', 'bit', ...
    'UnitAveragePower', true);

% Create resource grid
rg = OTFSResGrid(M, N);
rg.setPulse2Recta();
rg.map(x);

% OTFS modulation
otfs = OTFS();
otfs.modulate(rg);

% AWGN channel (no Doppler yet)
No = 10^(-SNR_dB/10);
otfs.passChannel(No);

% Demodulation
rg_rx = otfs.demodulate();

% Extract received symbols
y = rg_rx.demap();

% QAM detection
bits_hat = qamdemod(y, M_mod, ...
    'OutputType', 'bit', ...
    'UnitAveragePower', true);

bits_hat = bits_hat(:);

% BER
bits_compare = bits(1:min(length(bits), length(bits_hat)));
bits_hat = bits_hat(1:length(bits_compare));

BER = mean(bits_compare ~= bits_hat);

fprintf("SNR = %d dB\n", SNR_dB);
fprintf("BER = %.6f\n", BER);