#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LMD-DWT (Local Mean Decomposition + Discrete Wavelet Transform) 기반 EOG 아티팩트 제거 모듈
"""

import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
import os

# LMD와 DWT 관련 라이브러리 import
try:
    import pywt
    from scipy.stats import kurtosis
    from scipy.signal import welch
    import warnings
    warnings.filterwarnings('ignore')
    LMD_DWT_AVAILABLE = True
except ImportError:
    LMD_DWT_AVAILABLE = False
    print("⚠️ LMD/DWT 기능을 사용하려면 pywt, scipy가 필요합니다.")


def calculate_dispersion_entropy(signal_data, m=2, c=6):
    """
    Calculate Dispersion Entropy (DiEn) for a given signal.
    
    Parameters:
    -----------
    signal_data : array-like
        Input signal
    m : int
        Embedding dimension (default: 2)
    c : int
        Number of classes (default: 6)
    
    Returns:
    --------
    float : Dispersion entropy value
    """
    if len(signal_data) < m + 1:
        return 0.0
    
    # Normalize signal to [0, 1]
    signal_norm = (signal_data - np.min(signal_data)) / (np.max(signal_data) - np.min(signal_data) + 1e-10)
    
    # Map to classes [1, c]
    signal_mapped = np.floor(signal_norm * c) + 1
    signal_mapped = np.clip(signal_mapped, 1, c)
    
    # Create embedding vectors
    embedding_vectors = []
    for i in range(len(signal_mapped) - m + 1):
        embedding_vectors.append(signal_mapped[i:i+m])
    
    if len(embedding_vectors) == 0:
        return 0.0
    
    embedding_vectors = np.array(embedding_vectors)
    
    # Count unique patterns
    unique_patterns = {}
    for pattern in embedding_vectors:
        pattern_str = ','.join(map(str, pattern))
        unique_patterns[pattern_str] = unique_patterns.get(pattern_str, 0) + 1
    
    # Calculate probabilities
    total_patterns = len(embedding_vectors)
    probabilities = np.array(list(unique_patterns.values())) / total_patterns
    
    # Calculate entropy
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    
    return entropy


def extract_imf_features(imf, fs=256):
    """
    Extract features from an IMF for EOG artifact detection.
    
    Parameters:
    -----------
    imf : array-like
        Intrinsic Mode Function
    fs : int
        Sampling frequency (default: 256 Hz)
    
    Returns:
    --------
    dict : Dictionary containing extracted features
    """
    # Kurtosis
    kt = kurtosis(imf)
    
    # Power Spectral Density in 0.1-4 Hz band
    freqs, psd = welch(imf, fs=fs, nperseg=min(256, len(imf)//4))
    
    # Filter PSD to 0.1-4 Hz band
    freq_mask = (freqs >= 0.1) & (freqs <= 4.0)
    if np.any(freq_mask):
        psd_band = psd[freq_mask]
        psd_mean = np.mean(psd_band)
    else:
        psd_mean = 0.0
    
    # Dispersion Entropy
    dien = calculate_dispersion_entropy(imf)
    
    return {
        'kurtosis': kt,
        'psd_mean': psd_mean,
        'dispersion_entropy': dien
    }


def identify_eog_imfs(imfs, fs=256):
    """
    Identify IMFs contaminated with EOG artifacts using more conservative threshold rules.
    
    Updated Rules (More Conservative):
    - Kt > mean(Kt) + 1.5 * std(Kt)  # Increased threshold
    - PSD (0.1–4 Hz) > mean(PSD) + 0.5 * std(PSD)  # Added std factor
    - DiEn < mean(DiEn) − 1.5 * std(DiEn)  # Increased threshold
    
    An IMF is flagged as EOG-contaminated if ALL 3 conditions are satisfied (stricter).
    
    Parameters:
    -----------
    imfs : list of arrays
        List of Intrinsic Mode Functions
    fs : int
        Sampling frequency
    
    Returns:
    --------
    list : Indices of IMFs identified as EOG-contaminated
    """
    if not imfs:
        return []
    
    # Calculate features for all IMFs
    features_list = []
    for imf in imfs:
        features = extract_imf_features(imf, fs)
        features_list.append(features)
    
    # Calculate statistics across all IMFs
    kt_values = [f['kurtosis'] for f in features_list]
    psd_values = [f['psd_mean'] for f in features_list]
    dien_values = [f['dispersion_entropy'] for f in features_list]
    
    kt_mean, kt_std = np.mean(kt_values), np.std(kt_values)
    psd_mean, psd_std = np.mean(psd_values), np.std(psd_values)
    dien_mean, dien_std = np.mean(dien_values), np.std(dien_values)
    
    # Set more conservative thresholds
    kt_threshold = kt_mean + 1.5 * kt_std  # Increased from 1.0 to 1.5
    psd_threshold = psd_mean + 0.5 * psd_std  # Added std factor
    dien_threshold = dien_mean - 1.5 * dien_std  # Increased from 1.0 to 1.5
    
    print(f"   임계값 정보 (보수적 접근):")
    print(f"   - Kurtosis: {kt_threshold:.3f} (mean: {kt_mean:.3f}, std: {kt_std:.3f})")
    print(f"   - PSD (0.1-4Hz): {psd_threshold:.3f} (mean: {psd_mean:.3f}, std: {psd_std:.3f})")
    print(f"   - Dispersion Entropy: {dien_threshold:.3f} (mean: {dien_mean:.3f}, std: {dien_std:.3f})")
    
    # Identify EOG-contaminated IMFs using stricter voting system
    eog_imf_indices = []
    for i, features in enumerate(features_list):
        # Check each condition
        condition1 = features['kurtosis'] > kt_threshold
        condition2 = features['psd_mean'] > psd_threshold
        condition3 = features['dispersion_entropy'] < dien_threshold
        
        # Count satisfied conditions
        satisfied_conditions = sum([condition1, condition2, condition3])
        
        # Flag if ALL 3 conditions are satisfied (stricter than before)
        if satisfied_conditions == 3:
            eog_imf_indices.append(i)
            print(f"   IMF {i}: EOG 오염 식별 (모든 조건 만족: 3/3)")
            print(f"     - Kt: {features['kurtosis']:.3f} > {kt_threshold:.3f}: {condition1}")
            print(f"     - PSD: {features['psd_mean']:.3f} > {psd_threshold:.3f}: {condition2}")
            print(f"     - DiEn: {features['dispersion_entropy']:.3f} < {dien_threshold:.3f}: {condition3}")
        elif satisfied_conditions >= 2:
            print(f"   IMF {i}: 부분적 EOG 특성 (조건 만족: {satisfied_conditions}/3) - 제거하지 않음")
    
    return eog_imf_indices


def remove_eog_from_imf(imf, wavelet='db4', level=4):
    """
    Remove EOG artifacts from a single IMF using DWT with more conservative approach.
    
    Parameters:
    -----------
    imf : array-like
        Intrinsic Mode Function contaminated with EOG
    wavelet : str
        Wavelet type for DWT (default: 'db4')
    level : int
        Decomposition level (default: 4)
    
    Returns:
    --------
    array : Cleaned IMF
    """
    # Apply DWT
    coeffs = pywt.wavedec(imf, wavelet, level=level)
    
    # More conservative approach: attenuate instead of completely removing
    # Only remove a portion of the approximation coefficients
    attenuation_factor = 0.3  # Remove only 30% of low-frequency components
    coeffs[0] = coeffs[0] * (1 - attenuation_factor)
    
    # Also attenuate some detail coefficients that might contain EOG artifacts
    # Focus on levels 1-2 which often contain EOG artifacts
    for i in range(1, min(3, len(coeffs))):
        coeffs[i] = coeffs[i] * 0.8  # Attenuate by 20%
    
    # Reconstruct signal
    cleaned_imf = pywt.waverec(coeffs, wavelet)
    
    # Ensure same length as original
    if len(cleaned_imf) != len(imf):
        cleaned_imf = cleaned_imf[:len(imf)]
    
    return cleaned_imf


def find_local_extrema(signal):
    """
    Find local maxima and minima in the signal.
    
    Parameters:
    -----------
    signal : array-like
        Input signal
    
    Returns:
    --------
    tuple : (maxima_indices, minima_indices)
    """
    # Check for NaN or infinite values
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        print("⚠️ 신호에 NaN/Inf 값이 포함되어 있습니다.")
        return [], []
    
    # Find local maxima
    maxima_indices = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            maxima_indices.append(i)
    
    # Find local minima
    minima_indices = []
    for i in range(1, len(signal) - 1):
        if signal[i] < signal[i-1] and signal[i] < signal[i+1]:
            minima_indices.append(i)
    
    return maxima_indices, minima_indices


def smooth_signal(signal, window_size=5):
    """
    Apply moving average smoothing to the signal.
    
    Parameters:
    -----------
    signal : array-like
        Input signal
    window_size : int
        Size of the moving average window
    
    Returns:
    --------
    array : Smoothed signal
    """
    if window_size < 3:
        window_size = 3
    
    # Pad the signal for edge handling
    pad_size = window_size // 2
    padded = np.pad(signal, (pad_size, pad_size), mode='edge')
    
    # Apply moving average
    smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
    
    return smoothed


def lmd_decompose(signal, max_imfs=10, stop_thresh=0.01):
    """
    Local Mean Decomposition (LMD) implementation based on the research paper.
    
    Parameters:
    -----------
    signal : array-like
        Input signal (1D numpy array)
    max_imfs : int
        Maximum number of IMFs to extract
    stop_thresh : float
        Stopping threshold for residual signal
    
    Returns:
    --------
    tuple : (list of IMFs, residual signal)
    """
    print("🔄 LMD 분해 시작 (연구논문 기반 알고리즘)...")
    
    signal = np.array(signal, dtype=float)
    
    # Check for NaN or infinite values in input signal
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        print("❌ 입력 신호에 NaN/Inf 값이 포함되어 있습니다. 빈 IMF 리스트를 반환합니다.")
        return [], signal
    
    imfs = []
    residual = signal.copy()
    
    for r in range(max_imfs):
        print(f"  IMF {r+1} 추출 중...")
        
        # Check if residual is monotonic or too small
        if np.std(residual) < stop_thresh:
            print(f"    잔여 신호가 너무 작음 (std: {np.std(residual):.6f})")
            break
        
        # Step 1: Identify local extrema
        maxima_indices, minima_indices = find_local_extrema(residual)
        
        if len(maxima_indices) < 2 or len(minima_indices) < 2:
            print(f"    충분한 극값이 없음 (maxima: {len(maxima_indices)}, minima: {len(minima_indices)})")
            break
        
        # Step 2: Calculate local means and magnitudes
        all_extrema = sorted(maxima_indices + minima_indices)
        
        # Initialize arrays for local means and magnitudes
        local_means = np.zeros_like(residual)
        local_magnitudes = np.zeros_like(residual)
        
        # Calculate local means and magnitudes between consecutive extrema
        for i in range(len(all_extrema) - 1):
            start_idx = all_extrema[i]
            end_idx = all_extrema[i + 1]
            
            # Local mean: m_{r,i,b} = (q_{i,b} + q_{i,b+1}) / 2
            local_mean = (residual[start_idx] + residual[end_idx]) / 2
            
            # Local magnitude: p_{r,i,b} = |q_{i,b} - q_{i,b+1}| / 2
            local_magnitude = abs(residual[start_idx] - residual[end_idx]) / 2
            
            # Fill between extrema points
            local_means[start_idx:end_idx+1] = local_mean
            local_magnitudes[start_idx:end_idx+1] = local_magnitude
        
        # Step 3: Smooth local means and magnitudes
        smoothed_means = smooth_signal(local_means, window_size=5)
        smoothed_magnitudes = smooth_signal(local_magnitudes, window_size=5)
        
        # Ensure smoothed magnitudes are positive and handle numerical issues
        smoothed_magnitudes = np.maximum(smoothed_magnitudes, 1e-10)
        
        # Check for numerical stability
        if np.any(np.isnan(smoothed_means)) or np.any(np.isnan(smoothed_magnitudes)):
            print(f"    수치적 불안정성 감지. IMF {r+1} 추출 중단")
            break
        
        # Step 4: Sifting process
        h_r = residual.copy()
        f_r = np.zeros_like(residual)
        
        for sifting_iter in range(100):  # Max sifting iterations
            # Step 4a: Subtract smoothed local mean
            g_r = h_r - smoothed_means
            
            # Step 4b: Construct frequency modulated signal with safety check
            f_r_new = g_r / smoothed_magnitudes
            
            # Check for NaN or infinite values
            if np.any(np.isnan(f_r_new)) or np.any(np.isinf(f_r_new)):
                print(f"    수치적 불안정성 감지 (sifting {sifting_iter}). IMF {r+1} 추출 중단")
                break
            
            # Check convergence
            if np.std(f_r_new - f_r) < 1e-6:
                break
            
            f_r = f_r_new.copy()
            h_r = g_r.copy()
        
        # Step 5: Compute Product Function
        # PF_r(n) = h_r(n) * f_r(n)
        product_function = h_r * f_r
        
        # Check for NaN or infinite values in product function
        if np.any(np.isnan(product_function)) or np.any(np.isinf(product_function)):
            print(f"    Product Function에 NaN/Inf 값 감지. IMF {r+1} 추출 중단")
            break
        
        # Step 6: Extract IMF
        imf = product_function.copy()
        imfs.append(imf)
        
        # Step 7: Update residual
        residual = residual - imf
        
        # Check for NaN or infinite values in residual
        if np.any(np.isnan(residual)) or np.any(np.isinf(residual)):
            print(f"    잔여 신호에 NaN/Inf 값 감지. IMF {r+1} 추출 중단")
            break
        
        print(f"    IMF {r+1} 추출 완료 (std: {np.std(imf):.6f})")
        
        # Check if residual is monotonic
        if len(find_local_extrema(residual)[0]) < 2 and len(find_local_extrema(residual)[1]) < 2:
            print(f"    잔여 신호가 단조함")
            break
    
    print(f"✅ LMD 분해 완료: {len(imfs)}개 IMF 추출")
    return imfs, residual


def simple_lmd_decomposition(signal_data, max_imfs=10, tolerance=0.05):
    """
    Wrapper function to maintain compatibility with existing code.
    Now uses the correct LMD implementation.
    
    Parameters:
    -----------
    signal_data : array-like
        Input signal
    max_imfs : int
        Maximum number of IMFs to extract
    tolerance : float
        Tolerance for convergence (converted to stop_thresh)
    
    Returns:
    --------
    list : List of IMFs
    """
    # Convert tolerance to stop_thresh (approximate conversion)
    stop_thresh = tolerance * np.std(signal_data)
    
    imfs, residual = lmd_decompose(signal_data, max_imfs=max_imfs, stop_thresh=stop_thresh)
    
    return imfs


def remove_eog_artifacts_lmd_dwt(eeg_signal, fs=256, max_imfs=8, wavelet='db4', level=4):
    """
    Remove EOG artifacts from EEG signal using LMD and DWT.
    
    Updated Method:
    1. Apply LMD to decompose EEG into IMFs
    2. Extract features: Kurtosis, PSD (0.1-4 Hz), Dispersion Entropy
    3. Identify EOG-contaminated IMFs using voting system (≥2/3 conditions)
    4. Apply DWT to flagged IMFs and remove low-frequency components
    5. Reconstruct cleaned signal
    
    Parameters:
    -----------
    eeg_signal : array-like
        Raw EEG signal (1D array)
    fs : int
        Sampling frequency (default: 256 Hz)
    max_imfs : int
        Maximum number of IMFs to extract
    wavelet : str
        Wavelet type for DWT
    level : int
        DWT decomposition level
    
    Returns:
    --------
    array : Cleaned EEG signal
    """
    if not LMD_DWT_AVAILABLE:
        print("❌ LMD/DWT 기능을 사용할 수 없습니다. 원본 신호를 반환합니다.")
        return eeg_signal
    
    print("🔄 LMD-DWT EOG 제거 시작...")
    
    # Step 1: Decompose signal into IMFs using LMD
    print("  1. LMD 분해 중...")
    imfs = simple_lmd_decomposition(eeg_signal, max_imfs=max_imfs)
    print(f"     {len(imfs)}개의 IMF 추출 완료")
    
    if not imfs:
        print("⚠️ IMF를 추출할 수 없습니다. 원본 신호를 반환합니다.")
        return eeg_signal
    
    # Step 2: Identify EOG-contaminated IMFs using updated rules
    print("  2. EOG 오염 IMF 식별 중 (업데이트된 임계값 규칙 적용)...")
    eog_imf_indices = identify_eog_imfs(imfs, fs=fs)
    print(f"    EOG 오염 IMF {len(eog_imf_indices)}개 식별: {eog_imf_indices}")
    
    # Step 3: Clean EOG-contaminated IMFs using DWT
    cleaned_imfs = []
    for i, imf in enumerate(imfs):
        if i in eog_imf_indices:
            print(f"  3. IMF {i} DWT 정리 중...")
            cleaned_imf = remove_eog_from_imf(imf, wavelet=wavelet, level=level)
            cleaned_imfs.append(cleaned_imf)
        else:
            cleaned_imfs.append(imf)
    
    # Step 4: Reconstruct signal
    print("  4. 신호 재구성 중...")
    cleaned_signal = np.sum(cleaned_imfs, axis=0)
    
    # Check for NaN or infinite values in final signal
    if np.any(np.isnan(cleaned_signal)) or np.any(np.isinf(cleaned_signal)):
        print("⚠️ 최종 신호에 NaN/Inf 값이 포함되어 있습니다. 원본 신호를 반환합니다.")
        return eeg_signal
    
    print("✅ LMD-DWT EOG 제거 완료")
    return cleaned_signal


def apply_lmd_dwt_cleaning(raw_filtered, test_mode=False):
    """
    Apply LMD-DWT cleaning to remove EOG artifacts from EEG data.
    
    Parameters:
    -----------
    raw_filtered : mne.io.Raw
        Filtered raw EEG data
    test_mode : bool
        Test mode flag
    
    Returns:
    --------
    mne.io.Raw : Cleaned raw data
    """
    print("\n=== LMD-DWT EOG 제거 시작 ===")
    
    if not LMD_DWT_AVAILABLE:
        print("❌ LMD-DWT 기능을 사용할 수 없습니다. 원본 데이터를 반환합니다.")
        return raw_filtered
    
    if test_mode:
        print("테스트 모드: LMD-DWT 제거 없이 진행합니다.")
        return raw_filtered
    
    # Get all data (not just EEG)
    all_data = raw_filtered.get_data()
    eeg_picks = mne.pick_types(raw_filtered.info, eeg=True)
    
    # Clean only EEG channels
    cleaned_data = all_data.copy()
    for i, channel_idx in enumerate(eeg_picks):
        print(f"EEG 채널 {i+1}/{len(eeg_picks)} 처리 중...")
        channel_data = all_data[channel_idx]
        cleaned_channel = remove_eog_artifacts_lmd_dwt(
            channel_data, 
            fs=raw_filtered.info['sfreq']
        )
        cleaned_data[channel_idx] = cleaned_channel
    
    # Create new Raw object with cleaned data
    # Use the original info to maintain all channel information
    cleaned_raw = mne.io.RawArray(cleaned_data, raw_filtered.info)
    
    print("=== LMD-DWT EOG 제거 완료 ===")
    return cleaned_raw


def process_with_lmd_dwt(raw_filtered, test_mode=False, subject=None, output_dir=None):
    """
    Complete LMD-DWT processing pipeline.
    
    Parameters:
    -----------
    raw_filtered : mne.io.Raw
        Filtered raw EEG data
    test_mode : bool
        Test mode flag
    subject : str or int, optional
        Subject ID for visualization
    output_dir : str or Path, optional
        Output directory for visualization
    
    Returns:
    --------
    mne.io.Raw : Fully processed raw data
    """
    print("\n=== LMD-DWT 파이프라인 시작 ===")
    
    # 전처리 전 데이터 저장
    raw_before = raw_filtered.copy()
    
    # Apply LMD-DWT cleaning
    raw_final = apply_lmd_dwt_cleaning(raw_filtered, test_mode=test_mode)
    
    # 시각화 생성 (파라미터가 제공된 경우)
    if subject is not None and output_dir is not None:
        create_preprocessing_visualization(raw_before, raw_final, subject, output_dir, "lmd_dwt")
    
    print("=== LMD-DWT 파이프라인 완료 ===")
    return raw_final


def create_preprocessing_visualization(raw_before, raw_after, subject, output_dir, method_name="lmd_dwt"):
    """
    전처리 전후 비교 시각화 생성 및 저장
    
    Parameters:
    -----------
    raw_before : mne.io.Raw
        전처리 전 데이터
    raw_after : mne.io.Raw
        전처리 후 데이터
    subject : str or int
        참가자 ID
    output_dir : str or Path
        출력 디렉토리
    method_name : str
        아티팩트 제거 방법 이름
    """
    print(f"\n=== {method_name.upper()} 전처리 시각화 생성 ===")
    
    # 출력 디렉토리 생성
    viz_dir = Path(output_dir) / 'visualizations' / method_name
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 시각화 설정
    plt.style.use('default')
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle(f'Subject {subject} - {method_name.upper()} Preprocessing Comparison', fontsize=16)
    
    # 채널 이름
    ch_names = ['TP9', 'AF7', 'AF8', 'TP10']
    
    # 데이터 추출 (처음 10초)
    duration = 10  # 10초
    sfreq = raw_before.info['sfreq']
    n_samples = int(duration * sfreq)
    
    for i, ch_name in enumerate(ch_names):
        # 전처리 전 데이터
        data_before = raw_before.get_data(picks=ch_name)[0, :n_samples]
        time_before = np.arange(len(data_before)) / sfreq
        
        # 전처리 후 데이터
        data_after = raw_after.get_data(picks=ch_name)[0, :n_samples]
        time_after = np.arange(len(data_after)) / sfreq
        
        # 플롯
        axes[i, 0].plot(time_before, data_before, 'b-', linewidth=0.5, alpha=0.8)
        axes[i, 0].set_title(f'{ch_name} - Before {method_name.upper()}')
        axes[i, 0].set_ylabel('Amplitude (μV)')
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].plot(time_after, data_after, 'r-', linewidth=0.5, alpha=0.8)
        axes[i, 1].set_title(f'{ch_name} - After {method_name.upper()}')
        axes[i, 1].set_ylabel('Amplitude (μV)')
        axes[i, 1].grid(True, alpha=0.3)
        
        if i == 3:  # 마지막 행
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    # 파일 저장
    filename = f'subject_{subject}_{method_name}_comparison.png'
    filepath = viz_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 시각화 저장 완료: {filepath}")
    
    # 스펙트럼 밀도 비교 시각화
    create_spectral_comparison(raw_before, raw_after, subject, viz_dir, method_name)


def create_spectral_comparison(raw_before, raw_after, subject, viz_dir, method_name):
    """
    전처리 전후 스펙트럼 밀도 비교 시각화
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Subject {subject} - {method_name.upper()} Spectral Comparison', fontsize=16)
    
    ch_names = ['TP9', 'AF7', 'AF8', 'TP10']
    
    for i, ch_name in enumerate(ch_names):
        row = i // 2
        col = i % 2
        
        try:
            # 전처리 전 스펙트럼
            raw_before.plot_psd(picks=ch_name, ax=axes[row, col], show=False,
                               fmax=50, color='blue', alpha=0.7)
            
            # 전처리 후 스펙트럼
            raw_after.plot_psd(picks=ch_name, ax=axes[row, col], show=False,
                              fmax=50, color='red', alpha=0.7)
            
            # 범례 추가
            axes[row, col].plot([], [], color='blue', alpha=0.7, label='Before')
            axes[row, col].plot([], [], color='red', alpha=0.7, label='After')
            
            axes[row, col].set_title(f'{ch_name}')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"⚠️ {ch_name} 채널 스펙트럼 시각화 중 오류: {str(e)}")
            # 오류가 발생한 경우 빈 플롯 생성
            axes[row, col].text(0.5, 0.5, f'Error: {ch_name}', 
                               ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].set_title(f'{ch_name} (Error)')
    
    plt.tight_layout()
    
    # 파일 저장
    filename = f'subject_{subject}_{method_name}_spectral.png'
    filepath = viz_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 스펙트럼 시각화 저장 완료: {filepath}") 