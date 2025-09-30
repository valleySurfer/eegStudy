#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ICA-DWT Hybrid EOG Artifact Removal Module
Based on Prakash & Kumar (2024) method combining ICA and DWT for EOG removal
"""

import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Required libraries
try:
    import pywt
    from sklearn.decomposition import FastICA
    from scipy import signal
    import warnings
    warnings.filterwarnings('ignore')
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    print("⚠️ ICA-DWT Hybrid 기능을 사용하려면 pywt, sklearn, scipy가 필요합니다.")


def apply_ica_decomposition(eeg_data, n_components=None, random_state=42):
    """
    Step 1: Apply ICA to decompose EEG signals into independent components (ICs).
    
    Parameters:
    -----------
    eeg_data : array-like
        EEG data of shape (n_channels, n_samples)
    n_components : int, optional
        Number of components to extract (default: n_channels)
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    tuple : (ica_components, ica_mixing_matrix, ica_unmixing_matrix)
    """
    if n_components is None:
        n_components = eeg_data.shape[0]
    
    # Initialize FastICA
    ica = FastICA(
        n_components=n_components,
        random_state=random_state,
        max_iter=1000,
        tol=1e-7
    )
    
    # Apply ICA (transpose for sklearn format: samples x features)
    ica_components = ica.fit_transform(eeg_data.T)  # Shape: (n_samples, n_components)
    
    # Check if ICA was successful
    if ica_components is None:
        raise ValueError("ICA decomposition failed")
    
    # Get mixing and unmixing matrices
    mixing_matrix = ica.mixing_  # Shape: (n_channels, n_components)
    unmixing_matrix = ica.components_  # Shape: (n_components, n_channels)
    
    # Transpose back to (n_components, n_samples) for consistency
    ica_components = ica_components.T
    
    return ica_components, mixing_matrix, unmixing_matrix


def apply_dwt_to_ic(ic_signal, wavelet='db7', level=4):
    """
    Step 2: Apply DWT to a single IC.
    
    Parameters:
    -----------
    ic_signal : array-like
        Single IC signal (1D array)
    wavelet : str
        Mother wavelet type (default: 'db7')
    level : int
        Decomposition level (default: 4)
    
    Returns:
    --------
    tuple : (coefficients, wavelet_name)
    """
    # Apply DWT
    coefficients = pywt.wavedec(ic_signal, wavelet, level=level)
    return coefficients, wavelet


def suppress_artifacts_in_dwt(coefficients, level=4, threshold_mode='soft', threshold_factor=2.0):
    """
    Step 3: Suppress high-amplitude noise in the detail coefficients.
    
    Parameters:
    -----------
    coefficients : list
        DWT coefficients from pywt.wavedec
    level : int
        Level to focus on for artifact suppression
    threshold_mode : str
        'soft', 'hard', or 'zero' thresholding
    threshold_factor : float
        Threshold factor for artifact detection
    
    Returns:
    --------
    list : Modified coefficients
    """
    if len(coefficients) <= level:
        return coefficients
    
    # Get detail coefficients at specified level
    detail_coeffs = coefficients[level]
    
    # Calculate threshold based on median absolute deviation (MAD)
    mad = np.median(np.abs(detail_coeffs - np.median(detail_coeffs)))
    threshold = threshold_factor * mad
    
    # Apply thresholding
    if threshold_mode == 'soft':
        # Soft thresholding
        detail_coeffs_cleaned = np.sign(detail_coeffs) * np.maximum(
            np.abs(detail_coeffs) - threshold, 0
        )
    elif threshold_mode == 'hard':
        # Hard thresholding
        detail_coeffs_cleaned = detail_coeffs * (np.abs(detail_coeffs) > threshold)
    elif threshold_mode == 'zero':
        # Zero out large coefficients
        detail_coeffs_cleaned = detail_coeffs * (np.abs(detail_coeffs) <= threshold)
    else:
        detail_coeffs_cleaned = detail_coeffs
    
    # Update coefficients
    coefficients_cleaned = coefficients.copy()
    coefficients_cleaned[level] = detail_coeffs_cleaned
    
    return coefficients_cleaned


def reconstruct_ic_from_dwt(coefficients, wavelet):
    """
    Step 4: Reconstruct IC using inverse DWT.
    
    Parameters:
    -----------
    coefficients : list
        Modified DWT coefficients
    wavelet : str
        Wavelet name
    
    Returns:
    --------
    array : Reconstructed IC signal
    """
    # Apply inverse DWT
    reconstructed_signal = pywt.waverec(coefficients, wavelet)
    
    return reconstructed_signal


def apply_inverse_ica(cleaned_components, mixing_matrix):
    """
    Step 5: Apply inverse ICA to reconstruct EEG.
    
    Parameters:
    -----------
    cleaned_components : array-like
        Cleaned ICs of shape (n_components, n_samples)
    mixing_matrix : array-like
        ICA mixing matrix of shape (n_channels, n_components)
    
    Returns:
    --------
    array : Reconstructed EEG signal of shape (n_channels, n_samples)
    """
    # Apply inverse ICA: EEG = mixing_matrix * ICs
    reconstructed_eeg = np.dot(mixing_matrix, cleaned_components)
    
    return reconstructed_eeg


def calculate_snr(original_signal, denoised_signal):
    """
    Calculate Signal-to-Noise Ratio improvement.
    
    Parameters:
    -----------
    original_signal : array-like
        Original signal
    denoised_signal : array-like
        Denoised signal
    
    Returns:
    --------
    float : SNR improvement in dB
    """
    # Calculate noise (difference between original and denoised)
    noise = original_signal - denoised_signal
    
    # Calculate signal and noise power
    signal_power = np.mean(original_signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Calculate SNR
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = float('inf')
    
    return snr


def calculate_rmse(original_signal, denoised_signal):
    """
    Calculate Root Mean Square Error.
    
    Parameters:
    -----------
    original_signal : array-like
        Original signal
    denoised_signal : array-like
        Denoised signal
    
    Returns:
    --------
    float : RMSE value
    """
    rmse = np.sqrt(np.mean((original_signal - denoised_signal) ** 2))
    return rmse


def remove_eog_artifacts_ica_dwt_hybrid(eeg_data, fs=256, wavelet='db7', level=4, 
                                       threshold_mode='soft', threshold_factor=2.0,
                                       return_diagnostics=False):
    """
    Hybrid ICA-DWT method for EOG artifact removal.
    
    Based on Prakash & Kumar (2024): "An efficient approach for denoising EOG artifact 
    through optimal wavelet selection"
    
    Parameters:
    -----------
    eeg_data : array-like
        EEG data of shape (n_channels, n_samples)
    fs : int
        Sampling frequency in Hz
    wavelet : str
        Mother wavelet type (default: 'db7')
    level : int
        DWT decomposition level (default: 4)
    threshold_mode : str
        Thresholding mode: 'soft', 'hard', or 'zero'
    threshold_factor : float
        Threshold factor for artifact detection
    return_diagnostics : bool
        Whether to return diagnostic metrics
    
    Returns:
    --------
    array or tuple : Cleaned EEG data and optionally diagnostic metrics
    """
    if not HYBRID_AVAILABLE:
        print("❌ ICA-DWT Hybrid 기능을 사용할 수 없습니다. 원본 신호를 반환합니다.")
        return eeg_data
    
    print("🔄 ICA-DWT Hybrid EOG 제거 시작...")
    
    # Ensure input is numpy array
    eeg_data = np.array(eeg_data)
    original_shape = eeg_data.shape
    
    # Step 1: Apply ICA
    print("  1. ICA 분해 중...")
    ica_components, mixing_matrix, unmixing_matrix = apply_ica_decomposition(eeg_data)
    print(f"     {ica_components.shape[0]}개의 독립 성분 추출 완료")
    
    # Step 2-4: Apply DWT to each IC and suppress artifacts
    print("  2. DWT 적용 및 아티팩트 억제 중...")
    cleaned_components = np.zeros_like(ica_components)
    
    for i in range(ica_components.shape[0]):
        print(f"    IC {i+1}/{ica_components.shape[0]} 처리 중...")
        
        # Apply DWT
        coefficients, wavelet_name = apply_dwt_to_ic(
            ica_components[i], 
            wavelet=wavelet, 
            level=level
        )
        
        # Suppress artifacts
        cleaned_coefficients = suppress_artifacts_in_dwt(
            coefficients, 
            level=level, 
            threshold_mode=threshold_mode,
            threshold_factor=threshold_factor
        )
        
        # Reconstruct IC
        cleaned_ic = reconstruct_ic_from_dwt(cleaned_coefficients, wavelet_name)
        
        # Ensure same length
        if len(cleaned_ic) != len(ica_components[i]):
            cleaned_ic = cleaned_ic[:len(ica_components[i])]
        
        cleaned_components[i] = cleaned_ic
    
    # Step 5: Reconstruct EEG
    print("  3. EEG 신호 재구성 중...")
    cleaned_eeg = apply_inverse_ica(cleaned_components, mixing_matrix)
    
    # Calculate diagnostics if requested
    diagnostics = {}
    if return_diagnostics:
        print("  4. 진단 지표 계산 중...")
        
        # Calculate SNR improvement
        snr_improvement = calculate_snr(eeg_data, cleaned_eeg)
        diagnostics['snr_improvement_db'] = snr_improvement
        
        # Calculate RMSE
        rmse_value = calculate_rmse(eeg_data, cleaned_eeg)
        diagnostics['rmse'] = rmse_value
        
        # Calculate correlation
        correlations = []
        for i in range(eeg_data.shape[0]):
            corr = np.corrcoef(eeg_data[i], cleaned_eeg[i])[0, 1]
            correlations.append(corr)
        diagnostics['mean_correlation'] = np.mean(correlations)
        diagnostics['correlations_per_channel'] = correlations
        
        print(f"    SNR 개선: {snr_improvement:.2f} dB")
        print(f"    RMSE: {rmse_value:.6f}")
        print(f"    평균 상관계수: {diagnostics['mean_correlation']:.4f}")
    
    print("✅ ICA-DWT Hybrid EOG 제거 완료")
    
    if return_diagnostics:
        return cleaned_eeg, diagnostics
    else:
        return cleaned_eeg


def apply_ica_dwt_hybrid_cleaning(raw_filtered, test_mode=False, 
                                 wavelet='db7', level=4, threshold_mode='soft',
                                 threshold_factor=2.0, return_diagnostics=False):
    """
    Apply ICA-DWT Hybrid cleaning to remove EOG artifacts from EEG data.
    
    Parameters:
    -----------
    raw_filtered : mne.io.Raw
        Filtered raw EEG data
    test_mode : bool
        Test mode flag
    wavelet : str
        Mother wavelet type
    level : int
        DWT decomposition level
    threshold_mode : str
        Thresholding mode
    threshold_factor : float
        Threshold factor
    return_diagnostics : bool
        Whether to return diagnostic metrics
    
    Returns:
    --------
    mne.io.Raw : Cleaned raw data
    """
    print("\n=== ICA-DWT Hybrid EOG 제거 시작 ===")
    
    if not HYBRID_AVAILABLE:
        print("❌ ICA-DWT Hybrid 기능을 사용할 수 없습니다. 원본 데이터를 반환합니다.")
        return raw_filtered
    
    if test_mode:
        print("테스트 모드: ICA-DWT Hybrid 제거 없이 진행합니다.")
        return raw_filtered
    
    # Get EEG data and create EEG-only info
    eeg_picks = mne.pick_types(raw_filtered.info, eeg=True)
    eeg_data = raw_filtered.get_data(picks='eeg')
    eeg_info = mne.pick_info(raw_filtered.info, eeg_picks)
    
    # Apply hybrid cleaning
    if return_diagnostics:
        cleaned_data, diagnostics = remove_eog_artifacts_ica_dwt_hybrid(
            eeg_data, 
            fs=raw_filtered.info['sfreq'],
            wavelet=wavelet,
            level=level,
            threshold_mode=threshold_mode,
            threshold_factor=threshold_factor,
            return_diagnostics=True
        )
        
        # Print diagnostics
        print(f"\n진단 결과:")
        print(f"  SNR 개선: {diagnostics['snr_improvement_db']:.2f} dB")
        print(f"  RMSE: {diagnostics['rmse']:.6f}")
        print(f"  평균 상관계수: {diagnostics['mean_correlation']:.4f}")
    else:
        cleaned_data = remove_eog_artifacts_ica_dwt_hybrid(
            eeg_data, 
            fs=raw_filtered.info['sfreq'],
            wavelet=wavelet,
            level=level,
            threshold_mode=threshold_mode,
            threshold_factor=threshold_factor,
            return_diagnostics=False
        )
    
    # Create new Raw object with cleaned EEG data and EEG-only info
    cleaned_eeg_raw = mne.io.RawArray(cleaned_data, eeg_info)
    
    # If there are non-EEG channels, we need to preserve them
    non_eeg_picks = mne.pick_types(raw_filtered.info, eeg=False)
    if len(non_eeg_picks) > 0:
        # Get non-EEG data
        non_eeg_data = raw_filtered.get_data(picks=non_eeg_picks)
        non_eeg_info = mne.pick_info(raw_filtered.info, non_eeg_picks)
        
        # Create RawArray for non-EEG channels
        non_eeg_raw = mne.io.RawArray(non_eeg_data, non_eeg_info)
        
        # Combine EEG and non-EEG channels
        cleaned_raw = mne.io.RawArray(
            np.vstack([cleaned_data, non_eeg_data]), 
            raw_filtered.info
        )
    else:
        # Only EEG channels, use the cleaned data directly
        cleaned_raw = cleaned_eeg_raw
    
    print("=== ICA-DWT Hybrid EOG 제거 완료 ===")
    return cleaned_raw


def process_with_ica_dwt_hybrid(raw_filtered, test_mode=False, 
                               wavelet='db7', level=4, threshold_mode='soft',
                               threshold_factor=2.0, return_diagnostics=False,
                               subject=None, output_dir=None):
    """
    Complete ICA-DWT Hybrid processing pipeline.
    
    Parameters:
    -----------
    raw_filtered : mne.io.Raw
        Filtered raw EEG data
    test_mode : bool
        Test mode flag
    wavelet : str
        Mother wavelet type
    level : int
        DWT decomposition level
    threshold_mode : str
        Thresholding mode
    threshold_factor : float
        Threshold factor
    return_diagnostics : bool
        Whether to return diagnostic metrics
    subject : str or int, optional
        Subject ID for visualization
    output_dir : str or Path, optional
        Output directory for visualization
    
    Returns:
    --------
    mne.io.Raw : Fully processed raw data
    """
    print("\n=== ICA-DWT Hybrid 파이프라인 시작 ===")
    
    # 전처리 전 데이터 저장
    raw_before = raw_filtered.copy()
    
    # Apply ICA-DWT Hybrid cleaning
    raw_final = apply_ica_dwt_hybrid_cleaning(
        raw_filtered, 
        test_mode=test_mode,
        wavelet=wavelet,
        level=level,
        threshold_mode=threshold_mode,
        threshold_factor=threshold_factor,
        return_diagnostics=return_diagnostics
    )
    
    # 시각화 생성 (파라미터가 제공된 경우)
    if subject is not None and output_dir is not None:
        create_preprocessing_visualization(raw_before, raw_final, subject, output_dir, "ica_dwt_hybrid")
    
    print("=== ICA-DWT Hybrid 파이프라인 완료 ===")
    return raw_final


def create_preprocessing_visualization(raw_before, raw_after, subject, output_dir, method_name="ica_dwt_hybrid"):
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
    
    plt.tight_layout()
    
    # 파일 저장
    filename = f'subject_{subject}_{method_name}_spectral.png'
    filepath = viz_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 스펙트럼 시각화 저장 완료: {filepath}") 