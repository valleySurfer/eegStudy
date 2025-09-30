#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ICA (Independent Component Analysis) 기반 아티팩트 제거 모듈
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def create_preprocessing_visualization(raw_before, raw_after, subject, output_dir, method_name="ica"):
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


def apply_ica(raw_filtered, test_mode=False, subject=None, output_dir=None):
    """
    Apply ICA for artifact removal with interactive component selection.
    
    Parameters:
    -----------
    raw_filtered : mne.io.Raw
        Filtered raw EEG data
    test_mode : bool
        Test mode flag
    subject : str or int, optional
        Subject ID for saving ICA components
    output_dir : str or Path, optional
        Output directory for saving ICA components
    
    Returns:
    --------
    mne.io.Raw : Cleaned raw data
    """
    print("\n=== ICA 적용 시작 ===")
    
    # ICA 객체 생성
    ica = mne.preprocessing.ICA(
        n_components=4,  # MUSE S has 4 channels
        random_state=42,
        method='fastica'
    )
    print("ICA 객체 생성 완료")
    
    # ICA 피팅
    ica.fit(raw_filtered, picks='eeg')
    print("ICA 피팅 완료")
    
    # ICA 컴포넌트 제거
    print("\n=== ICA 컴포넌트 제거 ===")
    if test_mode:
        print("테스트 모드: 컴포넌트 제거 없이 진행합니다.")
        raw_cleaned = raw_filtered.copy()
    else:
        # ICA 컴포넌트 시각화 및 수동 선택
        raw_cleaned = _interactive_ica_selection(ica, raw_filtered, subject, output_dir)
    
    return raw_cleaned


def _interactive_ica_selection(ica, raw_filtered, subject=None, output_dir=None):
    """
    ICA 컴포넌트를 시각화하고 사용자가 수동으로 제거할 컴포넌트를 선택할 수 있게 합니다.
    
    Parameters:
    -----------
    ica : mne.preprocessing.ICA
        피팅된 ICA 객체
    raw_filtered : mne.io.Raw
        필터링된 원본 데이터
    subject : str or int, optional
        참가자 ID
    output_dir : str or Path, optional
        출력 디렉토리
    
    Returns:
    --------
    mne.io.Raw : 아티팩트가 제거된 데이터
    """
    print("ICA 컴포넌트 시각화를 시작합니다...")
    
    # ICA 컴포넌트 시각화
    try:
        print("ICA 컴포넌트 종합 분석 시각화를 생성합니다...")
        
        # 4개 컴포넌트에 대해 각각 3가지 시각화 (시간파형, 토포맵, 주파수 특성)
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        fig.suptitle(f'Subject {subject} - ICA Components Analysis\n(Time Series | Topography | Power Spectrum)', fontsize=16)
        
        # 각 컴포넌트 분석
        for i in range(4):
            # 1. 시간파형 (Time Series)
            sources = ica.get_sources(raw_filtered)
            data = sources.get_data(picks=i)[0, :]
            time = np.arange(len(data)) / sources.info['sfreq']
            
            # 처음 30초만 표시
            max_time = 30
            max_samples = int(max_time * sources.info['sfreq'])
            if len(data) > max_samples:
                data = data[:max_samples]
                time = time[:max_samples]
            
            axes[i, 0].plot(time, data, 'b-', linewidth=0.5)
            axes[i, 0].set_title(f'Component {i} - Time Series')
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].grid(True, alpha=0.3)
            
            # 깜빡임 패턴 강조 (큰 진폭 부분)
            threshold = np.std(data) * 2
            blink_indices = np.where(np.abs(data) > threshold)[0]
            if len(blink_indices) > 0:
                axes[i, 0].scatter(time[blink_indices], data[blink_indices], 
                                 color='red', s=20, alpha=0.7, label='Potential Blinks')
                axes[i, 0].legend()
            
            # 2. 공간 분포 (Topography) - MUSE S 채널별 가중치
            mixing_matrix = ica.mixing_matrix_
            channel_weights = np.abs(mixing_matrix[:, i])
            
            # MUSE S 채널 위치 시뮬레이션 (TP9, AF7, AF8, TP10)
            channel_positions = {
                'TP9': (-0.8, -0.6),   # 좌측 후두부
                'AF7': (-0.4, 0.3),    # 좌측 전두부
                'AF8': (0.4, 0.3),     # 우측 전두부
                'TP10': (0.8, -0.6)    # 우측 후두부
            }
            
            # 토포맵 생성
            x_pos = [pos[0] for pos in channel_positions.values()]
            y_pos = [pos[1] for pos in channel_positions.values()]
            
            # 간단한 토포맵 시각화
            scatter = axes[i, 1].scatter(x_pos, y_pos, c=channel_weights, s=200, 
                                       cmap='RdBu_r', alpha=0.8, edgecolors='black')
            axes[i, 1].set_title(f'Component {i} - Channel Weights')
            axes[i, 1].set_xlim(-1, 1)
            axes[i, 1].set_ylim(-1, 1)
            axes[i, 1].set_aspect('equal')
            axes[i, 1].grid(True, alpha=0.3)
            
            # 채널 이름 표시
            for j, (ch_name, pos) in enumerate(channel_positions.items()):
                axes[i, 1].annotate(ch_name, (pos[0], pos[1]), 
                                  xytext=(5, 5), textcoords='offset points', 
                                  fontsize=10, fontweight='bold')
            
            # 컬러바 추가
            cbar = plt.colorbar(scatter, ax=axes[i, 1], shrink=0.8)
            cbar.set_label('Weight Magnitude')
            
            # 3. 주파수 특성 (Power Spectrum)
            try:
                freqs, psd = mne.time_frequency.psd_welch(sources, picks=i, fmin=0.5, fmax=100, n_fft=1024)
            except AttributeError:
                # 대안: scipy를 사용한 PSD 계산
                from scipy import signal
                data = sources.get_data(picks=i)[0, :]
                freqs, psd = signal.welch(data, fs=sources.info['sfreq'], nperseg=1024)
                # 주파수 범위 필터링
                freq_mask = (freqs >= 0.5) & (freqs <= 100)
                freqs = freqs[freq_mask]
                psd = psd[freq_mask]
                psd = [psd]  # 리스트 형태로 변환
            
            axes[i, 2].semilogy(freqs, psd[0], 'r-', linewidth=1.5)
            axes[i, 2].set_title(f'Component {i} - Power Spectrum')
            axes[i, 2].set_xlabel('Frequency (Hz)')
            axes[i, 2].set_ylabel('Power Spectral Density')
            axes[i, 2].grid(True, alpha=0.3)
            
            # 주요 주파수 대역 강조
            # 알파파 (8-12 Hz)
            alpha_mask = (freqs >= 8) & (freqs <= 12)
            if np.any(alpha_mask):
                axes[i, 2].fill_between(freqs[alpha_mask], psd[0][alpha_mask], 
                                      alpha=0.3, color='green', label='Alpha (8-12 Hz)')
            
            # EMG 대역 (20-50 Hz)
            emg_mask = (freqs >= 20) & (freqs <= 50)
            if np.any(emg_mask):
                axes[i, 2].fill_between(freqs[emg_mask], psd[0][emg_mask], 
                                      alpha=0.3, color='orange', label='EMG (20-50 Hz)')
            
            # 깜빡임 대역 (1-3 Hz)
            blink_mask = (freqs >= 1) & (freqs <= 3)
            if np.any(blink_mask):
                axes[i, 2].fill_between(freqs[blink_mask], psd[0][blink_mask], 
                                      alpha=0.3, color='blue', label='Blink (1-3 Hz)')
            
            axes[i, 2].legend()
            
            if i == 3:  # 마지막 행
                axes[i, 0].set_xlabel('Time (s)')
        
        plt.tight_layout()
        
        # 파일로 저장
        if subject is not None and output_dir is not None:
            viz_dir = Path(output_dir) / 'visualizations' / 'ica'
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f'subject_{subject}_ica_components.png'
            filepath = viz_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ ICA 컴포넌트 시각화 저장: {filepath}")
        
        plt.show()
        
        # 아티팩트 식별 가이드 출력
        print("\n" + "="*80)
        print("ICA 컴포넌트 아티팩트 식별 가이드:")
        print("="*80)
        print("1. 시간파형 (Time Series) 확인:")
        print("   - 깜빡임: 큰 진폭의 파형이 간헐적으로 나타남 (빨간 점으로 표시)")
        print("   - 심장박동: 규칙적인 파형, 1-2Hz 리듬")
        print("   - 근전도: 고주파성 노이즈")
        print()
        print("2. 공간 분포 (Topography) 확인:")
        print("   - 눈 깜빡임: 전두부(AF7/AF8)에 강하게 분포")
        print("   - 수직 안구운동: 전두부 정중앙에 대칭 분포")
        print("   - 수평 안구운동: 좌우 측두부에 비대칭 분포")
        print()
        print("3. 주파수 특성 (Power Spectrum) 확인:")
        print("   - EMG: 30-100Hz에서 높은 전력 (주황색 영역)")
        print("   - 깜빡임: 1-3Hz에서 피크 (파란색 영역)")
        print("   - 알파파: 8-12Hz에서 피크 (초록색 영역) - 보존 고려")
        print("="*80)
        
        # 사용자에게 제거할 컴포넌트 선택 요청
        print("\n" + "="*50)
        print("ICA 컴포넌트 분석")
        print("="*50)
        print("위의 시각화를 보고 아티팩트로 보이는 컴포넌트를 선택하세요.")
        print("일반적으로 눈 깜빡임, 심장박동, 근육 움직임 등이 아티팩트입니다.")
        print("컴포넌트 번호는 0부터 시작합니다.")
        print("="*50)
        
        while True:
            try:
                user_input = input("제거할 컴포넌트 번호를 입력하세요 (예: 0,1 또는 0 1 또는 none): ").strip()
                
                if user_input.lower() in ['none', 'n', '']:
                    print("컴포넌트 제거 없이 진행합니다.")
                    ica.exclude = []
                    break
                
                # 다양한 입력 형식 처리
                if ',' in user_input:
                    exclude_components = [int(x.strip()) for x in user_input.split(',')]
                else:
                    exclude_components = [int(x) for x in user_input.split()]
                
                # 유효성 검사
                if all(0 <= comp < 4 for comp in exclude_components):
                    ica.exclude = exclude_components
                    print(f"제거할 컴포넌트: {exclude_components}")
                    break
                else:
                    print("❌ 잘못된 컴포넌트 번호입니다. 0-3 사이의 숫자를 입력하세요.")
                    
            except ValueError:
                print("❌ 잘못된 입력입니다. 숫자를 입력하세요.")
            except KeyboardInterrupt:
                print("\n사용자가 중단했습니다. 컴포넌트 제거 없이 진행합니다.")
                ica.exclude = []
                break
        
        # 선택된 컴포넌트 제거
        if ica.exclude:
            print(f"선택된 컴포넌트 {ica.exclude}를 제거합니다...")
            raw_cleaned = ica.apply(raw_filtered)
            print("✅ ICA 컴포넌트 제거 완료")
        else:
            print("컴포넌트 제거 없이 진행합니다.")
            raw_cleaned = raw_filtered.copy()
            
    except Exception as e:
        print(f"⚠️ ICA 시각화 중 오류 발생: {str(e)}")
        print("컴포넌트 제거 없이 진행합니다.")
        raw_cleaned = raw_filtered.copy()
    
    return raw_cleaned


def post_ica_processing(raw_cleaned):
    """
    Apply post-ICA processing steps.
    
    Parameters:
    -----------
    raw_cleaned : mne.io.Raw
        ICA cleaned raw data
    
    Returns:
    --------
    mne.io.Raw : Post-processed raw data
    """
    print("\n=== ICA 이후 후처리 시작 ===")
    
    # 1. Low-pass 필터 (40 Hz)
    print("Low-pass 필터 적용 중 (h_freq=40.0)...")
    raw_filtered = raw_cleaned.copy()
    raw_filtered.filter(l_freq=None, h_freq=40.0, picks='eeg')
    print("Low-pass 필터 완료")
    
    # 2. Notch 필터 (60 Hz)
    print("Notch 필터 적용 중 (60 Hz)...")
    raw_filtered.notch_filter(freqs=60, picks='eeg')
    print("Notch 필터 완료")
    
    # 3. Baseline correction
    print("Baseline correction 적용 중...")
    raw_baseline = raw_filtered.copy()
    # 전체 데이터에 대해 baseline correction 적용
    data = raw_baseline.get_data()
    baseline_mean = np.mean(data, axis=1, keepdims=True)
    data_corrected = data - baseline_mean
    raw_baseline = mne.io.RawArray(data_corrected, raw_baseline.info)
    print("Baseline correction 완료")
    
    print("\n=== ICA 이후 후처리 완료 ===")
    
    return raw_baseline


def process_with_ica(raw_filtered, test_mode=False, subject=None, output_dir=None):
    """
    Complete ICA processing pipeline.
    
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
    print("\n=== ICA 파이프라인 시작 ===")
    
    # 전처리 전 데이터 저장
    raw_before = raw_filtered.copy()
    
    # 1. ICA 적용
    raw_cleaned = apply_ica(raw_filtered, test_mode=test_mode, subject=subject, output_dir=output_dir)
    
    # 2. ICA 이후 후처리
    raw_final = post_ica_processing(raw_cleaned)
    
    # 3. 시각화 생성 (파라미터가 제공된 경우)
    if subject is not None and output_dir is not None:
        create_preprocessing_visualization(raw_before, raw_final, subject, output_dir, "ica")
    
    print("=== ICA 파이프라인 완료 ===")
    return raw_final 