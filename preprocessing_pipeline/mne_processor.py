#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MNE 객체 생성 및 기본 전처리 모듈
"""

import os
import numpy as np
import mne
from pathlib import Path
from marker_parser import create_annotations_from_markers


def create_mne_raw(data, config=None, save_raw=False, subject=None, output_dir=None):
    """
    Create MNE Raw object from EEG and marker data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        EEG 데이터
    config : PreprocessingConfig, optional
        전처리 설정 객체
    save_raw : bool
        Raw 객체 저장 여부
    subject : str
        참가자 ID
    output_dir : str or Path
        출력 디렉토리
    """
    # 설정 로드
    if config is None:
        from config import MUSE_CHANNELS, DEFAULT_SAMPLE_RATE
        muse_channels = MUSE_CHANNELS
        sfreq = DEFAULT_SAMPLE_RATE
    else:
        muse_channels = config.get_muse_channels()
        sfreq = config.sample_rate
    
    # Extract EEG data (excluding non-EEG columns)
    eeg_channels = muse_channels['eeg']
    eeg_array = data[eeg_channels].values.T  # MNE expects (n_channels, n_times)
    
    # Create channel names list with proper mapping
    ch_names = muse_channels['channel_names'].copy()
    ch_types = ['eeg'] * len(ch_names)
    
    # Check for additional sensor data
    additional_channels = muse_channels['additional_channels']
    
    for col, ch_type in additional_channels.items():
        if col in data.columns:
            ch_names.append(col)
            ch_types.append(ch_type)
            # Add to eeg_array
            additional_data = data[col].values.reshape(1, -1)
            eeg_array = np.vstack([eeg_array, additional_data])
    
    # Create MNE Raw object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_array, info)
    
    # Get EEG start time for marker alignment
    eeg_start_time = float(data['eeg_timestamp'].iloc[0])
    
    # Create annotations from markers
    annotations, valid_descriptions = create_annotations_from_markers(
        data, eeg_start_time, subject=subject, config=config
    )
    
    if annotations:
        # Create MNE Annotations object
        onsets = [ann['onset'] for ann in annotations]
        durations = [ann['duration'] for ann in annotations]
        descriptions = [ann['description'] for ann in annotations]
        
        mne_annotations = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions
        )
        
        # Add annotations to raw
        raw.set_annotations(mne_annotations)
        
        print(f"Processed {len(valid_descriptions)} valid markers out of {len(data[['marker', 'marker_timestamp']].dropna())} total markers")
        
        # Print example markers
        print("\nExample markers:")
        for i, desc in enumerate(valid_descriptions[:5]):
            marker_val = data[data['marker'].notna()]['marker'].iloc[i]
            print(f"Marker {marker_val}: {desc}")
    
    # Raw 객체 저장 (옵션)
    if save_raw and subject is not None and output_dir is not None:
        output_dir = Path(output_dir)
        raw_dir = output_dir / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        raw_filename = os.path.join(raw_dir, f'subject_{subject}_raw.fif')
        raw.save(raw_filename, overwrite=True)
        print(f"✅ Raw 객체 저장 완료: {raw_filename}")
    
    return raw


def preprocess_data(raw, config=None, test_mode=None):
    """
    Apply preprocessing steps to the raw data.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG 데이터
    config : PreprocessingConfig, optional
        전처리 설정 객체
    test_mode : bool, optional
        테스트 모드 여부
    """
    print("\n=== 전처리 시작 ===")
    
    # 설정 로드
    if config is None:
        from config import FILTER_CONFIG, DEFAULT_TEST_MODE
        filter_config = FILTER_CONFIG
        if test_mode is None:
            test_mode = DEFAULT_TEST_MODE
    else:
        filter_config = config.get_filter_config()
        if test_mode is None:
            test_mode = config.test_mode
    
    # 1. High-pass 필터링 (0.5 Hz)
    print("1. High-pass 필터링 적용 중... (0.5 Hz)")
    raw_filtered = raw.copy()
    raw_filtered.filter(
        l_freq=filter_config['highpass']['l_freq'], 
        h_freq=None, 
        picks='eeg'
    )
    
    return raw_filtered


def apply_post_artifact_filters(raw, config=None):
    """
    아티팩트 제거 후 적용할 필터들 (low-pass, notch)
    
    Parameters:
    -----------
    raw : mne.io.Raw
        아티팩트 제거된 Raw 데이터
    config : PreprocessingConfig, optional
        전처리 설정 객체
    
    Returns:
    --------
    mne.io.Raw
        추가 필터링된 Raw 데이터
    """
    print("\n=== 아티팩트 제거 후 필터링 ===")
    
    # 설정 로드
    if config is None:
        from config import FILTER_CONFIG
        filter_config = FILTER_CONFIG
    else:
        filter_config = config.get_filter_config()
    
    # 1. Low-pass 필터링 (70 Hz)
    print("1. Low-pass 필터링 적용 중... (70 Hz)")
    raw_filtered = raw.copy()
    raw_filtered.filter(
        l_freq=None, 
        h_freq=filter_config['lowpass']['h_freq'], 
        picks='eeg'
    )
    
    # 2. Notch 필터링 (60 Hz)
    print("2. Notch 필터링 적용 중... (60 Hz)")
    notch_config = filter_config['notch']
    raw_filtered.notch_filter(
        freqs=notch_config['freqs'], 
        picks='eeg'
    )
    
    return raw_filtered 