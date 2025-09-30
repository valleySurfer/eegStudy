#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
에포크 생성 및 저장 모듈
"""

import os
import numpy as np
import mne
from pathlib import Path


def create_epochs(raw_post_ica, raw, config=None):
    """Create epochs from the preprocessed data."""
    print("\n=== 에포크 생성 단계 시작 ===")
    
    # 설정 기본값
    if config is None:
        from config import PreprocessingConfig
        config = PreprocessingConfig()
    
    # 1. 이벤트 추출 (raw 기준)
    events, event_id = mne.events_from_annotations(raw)
    print(f"총 이벤트 수: {len(events)}")

    # 이벤트 샘플 인덱스 중복 여부 확인
    unique, counts = np.unique(events[:, 0], return_counts=True)
    duplicates = unique[counts > 1]
    if len(duplicates) > 0:
        if config.show_duplicate_events:
            print(f"⚠️ 중복된 이벤트 샘플 인덱스: {duplicates}")
            for dup in duplicates:
                dup_events = events[events[:, 0] == dup]
                print(f"샘플 인덱스 {dup}에서 중복 이벤트:")
                for e in dup_events:
                    # annotation description 출력
                    desc = None
                    for k, v in event_id.items():
                        if v == e[2]:
                            desc = k
                            break
                    print(f"  - event_id: {e[2]}, description: {desc}")
        else:
            print(f"⚠️ 중복된 이벤트 샘플 인덱스 {len(duplicates)}개 발견 (상세 정보는 비활성화됨)")
    else:
        print("중복 이벤트 없음")
    
    # 2. 'video_start/end' 또는 'resting' 이벤트 타입만 필터링 (trial 제외)
    main_keys = [k for k in event_id if '_video_' in k]
    resting_keys = [k for k in event_id if 'resting' in k]
    all_event_keys = main_keys + resting_keys
    print(f"video 이벤트 키 수: {len(main_keys)}")
    print(f"resting 이벤트 키 수: {len(resting_keys)}")
    print(f"총 분석 이벤트 키 수: {len(all_event_keys)}")
    
    # 분석할 이벤트가 없는 경우 처리
    if len(all_event_keys) == 0:
        print("⚠️ 분석할 수 있는 이벤트가 없습니다.")
        print("사용 가능한 이벤트 키들:")
        for key in event_id.keys():
            print(f"  - {key}")
        raise ValueError("분석할 수 있는 이벤트가 없습니다. 마커 파싱을 확인해주세요.")
    
    # 3. 이벤트별로 분리하여 에포크 생성 (baseline correction 없이)
    event_id = dict(event_id)  # Ensure event_id is a dict
    all_epochs = []
    
    # Main 이벤트 에포크 생성 (tmax=20)
    if main_keys:
        main_event_codes = [event_id[k] for k in main_keys]
        main_events = events[np.isin(events[:, 2], main_event_codes)]
        main_event_id = {k: event_id[k] for k in main_keys}
        
        if len(main_events) > 0:
            main_epochs = mne.Epochs(
                raw_post_ica,
                events=main_events,
                event_id=main_event_id,
                tmin=-0.2,
                tmax=20.0,
                baseline=None,  # baseline correction 없이 생성
                preload=True,
                reject=None,
                event_repeated='drop'
            )
            all_epochs.append(('main', main_epochs))
            print(f"Main 이벤트 에포크 수: {len(main_epochs)}")
    
    # Resting 이벤트 에포크 생성 (tmax=10)
    if resting_keys:
        resting_event_codes = [event_id[k] for k in resting_keys]
        resting_events = events[np.isin(events[:, 2], resting_event_codes)]
        resting_event_id = {k: event_id[k] for k in resting_keys}
        
        if len(resting_events) > 0:
            resting_epochs = mne.Epochs(
                raw_post_ica,
                events=resting_events,
                event_id=resting_event_id,
                tmin=-0.2,
                tmax=10.0,
                baseline=None,  # baseline correction 없이 생성
                preload=True,
                reject=None,
                event_repeated='drop'
            )
            all_epochs.append(('resting', resting_epochs))
            print(f"Resting 이벤트 에포크 수: {len(resting_epochs)}")
    
    # 모든 에포크 결합 (길이가 다르므로 별도로 처리)
    if len(all_epochs) == 0:
        raise ValueError("생성된 에포크가 없습니다.")
    
    total_events = sum(len(epo[1].events) for epo in all_epochs)
    print(f"\n총 생성된 에포크 수: {sum(len(epo[1]) for epo in all_epochs)}")
    print(f"제거된 에포크 수: {total_events - sum(len(epo[1]) for epo in all_epochs)}")
    print("=" * 40)
    
    return all_epochs


def apply_baseline_correction(epochs, subject):
    """
    Baseline correction 적용
    resting이 있는 참가자는 resting 에포크 기준으로 baseline correction
    
    Parameters:
    -----------
    epochs : list
        에포크 리스트 [(event_type, epochs), ...]
    subject : str
        참가자 ID
    
    Returns:
    --------
    list
        baseline correction이 적용된 에포크 리스트
    """
    print("\n=== Baseline Correction 적용 ===")
    
    # resting 에포크가 있는지 확인
    resting_epochs = None
    main_epochs = None
    
    for event_type, epo in epochs:
        if event_type == 'resting':
            resting_epochs = epo
        elif event_type == 'main':
            main_epochs = epo
    
    corrected_epochs = []
    
    # Main 에포크 baseline correction
    if main_epochs is not None:
        if resting_epochs is not None:
            # resting 에포크가 있으면 resting 기준으로 baseline correction
            print("Resting 에포크 기준으로 main 에포크 baseline correction 적용")
            # resting 에포크의 평균을 baseline으로 사용
            resting_baseline = resting_epochs.get_data().mean(axis=0)  # (channels, times)
            main_data = main_epochs.get_data()
            
            # baseline 구간 (tmin ~ 0)의 평균을 계산
            baseline_times = main_epochs.times
            baseline_indices = (baseline_times >= main_epochs.tmin) & (baseline_times <= 0)
            
            if np.any(baseline_indices):
                # 각 에포크의 baseline 구간 평균을 빼기
                baseline_mean = main_data[:, :, baseline_indices].mean(axis=2, keepdims=True)
                main_data_corrected = main_data - baseline_mean
                
                # 새로운 에포크 객체 생성
                main_epochs_corrected = main_epochs.copy()
                main_epochs_corrected._data = main_data_corrected
                corrected_epochs.append(('main', main_epochs_corrected))
                print("✅ Main 에포크 baseline correction 완료")
            else:
                print("⚠️ Baseline 구간을 찾을 수 없어 baseline correction을 건너뜁니다.")
                corrected_epochs.append(('main', main_epochs))
        else:
            # resting 에포크가 없으면 일반적인 baseline correction
            print("일반적인 baseline correction 적용 (tmin ~ 0)")
            main_epochs_corrected = main_epochs.copy()
            main_epochs_corrected.apply_baseline(baseline=(main_epochs.tmin, 0))
            corrected_epochs.append(('main', main_epochs_corrected))
            print("✅ Main 에포크 baseline correction 완료")
    
    # Resting 에포크 baseline correction
    if resting_epochs is not None:
        print("Resting 에포크 baseline correction 적용")
        resting_epochs_corrected = resting_epochs.copy()
        resting_epochs_corrected.apply_baseline(baseline=(resting_epochs.tmin, 0))
        corrected_epochs.append(('resting', resting_epochs_corrected))
        print("✅ Resting 에포크 baseline correction 완료")
    
    return corrected_epochs


def save_results(epochs, subject, output_dir, artifact_method=None):
    """Save the preprocessed epochs data."""
    print("\n=== 결과 저장 시작 ===")
    
    # 결과 저장
    output_dir = Path(output_dir)
    epochs_dir = output_dir / 'epochs'
    epochs_dir.mkdir(parents=True, exist_ok=True)
    
    # 전처리 방법을 파일명에 반영
    method_suffix = f"_{artifact_method}" if artifact_method else ""
    
    # 에포크 데이터 저장
    for event_type, epo in epochs:
        filename = f'subject_{subject}_{event_type}_epo{method_suffix}.fif'
        epo.save(os.path.join(epochs_dir, filename), overwrite=True)
        print(f"✅ Saved preprocessed data to {filename}")
    print("=" * 40)


def load_epochs(subject, output_dir, event_type='main'):
    """Load saved epochs data."""
    output_dir = Path(output_dir)
    epochs_dir = output_dir / 'epochs'
    epochs_file = epochs_dir / f'subject_{subject}_{event_type}_epo.fif'
    
    if epochs_file.exists():
        return mne.read_epochs(epochs_file)
    else:
        raise FileNotFoundError(f"Epochs file not found: {epochs_file}")


def get_epochs_info(epochs):
    """Get information about epochs."""
    if isinstance(epochs, list):
        # Multiple epochs (main, resting)
        info = {}
        for event_type, epo in epochs:
            info[event_type] = {
                'n_epochs': len(epo),
                'n_channels': epo.info['nchan'],
                'n_times': epo.times.shape[0],
                'duration': epo.tmax - epo.tmin,
                'event_types': list(epo.event_id.keys())
            }
        return info
    else:
        # Single epochs
        return {
            'n_epochs': len(epochs),
            'n_channels': epochs.info['nchan'],
            'n_times': epochs.times.shape[0],
            'duration': epochs.tmax - epochs.tmin,
            'event_types': list(epochs.event_id.keys())
        } 