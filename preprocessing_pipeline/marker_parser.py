#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
마커 파싱 및 이벤트 처리 모듈
"""

import pandas as pd


def parse_marker(marker, subject=None):
    """Parse marker into its components.
    Handles both 4-digit (routine markers) and 6-digit (video markers) formats.
    
    Parameters:
    -----------
    marker : int or float
        Marker value to parse
    subject : int, optional
        Subject ID to determine parsing logic
    """
    try:
        if pd.isna(marker):
            return None
        
        marker_num = int(marker)  # ensure int

        if marker_num < 100000:  # 6자리 미만은 무시
            return None
        
        # Resting 이벤트 처리 (501001: 시작, 502001: 끝)
        if marker_num in [501001, 502001]:
            return {
                'emotion': 'resting',
                'event_type': 'resting',
                'event_state': 'start' if marker_num == 501001 else 'end',
                'video_order': 0,
                'video_serial': 0,
                'condition': 'resting'
            }
        
        # 각 자리수 추출
        emotion_code = marker_num // 100000
        event_type_code = (marker_num % 100000) // 10000
        event_state_code = (marker_num % 10000) // 1000
        video_serial = (marker_num % 1000) // 100
        video_order = (marker_num % 100) // 10
        condition_code = marker_num % 10
        
        # 감정 매핑
        emotion_map = {
            1: 'neutral', 2: 'happy', 3: 'anger', 4: 'emotional'
        }
        emotion = emotion_map.get(emotion_code, 'unknown')
        
        # 이벤트 타입 매핑
        event_type_map = {
            1: 'blank', 2: 'main', 3: 'slider'
        }
        event_type = event_type_map.get(event_type_code, 'unknown')
        
        # 참가자별 이벤트 상태 매핑
        if subject in [9, 10, 11]:
            # 참가자 9, 10, 11번: event_type과 관계없이 고정된 매핑
            event_state_map = {
                1: 'trial_start', 2: 'trial_end', 3: 'image_start', 
                4: 'image_stop', 5: 'video_start', 6: 'video_stop'
            }
        else:
            # 참가자 3,4,5,6번 및 기타: event_type에 따라 다르게 매핑
            if event_type == 'blank':
                event_state_map = {
                    1: 'image_start', 2: 'image_end', 3: 'trial_start', 4: 'trial_end'
                }
            elif event_type == 'main':
                event_state_map = {
                    1: 'video_start', 2: 'video_end', 3: 'trial_start', 4: 'trial_end'
                }
            else:  # slider
                event_state_map = {
                    1: 'trial_start', 2: 'trial_end'
                }
        
        event_state = event_state_map.get(event_state_code, 'unknown')
        
        # 조건 매핑
        condition_map = {
            1: 'natural', 2: 'suppress'
        }
        condition = condition_map.get(condition_code, 'unknown')
        
        return {
            'emotion': emotion,
            'event_type': event_type,
            'event_state': event_state,
            'video_order': video_order,
            'video_serial': video_serial,
            'condition': condition
        }
        
    except Exception as e:
        print(f"❌ Error parsing marker {marker}: {str(e)}")
        return None


def create_annotations_from_markers(data, eeg_start_time, subject=None, config=None):
    """Create MNE annotations from marker data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data containing marker and timestamp columns
    eeg_start_time : float
        EEG recording start time
    subject : int, optional
        Subject ID for marker parsing logic
    config : PreprocessingConfig, optional
        전처리 설정 객체
    """
    annotations = []
    valid_descriptions = []
    
    # 마커 데이터 추출
    marker_data = data[['marker', 'marker_timestamp']].dropna()
    
    for idx, row in marker_data.iterrows():
        parsed = parse_marker(row['marker'], subject=subject)
        if parsed is not None:
            # 시간 계산 (초 단위)
            marker_time = (row['marker_timestamp'] - eeg_start_time) / 1000.0
            
            # 주석 설명 생성
            desc = f"{parsed['emotion']}_{parsed['event_type']}_{parsed['event_state']}_{parsed['video_serial']}_{parsed['video_order']}_{parsed['condition']}"
            valid_descriptions.append(desc)
            
            # 주석 추가
            annotations.append({
                'onset': marker_time,
                'duration': 0.0,
                'description': desc
            })
    
    return annotations, valid_descriptions


def analyze_markers(data, subject=None):
    """Analyze marker data and return statistics.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data containing marker and timestamp columns
    subject : int, optional
        Subject ID for marker parsing logic
    """
    marker_data = data[['marker', 'marker_timestamp']].dropna()
    total_markers = len(marker_data)
    
    # 유효한 마커 분석
    valid_markers = 0
    video_start_markers = 0
    marker_types = {}
    
    for idx, row in marker_data.iterrows():
        parsed = parse_marker(row['marker'], subject=subject)
        if parsed is not None:
            valid_markers += 1
            
            # 마커 타입별 카운트
            marker_key = f"{parsed['emotion']}_{parsed['event_type']}_{parsed['event_state']}"
            marker_types[marker_key] = marker_types.get(marker_key, 0) + 1
            
            # video_start 마커 카운트
            if parsed['event_state'] == 'video_start':
                video_start_markers += 1
    
    return {
        'total_markers': total_markers,
        'valid_markers': valid_markers,
        'video_start_markers': video_start_markers,
        'marker_types': marker_types
    } 