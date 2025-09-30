#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
설정 및 상수 모듈
"""

# 기본 설정
DEFAULT_SAMPLE_RATE = 256  # Hz
DEFAULT_TEST_MODE = True
DEFAULT_SHOW_DUPLICATE_EVENTS = False  # 중복 이벤트 표시 기본값

# 참가자 목록
SUBJECTS = ['3', '4', '5', '6', '9', '10', '11']

# 디렉토리 설정
DATA_DIR = 'D:/EmotionClassification/1_data'
OUTPUT_DIR = 'D:/EmotionClassification/2_output'

# MUSE S 채널 설정
MUSE_CHANNELS = {
    'eeg': ['ch_0', 'ch_1', 'ch_2', 'ch_3'],
    'channel_names': ['TP9', 'AF7', 'AF8', 'TP10'],
    'additional_channels': {
        'ppg_0': 'bio', 'ppg_1': 'bio', 'ppg_2': 'bio',
        'acc_x': 'misc', 'acc_y': 'misc', 'acc_z': 'misc',
        'gyro_x': 'misc', 'gyro_y': 'misc', 'gyro_z': 'misc'
    }
}

# 필터링 설정
FILTER_CONFIG = {
    'highpass': {'l_freq': 0.5},
    'lowpass': {'h_freq': 70.0},
    'notch': {'freqs': 60}
}

# ICA 설정
ICA_CONFIG = {
    'n_components': 4,
    'random_state': 42,
    'method': 'fastica'
}

# 에포크 설정
EPOCH_CONFIG = {
    'main': {
        'tmin': -0.2,
        'tmax': 20.0,
        'baseline': (None, 0)
    },
    'resting': {
        'tmin': -0.2,
        'tmax': 10.0,
        'baseline': (None, 0)
    }
}

# 마커 매핑 설정
MARKER_MAPPING = {
    'emotion': {
        1: 'neutral',
        2: 'happy', 
        3: 'anger',
        4: 'emotional'
    },
    'event_type': {
        1: 'blank',
        2: 'main',
        3: 'slider'
    },
    'condition': {
        1: 'natural',
        2: 'suppress'
    }
}

# 이벤트 상태 매핑 (참가자별)
EVENT_STATE_MAPPING = {
    'blank': {
        1: 'image_start',
        2: 'image_end', 
        3: 'trial_start',
        4: 'trial_end'
    },
    'main': {
        1: 'video_start',
        2: 'video_end',
        3: 'trial_start', 
        4: 'trial_end'
    },
    'slider': {
        1: 'trial_start',
        2: 'trial_end'
    }
}

# 특수 마커
SPECIAL_MARKERS = {
    'resting_start': 501001,
    'resting_end': 502001
}

# 로깅 설정
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# 아티팩트 제거 방법 설정
ARTIFACT_METHODS = {
    'ica': {
        'name': 'ICA (Independent Component Analysis)',
        'description': '전통적인 독립성분분석',
        'default': True
    },
    'lmd_dwt': {
        'name': 'LMD-DWT (Local Mean Decomposition + DWT)',
        'description': 'Local Mean Decomposition + Discrete Wavelet Transform',
        'default': False
    },
    'ica_dwt_hybrid': {
        'name': 'ICA-DWT Hybrid',
        'description': 'ICA와 DWT를 결합한 하이브리드 방법',
        'default': False
    }
}

# 출력 디렉토리 구조
OUTPUT_DIRS = {
    'raw': 'raw',
    'epochs': 'epochs',
    'visualizations': 'visualizations',
    'ica_viz': 'visualizations/ica',
    'lmd_dwt_viz': 'visualizations/lmd_dwt',
    'ica_dwt_hybrid_viz': 'visualizations/ica_dwt_hybrid'
}


class PreprocessingConfig:
    """전처리 설정 관리 클래스"""
    
    def __init__(self, **kwargs):
        """
        전처리 설정 초기화
        
        Parameters:
        -----------
        **kwargs : dict
            설정값들
        """
        # 기본값으로 초기화
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.test_mode = DEFAULT_TEST_MODE
        self.subjects = SUBJECTS.copy()
        self.data_dir = DATA_DIR
        self.output_dir = OUTPUT_DIR
        self.artifact_method = 'ica'
        self.show_duplicate_events = DEFAULT_SHOW_DUPLICATE_EVENTS
        
        # 사용자 정의값으로 업데이트
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"⚠️ 알 수 없는 설정: {key}")
    
    def get_muse_channels(self):
        """MUSE 채널 설정 반환"""
        return MUSE_CHANNELS
    
    def get_filter_config(self):
        """필터링 설정 반환"""
        return FILTER_CONFIG
    
    def get_ica_config(self):
        """ICA 설정 반환"""
        return ICA_CONFIG
    
    def get_epoch_config(self):
        """에포크 설정 반환"""
        return EPOCH_CONFIG
    
    def get_marker_mapping(self):
        """마커 매핑 설정 반환"""
        return MARKER_MAPPING
    
    def get_event_state_mapping(self):
        """이벤트 상태 매핑 설정 반환"""
        return EVENT_STATE_MAPPING
    
    def get_special_markers(self):
        """특수 마커 설정 반환"""
        return SPECIAL_MARKERS
    
    def get_artifact_methods(self):
        """아티팩트 제거 방법 설정 반환"""
        return ARTIFACT_METHODS
    
    def get_output_dirs(self):
        """출력 디렉토리 구조 반환"""
        return OUTPUT_DIRS
    
    def get_logging_config(self):
        """로깅 설정 반환"""
        return LOGGING_CONFIG
    
    def get_full_config(self):
        """전체 설정 반환"""
        return {
            'sample_rate': self.sample_rate,
            'test_mode': self.test_mode,
            'subjects': self.subjects,
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'artifact_method': self.artifact_method,
            'muse_channels': self.get_muse_channels(),
            'filter_config': self.get_filter_config(),
            'ica_config': self.get_ica_config(),
            'epoch_config': self.get_epoch_config(),
            'marker_mapping': self.get_marker_mapping(),
            'event_state_mapping': self.get_event_state_mapping(),
            'special_markers': self.get_special_markers(),
            'artifact_methods': self.get_artifact_methods(),
            'output_dirs': self.get_output_dirs(),
            'logging_config': self.get_logging_config()
        } 