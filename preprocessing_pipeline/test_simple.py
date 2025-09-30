#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
간단한 전처리 파이프라인 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eeg_preprocessing import EEGPreprocessor
from config import PreprocessingConfig

def test_single_subject():
    """단일 참가자 테스트"""
    print("=== 단일 참가자 테스트 시작 ===")
    
    # 설정 초기화
    config = PreprocessingConfig(
        subjects=['3'],  # 참가자 3만 테스트
        test_mode=True,  # 테스트 모드
        artifact_method='ica'  # 간단한 ICA 방법
    )
    
    # 전처리기 초기화
    preprocessor = EEGPreprocessor(config=config)
    
    # 단일 참가자 처리
    success = preprocessor.process_subject('3')
    
    if success:
        print("✅ 참가자 3 처리 성공!")
    else:
        print("❌ 참가자 3 처리 실패!")
    
    return success

if __name__ == "__main__":
    test_single_subject() 