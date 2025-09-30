#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EEG 전처리 메인 스크립트
모듈화된 구조를 사용하여 EEG 데이터를 전처리하고 에포크를 생성합니다.

주요 개선사항:
1. 설정 기반 초기화
2. 클래스 기반 설계
3. 중앙화된 설정 관리
4. 유연한 아티팩트 제거 방법 선택
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

# 설정 및 모듈 import
from config import PreprocessingConfig, SUBJECTS, DATA_DIR, OUTPUT_DIR, DEFAULT_TEST_MODE
from data_loader import find_subject_file, load_csv, validate_data_file
from mne_processor import create_mne_raw, preprocess_data, apply_post_artifact_filters
from ica_processor import process_with_ica
from lmd_dwt_processor import process_with_lmd_dwt
from ica_dwt_hybrid_processor import process_with_ica_dwt_hybrid
from epoch_processor import create_epochs, save_results, apply_baseline_correction
from analysis_utils import (
    analyze_event_counts, 
    create_processing_summary,
    validate_processing_pipeline
)


class EEGPreprocessor:
    """EEG 전처리 메인 클래스"""
    
    def __init__(self, config=None, data_dir=None, output_dir=None, 
                 test_mode=None, artifact_method=None):
        """
        EEG 전처리기 초기화
        
        Parameters:
        -----------
        config : PreprocessingConfig, optional
            전처리 설정 객체
        data_dir : str, optional
            데이터 디렉토리
        output_dir : str, optional
            출력 디렉토리
        test_mode : bool, optional
            테스트 모드 여부
        artifact_method : str, optional
            아티팩트 제거 방법
        """
        # 설정 객체 초기화
        if config is None:
            self.config = PreprocessingConfig()
        else:
            self.config = config
        
        # 디렉토리 설정
        if data_dir is None:
            self.data_dir = Path(self.config.data_dir)
        else:
            self.data_dir = Path(data_dir)
        
        if output_dir is None:
            self.output_dir = Path(self.config.output_dir)
        else:
            self.output_dir = Path(output_dir)
        
        # 기타 설정
        if test_mode is None:
            self.test_mode = self.config.test_mode
        else:
            self.test_mode = test_mode
        
        if artifact_method is None:
            self.artifact_method = self.config.artifact_method
        else:
            self.artifact_method = artifact_method
        
        # 출력 디렉토리 구조 생성
        self._create_output_directories()
    
    def _create_output_directories(self):
        """출력 디렉토리 구조 생성"""
        output_dirs = self.config.get_output_dirs()
        
        for dir_name, dir_path in output_dirs.items():
            full_path = self.output_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
        
        print("✅ 출력 디렉토리 구조 생성 완료")
    
    def process_subject(self, subject):
        """
        단일 참가자 데이터 처리
        
        Parameters:
        -----------
        subject : str
            참가자 ID
        
        Returns:
        --------
        bool
            처리 성공 여부
        """
        print(f"\n=== 참가자 {subject} 데이터 처리 시작 ===")
        print(f"아티팩트 제거 방법: {self.artifact_method.upper()}")
        
        # 데이터 파일 찾기
        data_file = find_subject_file(self.data_dir, subject)
        if data_file is None:
            print(f"❌ 참가자 {subject}의 데이터 파일을 찾을 수 없습니다.")
            return False
            
        print(f"데이터 파일: {data_file}")
        
        # 파일 검증
        is_valid, message = validate_data_file(data_file)
        if not is_valid:
            print(f"❌ 파일 검증 실패: {message}")
            return False
        
        try:
            # 데이터 로드
            data = load_csv(data_file)
            if data is None:
                return False
            
            # MNE Raw 객체 생성
            raw = create_mne_raw(
                data, 
                config=self.config,
                save_raw=True, 
                subject=subject, 
                output_dir=self.output_dir
            )
            
            # 1. High-pass 필터링 (0.5 Hz)
            raw_highpass = preprocess_data(
                raw, 
                config=self.config,
                test_mode=self.test_mode
            )
            
            # 2. 아티팩트 제거 (ICA, LMD-DWT, ICA-DWT 등)
            raw_artifact_removed = self._apply_artifact_removal(raw_highpass, subject)
            
            # 3. Low-pass 필터링 (70 Hz)
            # 4. Notch 필터링 (60 Hz)
            raw_final = apply_post_artifact_filters(raw_artifact_removed, config=self.config)
            
            # 5. 에포크 생성 (baseline correction 없이)
            epochs = create_epochs(raw_final, raw, config=self.config)
            
            # 6. Baseline correction 적용
            epochs_corrected = apply_baseline_correction(epochs, subject)
            
            # 7. 결과 저장
            save_results(epochs_corrected, subject, self.output_dir, artifact_method=self.artifact_method)
            
            print(f"✅ 참가자 {subject} 데이터 처리 완료")
            return True
            
        except Exception as e:
            print(f"❌ 참가자 {subject} 데이터 처리 중 오류 발생: {str(e)}")
            return False
    
    def _apply_artifact_removal(self, raw_filtered, subject):
        """
        아티팩트 제거 방법 적용
        
        Parameters:
        -----------
        raw_filtered : mne.io.Raw
            필터링된 Raw 데이터
        subject : str
            참가자 ID
        
        Returns:
        --------
        mne.io.Raw
            아티팩트 제거된 Raw 데이터
        """
        artifact_methods = self.config.get_artifact_methods()
        
        if self.artifact_method.lower() == 'ica':
            return process_with_ica(
                raw_filtered, 
                test_mode=self.test_mode, 
                subject=subject, 
                output_dir=self.output_dir
            )
        elif self.artifact_method.lower() == 'lmd_dwt':
            return process_with_lmd_dwt(
                raw_filtered, 
                test_mode=self.test_mode, 
                subject=subject, 
                output_dir=self.output_dir
            )
        elif self.artifact_method.lower() == 'ica_dwt_hybrid':
            return process_with_ica_dwt_hybrid(
                raw_filtered, 
                test_mode=self.test_mode, 
                subject=subject, 
                output_dir=self.output_dir
            )
        else:
            print(f"⚠️ 알 수 없는 아티팩트 제거 방법: {self.artifact_method}. ICA를 사용합니다.")
            return process_with_ica(
                raw_filtered, 
                test_mode=self.test_mode, 
                subject=subject, 
                output_dir=self.output_dir
            )
    
    def run_processing(self, subjects=None):
        """
        전체 전처리 실행
        
        Parameters:
        -----------
        subjects : list, optional
            처리할 참가자 리스트
        """
        if subjects is None:
            subjects = self.config.subjects
        
        print("=" * 60)
        print("EEG 전처리 파이프라인 시작")
        print("=" * 60)
        print(f"데이터 디렉토리: {self.data_dir}")
        print(f"출력 디렉토리: {self.output_dir}")
        print(f"아티팩트 제거 방법: {self.artifact_method}")
        print(f"테스트 모드: {'예' if self.test_mode else '아니오'}")
        print(f"처리할 참가자: {subjects}")
        
        # 파이프라인 검증
        if not validate_processing_pipeline(self.data_dir, self.output_dir):
            print("❌ 파이프라인 검증 실패. 프로그램을 종료합니다.")
            return
        
        # 각 참가자별로 처리
        success_count = 0
        total_count = len(subjects)
        
        for subject in subjects:
            if self.process_subject(subject):
                success_count += 1
        
        # 처리 결과 요약
        summary = create_processing_summary(subjects, self.data_dir, self.output_dir, self.artifact_method)
        
        print("\n" + "=" * 60)
        print("모든 처리 완료!")
        print(f"성공: {success_count}/{total_count} 참가자")
        print(f"성공률: {success_count/total_count*100:.1f}%")
        print(f"아티팩트 제거 방법: {self.artifact_method.upper()}")
        print(f"테스트 모드: {'예' if self.test_mode else '아니오'}")
        print("=" * 60)


def process_subject(subject, data_dir, output_dir, test_mode=DEFAULT_TEST_MODE, artifact_method='ica'):
    """
    단일 참가자 데이터 처리 (기존 함수 호환성 유지)
    
    Parameters:
    -----------
    subject : str
        참가자 ID
    data_dir : str or Path
        데이터 디렉토리
    output_dir : str or Path
        출력 디렉토리
    test_mode : bool
        테스트 모드 여부
    artifact_method : str
        아티팩트 제거 방법
    
    Returns:
    --------
    bool
        처리 성공 여부
    """
    config = PreprocessingConfig(
        test_mode=test_mode,
        artifact_method=artifact_method
    )
    
    preprocessor = EEGPreprocessor(
        config=config,
        data_dir=data_dir,
        output_dir=output_dir
    )
    
    return preprocessor.process_subject(subject)


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("EEG 전처리 파이프라인 시작")
    print("=" * 60)
    
    # 기본 설정으로 전처리기 초기화
    config = PreprocessingConfig()
    preprocessor = EEGPreprocessor(config=config)
    
    # 아티팩트 제거 방법 선택
    artifact_methods = config.get_artifact_methods()
    
    print("\n" + "="*50)
    print("아티팩트 제거 방법을 선택하세요:")
    for i, (method, info) in enumerate(artifact_methods.items(), 1):
        print(f"{i}. {info['name']}")
        print(f"   {info['description']}")
    print("="*50)
    
    while True:
        choice = input("선택 (1, 2, 또는 3): ").strip()
        if choice == '1':
            artifact_method = 'ica'
            print("✅ ICA 방법을 선택했습니다.")
            break
        elif choice == '2':
            artifact_method = 'lmd_dwt'
            print("✅ LMD-DWT 방법을 선택했습니다.")
            break
        elif choice == '3':
            artifact_method = 'ica_dwt_hybrid'
            print("✅ ICA-DWT Hybrid 방법을 선택했습니다.")
            break
        else:
            print("❌ 잘못된 선택입니다. 1, 2, 또는 3을 입력하세요.")
    
    # 테스트 모드 설정
    test_mode = config.test_mode
    user_input = input(f"\n테스트 모드로 실행하시겠습니까? (현재: {'예' if test_mode else '아니오'}) (y/n): ").strip().lower()
    if user_input == 'y':
        test_mode = True
        print("✅ 테스트 모드로 설정했습니다.")
    elif user_input == 'n':
        test_mode = False
        print("✅ 일반 모드로 설정했습니다.")
    else:
        print(f"⚠️ 잘못된 입력입니다. 기본값 ({'테스트 모드' if test_mode else '일반 모드'})을 사용합니다.")
    
    # 설정 업데이트
    config.test_mode = test_mode
    config.artifact_method = artifact_method
    
    # 전처리 실행 (업데이트된 설정으로 새로운 preprocessor 생성)
    preprocessor = EEGPreprocessor(
        config=config,
        test_mode=test_mode,
        artifact_method=artifact_method
    )
    preprocessor.run_processing()


if __name__ == "__main__":
    main() 