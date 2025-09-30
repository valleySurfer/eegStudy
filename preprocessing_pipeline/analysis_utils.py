#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
분석 및 유틸리티 함수 모듈
"""

from pathlib import Path
from data_loader import find_subject_file, load_csv
from marker_parser import parse_marker, analyze_markers


def analyze_event_counts(data_dir):
    """참가자별 이벤트 개수 분석"""
    print("\n=== 참가자별 이벤트 개수 분석 ===")
    
    subjects = ['3', '4', '5', '6', '9', '10']
    event_counts = {}
    
    for subject in subjects:
        data_file = find_subject_file(data_dir, subject)
        if data_file is None:
            print(f"❌ 참가자 {subject}의 데이터 파일을 찾을 수 없습니다.")
            continue
            
        try:
            data = load_csv(data_file)
            if data is None:
                continue
                
            # 마커 분석
            marker_stats = analyze_markers(data, subject=int(subject))
            event_counts[subject] = marker_stats
            
            print(f"\n참가자 {subject}:")
            print(f"  총 마커 수: {marker_stats['total_markers']}")
            print(f"  유효한 마커 수: {marker_stats['valid_markers']}")
            print(f"  video_start 마커 수: {marker_stats['video_start_markers']}")
            print(f"  마커 타입별 분포:")
            for marker_type, count in marker_stats['marker_types'].items():
                print(f"    {marker_type}: {count}")
                
        except Exception as e:
            print(f"❌ 참가자 {subject} 분석 중 오류 발생: {str(e)}")
            continue
    
    return event_counts


def create_processing_summary(subjects, data_dir, output_dir, artifact_method=None):
    """전체 처리 과정 요약 생성"""
    print("\n" + "="*60)
    print("전체 처리 과정 요약")
    print("="*60)
    
    summary = {
        'total_subjects': len(subjects),
        'processed_subjects': [],
        'failed_subjects': [],
        'total_epochs': {'main': 0, 'resting': 0}
    }
    
    for subject in subjects:
        # 에포크 파일 확인 (아티팩트 제거 방법 접미사 포함)
        epochs_dir = Path(output_dir) / 'epochs'
        
        # 아티팩트 제거 방법에 따른 파일명 패턴 생성
        if artifact_method:
            main_file = epochs_dir / f'subject_{subject}_main_epo_{artifact_method}.fif'
            resting_file = epochs_dir / f'subject_{subject}_resting_epo_{artifact_method}.fif'
        else:
            # 아티팩트 제거 방법이 지정되지 않은 경우, 모든 가능한 패턴 확인
            main_pattern = f'subject_{subject}_main_epo*.fif'
            resting_pattern = f'subject_{subject}_resting_epo*.fif'
            
            main_files = list(epochs_dir.glob(main_pattern))
            resting_files = list(epochs_dir.glob(resting_pattern))
            
            main_file = main_files[0] if main_files else None
            resting_file = resting_files[0] if resting_files else None
        
        if (main_file and main_file.exists()) or (resting_file and resting_file.exists()):
            summary['processed_subjects'].append(subject)
            
            # 에포크 개수 확인
            if main_file and main_file.exists():
                import mne
                main_epochs = mne.read_epochs(main_file)
                summary['total_epochs']['main'] += len(main_epochs)
            
            if resting_file and resting_file.exists():
                import mne
                resting_epochs = mne.read_epochs(resting_file)
                summary['total_epochs']['resting'] += len(resting_epochs)
        else:
            summary['failed_subjects'].append(subject)
    
    # 요약 출력
    print(f"총 참가자 수: {summary['total_subjects']}")
    print(f"성공적으로 처리된 참가자: {len(summary['processed_subjects'])}")
    print(f"처리 실패한 참가자: {len(summary['failed_subjects'])}")
    
    if summary['processed_subjects']:
        print(f"성공한 참가자: {', '.join(summary['processed_subjects'])}")
    
    if summary['failed_subjects']:
        print(f"실패한 참가자: {', '.join(summary['failed_subjects'])}")
    
    print(f"\n총 생성된 에포크:")
    print(f"  Main 에포크: {summary['total_epochs']['main']}개")
    print(f"  Resting 에포크: {summary['total_epochs']['resting']}개")
    
    return summary


def validate_processing_pipeline(data_dir, output_dir):
    """처리 파이프라인 검증"""
    print("\n=== 처리 파이프라인 검증 ===")
    
    # 1. 데이터 디렉토리 확인
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ 데이터 디렉토리가 존재하지 않습니다: {data_dir}")
        return False
    
    # 2. 출력 디렉토리 확인
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"❌ 출력 디렉토리가 존재하지 않습니다: {output_dir}")
        return False
    
    # 3. 필요한 하위 디렉토리 확인
    raw_dir = output_path / 'raw'
    epochs_dir = output_path / 'epochs'
    
    if not raw_dir.exists():
        print(f"⚠️ Raw 디렉토리가 없습니다: {raw_dir}")
    
    if not epochs_dir.exists():
        print(f"⚠️ Epochs 디렉토리가 없습니다: {epochs_dir}")
    
    # 4. 데이터 파일 확인
    csv_files = list(data_path.glob('*_*.csv'))
    if not csv_files:
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_dir}")
        return False
    
    print(f"✅ 데이터 파일 {len(csv_files)}개 발견")
    print(f"✅ 출력 디렉토리 준비 완료")
    
    return True 