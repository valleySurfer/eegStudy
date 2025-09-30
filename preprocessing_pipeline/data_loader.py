#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
데이터 로딩 및 파일 처리 모듈
"""

import os
import glob
import pandas as pd
from pathlib import Path


def load_csv(file_path):
    """Load CSV file and return as pandas DataFrame."""
    try:
        # marker 열을 정수형으로 읽기
        return pd.read_csv(file_path, dtype={'marker': 'Int64'})
    except Exception as e:
        print(f"❌ Error loading file {file_path}: {str(e)}")
        return None


def find_subject_file(file_dir, subject):
    """Find recording file for a specific subject."""
    file_pattern = os.path.join(file_dir, f'*_{subject}.csv')
    return next(glob.iglob(file_pattern), None)


def get_all_subject_files(data_dir):
    """Get all subject files from data directory."""
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob('*_*.csv'))
    
    subjects = []
    for file in csv_files:
        # 파일명에서 참가자 번호 추출 (예: muse_recording_20250611_124905_6.csv -> 6)
        subject_num = file.stem.split('_')[-1]
        if subject_num.isdigit():
            subjects.append(subject_num)
    
    return sorted(subjects)


def validate_data_file(data_file):
    """Validate if data file exists and is readable."""
    if data_file is None:
        return False, "데이터 파일을 찾을 수 없습니다."
    
    if not os.path.exists(data_file):
        return False, f"파일이 존재하지 않습니다: {data_file}"
    
    try:
        data = load_csv(data_file)
        if data is None:
            return False, "CSV 파일을 읽을 수 없습니다."
        return True, "성공"
    except Exception as e:
        return False, f"파일 검증 중 오류 발생: {str(e)}" 