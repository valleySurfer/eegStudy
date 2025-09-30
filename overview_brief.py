import mne
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from mne.preprocessing import ICA
from mne.decoding import CSP
from mne.time_frequency import tfr_morlet

# 1. 데이터 로드 및 기본 설정
sfreq = 250
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'EOG']
ch_types = ['eeg'] * 19 + ['eog']
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

n_samples = 20000
n_channels = len(ch_names)
np.random.seed(42)
raw_data = np.random.randn(n_channels, n_samples)
raw = mne.io.RawArray(raw_data, info)

# 2. 이벤트 생성 및 Epoch 추출
event_id = {'stim1': 1, 'stim2': 2}
tmin, tmax = -0.2, 0.8
n_events = 50
events = np.array([[i * 300 + 250, 0, np.random.randint(1, 3)] for i in range(n_events)])
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, preload=True, baseline=(None, 0), proj=False)

# 3. ICA를 이용한 눈깜빡임 아티팩트 제거
ica = ICA(n_components=15, max_iter='auto', random_state=97)
ica.fit(epochs)
eog_indices, eog_scores = ica.find_bads_eog(epochs, ch_name='EOG')
ica.exclude = eog_indices
ica.apply(epochs)

# 4. 웨이블릿 변환을 통한 밴드 분리 및 추출
freqs = np.logspace(*np.log10([4, 100]), num=20)
n_cycles = freqs / 2.
# [수정 사항] average=False 파라미터를 추가하여 각 에포크의 데이터를 보존
power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                   return_itc=False, decim=3, n_jobs=1, average=False)

def extract_band_power(power, fmin, fmax):
    # power.freqs에서 fmin과 fmax 사이의 주파수 인덱스를 찾음
    band_indices = np.where((power.freqs >= fmin) & (power.freqs <= fmax))[0]
    # power.data는 이제 (에포크, 채널, 주파수, 시간) 4차원이므로 기존 코드 사용 가능
    # 주파수 축(axis=2)을 기준으로 평균을 내어 (에포크, 채널, 시간) 3차원 배열 생성
    band_power = power.data[:, :, band_indices, :].mean(axis=2)
    return band_power

delta_power = extract_band_power(power, 1, 4)
theta_power = extract_band_power(power, 4, 8)
alpha_power = extract_band_power(power, 8, 13)
beta_power = extract_band_power(power, 13, 30)
gamma_power = extract_band_power(power, 30, 100)

# 5. CSP 특징 추출
X_bands = {
    'delta': delta_power,
    'theta': theta_power,
    'alpha': alpha_power,
    'beta': beta_power,
    'gamma': gamma_power
}
y = epochs.events[:, -1]

csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
X_csp_features = []

for band in X_bands:
    # 입력 데이터(X_bands[band])는 (에포크, 채널, 시간) 3차원으로 CSP에 적합
    csp.fit(X_bands[band], y)
    features = csp.transform(X_bands[band])
    X_csp_features.append(features)

X_combined = np.concatenate(X_csp_features, axis=1)

# 6. LDA 머신러닝 및 교차 검증
lda = LinearDiscriminantAnalysis()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(lda, X_combined, y, cv=cv, n_jobs=1)

print("LDA Classification Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
