import numpy as np
import pandas as pd
from pydub import AudioSegment
from pydub.generators import WhiteNoise, Sine, Square
import streamlit as st
import fitdecode
import io
import warnings

# fitdecodeの読み込み時に発生する型サイズの警告を無視する
warnings.filterwarnings('ignore', category=UserWarning, module='fitdecode')

class FitDataToMusicMapper:
    """
    FITデータを音楽パラメータに変換するマッピングクラス
    """
    def __init__(self, df: pd.DataFrame, bpm: int = 180):
        self.df = df
        self.bpm = bpm
        # 180 BPMの場合、1拍(四分音符)は 60 / 180 = 0.333秒 = 333ms
        self.beat_duration_ms = int((60 / self.bpm) * 1000)
        
        # データの正規化などの前処理
        self._normalize_data()

    def _normalize_data(self):
        """各種FITデータの正規化"""
        # 心拍数 (例: 80〜200の範囲を0.0〜1.0に正規化)
        hr_min, hr_max = 80, 200 
        if 'heart_rate' in self.df.columns:
            self.df['hr_norm'] = ((self.df['heart_rate'] - hr_min) / (hr_max - hr_min)).clip(0, 1)
        else:
            self.df['hr_norm'] = 0.5
            
        # 上下動 (vertical_oscillation) (動的に最小・最大を取得して正規化)
        if 'vertical_oscillation' in self.df.columns:
            vo = self.df['vertical_oscillation']
            vo_min = vo.min()
            vo_max = vo.max()
            if vo_max > vo_min:
                self.df['vo_norm'] = ((vo - vo_min) / (vo_max - vo_min)).clip(0, 1)
            else:
                self.df['vo_norm'] = 0.5
        else:
            self.df['vo_norm'] = 0.5
        
        # 標高 (全データの獲得標高と下降標高の平均をもとめた平均標高を求める)
        if 'elevation' in self.df.columns:
            diff = self.df['elevation'].diff().fillna(0)
            total_gain = diff[diff > 0].sum()
            total_loss = abs(diff[diff < 0].sum())
            self.avg_elevation_from_gain_loss = (total_gain + total_loss) / 2
            
            # 斜度 (上りか下りか) を差分として保存
            self.df['grade'] = diff
        else:
            self.avg_elevation_from_gain_loss = 0.0
            self.df['grade'] = 0.0

        # 速度
        if 'speed' in self.df.columns:
            speed_max = 10.0
            self.df['speed_norm'] = (self.df['speed'] / speed_max).clip(0, 1)
        else:
            self.df['speed_norm'] = 0.5
            
        # 緯度 (position_lat)
        if 'position_lat' in self.df.columns:
            # 緯度は-90から90の範囲。日本の緯度を考慮し、30-45度あたりを0-1に正規化
            lat_min, lat_max = 30, 45 
            self.df['lat_norm'] = ((self.df['position_lat'] - lat_min) / (lat_max - lat_min)).clip(0, 1)
        else:
            self.df['lat_norm'] = 0.5

    def get_music_params_at_second(self, sec: int):
        """指定秒数の音楽制御パラメータを取得"""
        if sec >= len(self.df):
            return None
        row = self.df.iloc[sec]
        
        synth_intensity = row.get('hr_norm', 0.5)
        vo_norm = row.get('vo_norm', 0.5)
        
        cadence = row.get('cadence', 0)
        hihat_on = 175 <= cadence <= 185
        
        fx_volume = row.get('speed_norm', 0.5)
        
        # 追加された音楽パラメータ (サイドチェイン強度)
        sidechain_intensity = row.get('sidechain_intensity', 0.0)
        
        # 緯度のパラメータを追加
        lat_norm = row.get('lat_norm', 0.5)
        
        # 斜度の判定 (0以上なら上昇/平坦、未満なら下降)
        is_uphill = row.get('grade', 0) >= 0
        
        return {
            'synth_intensity': synth_intensity,
            'is_uphill': is_uphill,
            'hihat_on': hihat_on,
            'fx_volume': fx_volume,
            'sidechain_intensity': sidechain_intensity,
            'lat_norm': lat_norm,
            'vo_norm': vo_norm
        }

def generate_heavy_kick(duration=0.25, sample_rate=44100):
    """
    プログレッシブハウスに適した、重くてパンチのあるヘビー・キックを生成する
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 1. Pitch Envelope (ピッチ・エンベロープ)
    # 開始周波数150Hz付近から40-50Hzへ指数関数的に急降下
    f_start = 150.0
    f_end = 45.0
    pitch_decay = 40.0
    f_env = (f_start - f_end) * np.exp(-pitch_decay * t) + f_end
    
    # 周波数を積分して位相を計算
    phase = 2 * np.pi * np.cumsum(f_env) / sample_rate
    
    # 2. Sine Wave Base
    waveform = np.sin(phase)
    
    # 3. Saturation / Clipping (歪み)
    # tanh関数でソフトクリッピングし倍音を付加
    drive = 3.0
    waveform = np.tanh(waveform * drive)
    
    # 4. Decay (減衰)
    # 滑らかに減衰するよう指数関数をかける
    amp_decay = 15.0
    envelope = np.exp(-amp_decay * t)
    waveform = waveform * envelope
    
    # int16 形式の ndarray に変換 (-32768 ~ 32767)
    waveform_int16 = np.int16(waveform * 32767)
    
    # pydub の AudioSegment オブジェクトに変換して返す
    kick = AudioSegment(
        waveform_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    return kick

class ProgressiveHouseGenerator:
    """
    パラメータに基づき、pydubとnumpyを用いてプログレッシブハウスのトラックを生成するクラス
    """
    def __init__(self, mapper: FitDataToMusicMapper):
        self.mapper = mapper
        self.sample_rate = 44100
        self.beat_ms = self.mapper.beat_duration_ms
        # 8分音符の長さ (四分音符の半分)
        self.eighth_note_ms = self.beat_ms // 2
        
    def _generate_kick(self, duration_ms):
        kick_sound = generate_heavy_kick(duration=0.25, sample_rate=self.sample_rate)
        
        kick_len = len(kick_sound)
        if kick_len < self.beat_ms:
            silence = AudioSegment.silent(duration=self.beat_ms - kick_len)
            one_beat = kick_sound + silence
        else:
            one_beat = kick_sound[:self.beat_ms]
            
        beats_needed = int(np.ceil(duration_ms / self.beat_ms))
        track = one_beat * beats_needed
        return track[:duration_ms]
        
    def _generate_hihat(self, duration_ms, active: bool):
        if not active:
            return AudioSegment.silent(duration=duration_ms)
        
        hh_sound = WhiteNoise().to_audio_segment(duration=50).high_pass_filter(5000).apply_gain(-10)
        half_beat = self.beat_ms // 2
        silence_before = AudioSegment.silent(duration=half_beat)
        silence_after = AudioSegment.silent(duration=self.beat_ms - half_beat - 50)
        one_beat = silence_before + hh_sound + silence_after
        
        beats_needed = int(np.ceil(duration_ms / self.beat_ms))
        track = one_beat * beats_needed
        return track[:duration_ms]
        
    def _generate_uphill_sine_track(self, total_duration_ms, avg_elev):
        """
        平均標高に基づき、上昇時の(8分休符 + 8分音符) x 4 のリズムを持つSine波トラックを生成する
        """
        normalized_elev = min(max(avg_elev, 0), 1000) / 1000.0
        freq = 261.63 * (2 ** (normalized_elev * 2))
        
        # 8分音符 (音を鳴らす) と 8分休符 (無音)
        note_duration = self.eighth_note_ms
        silence_duration = self.eighth_note_ms
        
        # 正弦波を生成 (音が大きすぎないようにゲインを調整)
        tone = Sine(freq).to_audio_segment(duration=note_duration).apply_gain(-15)
        
        # 音が滑らかに繋がるように短いフェードを追加
        fade_time = min(20, note_duration // 4)
        tone = tone.fade_in(fade_time).fade_out(fade_time)
        
        silence = AudioSegment.silent(duration=silence_duration)
        
        # (8分休符 + 8分音符) の1ペア (これで四分音符1個分 = 1拍)
        one_pair = silence + tone
        
        # トラック全体をカバーする長さまで繰り返す
        beats_needed = int(np.ceil(total_duration_ms / self.beat_ms))
        track = (one_pair * beats_needed)[:total_duration_ms]
        
        return track

    def _generate_downhill_sine_track(self, total_duration_ms, avg_elev):
        """
        平均標高に基づき、下降時の(4分休符 + 1拍3連符) のリズムを持つSine波トラックを生成する
        """
        normalized_elev = min(max(avg_elev, 0), 1000) / 1000.0
        freq = 261.63 * (2 ** (normalized_elev * 2))
        
        # 4分休符 (1拍)
        quarter_rest = AudioSegment.silent(duration=self.beat_ms)
        
        # 1拍3連符 (1拍を3等分)
        triplet_note_duration = self.beat_ms // 3
        # 少しスタッカートにして音を区切る
        tone_duration = triplet_note_duration - 10
        silence_duration = 10
        
        tone = Sine(freq).to_audio_segment(duration=tone_duration).apply_gain(-15)
        fade_time = min(10, tone_duration // 4)
        tone = tone.fade_in(fade_time).fade_out(fade_time)
        silence = AudioSegment.silent(duration=silence_duration)
        
        one_triplet_note = tone + silence
        triplet_beat = one_triplet_note * 3
        
        # 念のため1拍分の長さに厳密に合わせる
        if len(triplet_beat) < self.beat_ms:
            triplet_beat += AudioSegment.silent(duration=self.beat_ms - len(triplet_beat))
        triplet_beat = triplet_beat[:self.beat_ms]
        
        # (4分休符 + 1拍3連符) = 2拍分のパターン
        two_beat_pattern = quarter_rest + triplet_beat
        
        # トラック全体をカバーする長さまで繰り返す
        patterns_needed = int(np.ceil(total_duration_ms / (self.beat_ms * 2)))
        track = (two_beat_pattern * patterns_needed)[:total_duration_ms]
        
        return track
        
    @staticmethod
    def _generate_fx_noise(duration_ms, intensity):
        if intensity <= 0.05:
            return AudioSegment.silent(duration=duration_ms)
            
        noise = WhiteNoise().to_audio_segment(duration=duration_ms)
        target_db = -30 + (intensity * 20)
        noise = noise.apply_gain(target_db - noise.dBFS)
        return noise
        
    def _generate_synth_arp(self, duration_ms, intensity, vo_norm):
        num_notes = 2 if intensity < 0.5 else 4
        note_duration = self.beat_ms // num_notes
        
        # マイナーペンタトニックスケール (A Minor Pentatonic: A, C, D, E, G)
        # 複数オクターブ分の周波数を定義
        pentatonic_notes = [
            220.00, 261.63, 293.66, 329.63, 392.00, # A3, C4, D4, E4, G4
            440.00, 523.25, 587.33, 659.25, 783.99, # A4, C5, D5, E5, G5
            880.00, 1046.50, 1174.66, 1318.51, 1567.98 # A5, C6, D6, E6, G6
        ]
        
        # vo_norm (0.0 - 1.0) に応じて使用する音階の範囲（開始インデックス）を決定
        max_idx = len(pentatonic_notes) - 4
        start_idx = int(vo_norm * max_idx)
        start_idx = max(0, min(max_idx, start_idx))
        
        freqs = pentatonic_notes[start_idx : start_idx + 4]
        
        track = AudioSegment.silent(duration=0)
        
        loops = int(np.ceil(duration_ms / note_duration))
        for i in range(loops):
            f = freqs[i % len(freqs)]
            tone_len = max(5, note_duration - 10)
            tone = Sine(f).to_audio_segment(duration=tone_len).apply_gain(-10)
            silence = AudioSegment.silent(duration=note_duration - tone_len)
            track += (tone + silence)
            
        high_boost = intensity * 8
        track = track.apply_gain(high_boost)
        
        return track[:duration_ms]
        
    def _generate_lat_square(self, duration_ms, lat_norm):
        """
        緯度に応じて音程（周波数）が変わる矩形波（Square）を4拍子（4つ打ち）で生成する
        lat_norm (0.0 - 1.0) を基に、ベース音域の周波数（例: C2=65.41Hz 〜 C4=261.63Hz）にマッピング
        """
        min_freq = 65.41  # C2
        max_freq = 261.63 # C4
        freq = min_freq + (max_freq - min_freq) * lat_norm
        
        # 1拍あたりの音の長さ（スタッカート気味に短くする）
        note_duration = min(150, self.beat_ms - 20) 
        
        # 矩形波を生成し、少し音量を下げる（矩形波は目立つため）
        tone = Square(freq).to_audio_segment(duration=note_duration).apply_gain(-15)
        
        # 1拍分の長さに調整 (音 + 無音)
        silence_duration = self.beat_ms - note_duration
        if silence_duration > 0:
            silence = AudioSegment.silent(duration=silence_duration)
            one_beat = tone + silence
        else:
            one_beat = tone[:self.beat_ms]
            
        # 必要な拍数分繰り返す (4拍子で鳴らすため1拍ごとに鳴る)
        beats_needed = int(np.ceil(duration_ms / self.beat_ms))
        track = one_beat * beats_needed
        
        return track[:duration_ms]
        
    def _apply_sidechain(self, track: AudioSegment, intensity: float):
        """
        キックに合わせてボリュームをダッキングするサイドチェインエフェクト
        intensity (0.0 - 1.0) に応じてダッキングの深さを調整する
        """
        if intensity <= 0.0:
            return track
            
        # 最大20dB下げる
        duck_db = - (intensity * 20.0)
        duck_duration = 100  # キックが鳴る最初の100msをダッキング
        
        result = AudioSegment.silent(duration=0)
        # トラックを1拍ごとに分割して処理
        for i in range(0, len(track), self.beat_ms):
            beat = track[i:i + self.beat_ms]
            if len(beat) > duck_duration:
                # 前半部分をダッキング
                ducked = beat[:duck_duration].apply_gain(duck_db)
                rest = beat[duck_duration:]
                # フェードを入れてノイズを防ぐ
                ducked = ducked.fade_out(10)
                rest = rest.fade_in(10)
                result += (ducked + rest)
            else:
                result += beat.apply_gain(duck_db)
        return result

    def generate_track(self, progress_callback=None):
        """全データに基づき楽曲全体を生成"""
        total_seconds = len(self.mapper.df)
        final_mix = AudioSegment.silent(duration=total_seconds * 1000)
        
        # 上昇時・下降時それぞれの正弦波(Sine)トラックを全曲分あらかじめ生成
        # これにより、秒の切り替わりでリズムが崩れる（1秒＝3拍で中途半端に切れる）のを防ぎます
        avg_elev = self.mapper.avg_elevation_from_gain_loss
        sine_uphill_full = self._generate_uphill_sine_track(total_seconds * 1000, avg_elev)
        sine_downhill_full = self._generate_downhill_sine_track(total_seconds * 1000, avg_elev)
        
        for sec in range(total_seconds):
            params = self.mapper.get_music_params_at_second(sec)
            
            kick = self._generate_kick(1000)
            hihat = self._generate_hihat(1000, params['hihat_on'])
            fx = self._generate_fx_noise(1000, params['fx_volume'])
            synth = self._generate_synth_arp(1000, params['synth_intensity'], params['vo_norm'])
            lat_square = self._generate_lat_square(1000, params['lat_norm'])
            
            # その1秒間に対応する正弦波(Sine)トラックの部分を上昇・下降で切り替える
            if params['is_uphill']:
                sine_1sec = sine_uphill_full[sec * 1000 : (sec + 1) * 1000]
            else:
                sine_1sec = sine_downhill_full[sec * 1000 : (sec + 1) * 1000]
            
            # 各要素にサイドチェインを適用
            sidechain_int = params['sidechain_intensity']
            fx = self._apply_sidechain(fx, sidechain_int)
            synth = self._apply_sidechain(synth, sidechain_int)
            lat_square = self._apply_sidechain(lat_square, sidechain_int)
            sine_1sec = self._apply_sidechain(sine_1sec, sidechain_int)
            
            mix_1sec = kick.overlay(hihat).overlay(fx).overlay(synth).overlay(lat_square).overlay(sine_1sec)
            
            final_mix = final_mix.overlay(mix_1sec, position=sec * 1000)
            
            if progress_callback:
                progress_callback(sec + 1, total_seconds)
            
        return final_mix

def adjust_heart_rate_anomalies(df, threshold_bpm=10):
    """
    心拍数の欠落後に異常な値(小さすぎる値)が記録され、その後正常な値に戻る現象を補正する。
    また、値が欠損している区間（NaN）も、直前の値 (b) と復帰後の正常な値 (c) の間で補間する。
    """
    if 'heart_rate' not in df.columns:
        return df

    # NaNかどうかを判定するためのブール配列
    is_nan = df['heart_rate'].isna()

    # 欠損の開始位置と終了位置を見つけるためのグループ化
    # 欠損状態が変わるごとにgroupのIDが増加する
    nan_groups = (is_nan != is_nan.shift()).cumsum()

    # 補正済みのデータを格納するシリーズ
    corrected_hr = df['heart_rate'].copy()

    # 欠損しているグループのIDを取得
    missing_group_ids = nan_groups[is_nan].unique()

    for gid in missing_group_ids:
        missing_indices = df.index[nan_groups == gid]
        if len(missing_indices) == 0:
            continue

        first_missing_idx = missing_indices[0]
        last_missing_idx = missing_indices[-1]

        # 欠損の直前のインデックスを探す (b)
        loc_first_missing = df.index.get_loc(first_missing_idx)
        if loc_first_missing == 0:
            continue # 先頭が欠損の場合は無視

        b_idx = df.index[loc_first_missing - 1]
        b_val = df.at[b_idx, 'heart_rate']

        if pd.isna(b_val):
            continue # 直前も何らかの理由でNaNなら無視

        # 欠損の直後のインデックスを探す (a)
        loc_last_missing = df.index.get_loc(last_missing_idx)
        if loc_last_missing >= len(df) - 1:
            continue # 末尾が欠損の場合は無視

        a_idx = df.index[loc_last_missing + 1]
        a_val = df.at[a_idx, 'heart_rate']

        if pd.isna(a_val):
            continue

        # (a) が (b) より明らかに小さいかチェック
        if b_val - a_val >= threshold_bpm:
            # 異常な急降下があった場合
            # (c) を探す: a_idx より後で、値が b_val ± 2 になる最初のポイント
            c_idx = None
            c_val = None

            # a_idx の次から探索
            search_start_loc = loc_last_missing + 1

            for i in range(search_start_loc + 1, min(search_start_loc + 60, len(df))): # 最大60ポイント(約60秒)先まで探索
                current_idx = df.index[i]
                current_val = df.at[current_idx, 'heart_rate']

                if pd.isna(current_val):
                    continue

                if abs(current_val - b_val) <= 2:
                    c_idx = current_idx
                    c_val = current_val
                    break

            if c_idx is not None:
                # (b) の直後から (c) の直前までの補間を行う（NaNの部分も異常な急降下部分も含める）
                b_loc = df.index.get_loc(b_idx)
                c_loc = df.index.get_loc(c_idx)

                # 補間対象の要素数（両端 b, c を含まない間の要素数）
                num_points_to_interpolate = c_loc - b_loc - 1

                if num_points_to_interpolate > 0:
                    step = (c_val - b_val) / (num_points_to_interpolate + 1)
                    interpolated_values = [round(b_val + step * (i + 1)) for i in range(num_points_to_interpolate)]

                    # b_idx の次 から c_idx の直前までを置き換え
                    for i, val in enumerate(interpolated_values):
                        idx_to_replace = df.index[b_loc + 1 + i]
                        corrected_hr.at[idx_to_replace] = val
        else:
            # 異常な急降下がなかった場合でも、NaNの区間だけを (b) と (a) の間で補間する
            b_loc = df.index.get_loc(b_idx)
            a_loc = df.index.get_loc(a_idx)

            num_points_to_interpolate = a_loc - b_loc - 1
            if num_points_to_interpolate > 0:
                step = (a_val - b_val) / (num_points_to_interpolate + 1)
                interpolated_values = [round(b_val + step * (i + 1)) for i in range(num_points_to_interpolate)]

                for i, val in enumerate(interpolated_values):
                    idx_to_replace = df.index[b_loc + 1 + i]
                    corrected_hr.at[idx_to_replace] = val

    df['heart_rate'] = corrected_hr
    return df

def speed_to_pace_str(speed_ms):
    """m/s (秒速) を 1kmあたりのペース (M:S/km) に変換"""
    if pd.isna(speed_ms) or speed_ms is None or speed_ms <= 0.1:
        return None
    sec = 1000 / speed_ms
    minutes = int(sec // 60)
    seconds = int(sec % 60)
    return f"{minutes}:{seconds:02d}"

def semicircles_to_degrees(semicircles):
    """GarminのFITデータ特有のSemicircles単位を、一般的な緯度経度（Degrees）に変換する"""
    if pd.isna(semicircles) or semicircles is None:
        return None
    # 変換式: degrees = semicircles * ( 180 / 2^31 )
    return semicircles * (180.0 / (2**31))

def parse_fit_data(file_bytes) -> pd.DataFrame:
    """FITファイルをパースしてPandas DataFrameに変換し、_samlpe.py と同じ前処理を行う"""
    data = []
    
    with fitdecode.FitReader(file_bytes) as fit_file:
        for frame in fit_file:
            if isinstance(frame, fitdecode.records.FitDataMessage) and frame.name == 'record':
                record_data = {}
                for field in frame.fields:
                    record_data[field.name] = field.value
                data.append(record_data)

    if not data:
        # 空のDataFrameを返す
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # 英語の列名の中に空白が入っている場合、アンダースコアに変換し小文字に統一
    df.columns = [str(col).strip().replace(' ', '_').lower() for col in df.columns]

    # 不要な従来の 'altitude' と 'speed' を削除
    columns_to_drop = [col for col in ['altitude', 'speed'] if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    # enhanced_altitude を altitude (elevation用) にリネーム
    if 'enhanced_altitude' in df.columns:
        df = df.rename(columns={'enhanced_altitude': 'elevation'})
        
    # もし altitude が残っていて elevation がない場合
    if 'altitude' in df.columns and 'elevation' not in df.columns:
         df = df.rename(columns={'altitude': 'elevation'})

    # enhanced_speed を speed (m/sのまま、後でpace_strにも変換可能だが今回は数値として保持する)
    # 今回は音楽生成用に 'speed' というカラム名で数値を残す必要がある
    if 'enhanced_speed' in df.columns:
        df['speed'] = df['enhanced_speed']
        # 文字列のペースが欲しい場合は別カラムにする
        df['pace'] = df['enhanced_speed'].apply(speed_to_pace_str)
        df = df.drop(columns=['enhanced_speed'])

    # effort_pace を文字列 (M:S/km) に変換
    if 'effort_pace' in df.columns:
        df['effort_pace'] = df['effort_pace'].apply(speed_to_pace_str)

    # 位置情報(Semicircles)を緯度経度(Degrees)に変換
    if 'position_lat' in df.columns:
        df['position_lat'] = df['position_lat'].apply(semicircles_to_degrees)
    if 'position_long' in df.columns:
        df['position_long'] = df['position_long'].apply(semicircles_to_degrees)

    # ストライド(step_length)を mm から cm に変換
    if 'step_length' in df.columns:
        df['step_length'] = df['step_length'] / 10.0

    # ケイデンスを両足の歩数(spm)にするため2倍に変換
    if 'cadence' in df.columns:
        if 'fractional_cadence' in df.columns:
            df['cadence'] = (df['cadence'] + df['fractional_cadence']) * 2
        else:
            df['cadence'] = df['cadence'] * 2

    # timestampの処理とindexセット
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # FITファイルのタイムスタンプは通常UTCなので、JST(日本標準時)に変換する
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Tokyo')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Tokyo')

        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        df = df.set_index('timestamp')

    # 心拍数の異常値補正処理の追加
    df = adjust_heart_rate_anomalies(df, threshold_bpm=15)
    
    # 音楽生成ロジック(FitDataToMusicMapper)は英語の列名を期待しているため、
    # _sample.py のような日本語へのリネームはここでは行わず、元の英名カラムを維持します。
    
    # データマッピングと音楽的エフェクト用パラメータの生成
    df = apply_musical_fx_params(df)
    
    # 必要なカラムの欠損値を補完 (pandas 2.0+ 互換の ffill 使用)
    if 'heart_rate' in df.columns:
        df['heart_rate'] = df['heart_rate'].ffill().fillna(120)
    if 'elevation' in df.columns:
        df['elevation'] = df['elevation'].ffill().fillna(0)
    if 'position_lat' in df.columns:
        df['position_lat'] = df['position_lat'].ffill().fillna(35) # 日本の平均的な緯度で補完

    if 'cadence' in df.columns:
        df['cadence'] = df['cadence'].ffill().fillna(0)
    if 'speed' in df.columns:
        df['speed'] = df['speed'].ffill().fillna(0)

    return df

def apply_musical_fx_params(df: pd.DataFrame) -> pd.DataFrame:
    """
    ランニングフォームの質を音楽の表情に変えるためのパラメータを計算して付与する
    
    1. vertical_oscillation (上下動: mm単位)
    2. stance_time (接地時間: ms単位)
    3. running_power (パワー: W単位) -> サイドチェイン強度(sidechain_intensity)
    """
    # 各項目のユーザー基準値の定数定義
    base_power = 200.0                # 平均的なパワー (W)
    
    # 上下動・接地時間の補完（存在しない場合は0）
    for col in ['vertical_oscillation', 'stance_time']:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].ffill().fillna(0.0)
            
    # パワーの補完 (FITデータでは 'power' カラムに入ることが多い)
    if 'power' in df.columns and 'running_power' not in df.columns:
        df['running_power'] = df['power']
        
    if 'running_power' not in df.columns:
        df['running_power'] = 0.0
    else:
        df['running_power'] = df['running_power'].ffill().fillna(0.0)
        
    # パワー ➔ サイドチェイン強度の計算 (0.0 〜 1.0)
    # パワーが高いほど、テクノ特有のサイドチェイン効果（ボリュームダッキング）を深くする
    # 例として基準値(200W)の半分から1.5倍の範囲を0.0〜1.0にマッピングする
    min_power = base_power * 0.5
    max_power = base_power * 1.5
    
    if max_power > min_power:
        df['sidechain_intensity'] = ((df['running_power'] - min_power) / (max_power - min_power)).clip(0.0, 1.0)
    else:
        df['sidechain_intensity'] = 0.0
        
    return df

def resample_dataframe(df: pd.DataFrame, target_seconds: int) -> pd.DataFrame:
    """
    データフレーム全体を選択された秒数(target_seconds)にリサンプル（圧縮）する
    """
    total_len = len(df)
    if total_len <= target_seconds:
        # データの長さが指定秒数以下の場合はそのまま返す
        return df

    # グループ数を target_seconds に設定
    # index をもとにグループ化: group_id = index * target_seconds // total_len
    # 各グループの平均値をとることで圧縮する
    
    df_reset = df.reset_index(drop=True)
    # FITデータには数値以外のカラム（日時、文字列など）が含まれる可能性があるため
    # numeric_only=True を指定して平均値を計算します
    group_ids = (df_reset.index * target_seconds) // total_len
    resampled_df = df_reset.groupby(group_ids).mean(numeric_only=True)
    
    return resampled_df

def main():
    st.set_page_config(page_title="FIT to Music Generator", layout="centered")
    
    st.title("🏃 FIT to Music Generator 🎵")
    st.markdown("""
    Garmin等の **FITデータ** をアップロードして、あなたのランニングデータを
    **180 BPM の爽快なプログレッシブハウス** に変換します！
    
    * **心拍数** ➔ シンセの激しさ
    * **上下動 (Vertical Oscillation)** ➔ シンセサイザーの音階 (マイナーペンタトニックスケール)
    * **標高/斜度** ➔ 全データの平均標高をベースにした正弦波（Sine）の音程
      * 上昇時・平坦: 8分休符・8分音符の繰り返し
      * 下降時: 4分休符・1拍3連符の繰り返し
    * **緯度** ➔ 矩形波（スクエアベース）の音程
    * **ケイデンス** ➔ 180spm付近で裏打ちハイハット
    * **速度** ➔ 疾走感のあるライザー音 (ホワイトノイズ)
    * **パワー** ➔ キックに合わせたサイドチェイン（ダッキング）効果の深さ
    """)
    
    uploaded_file = st.file_uploader("FITファイルをアップロードしてください", type=["fit"])
    
    if uploaded_file is not None:
        st.success("ファイルを読み込みました！")
        
        try:
            with st.spinner("FITデータを解析中..."):
                file_bytes = io.BytesIO(uploaded_file.getvalue())
                df = parse_fit_data(file_bytes)
                
            st.write("### 解析されたデータ (一部)")
            st.dataframe(df.head())
            total_data_seconds = len(df)
            st.write(f"総データ秒数: {total_data_seconds} 秒")
            
            # 生成する長さを選択。デフォルトは60秒。
            max_seconds = st.slider(
                "生成する音楽の長さ(秒)を選択してください。全データをこの長さに圧縮(平均化)します。", 
                min_value=10, 
                max_value=min(600, total_data_seconds), 
                value=min(60, total_data_seconds), 
                step=10
            )
            
            if st.button("🎵 音楽を生成する"):
                # 全データを指定した秒数にリサンプル（平均化）する
                df_resampled = resample_dataframe(df, max_seconds)
                
                # リサンプル後のデータフレームをマッパーに渡す
                mapper = FitDataToMusicMapper(df_resampled, bpm=180)
                generator = ProgressiveHouseGenerator(mapper)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total):
                    progress = int((current / total) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"楽曲生成中... {current}/{total} 秒完了")

                track = generator.generate_track(progress_callback=update_progress)
                
                status_text.text("生成完了！書き出しています...")
                
                output_filename = "generated_track.wav"
                track.export(output_filename, format="wav")
                
                status_text.text("完成しました！")
                
                # Streamlitのaudioプレイヤーで再生
                st.audio(output_filename, format="audio/wav")
                
                # ダウンロードボタン
                with open(output_filename, "rb") as file:
                    st.download_button(
                        label="⬇️ WAVファイルをダウンロード",
                        data=file,
                        file_name="progressive_house_run.wav",
                        mime="audio/wav"
                    )
                    
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()