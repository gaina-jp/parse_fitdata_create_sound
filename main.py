import numpy as np
import pandas as pd
from pydub import AudioSegment
from pydub.generators import WhiteNoise, Sine
import streamlit as st
import fitdecode
import io

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
        # 心拍数 (例: 100〜180の範囲を0.0〜1.0に正規化)
        hr_min, hr_max = 100, 180
        if 'heart_rate' in self.df.columns:
            self.df['hr_norm'] = ((self.df['heart_rate'] - hr_min) / (hr_max - hr_min)).clip(0, 1)
        else:
            self.df['hr_norm'] = 0.5
        
        # 標高
        if 'elevation' in self.df.columns:
            elev_min, elev_max = self.df['elevation'].min(), self.df['elevation'].max()
            if elev_max > elev_min:
                self.df['elev_norm'] = (self.df['elevation'] - elev_min) / (elev_max - elev_min)
            else:
                self.df['elev_norm'] = 0.0

            # 斜度 (上りか下りか)
            self.df['grade'] = self.df['elevation'].diff().fillna(0)
        else:
            self.df['elev_norm'] = 0.5
            self.df['grade'] = 0.0

        # 速度
        if 'speed' in self.df.columns:
            speed_max = 10.0
            self.df['speed_norm'] = (self.df['speed'] / speed_max).clip(0, 1)
        else:
            self.df['speed_norm'] = 0.5

    def get_music_params_at_second(self, sec: int):
        """指定秒数の音楽制御パラメータを取得"""
        if sec >= len(self.df):
            return None
        row = self.df.iloc[sec]
        
        synth_intensity = row.get('hr_norm', 0.5)
        pad_volume = row.get('elev_norm', 0.5)
        is_uphill = row.get('grade', 0) > 0
        
        cadence = row.get('cadence', 0)
        hihat_on = 175 <= cadence <= 185
        
        fx_volume = row.get('speed_norm', 0.5)
        
        # 追加された音楽パラメータ (サイドチェイン強度)
        sidechain_intensity = row.get('sidechain_intensity', 0.0)
        
        return {
            'synth_intensity': synth_intensity,
            'pad_volume': pad_volume,
            'is_uphill': is_uphill,
            'hihat_on': hihat_on,
            'fx_volume': fx_volume,
            'sidechain_intensity': sidechain_intensity
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
        
    @staticmethod
    def _generate_pad(duration_ms, volume_norm, is_uphill):
        if is_uphill:
            freqs = [220.0, 261.63, 329.63]
        else:
            freqs = [261.63, 329.63, 392.00]
            
        pad = AudioSegment.silent(duration=duration_ms)
        for f in freqs:
            tone = Sine(f).to_audio_segment(duration=duration_ms)
            pad = pad.overlay(tone)
            
        target_db = -25 + (volume_norm * 20)
        pad = pad.apply_gain(target_db - pad.dBFS)
        return pad
        
    @staticmethod
    def _generate_fx_noise(duration_ms, intensity):
        if intensity <= 0.05:
            return AudioSegment.silent(duration=duration_ms)
            
        noise = WhiteNoise().to_audio_segment(duration=duration_ms)
        target_db = -30 + (intensity * 20)
        noise = noise.apply_gain(target_db - noise.dBFS)
        return noise
        
    def _generate_synth_arp(self, duration_ms, intensity):
        num_notes = 2 if intensity < 0.5 else 4
        note_duration = self.beat_ms // num_notes
        
        freqs = [440.0, 554.37, 659.25, 880.0]
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
        
        for sec in range(total_seconds):
            params = self.mapper.get_music_params_at_second(sec)
            
            kick = self._generate_kick(1000)
            hihat = self._generate_hihat(1000, params['hihat_on'])
            pad = self._generate_pad(1000, params['pad_volume'], params['is_uphill'])
            fx = self._generate_fx_noise(1000, params['fx_volume'])
            synth = self._generate_synth_arp(1000, params['synth_intensity'])
            
            # パッド、FX、シンセにサイドチェインを適用
            sidechain_int = params['sidechain_intensity']
            pad = self._apply_sidechain(pad, sidechain_int)
            fx = self._apply_sidechain(fx, sidechain_int)
            synth = self._apply_sidechain(synth, sidechain_int)
            
            mix_1sec = kick.overlay(hihat).overlay(pad).overlay(fx).overlay(synth)
            
            final_mix = final_mix.overlay(mix_1sec, position=sec * 1000)
            
            if progress_callback:
                progress_callback(sec + 1, total_seconds)
            
        return final_mix

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

def parse_fit_data(file_bytes) -> pd.DataFrame:
    """FITファイルをパースしてPandas DataFrameに変換する"""
    records = []
    
    with fitdecode.FitReader(file_bytes) as fit:
        for frame in fit:
            # データの含まれるフレームのみを取得
            if isinstance(frame, fitdecode.FitDataMessage):
                if frame.name == 'record':
                    data = {}
                    for field in frame.fields:
                        data[field.name] = field.value
                    records.append(data)
        
    df = pd.DataFrame(records)
    
    if len(df) > 0:
        # データマッピングと音楽的エフェクト用パラメータの生成
        df = apply_musical_fx_params(df)
        
        # 必要なカラムの欠損値を補完 (pandas 2.0+ 互換の ffill 使用)
        if 'heart_rate' in df.columns:
            df['heart_rate'] = df['heart_rate'].ffill().fillna(120)
        if 'elevation' in df.columns:
            df['elevation'] = df['elevation'].ffill().fillna(0)
        # FITデータでは標高は altitude という名前で保存されることが多いのでその対応も追加
        elif 'altitude' in df.columns:
            df['elevation'] = df['altitude'].ffill().fillna(0)
    
        if 'cadence' in df.columns:
            df['cadence'] = df['cadence'].ffill().fillna(0)
        if 'speed' in df.columns:
            df['speed'] = df['speed'].ffill().fillna(0)
        
    return df

def main():
    st.set_page_config(page_title="FIT to Music Generator", layout="centered")
    
    st.title("🏃 FIT to Music Generator 🎵")
    st.markdown("""
    Garmin等の **FITデータ** をアップロードして、あなたのランニングデータを
    **180 BPM の爽快なプログレッシブハウス** に変換します！
    
    * **心拍数** ➔ シンセの激しさ
    * **標高/斜度** ➔ パッド音量とコード進行（上り: マイナー, 下り: メジャー）
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