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
        
        return {
            'synth_intensity': synth_intensity,
            'pad_volume': pad_volume,
            'is_uphill': is_uphill,
            'hihat_on': hihat_on,
            'fx_volume': fx_volume
        }

class ProgressiveHouseGenerator:
    """
    パラメータに基づき、pydubとnumpyを用いてプログレッシブハウスのトラックを生成するクラス
    """
    def __init__(self, mapper: FitDataToMusicMapper):
        self.mapper = mapper
        self.sample_rate = 44100
        self.beat_ms = self.mapper.beat_duration_ms
        
    def _generate_kick(self, duration_ms):
        kick_sound = Sine(60).to_audio_segment(duration=100).apply_gain(5)
        silence = AudioSegment.silent(duration=self.beat_ms - 100)
        one_beat = kick_sound + silence
        
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
            
            mix_1sec = kick.overlay(hihat).overlay(pad).overlay(fx).overlay(synth)
            
            final_mix = final_mix.overlay(mix_1sec, position=sec * 1000)
            
            if progress_callback:
                progress_callback(sec + 1, total_seconds)
            
        return final_mix

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
            st.write(f"総データ秒数: {len(df)} 秒")
            
            # 長すぎると処理に時間がかかるため制限を設けるオプション
            max_seconds = st.slider("生成する長さ(秒)を選択してください", min_value=10, max_value=min(600, len(df)), value=min(60, len(df)), step=10)
            
            if st.button("🎵 音楽を生成する"):
                df_subset = df.head(max_seconds).copy()
                
                mapper = FitDataToMusicMapper(df_subset, bpm=180)
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