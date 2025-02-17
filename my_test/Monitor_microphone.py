import pyaudio
import wave
import threading

# 音频参数
CHUNK = 1024  # 每次读取的音频帧数
FORMAT = pyaudio.paInt16  # 音频格式
CHANNELS = 1  # 单声道
RATE = 16000  # 采样率
WAVE_OUTPUT_FILENAME = "./realtime_output.wav"  # 输出文件名

# 初始化PyAudio
p = pyaudio.PyAudio()
p.get_device_info_by_index(3)
# 打开音频流
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# 创建WAV文件
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)

print("开始实时录音，按Ctrl+C停止...")

try:
    while True:
        # 读取音频数据
        data = stream.read(CHUNK)
        # 将音频数据写入WAV文件
        wf.writeframes(data)
except KeyboardInterrupt:
    print("录音已停止。")

# 关闭音频流和文件
stream.stop_stream()
stream.close()
p.terminate()
wf.close()

print(f"音频已保存到 {WAVE_OUTPUT_FILENAME}")