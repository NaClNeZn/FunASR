import queue
import time
import torch
import sounddevice as sd
import numpy as np
from queue import Queue
from funasr import AutoModel

# 配置参数
SAMPLE_RATE = 16000
DEVICE = 3  # 麦克风设备
BUFFER_SECONDS = 3  # 处理粒度
MAX_QUEUE_SIZE = 10  # 防止内存溢出
HOTWORDS="闫乃新 于俊凯 吴桐" # 热词

# 初始化线程安全队列
audio_queue = Queue(maxsize=MAX_QUEUE_SIZE)

# 模型初始化
try:
    model = AutoModel(
        # 使用流式模型
        model="D:\\Codes\\mine\\FunASR\\model\\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        # 在线VAD
        # vad_model="D:\\Codes\\mine\\FunASR\\model\\speech_fsmn_vad_zh-cn-8k-common",
        # vad_kwargs={"max_single_segment_time": 30000},
        # 标点恢复
        punc_model="D:\\Codes\\mine\\FunASR\\model\\punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
        # 说话人
        spk_model="D:\\Codes\\mine\\FunASR\\model\\speech_campplus_sv_zh-cn_16k-common",
        # 有GPU时可启用加速
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
except Exception as e:
    print(f"模型初始化失败: {str(e)}")
    exit(1)


def audio_callback(indata, frames, time, status):
    """音频回调"""
    if status:
        print(f"音频采集错误: {status}")
    try:
        audio_queue.put_nowait(indata[:, 0].astype(np.float32))
    except queue.Full:
        print("警告：音频队列已满，丢弃数据")


def process_audio():
    """处理音频的函数"""
    buffer = np.array([], dtype=np.float32)
    while True:
        try:
            chunk = audio_queue.get(timeout=1)
            buffer = np.concatenate([buffer, chunk])

            # 动态处理机制
            while len(buffer) >= SAMPLE_RATE * BUFFER_SECONDS:
                process_chunk = buffer[:int(SAMPLE_RATE * BUFFER_SECONDS)]
                buffer = buffer[int(SAMPLE_RATE * BUFFER_SECONDS):]

                # 识别处理
                result = model.generate(
                    input=process_chunk,
                    chunk_size=[5, 10, 5],
                    hotwords=HOTWORDS
                )

                # 结果处理
                if result and "text" in result[0]:
                    display_result(result[0]["text"])

        except queue.Empty:
            continue


def display_result(text):
    """结果显示"""
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] 识别结果: {text}")


if __name__ == "__main__":
    print("可用音频设备：")
    print(sd.query_devices())

    try:
        with sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=int(SAMPLE_RATE * BUFFER_SECONDS),
                device=DEVICE,
                channels=1,
                callback=audio_callback
        ):
            process_audio()
    except KeyboardInterrupt:
        print("\n正常退出")