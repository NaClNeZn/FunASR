import sounddevice as sd
import numpy as np
from funasr import AutoModel

# 配置参数
SAMPLE_RATE = 16000  # FunASR模型通常需要16kHz采样率
CHUNK_DURATION = 3.0  # 每次处理的音频块长度（秒）
DEVICE = 3  # 使用默认麦克风设备（可通过sd.query_devices()查看设备列表）
HOTWORDS = ["闫乃新", "于俊凯", "吴桐"] # 热词列表（根据需求自定义）
current_cache = {} # 初始化缓存

# 初始化FunASR流式模型
model = AutoModel(
    # 使用流式模型
    model="D:\\Codes\\mine\\FunASR\\model\\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    # 在线VAD
    vad_model="D:\\Codes\\mine\\FunASR\\model\\speech_fsmn_vad_zh-cn-8k-common",
    vad_kwargs={"max_single_segment_time": 30000},
    # 标点恢复
    punc_model="D:\\Codes\\mine\\FunASR\\model\\punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
    # 说话人
    spk_model="D:\\Codes\\mine\\FunASR\\model\\speech_campplus_sv_zh-cn_16k-common",
    # 有GPU时可启用加速
    device="cuda:0",
    batch_size=1,
    quantize=True
)

# 创建音频缓冲区
audio_buffer = np.array([], dtype=np.float32)


def audio_callback(indata, frames, time, status):
    """麦克风音频数据回调函数"""
    global audio_buffer
    audio_chunk = indata[:, 0].astype(np.float32)  # 转换为单声道
    audio_buffer = np.concatenate([audio_buffer, audio_chunk])


try:
    print("🎤 开始录音（按CTRL+C停止）...")

    # 开始音频流捕获
    with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
            device=DEVICE,
            channels=1,  # 单声道
            callback=audio_callback
    ):
        while True:
            # 等待足够长度的音频数据
            if len(audio_buffer) >= SAMPLE_RATE * CHUNK_DURATION:
                # 提取当前块并清空缓冲区
                current_chunk = audio_buffer[:int(SAMPLE_RATE * CHUNK_DURATION)]
                audio_buffer = audio_buffer[int(SAMPLE_RATE * CHUNK_DURATION):]

                # 执行ASR推理（流式模式）
                result = model.generate(
                    input=current_chunk,
                    cache=current_cache,  # 保持上下文缓存
                    is_final=False,  # 持续流式处理
                    chunk_size=[5, 10, 5],  # 流式处理窗口配置
                    hotword=HOTWORDS, # 热词
                )

                # 更新缓存
                if result and 'cache' in result[0]:
                    current_cache = result[0]['cache']

                # 打印实时结果（自动叠加更新）
                if result and "text" in result[0]:
                    print("\r识别结果:", result[0]["text"], end="", flush=True)

except KeyboardInterrupt:
    # 处理最终片段
    if len(audio_buffer) > 0:
        result = model.generate(input=audio_buffer, is_final=True)
        if result and "text" in result[0]:
            print("\n最终结果:", result[0]["text"])
    print("\n录音已停止")

except Exception as e:
    print(f"发生错误: {str(e)}")