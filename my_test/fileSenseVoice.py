from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

"""
本地文件完整输出文字
"""

model_dir = "D:\Codes\mine\FunASR\model\SenseVoiceSmall"

model = AutoModel(
    model=model_dir,
    # vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

# en
res = model.generate(
    # input=f"{model.model_path}/example/en.mp3",
    input=f"D:\\temp\\133159_use11.34s-audio0s-seed1-te0.3-tp0.7-tk20-textlen20-34245-merge.wav",
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)