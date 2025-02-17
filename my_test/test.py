import pyaudio

def list_microphones():
    # 初始化PyAudio
    p = pyaudio.PyAudio()

    # 获取设备总数
    device_count = p.get_device_count()
    print(f"总设备数：{device_count}")

    # 遍历所有设备并打印信息
    for i in range(device_count):
        device_info = p.get_device_info_by_index(i)
        device_name = device_info['name']
        device_channels = device_info['maxInputChannels']
        device_rate = device_info['defaultSampleRate']

        # 检查是否为麦克风（输入设备）
        if device_channels > 0:
            print(f"设备索引：{i}")
            print(f"设备名称：{device_name}")
            print(f"最大输入通道数：{device_channels}")
            print(f"默认采样率：{int(device_rate)} Hz")
            print("-" * 40)

    # 关闭PyAudio
    p.terminate()

if __name__ == "__main__":
    list_microphones()