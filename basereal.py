from ttsreal import EdgeTTS


class BaseReal:
    def __init__(self, opt):
        self.opt = opt
        self.sample_rate = 16000
        self.chunk = self.sample_rate // opt.fps
        self.tts = EdgeTTS(opt, self)
