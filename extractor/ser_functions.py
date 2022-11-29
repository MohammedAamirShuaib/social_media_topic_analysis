import os
import numpy as np
from pydub import AudioSegment
import librosa
from scipy import signal
from scipy.io import wavfile
import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.hub import load_state_dict_from_url
import matplotlib.pyplot as plt
from scipy.special import softmax
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
plt.style.use('ggplot')


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
__all__ = ['AlexNet', 'alexnet']
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth', }
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True)


def audio2spectrogram(filepath):
    window_size = 40
    step_size = 20
    eps = 1e-10
    samplerate, test_sound = wavfile.read(filepath, mmap=True)
    nperseg = int(round(window_size * samplerate / 1e3))
    noverlap = int(round(step_size * samplerate / 1e3))
    freqs, _, spec = signal.spectrogram(test_sound,
                                        fs=samplerate,
                                        window='hann',
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        detrend=False)
    spectrogram = np.log(spec.T.astype(np.float32) + eps)
    return spectrogram


def get_3d_spec(spectrogram, moments=None):
    N_CHANNELS = 3
    if moments is not None:
        (base_mean, base_std, delta_mean, delta_std,
         delta2_mean, delta2_std) = moments
    else:
        base_mean, delta_mean, delta2_mean = (0, 0, 0)
        base_std, delta_std, delta2_std = (1, 1, 1)
        h, w = spectrogram.shape
        right1 = np.concatenate(
            [spectrogram[:, 0].reshape((h, -1)), spectrogram], axis=1)[:, :-1]
        delta = (spectrogram - right1)[:, 1:]
        delta_pad = delta[:, 0].reshape((h, -1))
        delta = np.concatenate([delta_pad, delta], axis=1)
        right2 = np.concatenate(
            [delta[:, 0].reshape((h, -1)), delta], axis=1)[:, :-1]
        delta2 = (delta - right2)[:, 1:]
        delta2_pad = delta2[:, 0].reshape((h, -1))
        delta2 = np.concatenate([delta2_pad, delta2], axis=1)
        base = (spectrogram - base_mean) / base_std
        delta = (delta - delta_mean) / delta_std
        delta2 = (delta2 - delta2_mean) / delta2_std
        stacked = [arr.reshape((h, w, 1)) for arr in (base, delta, delta2)]
    return np.concatenate(stacked, axis=2)


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((12, 12))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        print('features', x.shape)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=5):
        super(ModifiedAlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.Sigmoid = nn.Sigmoid

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=2)
        x = torch.sum(x, dim=2)
        x = self.classifier(x)
        return x


def modifiedAlexNet(pretrained=False, progress=True, **kwargs):
    model_modified = ModifiedAlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model_modified.load_state_dict(state_dict)
    return model_modified


outputs_text = []


def hook_text(module, input, output):
    outputs_text.clear()
    outputs_text.append(output)
    return None


outputs_audio = []


def hook_audio(module, input, output):
    outputs_audio.clear()
    outputs_audio.append(output)
    return None


class CombinedAudioTextModel(nn.Module):
    def __init__(self, num_classes=5):
        super(CombinedAudioTextModel, self).__init__()
        self.num_classes = num_classes
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.text_model = torch.load(
            'models/model_text_best.pt', map_location=torch.device('cpu'))
        self.audio_model = ModifiedAlexNet()
        PATH = 'models/model_audio_best.pt'
        torch.save(self.audio_model.state_dict(), PATH)
        self.audio_model.load_state_dict(torch.load(PATH))
        self.text_model.bert.pooler.register_forward_hook(hook_text)
        self.audio_model.features.register_forward_hook(hook_audio)

        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.audio_model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(.5)
        self.linear = nn.Linear(1024, num_classes)
        self.Sigmoid = nn.Sigmoid

    def forward(self, text, audio):
        self.text_model(text)
        self.audio_model(audio)
        audio_embed = outputs_audio[0]
        text_embed = outputs_text[0]
        audio_embed = torch.flatten(audio_embed, start_dim=2)
        audio_embed = torch.sum(audio_embed, dim=2)
        concat_embded = torch.cat((text_embed, audio_embed), 1)
        x = self.dropout(concat_embded)
        x = self.linear(x)
        return x


model = CombinedAudioTextModel()
PATH1 = 'models/Combined_model_audio_text.pt'
torch.save(model.state_dict(), PATH1)
model.load_state_dict(torch.load(PATH1))
model.eval()
model.to('cpu')

HF_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "superb/wav2vec2-base-superb-er")
HF_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "superb/wav2vec2-base-superb-er")


def predict_bert_emotion(file_path, utter):
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    y = y * 32767 / max(0.01, np.max(np.abs(y)))
    audio_filepath = file_path.split(".wav")[0] + ".wav"
    wavfile.write(file_path, sr, y.astype(np.int16))
    indextolabel = {0: 'ang', 1: 'hap', 3: 'neu', 2: 'sad', 4: 'exc'}
    if (audio2spectrogram(file_path)).shape[0] != 0:
        prob = None
        preds = None
        spector = audio2spectrogram(file_path)
        spector = get_3d_spec(spector)
        npimg = np.transpose(spector, (2, 0, 1))
        input_tensor = torch.tensor(npimg)
        # create a mini-batch as expected by the model
        sprectrome = input_tensor.unsqueeze(0)
        input_ids = torch.tensor(tokenizer.encode(
            utter, add_special_tokens=True, truncation=True)).unsqueeze(0)
        with torch.no_grad():
            if (sprectrome.shape[2] > 10):
                output = model(input_ids, sprectrome)
                m = nn.Softmax(dim=1)
                probs = m(output)
                preds = torch.argmax(probs)
                prob = torch.max(probs)
                probs = probs.tolist()[0]
                emotion = {}
            for i in range(len(probs)):
                emotion[indextolabel[i]] = round(probs[i] * 100, 2)
                emotions = emotion
        return emotions


def predict_HF(filepath):
    speech, _ = librosa.load(filepath, sr=16000, mono=True)
    inputs = HF_feature_extractor(
        speech, sampling_rate=16000, padding=True, return_tensors="pt")
    logits = HF_model(**inputs).logits
    logits = logits.tolist()[0]
    logits = softmax(logits, axis=0)
    emotion_dict = {'neu': logits[0], 'hap': logits[1],
                    'ang': logits[2], 'sad': logits[3]}
    return emotion_dict


def Get_Emotions(audio_filepath, start_time, end_time, utter, model):
    try:
        fname = os.path.splitext(os.path.basename(audio_filepath))[0]
        os.makedirs("crop_audio", exist_ok=True)
        output_path = 'crop_audio/'
        t1, t2 = start_time, end_time
        newAudio = AudioSegment.from_file(audio_filepath)
        newAudio = newAudio[t1:t2]
        newAudio.export(output_path+str(t1)+'_'+str(t2) +
                        '_'+fname+".wav", format="wav")
        file_path = output_path+str(t1)+'_'+str(t2)+'_'+fname+".wav"
        if model == "HuggingFace":
            emotion_dict = predict_HF(file_path)
        if model == "BERT_PT":
            emotion_dict = predict_bert_emotion(file_path, utter)
        if model == "Both":
            HF_emotion_dict = predict_HF(file_path)
            BERT_emotion_dict = predict_bert_emotion(file_path, utter)
            emotion_dict = {"HuggingFace": HF_emotion_dict,
                            "BERT_PT": BERT_emotion_dict}
        os.remove(file_path)
        return emotion_dict
    except:
        os.remove(file_path)
        return {}


def emotion_results(dict, emotion):
    try:
        score = dict[emotion]
        return score
    except:
        return None
