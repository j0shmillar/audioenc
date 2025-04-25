import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from tinyseanet import TinySEANetEncoder

class TempDataset(Dataset):
    def __init__(self, data_info, processor, sample_rate=24000, fixed_length=None):
        self.data_info = data_info
        self.processor = processor
        self.sample_rate = sample_rate
        if fixed_length is None:
            self.fixed_length = 3 * self.sample_rate  # 3 seconds of audio
        else:
            self.fixed_length = fixed_length

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        audio_path = self.data_info[idx]
        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(waveform)

        if waveform.shape[1] < self.fixed_length:
            padding = self.fixed_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > self.fixed_length:
            waveform = waveform[:, :self.fixed_length]

        input_values = self.processor(waveform.squeeze(0), sampling_rate=self.sample_rate, return_tensors="pt")["input_values"]

        return input_values.squeeze(0)

def evaluate_model(student_encoder, student_decoder, teacher, test_loader):
    with torch.no_grad():
        for idx, input_audio in tqdm(enumerate(test_loader)):
            if idx >= 10:
                break

            if input_audio.dim == 2:
               input_audio = input_audio.unsqueeze(0) 

            teacher_encoded_audio = teacher.encode(input_audio)
            print("teacher")
            print(teacher_encoded_audio)
            teacher_decoded_audio = teacher.decode(teacher_encoded_audio.audio_codes, teacher_encoded_audio.audio_scales)[0]

            student_encoded_audio = student_encoder(input_audio)
            print("student")
            print(student_encoded_audio)
            student_decoded_audio = student_decoder(student_encoded_audio)

            teacher_decoded_audio = teacher_decoded_audio.squeeze(0)
            student_decoded_audio = student_decoded_audio.squeeze(0)
            input_audio = input_audio.squeeze(0)

            torchaudio.save(f"outputs/original{idx}.wav", input_audio, 24000)
            torchaudio.save(f"outputs/teacher_decoded{idx}.wav", teacher_decoded_audio, 24000)
            torchaudio.save(f"outputs/student_decoded{idx}.wav", student_decoded_audio, 24000)
    print("Eval complete")

folder_path = "recordings"
data_info = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
_, test_data = train_test_split(data_info, test_size=0.2, random_state=42)

print(f"test data: {len(test_data)}")

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained('facebook/encodec_24khz')

test_dataset = TempDataset(test_data, processor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

student_encoder = TinySEANetEncoder()
student_encoder.load_state_dict(torch.load("student_encoder.pt", weights_only=True))
student_encoder.eval()

from transformers import EncodecModel
model = EncodecModel.from_pretrained('facebook/encodec_24khz')
teacher_path = "fine_tuned_encodec"
teacher = EncodecModel.from_pretrained(teacher_path)
teacher.eval()

student_decoder = teacher.decoder
student_decoder.eval()

evaluate_model(student_encoder, student_decoder, teacher, test_loader)
