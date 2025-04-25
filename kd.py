import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
from transformers import EncodecModel, AutoProcessor

from tinyseanet import TinySEANetEncoder

# TODO
# Add an STFT loss: distance between spectrograms
# Use a perceptual loss (e.g., from a pre-trained audio model)
# Include latent alignment loss: MSE between encoder outputs
# For on MCU:
#  - Shrink input dims
#  - No autoproc!

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

def train_student_encoder(finetuned_teacher_path, save_path, num_epochs=150, lr=1e-3, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    teacher = EncodecModel.from_pretrained(finetuned_teacher_path).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    student_encoder = TinySEANetEncoder().to(device)
    student_encoder.train()
    for param in student_encoder.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(student_encoder.parameters(), lr=lr)

    folder_path = "recordings" 
    data_info = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    train_data, test_data = train_test_split(data_info, test_size=0.2, random_state=42)

    print(f"train data: {len(train_data)}, test data: {len(test_data)}")

    processor = AutoProcessor.from_pretrained('facebook/encodec_24khz', use_fast=False)

    train_dataset = TempDataset(train_data, processor)
    test_dataset = TempDataset(test_data, processor)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0

        for batch in tqdm(train_loader):
            batch = batch.to(device)  # shape (B, 1, T)

            if batch.dim() == 2:
                batch = batch.unsqueeze(1)

            def stft_loss(x, y):
                x_stft = torch.stft(x.squeeze(1), n_fft=512, hop_length=160, 
                                   win_length=512, return_complex=True)
                y_stft = torch.stft(y.squeeze(1), n_fft=512, hop_length=160, 
                                   win_length=512, return_complex=True)
                return F.mse_loss(torch.abs(x_stft), torch.abs(y_stft))
            
            with torch.no_grad():
                teacher_latents = teacher.encode(batch)
                teacher_recon = teacher.decode(teacher_latents.audio_codes, teacher_latents.audio_scales)[0]
                teacher_codes = teacher_latents.audio_codes.detach()
                teacher_recon = teacher_recon.detach()

            student_latents_float = student_encoder(batch)  # requires grad
            student_latents_long = student_latents_float.detach().unsqueeze(0).long()
            student_recon = teacher.decode(student_latents_long, [None])[0].detach()

            latent_loss = F.mse_loss(student_latents_float, teacher_codes.float().squeeze(0))
            spec_loss = stft_loss(student_recon, teacher_recon)
            loss = spec_loss + latent_loss

            latent_loss = F.mse_loss(student_latents_float, teacher_codes.float().squeeze(0))
            recon_loss = F.mse_loss(student_recon, teacher_recon)
            spec_loss = stft_loss(student_recon, teacher_recon)
            loss = recon_loss + 1e-4 * latent_loss # also try just stft_loss

            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_encoder.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

    torch.save(student_encoder.state_dict(), save_path)

if __name__ == "__main__":
    finetuned_teacher_path = "fine_tuned_encodec"   
    save_path = "student_encoder.pt"
    train_student_encoder(finetuned_teacher_path, save_path)

