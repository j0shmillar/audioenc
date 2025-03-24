
import os
import sys
import tarfile
import librosa
import librosa.display
import tempfile
import requests

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split

import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

from autoenc import Autoencoder, ConvAutoencoder

# TODO
# am I downsampling okay?
    # try without downsampling 1st
# birdnetlib - use actual coords
# try rave or another actual model

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class Config:
    SAMPLE_RATE = 44100
    DURATION = 1
    
    BATCH_SIZE = 64
        
    LATENT_DIM = 512
    ENCODER_LAYERS = [64, 32, 32] 
    DECODER_LAYERS = [32, 64, 128, 256, 512, 1024, 512, 256, 256]  
    
    LEARNING_RATE = 0.001
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    MODELS_DIR = "models"
    RESULTS_DIR = "results"
    OUTPUT_DIR = "outputs" 

config = Config()
os.makedirs(config.MODELS_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True) 

class TempDataset(Dataset):
    def __init__(self, data_info, transform=None, fixed_duration=5):
        self.data_info = data_info
        self.transform = transform
        self.fixed_duration = fixed_duration
        self.sample_rate = config.SAMPLE_RATE
        self.downsampled_rate = 11025  # 44100 / 4
        self.target_length = self.fixed_duration * self.downsampled_rate
    
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        audio_path = self.data_info[idx]
        waveform, _ = torchaudio.load(audio_path)

        waveform = waveform.mean(dim=0)
        
        if waveform.shape[0] < self.sample_rate * self.fixed_duration:
            padded = torch.zeros(self.sample_rate * self.fixed_duration)
            padded[:waveform.shape[0]] = waveform
            waveform = padded
        else:
            waveform = waveform[:self.sample_rate * self.fixed_duration]
        
        # if waveform.abs().max() > 0:
        #     waveform = waveform / waveform.abs().max()
        
        downsampled_waveform = waveform[::4]
        
        if downsampled_waveform.shape[0] != self.target_length:
            padded = torch.zeros(self.target_length)
            padded[:min(downsampled_waveform.shape[0], self.target_length)] = downsampled_waveform[:min(downsampled_waveform.shape[0], self.target_length)]
            downsampled_waveform = padded

        return downsampled_waveform, waveform

def train_autoencoder(model, train_loader, val_loader, config):
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 5
    
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")

        for batch_idx, (ds_input_data, input_data) in enumerate(progress_bar):
            input_data = input_data.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            reconstructed, latent = model(input_data)

            # if reconstructed.shape != input_data.shape:
            #     reconstructed = nn.functional.interpolate(
            #         reconstructed.unsqueeze(1), 
            #         size=input_data.shape[1], 
            #         mode='linear', 
            #         align_corners=False
            #     ).squeeze(1)

            loss = criterion(reconstructed, input_data)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            current_loss = loss.item()
            train_loss += current_loss
            progress_bar.set_postfix({'loss': f"{current_loss:.4f}"})
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for _, input_data in val_loader:
                input_data = input_data.to(config.DEVICE)
                
                reconstructed, _ = model(input_data)
                
                # if reconstructed.shape != input_data.shape:
                #     reconstructed = nn.functional.interpolate(
                #         reconstructed.unsqueeze(1), 
                #         size=input_data.shape[1], 
                #         mode='linear', 
                #         align_corners=False
                #     ).squeeze(1)
                
                loss = criterion(reconstructed, input_data)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{config.EPOCHS}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{config.MODELS_DIR}/autoencoder_best.pt")
            print(f"Model saved at epoch {epoch+1}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f"{config.MODELS_DIR}/autoencoder_epoch_{epoch+1}.pt")
    
    torch.save(model.state_dict(), f"{config.MODELS_DIR}/autoencoder_final.pt")
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('loss')
    plt.savefig(f"{config.RESULTS_DIR}/training_loss.png")
    
    return model, history

def download_birdnet():
    url = "https://github.com/kahst/BirdNET-Analyzer/archive/refs/heads/main.tar.gz"
    response = requests.get(url, stream=True)
    file = tarfile.open(fileobj=response.raw, mode="r|gz")
    file.extractall(path=".")

def save_temp_audio(audio, sr):
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, audio, sr, format='WAV')
    return temp_file.name

def save_audio_files(original_audio, reconstructed_audio, sample_id, sr):
    """
    save original and reconstructed audio as WAV files
    
    - original_audio: numpy array of original audio
    - reconstructed_audio: numpy array of reconstructed audio
    - sample_id: identifier for the audio sample
    - sr: sample rate
    """
    original_dir = os.path.join(config.OUTPUT_DIR, "original")
    reconstructed_dir = os.path.join(config.OUTPUT_DIR, "reconstructed")
    
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(reconstructed_dir, exist_ok=True)
    
    if isinstance(original_audio, torch.Tensor):
        original_audio = original_audio.cpu().numpy()
    
    if isinstance(reconstructed_audio, torch.Tensor):
        reconstructed_audio = reconstructed_audio.cpu().numpy()
    
    # if original_audio.max() > 1.0 or original_audio.min() < -1.0:
    #     original_audio = np.clip(original_audio / max(abs(original_audio.max()), abs(original_audio.min())), -1.0, 1.0)
    
    # if reconstructed_audio.max() > 1.0 or reconstructed_audio.min() < -1.0:
    #     reconstructed_audio = np.clip(reconstructed_audio / max(abs(reconstructed_audio.max()), abs(reconstructed_audio.min())), -1.0, 1.0)
    
    original_path = os.path.join(original_dir, f"sample_{sample_id}_original.wav")
    sf.write(original_path, original_audio, sr, format='WAV')
    
    reconstructed_path = os.path.join(reconstructed_dir, f"sample_{sample_id}_reconstructed.wav")
    sf.write(reconstructed_path, reconstructed_audio, sr, format='WAV')
    
    return original_path, reconstructed_path

def evaluate_model(model, test_loader, config):
    model.eval()
    evaluation_results = []
    
    print(f"saving audio files to {config.OUTPUT_DIR}")
    
    with torch.no_grad():
        for idx, (_, input_data) in enumerate(test_loader):
            if idx >= 10: 
                break
                
            input_data = input_data.to(config.DEVICE)
            
            reconstructed_data, _ = model(input_data)
            
            original_np = input_data[0].cpu().numpy()
            
            reconstructed_data = reconstructed_data.cpu()
            
            reconstructed_np = reconstructed_data[0].numpy()
            if len(reconstructed_np) != len(original_np):
                reconstructed_np = librosa.resample(
                    reconstructed_np, 
                    orig_sr=config.SAMPLE_RATE, 
                    target_sr=config.SAMPLE_RATE   
                )
                
                if len(reconstructed_np) > len(original_np):
                    reconstructed_np = reconstructed_np[:len(original_np)]
                elif len(reconstructed_np) < len(original_np):
                    padding = np.zeros(len(original_np) - len(reconstructed_np))
                    reconstructed_np = np.concatenate([reconstructed_np, padding])
            
            original_path, reconstructed_path = save_audio_files(
                original_np, reconstructed_np, idx, config.SAMPLE_RATE
            )
            
            print(f"\nevaluating sample {idx+1}/10:")
            print(f"  original saved to: {original_path}")
            print(f"  reconstructed saved to: {reconstructed_path}")
            
            try:
                with HiddenPrints():
                    results_reconstructed = analyze_with_birdnet(
                        reconstructed_np, config.SAMPLE_RATE,
                        lat=47.16, lon=13.38, week=20, sensitivity=1.0, min_conf=0.1
                    )
                    
                    results_original = analyze_with_birdnet(
                        original_np, config.SAMPLE_RATE,
                        lat=47.16, lon=13.38, week=20, sensitivity=1.0, min_conf=0.1
                    )
                
                print("\noriginal audio detections:")
                for detection in results_original:
                    print(f" {detection['common_name']} ({detection['scientific_name']}): {detection['confidence']:.2f}")
                
                print("\nreconstructed audio detections:")
                for detection in results_reconstructed:
                    print(f"  {detection['common_name']} ({detection['scientific_name']}): {detection['confidence']:.2f}")
                
                original_species = {d['scientific_name']: d['confidence'] for d in results_original}
                reconstructed_species = {d['scientific_name']: d['confidence'] for d in results_reconstructed}
                
                common_species = set(original_species.keys()) & set(reconstructed_species.keys())
                match_percentage = len(common_species) / max(len(original_species), 1) * 100

                evaluation_results.append({
                    'sample_id': idx,
                    'original_detections': results_original,
                    'reconstructed_detections': results_reconstructed,
                    'match_percentage': match_percentage,
                    'original_audio_path': original_path,
                    'reconstructed_audio_path': reconstructed_path
                })
                
            except Exception as e:
                print(f"Error in BirdNET analysis: {e}")
    
    with open(f"{config.RESULTS_DIR}/evaluation_results.json", "w") as f:
        import json
        json.dump(evaluation_results, f, indent=4)
    
    print(f"audio files saved in {config.OUTPUT_DIR}")
    
    if evaluation_results:
        avg_match_rate = sum(r['match_percentage'] for r in evaluation_results) / len(evaluation_results)
        print(f"\nAverage species match rate across all samples: {avg_match_rate:.1f}%")
    
    return evaluation_results

def analyze_with_birdnet(audio, sr, lat=None, lon=None, week=None, sensitivity=1.0, min_conf=0.1):
    """
    analyze in-memory audio using BirdNET
    
    Parameters:
    - audio: numpy array of audio samples
    - sr: sample rate
    - lat, lon: Optional latitude and longitude for location-specific analysis
    - week: Week of the year (1-52) for time-specific analysis
    - sensitivity: Detection sensitivity (0.5-1.5)
    - min_conf: Minimum confidence threshold
    """
    temp_audio_path = save_temp_audio(audio, sr)
    
    try:
        analyzer = Analyzer()

        recording = Recording(
            analyzer,
            temp_audio_path,
            lat=35.4244,
            lon=-120.7463,
            date=datetime(year=2022, month=5, day=10), 
            min_conf=0.1,
        )
        recording.analyze()
        
        return recording.detections
    
    finally:
        os.unlink(temp_audio_path)

def main():
    # data_info = download_xeno_canto_data()
    folder_path = "recordings" 
    data_info = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    train_data, test_data = train_test_split(data_info, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    
    print(f"train data: {len(train_data)}, val data: {len(val_data)}, test data: {len(test_data)}")
    
    train_dataset = TempDataset(train_data, fixed_duration=config.DURATION)
    val_dataset = TempDataset(val_data, fixed_duration=config.DURATION)
    test_dataset = TempDataset(test_data, fixed_duration=config.DURATION)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # sample_input, _ = next(iter(train_loader)) # downsampled
    _, sample_input = next(iter(train_loader)) # original
    input_length = sample_input.shape[1]
    
    print(f"input sample shape: {sample_input.shape}")

    # model = Autoencoder(
    #     input_dim=input_length,
    #     latent_dim=config.LATENT_DIM,
    #     encoder_layers=config.ENCODER_LAYERS,
    #     decoder_layers=config.DECODER_LAYERS
    # ).to(config.DEVICE)

    model = ConvAutoencoder(
        input_length=input_length,
        latent_dim=config.LATENT_DIM
    ).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"model created with {total_params:,} total params")
    print(f"trainable params: {trainable_params:,}")
    
    model, _ = train_autoencoder(model, train_loader, val_loader, config)
    
    print(f"saving audio files to {config.OUTPUT_DIR}")

    evaluate_model(model, test_loader, config)

if __name__ == "__main__":
    main()