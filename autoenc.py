import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_layers, decoder_layers):
        super(Autoencoder, self).__init__()

        # encoder   
        encoder_modules = []
        prev_dim = input_dim
        
        for dim in encoder_layers:
            encoder_modules.append(nn.Linear(prev_dim, dim))
            encoder_modules.append(nn.ReLU())
            prev_dim = dim
        
        encoder_modules.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_modules)
        
        # decoder
        decoder_modules = []
        prev_dim = latent_dim
        
        for dim in decoder_layers:
            decoder_modules.append(nn.Linear(prev_dim, dim))
            decoder_modules.append(nn.ReLU())
            prev_dim = dim
        
        decoder_modules.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_modules)
    
    def decode(self, z):
        decoded = self.decoder(z)
        return decoded


    def forward(self, x):
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, z  

class ConvAutoencoder(nn.Module):
    def __init__(self, input_length=55125, latent_dim=96):
        super(ConvAutoencoder, self).__init__()
        
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),

            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
        
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )
        
        self.conv_output_size = self._get_conv_output_size(input_length)
        
        self.fc_encoder = nn.Linear(self.conv_output_size, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.conv_output_size)
    
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(128, 64, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(32, 16, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(16, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def _get_conv_output_size(self, input_length):
        x = torch.randn(1, 1, input_length)
        x = self.encoder(x)
        return x.shape[1] * x.shape[2]
        
    def encode(self, x):
        x = x.view(x.size(0), 1, -1)
        x = self.encoder(x)

        x = x.view(x.size(0), -1)
        
        z = self.fc_encoder(x)
        return z
    
    def decode(self, z):
        x = self.fc_decoder(z)
        
        batch_size = z.size(0)
        x = x.view(batch_size, 256, -1)
        
        output = self.decoder(x)
        
        reconstructed = output.squeeze(1)
        
        if reconstructed.shape[1] != self.input_length:
            reconstructed = nn.functional.interpolate(
                output, size=self.input_length, mode='linear', align_corners=False
            ).squeeze(1)
            
        return reconstructed
    
    def forward(self, x):
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, z