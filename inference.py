import argparse
import json
import os
import sys
from pathlib import Path
import soundfile as sf
import torch
import numpy as np
import yaml
import librosa

from datetime import datetime

t1 = datetime.now()

AASIST_DIR = Path(__file__).resolve().parent
if str(AASIST_DIR) not in sys.path:
    sys.path.insert(0, str(AASIST_DIR))

from aasist.models.AASIST import Model as AASISTModel


def load_model(model_path, config_path, device):
    config = yaml.safe_load(open(config_path, 'r'))
    model_config = config.get('model_config', {})
    model_config.setdefault('nb_samp', 64000) 
    model_config.setdefault('device', device)

    model = AASISTModel(model_config) 

    model_state_dict = torch.load(model_path, map_location=device, weights_only=True)
    new_state_dict = {}
    for k, v in model_state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)


    model.to(device) 
    model.eval()
    print(f"Model loaded from {model_path} and configured using {config_path}")
    return model, config

def preprocess_audio(audio_path, config, device):
    data_config = config.get('data_config', {})
    feat_config = config.get('feat_config', {})

    target_sr = feat_config.get('sample_rate', 16000)
    num_required_samples = feat_config.get('nb_samp', 64000) 

    try:
        audio, sr = sf.read(audio_path)

        if sr != target_sr:
            raise ValueError(f"Sample rate mismatch: Expected {target_sr}Hz, got {sr}Hz. Resampling required.")

        if audio.ndim > 1:
            audio = audio[:, 0]

        current_len = len(audio)
        if current_len > num_required_samples:
            audio = audio[:num_required_samples]
        elif current_len < num_required_samples:
            padding = num_required_samples - current_len
            audio = np.pad(audio, (0, padding), 'constant')
        
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)

        return audio_tensor

    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        return None

def run_inference(model, audio_tensor, device):
    if audio_tensor is None:
        return None

    with torch.no_grad():
        raw_score = model(audio_tensor)
        if isinstance(raw_score, tuple):
            raw_score = raw_score[0]
        #print(f"Raw score shape: {raw_score.shape}")
        score = raw_score.mean().squeeze().cpu().item()

    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AASIST Single Audio Inference Script")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pre-trained AASIST model checkpoint (.pth)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the model configuration file (.conf or .yaml)')
    parser.add_argument('--input_audio', type=str, required=True,
                        help='Path to the input audio file (.wav, .flac, etc.)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use for inference (cpu or cuda)')

    args = parser.parse_args()

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU: ", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("Using CPU")
        if args.device == 'cuda':
            print("CUDA not available, falling back to CPU.")

    try:
        model, config = load_model(args.model_path, args.config, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    audio_tensor = preprocess_audio(args.input_audio, config, device)
    if audio_tensor is None:
        sys.exit(1)

    score = run_inference(model, audio_tensor, device)

    if score is not None:
        print(f"\n--- Inference Result ---")
        print(f"Input File: {args.input_audio}")
        print(f"Raw Model Score: {score:.4f}")

        threshold = 0.0 
        if score > threshold:
            print(f"Interpretation: Likely BONAFIDE (Score > {threshold})")
        else:
            print(f"Interpretation: Likely SPOOFED (Score <= {threshold})")
        print(f"------------------------")
    else:
        print("Inference failed.")


print(f"Total Time Taken: {datetime.now() - t1}")