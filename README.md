# Audio Deepfake Detection

## Overview 

This repository contains the implementation for Momenta’s Audio Deepfake Detection assessment. Having Explored various SotA architectures, I have decided to go with the AASIST-L model - A graph attention-based architecture.

## Research & Selection

#### Types of spoofing techniques: 
1. **Logical Access (LA):** Generated through Text-to-Speech (TTS) and Voice Conversion (VC).
2. **Physical Access (PA):** Spoofed speech captured in real physical spaces and replayed using devices.
3. **Deepfake (DF):** Manipulated, compressed speech data.

### Use case:
1. To detect AI-Generated Human-speech
2. Potential for near real time prediction and Mobile Deployment
3. Effectiveness on real conversations

The type of dataset current to our use case is LA and DF.


Without having implemented the top approaches, it's difficult 
### Identified Approaches:
#### 1. [RawGAT-ST](https://arxiv.org/pdf/2107.12710):
Utilizes a spectro-temporal graph attention network (GAT), enabling it to learn relations between spectral (frequency) and temporal (time) features in audio, combines them for enhanced speech spoof detection.

**Reported Performance Metrics:**
The RawGAT-ST achieved a 1.06% equal error rate (EER) on the ASVspoof 2019 logical access dataset, setting one of the best reported performances for anti-spoofing tasks. Baseline RawGAT params ~ 437K 

**Relevance**: 
- Graph structure helps capture dynamic characteristics of speech
- End to end mechanism, no reliance on handcrafted features.

**Limitations:**
- Heavy structure
- still not optimized for low latency inference

#### 2. [SpecRNet](https://arxiv.org/pdf/2210.06105):
Applies a spectrogram based res-net architecture. Processes features in the spectral domain via lightweight attention modules. Designed for deployability without needing raw waveform processing.

**Reported Performance Metrics:**
- Parameters: ~277K

**Why It’s Promising:**  
- Simple architecture
- Provided a 40% processing time decrease compared to LCNN architectures 

**Limitations:** Offers a balance between efficiency and accuracy, but performance may vary with different types of deepfake attacks

#### 3. [AASIST](https://arxiv.org/pdf/2110.01200):
Implements a RawNet stule front-end and a and a Graph Attention Network (GAT) type architecture. The heterogeneous stacking graph attention layer (HS-GAL) models both spectral and temporal graphs and the MGO (Max Graph Operation) enables it to capture the most the most significant spoofing artefacts.


**Reported Performance Metrics:**
- Parameters: 297k
- EER on LA  ASVSpoof 2019: **0.83%**  
- min-tDCF LA ASVSpoof 2019: **0.0275**
- AASIST-L: This lightweight variant of AASIST has 85K parameters. It achieves an EER of **0.99%** and a min t-DCF of **0.0309**.

**Why It’s Promising:**  
- Performed better than RAWGAT-ST in majority of systems 
- The graph based temporal modeling suggest good generalization for LA and DF datasets (relevant to our use case)
- The 85K parameter model, trained using mixed precision training offers a more compact, efficient solution for real world deployment without compromising performance, making it suitable for constrained environments.

**Limitations:**  
- Requires large, diverse datasets for proper generalizability
- Training complexity
- Will still face challenges in achieving real-time processing. Will need to be optimized further.


## 📊 Summary Table

| Model         | Parameters | EER (%) | min-tDCF |
|---------------|------------|---------|----------|
| **AASIST**    | ~297K      | 0.83    | 0.0275   |
| **AASIST-L**  | ~85K       | 0.99    | 0.0309   |
| **RawGAT-ST** | ~437K      | 1.06    | 0.0335   |


*While powerful models like wav2vec 2.0-based systems show excellent performance on benchmarks (e.g., ASVspoof 2021), they were not considered due to their high inference latency and size, which are not aligned with real-time mobile deployment goals.*

Evaluation Metrics:
1. Equal Error Rate (EER): When the false acceptance rate (FAR) and false rejection rate (FRR) are equal
2. Tandem detection cost function (t-DCF): A performance metric that balances spoof detection with speaker verification errors. Lower is better.


## Implementation
This project builds on the official [AASIST repository by ClovaAI](https://github.com/clovaai/aasist).  
Both **AASIST** and its lighter version **AASIST-L** were re-trained on the **ASVspoof 2019 LA** dataset with custom modifications to support training on limited resources and ensure reproducibility.
Dataset

- **Dataset Used**: [ASVspoof 2019 Logical Access (LA)](https://datashare.ed.ac.uk/handle/10283/3336)

| Feature           | Detail                       |
|------------------|------------------------------|
| GPU              | NVIDIA RTX 4070 (8 GB VRAM)  |
| Python Version   | 3.9.21                       |
| CUDA Version     | 12.1                         |
| Training Epochs  | 10                           |
| Final Batch Size | 8                            |

---

**Implementation challenges**
- Deprecation warnings, version conflicts, and missing imports — even with recommended package versions
- The original `download_dataset.py` script failed to maintain the required folder structure.
- Long training times, memory constraints.
- Progress could be lost if there were an issue at any given epoch.

**Fixes**
- Manual Deprecation fixes, code replacements, importing necessary libraries and saving a fresh requirements.txt
- Modified the script to extract and organize data to match the expected input pipeline format.
- Model training at custom configurations.
- Implemented a checkpoint mechanism which saves the current model state after each epoch.



