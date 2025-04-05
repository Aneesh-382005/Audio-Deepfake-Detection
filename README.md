# Audio Deepfake Detection

## Overview 

This repository contains the implementation for Momenta’s Audio Deepfake Detection assessment. Having Explored various SoTA architectures, I have decided to go with the AASIST-L model - A graph attention-based architecture.

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

| Configurations   | Detail                       |
|------------------|------------------------------|
| GPU              | NVIDIA RTX 4070 (8 GB VRAM)  |
| Python Version   | 3.9.21                       |
| CUDA Version     | 12.1                         |
| Training Epochs  | 10                           |
| Final Batch Size | 8                            |




---

**Model Analysis**

The AASIST model was choosen for implementation because of it's SoTA performance over multiple datasets. It is a significant upgrade over the RawNet2 and RawGAT-ST architectures. Additionally the lightweight variant, AASIST-L, offers a compact solution of just 85K parameters, making it suitable for environments with limited resources.

**Reasons to implement**
- **State of the Art Performance:** The model performs well on both LA and DF datasets and holds a decent ranking in the [ASVSpoof 2021 benchmarks](https://paperswithcode.com/sota/audio-deepfake-detection-on-asvspoof-2021)
- **Efficiency**: AASIST is a single end to end system which simplifies development as compared to models that require feature engineering or complex pipelines.

**Key Components**
- **RawNet** based encoder for extracting high-level feature maps from raw input waveforms. This encoder treats the output of the initial sinc-convolution layer as a 2D image (like a spectrogram) and uses residual blocks to learn relevant audio representations.
- **Graph Combination:** A heterogeneous graph is composed using the two different graphs that each model spectral and temporal domains.
- **Heterogeneous Stacking Graph Attention Layer (HS-GAL):** This layer incorporates a modified Attention Mechanism and an additional stack node. It enables the modelling of the two graphs with different nodes and dimensionalities.
- **Max Graph Operation (MGO) and Readout Scheme:** Involves two branches, each containing two HS-GALs and graph pooling layers, followed by a max() operation. This allows model to learn diverse groups of spoofing artefacts. Thestack Node-based readout scheme aggregates information in the graph representations, enhancing the model to capture complex patterns in data.

 
**Setup Instructions**
```
git clone https://github.com/Aneesh-382005/Audio-Deepfake-Detection.git
cd Audio-Deepfake-Detection
conda create -n aasist python=3.9
conda activate aasist
pip install -r requirements.txt
cd aasist
python.exe download_dataset.py #downloads and unzips the dataset.
```

This [README](https://github.com/Aneesh-382005/Audio-Deepfake-Detection/blob/main/aasist/README.md) contains the commands for the training baselines to train RawNet2, RawGAT-ST AASIST, and AASIST-L

**My Implementation**

I trained and evaluated both AASIST and AASIST-L for 10 epochs on the ASVspoof 2019 LA dataset, with batch size 8. Below are the summarized performance results and artifacts:

Below are the configurations and results for:

**AASIST** : [`AASISTcustom.conf`](https://github.com/Aneesh-382005/Audio-Deepfake-Detection/blob/main/aasist/config/AASISTcustom.conf)

`python main.py --config ./config/AASISTcustom.conf `

**AASIST - L** : [`AASIST-Lcustom.conf`](https://github.com/Aneesh-382005/Audio-Deepfake-Detection/blob/main/aasist/config/AASIST-Lcustom.conf)

`python main.py --config ./config/AASIST-Lcustom.conf`

---

| Model       | Epochs | Batch Size | EER (%) | t-DCF  | Best Weights | Metrics Log |
|-------------|:------:|:----------:|:-------:|:------:|:-------------|:-------------|
| **AASIST**   | 10     | 8          | **2.432** | **0.06331** | [`best.pth`](https://github.com/Aneesh-382005/Audio-Deepfake-Detection/blob/main/aasist/exp_result/LA_AASISTcustom_ep10_bs8/weights/best.pth) |  [`t-DCF_EER`](https://github.com/Aneesh-382005/Audio-Deepfake-Detection/blob/main/aasist/exp_result/LA_AASISTcustom_ep10_bs8/metrics/t-DCF_EER_008epo.txt) |
| **AASIST-L** | 10     | 8          | **3.373**    | **0.10398**  | [`best.pth`](https://github.com/Aneesh-382005/Audio-Deepfake-Detection/blob/main/aasist/exp_result/LA_AASIST-Lcustom_ep10_bs8/t-DCF_EER.txt) |  [`t-DCF_EER`](https://github.com/Aneesh-382005/Audio-Deepfake-Detection/blob/main/aasist/exp_result/LA_AASIST-Lcustom_ep10_bs8/t-DCF_EER.txt) |

---

**Implementation challenges**
- Deprecations , version conflicts, and missing imports — even with recommended package versions
- The original `download_dataset.py` script failed to maintain the required folder structure.
- Long training times, memory constraints.
- Progress could be lost if there were an issue at any given epoch.

**Fixes**
- Manual Deprecation fixes, code replacements, importing necessary libraries and saving a fresh requirements.txt
- Modified the script to extract and organize data to match the expected input pipeline format.
- Model training at custom configurations.
- Implemented a checkpoint mechanism which saves the current model state after each epoch. Keeps track of the last 5 checkpoints. This is especially useful if a previous epoch shows promising performance.




