# Audio Deepfake Detection

## Overview 

This repository contains the implementation for Momenta’s Audio Deepfake Detection assessment. Having Explored various SoTA architectures, I have decided to go with the **AASIST-L** model - A graph attention-based architecture.

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

---

## Implementation
This project builds on the official [AASIST repository by ClovaAI](https://github.com/clovaai/aasist).  
Both **AASIST** and its lighter version **AASIST-L** were re-trained on the **ASVspoof 2019 LA** dataset with custom modifications to support training on limited resources and ensure reproducibility.
Dataset

- **Dataset Used**: [ASVspoof 2019 Logical Access (LA)](https://datashare.ed.ac.uk/handle/10283/3336)

| Configurations   | Detail                       |
|------------------|------------------------------|
| GPU              | **NVIDIA RTX 4070 (8 GB VRAM)**  |
| Python Version   | **3.9.21**                       |
| CUDA Version     | **12.1**                         |
| Training Epochs  | 10                           |
| Final Batch Size | 8                            |



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

This [README](https://github.com/Aneesh-382005/Audio-Deepfake-Detection/blob/main/aasist/README.md) contains the commands for the training baselines to train **RawNet2, RawGAT-ST AASIST, and AASIST-L**

**My Implementation**

I trained and evaluated both AASIST and AASIST-L for 10 epochs on the ASVspoof 2019 LA dataset, with batch size 8. Below are the summarized performance results and artifacts:

Below are the configurations and results:

**AASIST** : [`AASISTcustom.conf`](https://github.com/Aneesh-382005/Audio-Deepfake-Detection/blob/main/aasist/config/AASISTcustom.conf)

`python main.py --config ./config/AASISTcustom.conf `

**AASIST - L** : [`AASIST-Lcustom.conf`](https://github.com/Aneesh-382005/Audio-Deepfake-Detection/blob/main/aasist/config/AASIST-Lcustom.conf)

`python main.py --config ./config/AASIST-Lcustom.conf`



| Model       | Epochs | Batch Size | EER (%) | t-DCF  | Best Weights | Metrics Log |
|-------------|:------:|:----------:|:-------:|:------:|:-------------|:-------------|
| **AASIST**   | 10     | 8          | **2.432** | **0.06331** | [`best.pth`](https://github.com/Aneesh-382005/Audio-Deepfake-Detection/blob/main/aasist/exp_result/LA_AASISTcustom_ep10_bs8/weights/best.pth) |  [`t-DCF_EER`](https://github.com/Aneesh-382005/Audio-Deepfake-Detection/blob/main/aasist/exp_result/LA_AASISTcustom_ep10_bs8/metrics/t-DCF_EER_008epo.txt) |
| **AASIST-L** | 10     | 8          | **3.373**    | **0.10398**  | [`best.pth`](https://github.com/Aneesh-382005/Audio-Deepfake-Detection/blob/main/aasist/exp_result/LA_AASIST-Lcustom_ep10_bs8/t-DCF_EER.txt) |  [`t-DCF_EER`](https://github.com/Aneesh-382005/Audio-Deepfake-Detection/blob/main/aasist/exp_result/LA_AASIST-Lcustom_ep10_bs8/t-DCF_EER.txt) |

---


### Inference

To run the inference, go back to the root directory

```bash
cd .. 
D:\Audio-Deepfake-Detection>python inference.py ^
  --model_path <path\to\the\model.pth> ^
  --config <config path.conf> ^
  --input_audio path\to\your\audio.wav ^
  --device <device>
```

`--device cpu` for running the inference on the CPU

`--device cuda` for GPU inference.


**Model Path:**

**AASIST-L**: `aasist\models\weights\AASIST-L.pth`

**AASIST**: `aasist\models\weights\AASIST.pth`

**Config Path:**

**AASIST-L**: `aasist\config\AASIST-L.conf`

**AASIST**: `aasist\config\AASIST.conf`

**Paths to my trained models:**

**AASIST-L**:  `aasist\exp_result\LA_AASIST-Lcustom_ep10_bs8\weights\best.pth`

**AASIST**: `aasist\exp_result\LA_AASISTcustom_ep10_bs8\weights\best.pth`

---

### *Upon testing on .wav files in the range 200KB to 4MB, The inference time on my NVIDIA RTX 4070 was **0.3 seconds** on average. Upon optimization, this shows great promise for mobile deployment.*

---

**Implementation challenges**
- Deprecations , version conflicts, and missing imports — even with recommended package versions
- The original `download_dataset.py` script failed to maintain the required folder structure.
- Long training times, memory constraints.
- Progress could be lost if there were an issue at any given epoch.
- No script for Inference

**Fixes**
- Manual Deprecation fixes, code replacements, importing necessary libraries and saving a fresh requirements.txt
- Modified the script to extract and organize data to match the expected input pipeline format.
- Model training at custom configurations.
- Implemented a checkpoint mechanism which saves the current model state after each epoch. Keeps track of the last 5 checkpoints. This is especially useful if a previous epoch shows promising performance.
- Added an `inference.py`

---

### **Observed Strengths and Weaknesses**

**Strengths**:
- **Lightweight Architecture**: At just 85K parameters (~332KB), AASIST-L is the most viable model for mobile deployment.
- **Competitive Performance**: Achieve an EER of 0.99% and min t-DCF of 0.0309 on the ASVspoof 2019 LA dataset, outperforming many larger models.
- **Generalization**: The graph based architecture, heterogeneous stacking enables it to handle both logical access (LA) and deepfake (DF) types effectively.
- **Training Stability**: Despite long training times, the model trained stably with no divergence across epochs.
- **Compact & Deployable**: Thanks to half-precision training and a fixed architecture, it’s inference-ready on constrained devices (mobile/edge).

**Weaknesses:**
- **Long Training Time**: Even with mixed precision, training took over 4 hours for 10 epochs on a ~7GB dataset.
- **Noisy Real Data**: Real-world audio with background noise or compression artifacts could reduce detection accuracy due to overfitting to curated datasets.
- **Limited Explainability**: It is still challenging to interpret the model, explaining which features cause the final activation.
- **Dataset Dependence**: The performance relies on dataset diversity. The dataset should contain samples of various spoofing methods like TTS, VC(Voice Conversion), SS(Speech Synthesis), etc

**Suggestions for Future Improvements:**
- **Quantization and Pruning**: INT8 quantization to prepare it for mobile hardware like **Snapdragon DSP** or **Apple Neural Engine**. Getting rid of the unneccessary weights.
- **On-Device Benchmarking**: Evaluate latency and resources on actual devices (e.g., Android, iOS) to identify bottlenecks and optimize accordingly. **ONNX Runtime/ TensorRT** deployment to optimize it for mobiles. 
- **Data Augmentation**: Incorporate more real life augmentations — background noise, reverberation, compression to improve generalization and staying updated with the latest spoofing techniques.
- **Real-World Data Fine-Tuning**: Permission-based data aquisition for retraining and improving current datasets
- **Constant Research:** Experimentation with various techniques and ensemble methods to further improve the metrics.
- **Exploring cross language**: Exploring foreign function interfaces in Java to push performance ctritical parts of the code to C++/Rust.
- **Knowledge Distillation:** Distilling knowledge from a large ensemble (eg. AASIST + XLSR).


**Future dataset improvements:**
- Train on the Latest ASVspoof datasets like [ASVSpoof 5](https://zenodo.org/records/14498691)
- **Diverse and Representative Datasets**
- **Synthetic Data Generation**
- **Adversarial Training** - Incorporating examples in the datasets to learn to recignize adversatial manipulations as spoof

### Deploying in a Production Environment:
Deploying this model for real-world use involves multiple steps due to the complexity. Here's how I would approach it:
- **Model Optimization**
  - Quantization for speed and size reduction
  - Convert to ONNX format for cross-platform deployment
  - AASIST-L is already designed with only ~85K parameters, making it more deployment-friendly.

- **Format Conversion**
  - Convert the trained model to **ONNX**, **TorchScript**, or **TFLite**, depending on the chosen deployment framework.
  - Use **TensorFlow Lite** (TFLite) or **TorchScript + JNI** For Android.
  - Use **Core ML** or **TFLite** with Metal backend.
  - No ONNX cloud runtimes or web APIs are used – models stay local on the device.

- **Use native libraries or bindings:**
  - **C++ or Rust** for latency-critical pre-processing steps.
  - Bind with JNI (Android) or FFI (iOS) as needed.
  
- **CI/CD Pipeline**
  - **GitHub Actions** for CI
  - **Gradle** for Android builds
  - **Python scripts** for model conversion
  - **Local testing** on device or emulator 

- **Privacy and Security**
  - All processing is **on-device**, no audio or metadata is sent over the internet.
  - No external API calls, no logging to cloud, no inference tracking.
- **Future Update Strategy**
  - Use app store updates to deliver newer models.
  - Implement version-check logic to **select appropriate model per device** (based on hardware capability).


## Executive Summary

This project explores and implements **AASIST-L**, a lightweight graph attention-based model for detecting audio deepfakes. After comparing multiple state-of-the-art methods (RawGAT-ST, SpecRNet, AASIST), I chose AASIST-L for its excellent balance between performance and deployability.  

Trained on the **ASVspoof 2019 LA dataset**, for 10 epochs AASIST-L achieved an **EER of 3.37%** and **t-DCF of 0.10** under constrained settings, while it having just **85K parameters**, making it highly suitable for mobile or real-time deployment.

Future improvements include INT8 quantization, ONNX/TensorRT deployment, real-world data fine-tuning, and model compression.
