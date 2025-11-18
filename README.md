# Audio Deepfake Detection with AASIST-L

This repository presents a comprehensive implementation and evaluation of a state-of-the-art audio deepfake detection system. The project leverages the **AASIST-L** model, a lightweight, graph attention-based architecture, chosen for its exceptional balance of high performance and computational efficiency.

The model was trained and evaluated on the **ASVspoof 2019 Logical Access (LA) dataset**. Under resource-constrained training conditions (10 epochs), the AASIST-L model achieved a promising **3.37% Equal Error Rate (EER)**, demonstrating its viability for real-world applications.

## Key Features
- **State-of-the-Art Model**: Implements the AASIST and AASIST-L architectures, which are top performers on industry-standard anti-spoofing benchmarks.
- **Lightweight & Efficient**: The final AASIST-L model contains only **~85K parameters**, making it ideal for on-device and real-time applications.
- **End-to-End Pipeline**: A complete workflow from dataset preparation and training to inference is provided.
- **Fast Inference**: Achieves an average inference time of **~0.3 seconds** on an NVIDIA RTX 4070 GPU, showcasing its potential for near real-time detection.
- **Reproducibility**: Includes detailed setup instructions, training configurations, and pre-trained model weights.

---

## Model Selection and Rationale

The selection of AASIST-L was the result of a thorough review of current anti-spoofing architectures, focusing on the detection of Logical Access (LA) and Deepfake (DF) attacks.

#### Architectural Comparison

Several models were considered, with performance and efficiency being the primary criteria.

| Model | Parameters | Reported EER (%) | Reported min-tDCF | Suitability |
|:---|:---|:---:|:---:|:---|
| **AASIST** | ~297K | **0.83** | **0.0275** | State-of-the-art performance, but heavier than the 'L' variant. |
| **AASIST-L** | **~85K** | 0.99 | 0.0309 | **Optimal balance** of high accuracy and low resource usage. Ideal for deployment. |
| **RawGAT-ST**| ~437K | 1.06 | 0.0335 | Strong GAT-based model but outperformed and larger than AASIST. |
| **SpecRNet** | ~277K | - | - | Simpler architecture but less robust against diverse spoofing attacks. |

*Models like `wav2vec 2.0` were excluded due to high inference latency and large model sizes, which conflict with the project's goal of mobile deployment readiness.*

### Why AASIST-L?

**AASIST-L** was the clear choice for this implementation for three key reasons:
1.  **Top-Tier Performance:** It outperforms many larger models on the ASVspoof 2019 LA benchmark.
2.  **Extreme Efficiency:** With only **85K parameters**, it is exceptionally well-suited for environments with limited computational resources, such as mobile or edge devices.
3.  **Advanced Architecture:** Its use of a graph-based temporal modeling approach suggests strong generalization capabilities for both known and emerging deepfake techniques.

---

## Implementation and Results

This project is built upon the official [AASIST repository by ClovaAI](https://github.com/clovaai/aasist). Both AASIST and AASIST-L were re-trained from scratch to validate their performance and adapt the pipeline for custom use.

### Environment & Setup

#### 1. Clone the repository:
```bash
git clone https://github.com/Aneesh-382005/Audio-Deepfake-Detection.git
cd Audio-Deepfake-Detection
```

#### 2. Create and activate the Conda environment:
```bash
conda create -n aasist python=3.9
conda activate aasist
```

#### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

#### 4. Download and prepare the dataset:
```bash
cd aasist
python download_dataset.py
```

### Training

The models were trained for 10 epochs using the configurations and commands below.

**AASIST**
- **Config:** [`AASISTcustom.conf`](https://github.com/Aneesh-382005/Audio-Deepfake-Detection/blob/main/aasist/config/AASISTcustom.conf)
- **Command:**
  ```bash
  python main.py --config ./config/AASISTcustom.conf
  ```

**AASIST-L**
- **Config:** [`AASIST-Lcustom.conf`](https://github.com/Aneesh-382005/Audio-Deepfake-Detection/blob/main/aasist/config/AASIST-Lcustom.conf)
- **Command:**
  ```bash
  python main.py --config ./config/AASIST-Lcustom.conf
  ```

### Performance on ASVspoof 2019 LA (10 Epochs)

| Model | EER (%) | t-DCF | Best Weights | Metrics Log |
|:---|:---:|:---:|:---:|:---|
| **AASIST** | **2.432** | **0.0633** | [`best.pth`](https://github.com/Aneesh-382005/Audio-Deepfake-Detection/blob/main/aasist/exp_result/LA_AASISTcustom_ep10_bs8/weights/best.pth) | [`log.txt`](https://github.com/Aneesh-382005/Audio-Deepfake-Detection/blob/main/aasist/exp_result/LA_AASISTcustom_ep10_bs8/metrics/t-DCF_EER_008epo.txt) |
| **AASIST-L**| **3.373** | **0.1040** | [`best.pth`](https://github.com/Aneesh-382005/Audio-Deepfake-Detection/blob/main/aasist/exp_result/LA_AASIST-Lcustom_ep10_bs8/weights/best.pth) | [`log.txt`](https://github.com/Aneesh-382005/Audio-Deepfake-Detection/blob/main/aasist/exp_result/LA_AASIST-Lcustom_ep10_bs8/t-DCF_EER.txt) |

---

## Usage for Inference

An `inference.py` script is provided for easy testing on single audio files.

### Command Structure
From the root directory (`Audio-Deepfake-Detection`), run the following command:
```bash
python inference.py ^
  --model_path <path_to_model_weights.pth> ^
  --config <path_to_model_config.conf> ^
  --input_audio <path_to_your_audio.wav> ^
  --device <cpu_or_cuda>
```

### Model & Config Paths

**Official Pre-trained Models:**
- **AASIST-L Model:** `aasist\models\weights\AASIST-L.pth`
- **AASIST-L Config:** `aasist\config\AASIST-L.conf`
- **AASIST Model:** `aasist\models\weights\AASIST.pth`
- **AASIST Config:** `aasist\config\AASIST.conf`

**My Trained Models (10 Epochs):**
- **AASIST-L Model:** `aasist\exp_result\LA_AASIST-Lcustom_ep10_bs8\weights\best.pth`
- **AASIST Model:** `aasist\exp_result\LA_AASISTcustom_ep10_bs8\weights\best.pth`

> **Inference Performance Note**
> On an NVIDIA RTX 4070, inference on WAV files ranging from 200KB to 4MB took an average of **0.3 seconds**. This demonstrates excellent potential for real-time mobile deployment after further optimization.

---

## Project Analysis

### Development Challenges & Solutions

- **Challenge:** Environment setup issues due to deprecations and version conflicts.
  - **Solution:** Manually resolved dependencies, fixed deprecated code, and generated a stable `requirements.txt`.

- **Challenge:** The original dataset script failed to create the correct directory structure.
  - **Solution:** Modified the `download_dataset.py` script to correctly extract and organize data for the training pipeline.

- **Challenge:** Risk of losing progress during long training runs due to memory constraints or interruptions.
  - **Solution:** Implemented a robust checkpointing mechanism that saves the model state after each epoch and retains the last five checkpoints.

- **Challenge:** The original repository lacked a script for single-file inference.
  - **Solution:** Developed a user-friendly `inference.py` script to facilitate easy model testing.

### Strengths and Weaknesses of AASIST-L

**Strengths:**
- **Deployment-Ready:** The extremely small footprint (~332KB) makes it a prime candidate for mobile and edge computing.
- **Robust Architecture:** The graph-based design is effective at capturing complex spectro-temporal artifacts characteristic of spoofed audio.
- **Stable Training:** The model exhibits stable and convergent training behavior.

**Weaknesses:**
- **Dataset Sensitivity:** Performance is highly dependent on the diversity of the training data. It may struggle with out-of-distribution noise or novel spoofing attacks.
- **Interpretability:** As with many deep learning models, interpreting *why* a specific prediction was made remains challenging.
- **Training Time:** Despite its small size, training is computationally intensive, requiring significant time and resources.

---

## Future Work and Deployment Strategy

### Roadmap for Future Improvements
- **Quantization & Pruning:** Apply post-training INT8 quantization and weight pruning to further reduce model size and accelerate inference on specialized hardware like the Snapdragon DSP or Apple Neural Engine.
- **On-Device Benchmarking:** Profile latency and resource consumption on target Android and iOS devices to identify and resolve performance bottlenecks.
- **Format Conversion:** Deploy using mobile-optimized runtimes like **ONNX Runtime**, **TensorFlow Lite**, or **Core ML**.
- **Data Augmentation:** Enhance model robustness by augmenting the training data with real-world noise, reverberation, and various compression artifacts.
- **Knowledge Distillation:** Train AASIST-L to mimic the outputs of a larger, more powerful ensemble of models to improve its accuracy without increasing its size.

### Production Deployment Strategy
1.  **Model Optimization:** Convert the final PyTorch model to an optimized format (**ONNX** or **TFLite**).
2.  **On-Device Integration:**
    -   **Android:** Use **TFLite** or package the **ONNX Runtime** with the application. For performance-critical pre-processing, use C++ code integrated via the JNI.
    -   **iOS:** Convert the model to **Core ML** for native, hardware-accelerated performance.
3.  **Privacy-First Design:** Ensure all audio processing and inference occur **100% on-device** to protect user privacy. No data should be sent to a server.
4.  **CI/CD Pipeline:** Automate the model conversion and build process using tools like **GitHub Actions** to ensure consistency and reliability.
5.  **Model Updates:** Distribute updated models through app store releases, with in-app logic to manage model versions and ensure compatibility.
