## LOCAS: Multilabel RNA Localization with Supervised Contrastive Learning
Traditional approaches to predicting mRNA subcellular localization often fail to address the complexity of multiple compartmentalization, limiting biological insights. While recent multi-label models have shown progress, challenges persist in accurately capturing intricate localization patterns. We introduce LOCAS (Localization with Supervised Contrastive Learning), a novel framework that in corporates an RNA language model to generate initial embeddings and supervised contrastive learning (SCL) to identify distinct RNA clusters based on sequence similarity. LOCAS also uses a multi-label classification head (ML-Decoder) with cross-attention, enabling accurate multi-compartment predic tions. Our contributions include: (1) the first integration of RNA language models to create a nuanced embedding space for RNA sequences, (2) an SCL approach that detects overlapping localization pat terns with a multi-label similarity threshold, and (3) a multi-label classification head tailored for RNA localization. Comprehensive experiments, including extensive ablation studies and optimized threshold tuning, confirm LOCAS achieves state-of-the-art accuracy across all metrics, setting a new standard in multi-compartment mRNA localization.
### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abrarrahmanabir/LOCAS.git
   cd LOCAS


2. **Install dependencies**:

   ```bash
   pip install torch torchvision tqdm matplotlib pandas numpy scikit-learn
   ```
### RiNALMo Embedding Generation
   ```python
   python train_final.py


### How to Train
We give an example dataset named 'dataset.csv' and corresponding language model embeddings in 'dataset.npy'.
To start the training process, execute the following command:

   ```bash
   python train_final.py

## How to Run Inference

### Requirements:
- Pre-trained model: `Trained_models` folder contains the pretrained encoder and classifier.
- Input dataset: `Data/merged_output.csv`
- Language model embeddings must be loaded as part of the inference process.


### Steps to Run Inference:

1. **Prepare the Input Data**:
   Ensure that the input file `Data/merged_output.csv` is available in the repository's `Data/` folder. This CSV file contains the RNA sequence and localization labels.

2. **Load the Pre-trained Model**:
   The pre-trained models  are included in the `Trained_models`. The inference script will load this model automatically to perform predictions.

3. **Load Language Model Embeddings**:
   For the inference process, the necessary language model embeddings  `emb.pkl` have to be loaded for prediction.

4. **Run the Inference**:
   To perform inference on the provided dataset, run the following command:
   
   ```bash
   python inference.py 



### Model Architecture
![Model Architecture](overall_training.png)






