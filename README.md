# Tesi_Script_Colab
English: by Riccardo Venturi cagliari 2026
# Comparative Study of Deep Learning Pipelines for Automatic Assessment of Drilling-Induced Damage in CFRP Composites

This repository contains the source code developed for the **Bachelor’s thesis** (though it’s honestly Master’s-level) in Mechanical Engineering by **Riccardo Venturi**, defended at the **University of Cagliari** in the Academic Year 2023–2024.

The project addresses the challenge of automatic quantification of drilling-induced damage (delamination) in carbon fiber reinforced polymer (CFRP) materials, a critical analysis for high-technology sectors such as aerospace and automotive industries.

## 📜 Table of Contents

- [Context and Objectives](#context-and-objectives)
- [Repository Structure](#repository-structure)
- [Pipeline Workflow (ROIA)](#pipeline-workflow-roia)
- [⚙️ Installation and Setup](#️-installation-and-setup)
- [🚀 Usage](#-usage)
- [📄 Citation](#-citation)
- [Model Weights](#-weightsdataset)
- [⚖️ License](#️-license)

---

## Context and Objectives

Traditional inspection of subsurface defects in CFRP composites, often based on Non-Destructive Testing (NDT), is a manual, subjective process with low reproducibility.  
The objective of this thesis was the development and validation of an **end-to-end computational pipeline** to automate damage analysis.

Starting from an experimental dataset of radiographic scans and process data (drilling force), the work focused on the development of the **ROIA (Radiograph-Only Integrated Analysis)** architecture: a sequence of Computer Vision and Machine Learning modules capable of:

1. Extracting and normalizing data from heterogeneous raw scans.
2. Segmenting the drilled hole and delamination damage with high precision.
3. Extracting a robust set of quantitative, engineering-relevant features.
4. Imputing missing process data.
5. Predicting future damage evolution based on historical data.

---

## 📂 Repository Structure

The repository is organized into folders reflecting the main phases of the analysis pipeline.

---

## Pipeline Workflow (ROIA)

The main pipeline developed (ROIA) operates exclusively in the radiographic domain and concatenates the following modules in an MLOps chain:

1. **Detection and Normalization (Module `01`)**:  
   A **YOLOv8** network localizes the holes in raw scans. A K-Means clustering algorithm enforces a reproducible **boustrophedonic** ordering. Finally, a **scale-normalized cropping algorithm** generates a dataset of 512×512 patches, ensuring that each hole has a constant apparent size.

2. **Damage Segmentation (Module `02`)**:  
   The normalized patches are processed by a **UNet++** network. The model is first pre-trained on a large set of automatically generated masks (`Mask Factory`) and then specialized via a **hybrid “foveal” training**, combining full images (for global context) and 128×128 detail patches (for micro-scale damage structures).

3. **Feature Extraction (Module `03`)**:  
   A **per-image auto-calibration algorithm** analyzes the segmentation masks. Using the known hole diameter (6 mm) as an internal “ruler,” it converts pixel-based measurements (areas, diameters, Hu moments) into physical units (mm², mm), ensuring metrological robustness.

4. **Imputation and Prediction (Module `04`)**:
   * A **Multi-Layer Perceptron (MLP)** is used to impute missing drilling force values, exploiting correlations with geometric damage features.
   * A **Long Short-Term Memory (LSTM)** network, trained with a **Quantile Loss**, models the sequential evolution of damage. This approach predicts not only the trend but also a prediction interval (risk), capturing extreme events.

---

## ⚙️ Installation and Setup

To run the scripts and notebooks in this repository, it is recommended to create a virtual environment to manage dependencies.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Riccardo-Venturi/Tesi_Script_Colab.git
   cd Tesi_Script_Colab


2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   It is strongly recommended to create a `requirements.txt` file with the required libraries. The main dependencies include:

   ```bash
   pip install torch torchvision ultralytics segmentation-models-pytorch albumentations opencv-python-headless pandas scikit-learn matplotlib lightgbm tensorflow
   ```

---

## 🚀 Usage

To reproduce the thesis results, run the notebooks and scripts following the numerical order of the folders:

1. **Phase 1 (`01_data_preprocessing`)**: Generate normalized and ordered patches starting from raw scans.
2. **Phase 2 (`02_segmentation_unet++`)**: Use the patches to train the segmentation model.
3. **Phase 3 (`03_feature_extraction`)**: Run the feature extractor on the generated masks.
4. **Phase 4 (`04_predictive_modeling_lstm`)**: Use the final feature CSV to train the predictive model.

**Note:** Data paths inside the notebooks are hardcoded for the Google Colab environment with Google Drive mounted. These paths must be adapted for a local setup.

---

## Weights / Models / Dataset

---

If you are looking for model weights and datasets, they are available on Hugging Face.

# CFRP-ROIA: Machine Learning Pipelines for CFRP Damage Analysis

**Author:** Riccardo Venturi, Cagliari, 2025
**Thesis Repository (code only)** — This repository contains the scripts used for training and evaluating all pipelines (YOLOv8, UNet++, MLP–LSTM).

🧠 **Model Weights and Dataset**
The trained weights and processed datasets are publicly available at:

> Venturi R. (2025). *CFRP-ROIA Weights and Datasets.* Hugging Face Hub.
> url          = { [https://huggingface.co/Riccardo99999/CFRP-ROIA-weights](https://huggingface.co/Riccardo99999/CFRP-ROIA-weights) },
> doi          = { 10.57967/hf/6729 },

---

```bibtex
@misc{riccardoventuri_2025,
  author       = {Riccardo Venturi},
  title        = {RPOS-dataset (Revision 2709c43)},
  year         = 2025,
  url          = {https://huggingface.co/datasets/Riccardo99999/RPOS-dataset},
  doi          = {10.57967/hf/6749},
  publisher    = {Hugging Face}
}
```

---

## 📄 Citation

If you use this code in your research, please cite the work as follows:

```bibtex
@mastersthesis{Venturi2024CFRP,
  author    = {Riccardo Venturi},
  title     = {Comparative study of deep learning pipelines for the automatic assessment of drilling-induced damage in CFRP composites},
  school    = {University of Cagliari},
  year      = {2024},
  address   = {Cagliari, Italy},
  month     = {July},
  howpublished = {\url{https://github.com/Riccardo-Venturi/Tesi_Script_Colab}}
}
```

Matricola 65221

Italiano


# Tesi_Script_Colab

# Studio Comparativo di Pipeline di Deep Learning per la Valutazione Automatica del Danno da Foratura in Compositi CFRP

Questo repository contiene il codice sorgente sviluppato per la tesi di Laurea Triennale(ma è figa da magistrale) in Ingegneria Meccanica di **Riccardo Venturi**, discussa presso l'**Università degli Studi di Cagliari** nell'Anno Accademico 2023-2024.

Il progetto affronta la sfida della quantificazione automatica del danno indotto dalla foratura (delaminazione) in materiali compositi a fibra di carbonio (CFRP), un'analisi critica per settori ad alta tecnologia come l'aerospaziale e l'automotive.
## 📜 Indice

- [Contesto e Obiettivi](#contesto-e-obiettivi)
- [Struttura del Repository](#struttura-del-repository)
- [Flusso di Lavoro della Pipeline (ROIA)](#flusso-di-lavoro-della-pipeline-roia)
- [⚙️ Installazione e Setup](#️-installazione-e-setup)
- [🚀 Utilizzo](#-utilizzo)
- [📄 Citazione](#-citazione)
- [Pesi modelli](#-Pesi/dataset)
- [⚖️ Licenza](#️-licenza)

---

## Contesto e Obiettivi

L'ispezione tradizionale dei difetti sub-superficiali nei compositi CFRP, spesso basata su Controlli Non Distruttivi (NDT), è un processo manuale, soggettivo e a bassa riproducibilità. L'obiettivo di questa tesi è stato lo sviluppo e la validazione di una pipeline computazionale **end-to-end** per automatizzare l'analisi del danno.

Partendo da un dataset sperimentale di scansioni radiografiche e dati di processo (forza di foratura), il lavoro si è focalizzato sullo sviluppo dell'architettura **ROIA (Radiograph-Only Integrated Analysis)**, una sequenza di moduli di Computer Vision e Machine Learning in grado di:
1.  Estrarre e normalizzare i dati da scansioni grezze eterogenee.
2.  Segmentare con alta precisione il foro e il danno da delaminazione.
3.  Estrarre un set robusto di feature quantitative e ingegneristicamente significative.
4.  Imputare dati di processo mancanti.
5.  Predire l'evoluzione futura del danno basandosi sulla storia pregressa.

---

## 📂 Struttura del Repository

Il repository è organizzato in cartelle che rispecchiano le fasi principali della pipeline di analisi.

---

## Flusso di Lavoro della Pipeline (ROIA)

La pipeline principale sviluppata (ROIA) opera esclusivamente sul dominio radiografico e concatena i seguenti moduli in una catena MLOps:

1.  **Rilevamento e Normalizzazione (Modulo `01`)**: Una rete **YOLOv8** localizza i fori nelle scansioni grezze. Un algoritmo di clustering K-Means impone un ordine **bustrofedico** riproducibile. Infine, un algoritmo di **cropping con normalizzazione di scala** genera un dataset di patch 512x512, garantendo che ogni foro abbia una dimensione apparente costante.

2.  **Segmentazione del Danno (Modulo `02`)**: Le patch normalizzate vengono processate da una rete **UNet++**. Il modello viene prima pre-addestrato su un vasto set di maschere generate automaticamente (`Mask Factory`) e poi specializzato tramite un **addestramento ibrido "foveale"**, che combina immagini intere (per il contesto) e patch di dettaglio 128x128 (per le micro-strutture del danno).

3.  **Estrazione Feature (Modulo `03`)**: Un algoritmo di **auto-calibrazione per-immagine** analizza le maschere di segmentazione. Usando il diametro noto del foro (6 mm) come "righello interno", converte le misure in pixel (aree, diametri, Momenti di Hu) in unità fisiche (mm², mm), garantendo robustezza metrologica.

4.  **Imputazione e Predizione (Modulo `04`)**:
    *   Un **Multi-Layer Perceptron (MLP)** viene usato per imputare i valori mancanti della forza di processo, basandosi sulle correlazioni con le feature geometriche del danno.
    *   Una rete **Long Short-Term Memory (LSTM)**, addestrata con una **Quantile Loss**, modella l'evoluzione sequenziale del danno. Questo non solo predice il trend, ma stima anche un intervallo di predizione (rischio), catturando eventi estremi.

---

## ⚙️ Installazione e Setup

Per eseguire gli script e i notebook in questo repository, si consiglia di creare un ambiente virtuale per gestire le dipendenze.

1.  **Clonare il repository:**
    ```bash
    git clone https://github.com/Riccardo-Venturi/Tesi_Script_Colab.git
    cd Tesi_Script_Colab
    ```

2.  **Creare un ambiente virtuale (consigliato):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Su Windows: venv\Scripts\activate
    ```

3.  **Installare le dipendenze:**
    È fortemente consigliato creare un file `requirements.txt` con le librerie necessarie. Le dipendenze principali includono:
    ```bash
    pip install torch torchvision ultralytics segmentation-models-pytorch albumentations opencv-python-headless pandas scikit-learn matplotlib lightgbm tensorflow
    ```

---

## 🚀 Utilizzo

Per riprodurre i risultati della tesi, eseguire i notebook e gli script seguendo l'ordine numerico delle cartelle:

1.  **Fase 1 (`01_data_preprocessing`):** Partire dalle scansioni grezze per generare le patch normalizzate e ordinate.
2.  **Fase 2 (`02_segmentation_unet++`):** Utilizzare le patch per addestrare il modello di segmentazione.
3.  **Fase 3 (`03_feature_extraction`):** Eseguire l'estrattore di feature sulle maschere prodotte.
4.  **Fase 4 (`04_predictive_modeling_lstm`):** Utilizzare il CSV finale delle feature per addestrare il modello predittivo.

**Nota:** I percorsi ai dati all'interno dei notebook sono hardcoded per l'ambiente Google Colab con Google Drive montato. Sarà necessario adattare questi percorsi al proprio setup locale.

---
## Pesi/modelli/dataset
-------------------------------------------------------------------------------------------------
If you look for weights and dataset, avaible on huggingface
# CFRP-ROIA: Machine Learning Pipelines for CFRP Damage Analysis

**Author:** Riccardo Venturi  cagliari 2025
**Thesis Repository (code only)** — This repository contains the scripts used for training and evaluating all pipelines (YOLOv8, UNet++, MLP–LSTM).

🧠 **Model Weights and Dataset**  
The trained weights and processed datasets are publicly available at:

> Venturi R. (2025). *CFRP-ROIA Weights and Datasets.* Hugging Face Hub.,
> url          = { https://huggingface.co/Riccardo99999/CFRP-ROIA-weights },
> doi          = { 10.57967/hf/6729 },

----------------------------------------------------------------------------------------------------
@misc{riccardoventuri_2025,
	author       = { RiccardoVenturi },
	title        = { RPOS-dataset (Revision 2709c43) },
	year         = 2025,
	url          = { https://huggingface.co/datasets/Riccardo99999/RPOS-dataset },
	doi          = { 10.57967/hf/6749 },
	publisher    = { Hugging Face }
}

---
## 📄 Citazione

Se utilizzi questo codice per la tua ricerca, ti preghiamo di citare il lavoro come segue:

```bibtex
@mastersthesis{Venturi2024CFRP,
  author    = {Riccardo Venturi},
  title     = {Studio comparativo di pipeline di deep learning per la valutazione automatica del danno da foratura in compositi CFRP},
  school    = {Università degli Studi di Cagliari},
  year      = {2024},
  address   = {Cagliari, Italia},
  month     = {Luglio},
  howpublished = {\\url{https://github.com/Riccardo-Venturi/Tesi_Script_Colab}}
}
65221
