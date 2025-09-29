# Tesi_Script_Colab
# Studio Comparativo di Pipeline di Deep Learning per la Valutazione Automatica del Danno da Foratura in Compositi CFRP

Questo repository contiene il codice sorgente sviluppato per la tesi di Laurea Magistrale in Ingegneria Meccanica di **Riccardo Venturi**, discussa presso l'**Università degli Studi di Cagliari** nell'Anno Accademico 2023-2024.

Il progetto affronta la sfida della quantificazione automatica del danno indotto dalla foratura (delaminazione) in materiali compositi a fibra di carbonio (CFRP), un'analisi critica per settori ad alta tecnologia come l'aerospaziale e l'automotive.

---

## 📜 Indice

- [Contesto e Obiettivi](#contesto-e-obiettivi)
- [Struttura del Repository](#struttura-del-repository)
- [Flusso di Lavoro della Pipeline (ROIA)](#flusso-di-lavoro-della-pipeline-roia)
- [⚙️ Installazione e Setup](#️-installazione-e-setup)
- [🚀 Utilizzo](#-utilizzo)
- [📄 Citazione](#-citazione)
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
