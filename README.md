# SpiroFusion-Net: Multimodal Deep Learning on Raw Spirograms & Linked EHR

Official open-source implementation and supplementary repository for the **IEEE Journal of Biomedical and Health Informatics (J-BHI)** study:

> **SpiroFusion-Net: Cross-Modal Deep Learning on Raw Spirograms and EHR for COPD Detection, Blow Salvage, and Survival Prediction**
> *Md Emran Hasan, Yue-Fang Wu, and Dong-Jun Yu*.
> **Repository:** https://github.com/emrancub/SpiroFusion-Net

---

## Key Highlights
- **Dual Population Cohorts (`N = 462,825` combined spirograms):**
  - **Primary Discovery Cohort:** UK Biobank Application Number 419012 (`N = 453,186` valid spirograms; `22,392` COPD cases; `47,997` deaths over 16.0 years; held-out test set `N = 67,980`).
  - **Independent External Validation Cohort:** US CDC National Health and Nutrition Examination Survey (NHANES 2007–2012, Cycles E, F, G; `N = 9,639` multi-ethnic US adults; held-out test set `N = 4,820`).
- **Physiologically Grounded 4-Channel Waveform Tensor (`4 x 1,000`) & 12 Small-Airway Curve Biomarkers:**
  - Encodes absolute volume `V(t)`, fractional volume `V(t)/FVC`, expiratory flow `dV/dt`, and flow acceleration `d2V/dt2` alongside `FEF25-75%`, `FEF50%`, `FEF75%`, central/distal loop concavity indices, and `AUFVL`.
- **QC-Aware Bidirectional Cross-Modal Attention:**
  - Dynamically fuses a Multi-Scale 1D Squeeze-and-Excitation Residual Network (`SE-ResNet-1D`) with a Gated Residual Tabular Network (`GRN`), rescuing diagnostic performance in suboptimal/rejected blows (`EXPFLOW` / `BELOW6SEC` and NHANES QC Grades C & D).

## Main Results Summary
| Evaluation Benchmark | Clinical GOLD | LightGBM | 1D-Transformer (PatchTST) | **SpiroFusion-Net (Proposed)** |
| :--- | :---: | :---: | :---: | :---: |
| **UK Biobank Test AUROC (`N = 67,980`)** | 0.7821 | 0.8688 | 0.7988 | **0.8672** |
| **UK Biobank Test AUPRC** | 0.2125 | 0.4068 | 0.3055 | **0.4278** |
| **UK Biobank Suboptimal Blow Salvage AUROC (`N = 11,996`)** | 0.6864 | 0.8012 | 0.7640 | **0.8222** |
| **UK Biobank 16-Year Mortality C-Index (`47,997` deaths)** | 0.5980 | 0.7070 | 0.6204 | **0.7454** |
| **External US CDC NHANES Overall AUROC (`N = 4,820`)** | 0.8932 | 0.9197 | 0.8931 | **0.9296** |
| **External US CDC NHANES AUPRC** | 0.8099 | 0.8482 | 0.7980 | **0.8810** |
| **External US CDC NHANES Post-Bronchodilator AUROC** | 0.9832 | 0.9717 | 0.9774 | **0.9946** |
| **External US CDC NHANES Suboptimal Blow Salvage AUROC** | 0.8985 | 0.9600 | 0.9073 | **0.9453** |

## Repository Structure
- `config.py`: Central configuration for UK Biobank and US CDC NHANES schemas and hyperparameters.
- `data/`: 4-channel 10-ms waveform preprocessing (`preprocessing.py`), 12 small-airway curve biomarker extraction, and PyTorch `Dataset` loader (`dataset.py`).
- `models/`: `SpiroFusionNet` architecture (`multimodal_net.py`) and 11 clinical, classical ML, GBDT, and unimodal deep baselines (`tabular_baselines.py`).
- `training/`: Multi-task Focal-BCE + FP32 Cox proportional hazards trainer (`trainer.py`).
- `evaluation/`: Bootstrap 95% CIs, DeLong tests, C-index, Bayes prior calibration (`metrics.py`), and publication figure/table generators (`publication_outputs.py`).
- `run_nhanes_external_validation.py`: Automated downloader and PCHIP curve builder for US CDC NHANES 2007–2012 (`.XPT` modules).
- `run_pipeline.py`: End-to-end training, evaluation, external validation, and figure/table generation script.
- `Supplementary_Material_IEEE_JBHI.pdf`: Extended supporting tables (`Table S1`–`Table S4`, including SOTA comparison) and figures (`Fig. S1`–`Fig. S8`).
- `outputs/figures/` & `outputs/tables/`: All 15 figures (`.pdf` and `.png`) and 10 CSV tables.

## Data Availability & Acknowledgments
All source code, figures, and supplementary material required to reproduce the main results of this study are freely available at https://github.com/emrancub/SpiroFusion-Net. Individual-level primary biomedical datasets are available from the UK Biobank (https://www.ukbiobank.ac.uk/) upon formal research application and approval (UK Biobank Project Application Number 419012), and the external validation datasets are publicly available from the US CDC National Health and Nutrition Examination Survey (https://wwwn.cdc.gov/nchs/nhanes/).

## 📜 Citation

If you use this framework or ideas from this work, please cite:

> The final BibTeX entry and author list will be updated after acceptance.


## 📧 Contact

For questions, feedback, or collaboration:

* **Md Emran Hasan** – [mdemranhasan@njust.edu.cn](mailto:mdemranhasan@njust.edu.cn) or/and, **Dong-Jun Yu** – [njyudj@njust.edu.cn](mailto:njyudj@njust.edu.cn)

You are welcome to open an issue or pull request if you have suggestions or would like to contribute improvements after the journal’s decision.