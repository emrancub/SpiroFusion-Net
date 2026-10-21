# -*- coding: utf-8 -*-
import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Primary dataset path inside this project: data/data_participant.csv
    LOCAL_DATA_PATH = os.path.join(BASE_DIR, "data", "data_participant.csv")
    SAMPLE_DATA_PATH = os.path.join(BASE_DIR, "data", "data_participant_sample_5913.csv")

    # External US CDC NHANES (2007-2012) directory & mapped CSV path
    NHANES_DIR = os.path.join(BASE_DIR, "nhanes_external_data")
    NHANES_CSV_PATH = os.path.join(NHANES_DIR, "nhanes_mapped_ukb_schema.csv")

    # Candidate paths on csbio3090 server for the full UK Biobank CSV
    SERVER_CANDIDATE_PATHS = [
        "/home/emran/data_participant_full_backup.csv",
        "/home/emran/ukb_spiro_multimodal_ml/data/data_participant.csv",
        "/home/emran/data_participant.csv",
        os.path.expanduser("~/data_participant_full_backup.csv"),
        os.path.expanduser("~/ukb_spiro_multimodal_ml/data/data_participant.csv"),
        os.path.expanduser("~/data_participant.csv"),
    ]

    @classmethod
    def resolve_data_path(cls, custom_path=None):
        """
        Resolves the primary dataset path:
        1. If user passed --data_path <path> and it exists, use that directly.
        2. Otherwise, check local data/data_participant.csv and SERVER_CANDIDATE_PATHS
           and automatically select the largest available file (the full UKB cohort).
        3. Fall back to data/data_participant_sample_5913.csv if no primary file is found.
        """
        if custom_path and os.path.exists(custom_path):
            return os.path.abspath(custom_path)

        candidates = []
        if os.path.exists(cls.LOCAL_DATA_PATH):
            candidates.append((os.path.getsize(cls.LOCAL_DATA_PATH), cls.LOCAL_DATA_PATH))
        for sp in cls.SERVER_CANDIDATE_PATHS:
            if os.path.exists(sp):
                candidates.append((os.path.getsize(sp), sp))
        if os.path.exists(cls.SAMPLE_DATA_PATH):
            candidates.append((os.path.getsize(cls.SAMPLE_DATA_PATH), cls.SAMPLE_DATA_PATH))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
        return cls.LOCAL_DATA_PATH

    DATA_PATH = LOCAL_DATA_PATH
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
    TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")

    ID_COL = "eid"
    WAVEFORM_COL = "p3066_i0_a0"

    CONT_COLS = [
        "p21003_i0",    # Age at assessment (years)
        "p21001_i0",    # BMI (kg/m2)
        "p20161_i0",    # Pack years of smoking
        "p3062_i0_a0",  # FVC (Litres)
        "p3063_i0_a0",  # FEV1 (Litres)
        "p3064_i0_a0",  # PEF (L/min)
        "p20150_i0",    # FEV1 best measure (Litres)
        "p20151_i0",    # FVC best measure (Litres)
        "p20256",       # FEV1 GLI Z-score
        "p20257",       # FVC GLI Z-score
        "p20258",       # FEV1/FVC GLI Z-score
        "p42039",       # GP prescription count
        "p42040"        # GP clinical event record count
    ]

    CAT_COLS = [
        "p21000_i0",    # Ethnic background
        "p20116_i0",    # Smoking status (Never, Previous, Current)
        "p2129_i0",     # Spirometry completion status
        "p20031_i0_a0", # Blow quality error flag (ACCEPT, EXPFLOW, BELOW6SEC)
        "p20002_i0_a0"  # Baseline self-reported non-cancer comorbidity
    ]

    TARGET_COPD_DATE = "p131492"
    TARGET_COPD_SOURCE = "p131493"
    TARGET_DEATH_DATE = "p40000_i0"
    TARGET_DEATH_CAUSE = "p40001_i0"

    TARGET_WAVEFORM_LENGTH = 1000
    SAMPLING_INTERVAL_MS = 10
    # 4 Physiological Channels:
    # Ch0: Absolute Volume V(t)/6.0L, Ch1: Relative Volume V(t)/FVC,
    # Ch2: Expiratory Flow Rate dV/dt (L/s), Ch3: Flow Acceleration d2V/dt2
    NUM_WAVEFORM_CHANNELS = 4

    CNN_EMBED_DIM = 192
    TABULAR_EMBED_DIM = 192
    FUSION_EMBED_DIM = 192
    DROPOUT = 0.20

    SEED = 42
    BATCH_SIZE = 128
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
