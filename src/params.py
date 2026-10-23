INPUT_FOLDER = "../input/"
DATA_FOLDER = "../data/"            # files kept in the repository (labels, ROI boxes, knee sides)
LOG_FOLDER = "../logs/"
OUTPUT_FOLDER = "../output/"

TRAIN_CSV = INPUT_FOLDER + "train.csv"
LABELS_FOLDER = DATA_FOLDER + "labels/"
CLAUDE_LABELS_CSV = LABELS_FOLDER + "full_claude_context_v2_labels.csv"
SERIES_INDEX_CSV = OUTPUT_FOLDER + "series_index.csv"

# Raw series live in OUTPUT_FOLDER/series_npy as (slices, rows, cols) uint16/int16,
# 523 GB on NFS. Training reads a uint8 copy resized to CACHE_SIZE instead.
CACHE_SIZE = 384
CACHE_MAX_SLICES = 64
CACHE_DIR = OUTPUT_FOLDER + f"cache_{CACHE_SIZE}/"

CLASSES = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion",
    "Fracture",
]

ZONES = {
    "ACL": "anterior cruciate ligament",
    "MCL": "medial collateral ligament",  # also called tibial collateral ligament
    "Medial Meniscus": "medial meniscus",
    "Lateral Meniscus": "lateral meniscus",
    "Medial OA": "medial compartment",
    "Lateral OA": "lateral compartment",
    "PF OA": "patellofemoral compartment",
    "Effusion": "capsule",  # Excell of fluid
    "Synovitis": "capsule",  # Inflammation of lining
    "Baker's": "behind the knee",
    "Contusion": "bone",
    "Fracture": "bone",
}

NUM_CLASSES = len(CLASSES)

PLANES = ["Sagittal", "Coronal", "Axial"]
