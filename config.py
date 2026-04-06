# Project
PROJECT_NAME = "Suicidal Ideation Detection in Social Media"
DESCRIPTION = "Applied NLP and machine learning on Twitter/Reddit data to detect suicidal ideation, classifying posts into three risk levels to enable early mental health intervention."

# Paths
DATA_DIR = "data/"
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"
OUTPUTS_DIR = "outputs/"
MODELS_DIR = "outputs/models/"
PLOTS_DIR = "outputs/plots/"
RESULTS_DIR = "outputs/results/"

# Dataset
DATASET_FILENAME = "social_media_suicide_detection.csv"
SYNTHETIC_FILENAME = "synthetic_social_media_detection.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "risk_level"

# Risk Level Mapping
RISK_LEVELS = {0: "low_risk", 1: "moderate_risk", 2: "high_risk"}
RISK_LABELS = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
RISK_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
NUM_CLASSES = 3

# Preprocessing
MIN_TEXT_LENGTH = 3
REMOVE_STOPWORDS = True
LEMMATIZE = True
HANDLE_SOCIAL_MEDIA = True  # handle @mentions, #hashtags, RTs

# TF-IDF
TFIDF_MAX_FEATURES = 50000
TFIDF_NGRAM_RANGE = (1, 2)

# Classical ML
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Deep Learning
MAX_SEQUENCE_LENGTH = 128
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
PATIENCE = 3

# Transformer
TRANSFORMER_MODEL = "distilbert-base-uncased"
TRANSFORMER_MAX_LENGTH = 128
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_EPOCHS = 3
TRANSFORMER_LR = 2e-5

# Logging
LOG_LEVEL = "INFO"
