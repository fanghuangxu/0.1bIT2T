class ModelConfig:
    MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
    IMAGE_MODEL_NAME = "Salesforce/blip-image-captioning-small"
    DEVICE = "cpu"
    MAX_LENGTH = 512
    TEMPERATURE = 0.7
    TOP_P = 0.9
    BATCH_SIZE = 2
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    GRADIENT_ACCUMULATION_STEPS = 4
    QUANTIZATION_BITS = 4
    USE_MOE = True