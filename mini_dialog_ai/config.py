class ModelConfig:
    MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"
    IMAGE_MODEL_NAME = "Salesforce/blip-image-captioning-small"
    DEVICE = "cpu"
    MAX_LENGTH = 256
    TEMPERATURE = 0.7
    TOP_P = 0.9
    BATCH_SIZE = 1
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 1
    GRADIENT_ACCUMULATION_STEPS = 2
    QUANTIZATION_BITS = None
    USE_MOE = False