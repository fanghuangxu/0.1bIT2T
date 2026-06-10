import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def load_model(model_path):
    print(f"Loading model from: {model_path}")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    return model, tokenizer

def chat(model, tokenizer, prompt):
    model.eval()
    
    messages = [
        {"role": "system", "content": "你是一个友好的助手。"},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(text, "").strip()
    return response

def main():
    model_path = "./models/qwen-moe-dialog"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using: python train.py")
        return
    
    model, tokenizer = load_model(model_path)
    
    print("Model loaded! Start chatting...")
    print("Type 'exit' to quit.\n")
    
    while True:
        prompt = input("You: ")
        if prompt.lower() == 'exit':
            print("Goodbye!")
            break
        
        response = chat(model, tokenizer, prompt)
        print(f"AI: {response}\n")

if __name__ == "__main__":
    main()