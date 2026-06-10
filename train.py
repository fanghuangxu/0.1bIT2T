import os
import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from mini_dialog_ai.hf_dataset import HFDataProcessor

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def main():
    parser = argparse.ArgumentParser(description="Train Qwen MoE Dialog Model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-MoE-2.7B-Instruct", help="Model name")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrachat_200k", help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="./models/qwen-moe-dialog", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--quantization", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_name}")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ) if args.quantization else None
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map=args.device,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print(f"Loading dataset: {args.dataset_name}")
    dataset = HFDataProcessor.load_text_dataset(args.dataset_name, tokenizer, args.max_length)
    dataloader = HFDataProcessor.create_dataloader(dataset, batch_size=args.batch_size)
    
    print(f"Dataset size: {len(dataset)}")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training complete!")
    
    test_inference(model, tokenizer)

def test_inference(model, tokenizer):
    print("\nTesting inference...")
    model.eval()
    
    messages = [
        {"role": "system", "content": "你是一个友好的助手。"},
        {"role": "user", "content": "你好！"}
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
    print(f"AI: {response}")

if __name__ == "__main__":
    main()