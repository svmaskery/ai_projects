from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch
from trl import setup_chat_format

# Initialize models
base_model = 'HuggingFaceTB/SmolLM2-135M-instruct'
new_model = './lora-sft-1/checkpoint-1500/'

# Reload tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model)

base_model_reload = AutoModelForCausalLM.from_pretrained(
        base_model,
        return_dict=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
)


# Merge adapter with base model
model = PeftModel.from_pretrained(base_model_reload, new_model)

model = model.merge_and_unload()

# messages = [{"role": "user", "content": "Hello doctor, I have bad acne. How do I get rid of it?"}]
instruction = "You are a helpful assistant. Be polite and respectful while answering questions."
messages = [{"role": "system", "content": instruction},
    {"role": "user", "content": "I'm very depressed. How do I find someone to talk to?"}]

prompt = tokenizer.apply_chat_template(messages, tokenize=False)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

outputs = pipe(prompt, max_new_tokens=120, do_sample=True, temperature=0.5, top_k=5)
print(outputs[0]["generated_text"])