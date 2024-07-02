from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "google/mt-dnn-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
