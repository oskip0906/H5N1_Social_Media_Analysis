import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Read the CSV file using pandas
df = pd.read_csv('sentiment_analysis/model/training_dataset.csv', encoding='utf-8')
# print(df.head())

dataset = Dataset.from_pandas(df)

# Define training and testing split
dataset = dataset.train_test_split(test_size=0.2)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Six categories: sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5).

# Prepare the dataset for PyTorch
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Intilize the model and move it to the GPU
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
model.to(torch.device('cuda'))

# Define training arguments
training_args = TrainingArguments(
    eval_strategy='epoch',
    optim="adamw_torch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('sentiment_analysis/model/saved_model')
tokenizer.save_pretrained('sentiment_analysis/model/saved_tokenizer')