""""Hugging Face -> Github for AI Models
1 . Pipeline-> NLP Task-> Tokenize -> Data Set Load-> Fine Tune"""

import transformers
from transformers import pipeline
"""NLP Tasks -> Classification , Summarization , Translation 

How to do it Hugging face Models -> Read Docs 

TASK_VARIABLE=Pipeline ('TASK PERFORMED ', MODEL_NAME )"""


# # 1. Classification 
# classfy=pipeline("sentiment-analysis")
# result=classfy(sentence) simmilar for rest of tasks 
#---------------------------------------------------------------------------------------------------------------------------------------
# 2.Vectorization (Tokenization)

"""HOW ->  (we use Autotokenizer/pre trained tokenizer ) -> tokenize ( break words )->convert to ids (tokens) """

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sentence = "Hi My name is Anurag I love Dance"
tokens = tokenizer.tokenize(sentence)
print(tokens)

#TOKENS -> IDS 
ids=tokenizer.convert_tokens_to_ids(tokens)
print(ids)

#Decode IDS-> Tokens
decoded=tokenizer.decode(ids)
print(decoded)

#------------------------------------------------------------------------------------------------------------------------------------
# Data set load 
 
from datasets import load_dataset
dataset=load_dataset('imdb')
dataset
# https://huggingface.co/datasets/stanfordnlp/imdb 

# 1. Preprocessing Data set
def tokenize_function(example):
    return tokenizer(example['text'],padding='max_length',truncation=True)
toknize_data=dataset.map(tokenize_function,batched=True)
# use of map function -> {'text': 'This movie is amazing!', 'label': 1, 'input_ids': [101, 2023, 3185, 2003, 6429, 102]} we can also use for loop instead

# toknize_data['train'][0]

# 2. Providing Training Arguments   -> we will specify hyper parameters and training setting
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    eval_strategy ="epoch",     # Evaluate every epoch
    learning_rate=2e-5,              # Learning rate -> how fast model updates weight 
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=16,   # Batch size for evaluation
    num_train_epochs=1,              # Number of training epochs
    weight_decay=0.01,               # Strength of weight decay
)
training_args


# 3.Initialize the Model  
"""we need to use AutoModel Sequence Classification  as we are doing text classification """
from transformers import AutoModelForSequenceClassification, Trainer
from transformers import BertTokenizer

# Load the pre-trained model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=toknize_data['train'],
    eval_dataset=toknize_data['test']
)

trainer.train() # calling the train 
result=trainer.evaluate()
print(result)

