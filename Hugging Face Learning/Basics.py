"""Hugging Face -> Github for AI Models
1 . Pipeline-> NLP Task-> Tokenize -> Data Set Load-> Fine Tune"""

import transformers
from transformers import pipeline
"""NLP Tasks -> Classification , Summarization , Translation 

How to do it Hugging face Models -> Read Docs 

TASK_VARIABLE=TASK_NAME('TASK PERFORMED ', MODEL_NAME )"""


# # 1. Classification 
# classfy=pipeline("sentiment-analysis")
# result=classfy(sentence) simmilar for rest of tasks 

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




 