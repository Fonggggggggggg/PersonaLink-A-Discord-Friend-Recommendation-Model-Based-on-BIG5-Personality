#hugging face website
#https://huggingface.co/KevSun/Personality_LM
#Do note that using longer, more complex sentences is likely to show more variation in the outputs.
#Be sure to have the transformers model installed before running it
#re and warnings are optional
import warnings
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
#import re
import os
import torch

#It is normal that error will still appear, just ignore it
warnings.filterwarnings('ignore')


###############Prediction model settings##########################
model = AutoModelForSequenceClassification.from_pretrained("KevSun/Personality_LM", ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained("KevSun/Personality_LM")
##################################################################


class TraitsResult:
    def __init__(self, txt_file_path, tokenizer, model):
        self.txt_file_path = txt_file_path
        self.tokenizer = tokenizer
        self.model = model

        if not os.path.exists(self.txt_file_path):
            # File doesn't exist, create it 
            print(f"File '{self.txt_file_path}' created.")
        else:
            print(f"File '{self.txt_file_path}' already exists.")

        with open(self.txt_file_path, 'r', encoding='utf-8') as file:
            new_text = file.read()

        # Encode the text using the same tokenizer used during training
        self.encoded_input = tokenizer(new_text, return_tensors='pt', padding=True, truncation=True, max_length=64)

    def predict(self):
        with torch.no_grad():
            outputs = self.model(**self.encoded_input)

        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_scores = predictions[0].tolist()

        trait_names = ["agreeableness", "openness", "conscientiousness", "extraversion", "neuroticism"]

        traitsResults = []
        for trait, score in zip(trait_names, predicted_scores):
            traitsResult = f"{trait}: {score*100:.1f}"
            traitsResults.append(traitsResult)

        return traitsResults

# Create an instance of TraitsResult for testing the model
#traits_result = TraitsResult(txt_file_path='fong0.0.txt', tokenizer=tokenizer, model=model)
# Call the predict method on the instance
#result = traits_result.predict()
#print(result)