from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import torch_neuron

tokenizer = BertTokenizer.from_pretrained('twmkn9/bert-base-uncased-squad2')
model = BertForQuestionAnswering.from_pretrained(
    'twmkn9/bert-base-uncased-squad2', return_dict=False)

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
inputs = tokenizer(question, text, return_tensors='pt')

max_length = 128

neuron_model = torch.neuron.trace(
    model,
    example_inputs=(
        inputs['input_ids'], inputs['attention_mask']),
    padding="max_length",
    max_length=max_length,
    truncation=True,
    verbose=1)

outputs = neuron_model(*(inputs['input_ids'], inputs['attention_mask']))

print(outputs)
