from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.neuron

# BERTモデル及び、tokenizerのの読み込み
bert_path = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = AutoTokenizer.from_pretrained(bert_path)
tokenizer2 = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained(bert_path, return_dict=False)


# コンパイル時に使用するダミーtext。モデルに合わせたtokenizer.mask_tokenを使用すること
sample_input = f"これはtrace時に{tokenizer.mask_token}使うためのダミーテキストですねこれは激熱"

# 入力の最大系列長を事前に指定する
# inputs['input_ids'].size() -> torch.Size([1, 256])
# に統一している。もっと長い入力でコンパイルをかけたい場合は、max_lengthを変更して
max_length = 128
inputs = tokenizer.encode_plus(
   text=sample_input,
   return_tensors="pt",
   padding="max_length",
   max_length=max_length,
   truncation=True
)
tuple_inputs = inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]
print("input_ids:", inputs["input_ids"].size())
print("attention_mask:", inputs["attention_mask"].size())

# コンパイル (Neuron)
model_neuron = torch.neuron.trace(model, tuple_inputs)
# torch.jitの場合 (CPU)
# traced_model = torch.jit.trace(model, example_input, strict=False)

# neuton実行用モデルの保存
# model_neuron.save("bert_neuron.pt")

#実験
outputs = model_neuron(*tuple_inputs)
print(outputs)