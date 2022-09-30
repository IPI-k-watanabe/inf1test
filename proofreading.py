from typing import List

def make_tokens(text:str, tokenizer)->List:
    encoded_text = tokenizer(text)
    return tokenizer.convert_ids_to_tokens(encoded_text.input_ids)[1:-1]

def insert_mask_to_tokens(i: int, mask_token: str, tokens: List)-> List: # tokensのi番目にmask_tokenを挿入する
    masked_token=tokens.copy()
    masked_token.insert(i, mask_token)
    return masked_token

def replace_mask_to_tokens(i: int, mask_token: str, tokens: List)-> List: # tokensのi番目をmask_tokenに入れ替える
    masked_token=tokens.copy()
    masked_token[i] = mask_token
    return masked_token

def make_senetence_from_tokens(tokens:List)->str:
  sentence = "".join(tokens)
  def clean_sentence()-> str:
    return sentence.replace('#', '')
  return clean_sentence()