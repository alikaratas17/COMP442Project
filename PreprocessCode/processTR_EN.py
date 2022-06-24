import numpy as np

input_file_name = "trOriginal.txt"
text = []
with open(input_file_name,"r") as f:
  for x in f:
    text.append(x.strip())
np_text = np.array(text)

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
tokenizer = MBart50TokenizerFast.from_pretrained("mbart50-scratch-wmt16-tr-en")
model = MBartForConditionalGeneration.from_pretrained("mbart50-scratch-wmt16-tr-en")
model = model.to('cuda')
total = np_text.shape[0]
B = 32
translations = []
i = 0
with torch.no_grad():
  while True:
    s = i * B
    e = s + B
    if e > total:
      e = total
    current = np_text[s:e]
    print("{}%".format(100.*i*B/total))
    pt_batch = tokenizer(current.tolist(), padding=True,  return_tensors="pt").to('cuda')
    generated_tokens = model.generate(**pt_batch).to('cpu')
    outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    translations = translations + outputs
    i +=1
    if e == total:
      break
fout = open("enOutput.txt","w")
for t in translations:
  fout.write(t) # Removed .lower()
  fout.write("\n")
fout.close()
