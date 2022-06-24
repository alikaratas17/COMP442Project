import json
def JSONToInput():
  input_data_file = "./wmt16_en_tr/train.json"
  tr_sentences = []
  en_sentences = []
  with open(input_data_file,"r") as f:
    for line in f:
      item = json.loads(line.strip())['translation']
      tr_sentences.append(item['tr'])
      en_sentences.append(item['en'])
      
  #print(len(tr_sentences),len(en_sentences))
  with open("./wmt16_en_tr/trOriginal.txt","w") as f:
    for x in tr_sentences:
      f.write(x)
      f.write("\n")
  with open("./wmt16_en_tr/enOriginal.txt","w") as f:
    for x in en_sentences:
      f.write(x)
      f.write("\n")

def outputToJSON(tr_file,en_file,output_filename):
  tr = []
  en = []
  with open(tr_file,"r") as f:
    for line in f:
      tr.append(line.strip())
  with open(en_file,"r") as f:
    for line in f:
      en.append(line.strip())
  with open(output_filename,"w") as f:
    for i in range(len(tr)):
      item ={"translation":{"en":en[i],"tr":tr[i]}}
      json.dump(item,f,ensure_ascii=False)
      f.write("\n")

#JSONToInput()
outputToJSON("./trOutput.txt","./enOriginal.txt","./train_en_tr.json")
outputToJSON("./trOriginal.txt","./enOutput.txt","./train_tr_en.json")
