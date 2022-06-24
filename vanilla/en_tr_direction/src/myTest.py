from dataset import get_test_data#,getTestDataset
from architectures.machine_translation_transformer import MachineTranslationTransformer
import torch
from nltk.translate.bleu_score import corpus_bleu
from tokenizers import Tokenizer
from tqdm import tqdm
test_data = get_test_data()
run_name = "en_tr_vanilla"
model_path = "../runs/"+run_name+"/model_ckpt_best.pt"
#tst_dataset = getTestDataset()
tok_path = "../runs/"+run_name+"/tokenizer.json"
tokenizer = Tokenizer.from_file(tok_path)
model = MachineTranslationTransformer(d_model=512,n_blocks=6,src_vocab_size=32000,trg_vocab_size=32000,n_heads=4,d_ff=2048,dropout_proba=0.1)
model.load_state_dict(torch.load(model_path))
preds = []
reals = []
model.eval()
with torch.no_grad():
  for x in tqdm(test_data):
    reals.append(x['translation_trg'])
    preds.append(model.translate(x['translation_src'],tokenizer))
reals2 = [tokenizer.decode(tokenizer.encode(x).ids,skip_special_tokens=False).split(" ")[1:-1] for x in reals] # Removing BOS and EOS tokens
preds = [x.split(" ")[1:-1] for x in preds] # Removing BOS and EOS tokens
reals2 = [[x] for x in reals2]
BLEU2 = corpus_bleu(reals2,preds,[(1.0,),(0.5,0.5),(0.333,0.333,0.334),(0.25,0.25,0.25,0.25)])
print("BLEU SCORE FOR decode(encode()).split()")
print(BLEU2)
reals = [[x.split(" ")[1:-1]] for x in reals]
BLEU = corpus_bleu(reals,preds,[(1.0,),(0.5,0.5),(0.333,0.333,0.334),(0.25,0.25,0.25,0.25)])
print("BLEU SCORE FOR .split() [no encode decode for real data]")
print(BLEU)
