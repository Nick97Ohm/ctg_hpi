import subprocess
import argparse
import os
import bert_score
import logging
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer
import numpy as np 
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

logging.basicConfig(level='ERROR')
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description="metrics")
parser.add_argument("--checkpoint_dir", type=str)
parser.add_argument("--clsf_name", type=str, default="textattack/bert-base-uncased-yelp-polarity")
parser.add_argument("--ext_clsf_name", type=str, default="textattack/bert-base-uncased-yelp-polarity")
parser.add_argument("--attr_file", type=str, default="/home/user/dir_projects/dir.bert_sample/sent_anlys/batched_MH/data/yelp/test_li.attr")
parser.add_argument("--text_file", type=str, default="/home/user/dir_projects/dir.bert_sample/sent_anlys/batched_MH/data/yelp/test_li.txt")
parser.add_argument("--ref_file", type=str, default="/home/user/dir_projects/dir.bert_sample/sent_anlys/batched_MH/data/yelp/test_li_reference.txt")
parser.add_argument("--ext_clsf", action="store_true")
parser.add_argument("--reverse", action="store_true")
parser.add_argument("--form_em_lstm", action="store_true")


args = parser.parse_args()


cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

CLS = "[CLS]"
SEP = "[SEP]"
EOT_TOKEN = '<|endoftext|>'

# Calculating the GPT-2 perplexity score
def perplexity_fudge(checkpoint_dir,file_name='opt_samples.txt'):
    # calculate perplexity 
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to(device)
    gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with torch.no_grad():
        with open(f"{checkpoint_dir}/{file_name}", "r") as input_file:
                ppl = []
                sos_token = gpt_tokenizer.decode([0])
                for line in input_file:
                    sentence=line[:-1]
                    sentence=sentence.lower()
                    sentence = tokenizer.tokenize(sentence)
                    sentence = tokenizer.convert_tokens_to_ids(sentence)
                    sentence = tokenizer.decode(sentence)
                    sentence=sentence.replace(' \' ','\'')
                    sentence=sentence.replace(' - ','-')
                    sentence=sentence.replace(' .','.')
                    sentence=sentence.replace(' ,',',')
                    
                    full_tensor_input = gpt_tokenizer.encode(sos_token + sentence.replace(EOT_TOKEN, ' ').strip(), return_tensors='pt').to(device)
                    full_loss = gpt_model(full_tensor_input, labels=full_tensor_input)[0].mean()
                    ppl.append(torch.exp(full_loss).flatten().cpu().item())
    return np.median(ppl)



def get_external_cls(checkpoint_dir, disc_name, file_name='opt_samples.txt', out_file_name='opt_cls_ext.txt'):
    tokenizer_disc = BertTokenizer.from_pretrained(disc_name)  # finiteautomata/bertweet-base-sentiment-analysis ("textattack/bert-base-uncased-imdb")
    model_disc = AutoModelForSequenceClassification.from_pretrained(disc_name).to(device) #("textattack/bert-base-uncased-imdb").to(device)

    with open(f"{checkpoint_dir}/{out_file_name}", "w+") as f_pred_attr,open(f"{checkpoint_dir}/{file_name}", "r") as input_file:
        for i,line in enumerate((input_file)):
            
            seed_text = line[:-1].lower()

            seed_text = tokenizer_disc.tokenize(seed_text)
            batch = torch.tensor(get_init_text(seed_text, tokenizer_disc=tokenizer_disc, max_len=15, batch_size=1)).to(device)

            pred = disc_cls(batch,model_disc,tokenizer_disc)
            f_pred_attr.write(str(pred[0])+'\n')
            f_pred_attr.flush()


# Calculate Hamming-Distance
def get_hamming_dist_len_diff(checkpoint_dir, disc_name, src_file, file_name='opt_samples.txt', out_file_name='opt_cls_ext.txt'):
    tokenizer_disc = BertTokenizer.from_pretrained(disc_name)
   
    cnt= 0
    dist =0

    with open(f"{checkpoint_dir}/{file_name}", "r") as input_file, open(f"{src_file}", "r") as src_file:
        for i,(line, line_src) in enumerate(zip(input_file, src_file)):
            
            text = line[:-1].lower()
            line_src = line_src[:-1].lower()

            text = tokenizer_disc.tokenize(text)
            line_src = tokenizer_disc.tokenize(line_src)
            
            if len(text) == len(line_src):
                dist += sum([a !=b for (a,b) in zip(text,line_src)])

            
            else:
                dist += max(len(text), len(line_src))
                
                
                if len(line_src) > len(text):
                    for element in text:
                            if element in line_src:
                                dist -= 1

                else:
                    for element in line_src:
                        if element in text:
                            dist -= 1

            cnt+= 1

    return dist/cnt


def tokenize_batch(batch,tokenizer_disc):
    return [tokenizer_disc.convert_tokens_to_ids(sent) for sent in batch]

def get_init_text(seed_text, max_len, tokenizer_disc, batch_size=1):
    batch = [[CLS]+ seed_text  + [SEP]  for _ in range(batch_size)]

    return tokenize_batch(batch, tokenizer_disc)

def disc_cls(batch,model_disc,tokenizer_disc):

  output = model_disc(batch)['logits']
  pred = np.argmax(np.array(output.log_softmax(dim=-1).cpu().detach()),axis=-1)

  return pred


def get_cls_scores(attr_file,transfered_attr,file_name='opt_cls.txt', reverse = False):
    all_sp_cor =[0,0]
    all_sp_cnt = [0,0]
    all_cor = 0
    all_cnt = 0

    with open(attr_file, "r") as attr , open(f"{transfered_attr}/{file_name}", "r") as transfered_attr :
        for attr,trans in zip (attr,transfered_attr):
            trg = 1- int(attr[:-1])
            if reverse:
                
                tran= 1-int(trans[:-1])
                
            else:
                tran = int(trans[:-1])

            all_cnt +=1
            all_sp_cnt[trg] += 1


            if (tran == trg):
                all_cor +=1
                all_sp_cor[tran] += 1

    _0_acc = all_sp_cor[0]/all_sp_cnt[0] if all_sp_cnt[0] else 0
    _1_acc = all_sp_cor[1]/all_sp_cnt[1] if all_sp_cnt[1] else 0
    return all_cor/all_cnt

# Calculating Bert-Score
def get_bert_score(src_file,transfile):
    bert_scorer = bert_score.BERTScorer(lang='en')
    avg =0
    cnt = 0
    with open(src_file, "r") as src , open(f"{transfile}/opt_samples.txt", "r") as trans :
        for src_sample, trans_sample in zip (src,trans):
            P, R, F1 = bert_scorer.score([trans_sample[:-1]], [src_sample[:-1]], verbose=False, batch_size=1)
            avg += F1
            cnt += 1

    return avg.item()/cnt


# Getting necessary paths
checkpoint_dir = args.checkpoint_dir
attr_file = args.attr_file
txt_file= args.text_file
ref_file= args.ref_file



# Internal Classifier Accuracy
get_external_cls(checkpoint_dir, args.clsf_name, file_name='opt_samples.txt',out_file_name='opt_cls_met.txt')
acc = get_cls_scores(attr_file, checkpoint_dir,file_name='opt_cls_met.txt')

# External Classifier Accuracy
acc_ext = 1
if args.ext_clsf or args.lstm_clsf:
    if not os.path.exists(f'{checkpoint_dir}/opt_cls_ext.txt'):
        get_external_cls(checkpoint_dir, args.ext_clsf_name, file_name='opt_samples.txt')
    elif True:
        get_external_cls(checkpoint_dir, args.ext_clsf_name, file_name='opt_samples.txt')
    
    acc_ext = get_cls_scores(attr_file,checkpoint_dir,file_name='opt_cls_ext.txt', reverse = args.reverse)


# Hamming-Distance
dist_ham = get_hamming_dist_len_diff(checkpoint_dir, args.clsf_name, src_file=args.text_file)

# GPT-2 perplexity
gpt_mean_score = perplexity_fudge(checkpoint_dir,file_name='opt_samples.txt')

# Bertscore
bertsc = get_bert_score(txt_file,checkpoint_dir)


with open(f"{checkpoint_dir}/metrics.txt", "w+") as file:
    file.write(','.join([str(acc), str(acc_ext),str(dist_ham),str(gpt_mean_score), str(bertsc)]))
    file.flush()
    file.close()
    print(','.join([str(acc), str(acc_ext),str(dist_ham), str(gpt_mean_score), str(bertsc)]))

        



