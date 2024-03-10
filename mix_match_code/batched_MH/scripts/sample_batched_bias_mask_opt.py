import os
import random

import numpy as np
import transformers
from numpy.core.fromnumeric import shape
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, BertTokenizer, \
    BertModel, GPT2LMHeadModel, GPT2TokenizerFast
from torch.distributions.categorical import Categorical
from datetime import datetime
import random
import pandas as pd
from pyplexity import PerplexityModel

import argparse

################

parser = argparse.ArgumentParser(description="style transfer")

parser.add_argument("--dict_path", type=str, help="dir", default='./data/bias/agency_power.csv')

parser.add_argument("--max_iter", type=int, help="number of changes to make in the gibbs chain", default=100)
parser.add_argument("--n_samples", type=int, help="number of changes to make in the gibbs chain", default=20)
parser.add_argument("--batch_size", type=int, help="number of changes to make in the gibbs chain", default=20)

parser.add_argument("--temperature", type=float, help="number of changes to make in the gibbs chain", default=1.0)
parser.add_argument("--degenerate", action='store_true')
parser.add_argument("--block", action='store_true')
parser.add_argument("--shuffle_positions", action='store_true')

###degenerate gibbs sampler
parser.add_argument("--top_k", type=int, help="top_k sampler-so far only degenerate support", default=40)
parser.add_argument("--burnin", type=int, help="burn in for degenerate support", default=250)

parser.add_argument("--data_path", type=str, help="dir", default='./data/yelp')
parser.add_argument("--src_path", type=str, help="dir", default='./data/yelp')
parser.add_argument("--attr_path", type=str, help="dir", default='./data/yelp')

parser.add_argument("--data_name", type=str, help="dir", default='yelp')

parser.add_argument("--out_path", type=str, help="dir", default='./batched')
parser.add_argument("--model_path", type=str, help="dir", default='bert-base-uncased')
parser.add_argument("--tok_path", type=str, help="dir", default='bert-base-uncased')

# disc

parser.add_argument("--disc_name", type=str, help="disc dir", default='imdb')
parser.add_argument("--disc_dir", type=str, help="disc dir", default='textattack/bert-base-uncased-imdb')

# hyper params
parser.add_argument("--alpha", type=float, help="knob", default=1)  # disc
parser.add_argument("--beta", type=float, help="knob", default=1)
parser.add_argument("--delta", type=float, help="knob", default=1)  # hamming
parser.add_argument("--gamma", type=float, help="knob", default=0)  # bluert score
parser.add_argument("--theta", type=float, help="knob", default=0)  # bertscore

args = parser.parse_args()

##################

cuda = torch.cuda.is_available()
print(cuda)
device = 'cuda' if cuda else 'cpu'

# Load pre-trained model (weights)
model_version = args.model_path  # os.environ["MODEL_PATH"]
model = AutoModelForMaskedLM.from_pretrained(model_version)
model.eval()

# perplexity model
model_per = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")

model_perp = PerplexityModel.from_str("bigrams-cord19")

if cuda:
    model = model.cuda()

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained(args.tok_path)

if args.gamma:
    model_bleurt = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512")
    model_bleurt.eval()

    if cuda:
        model_bleurt = model_bleurt.cuda()

if args.theta:
    model_bert = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    model_bert.eval()

    if cuda:
        model_bert = model_bert.cuda()

global_max_value = None
global_min_value = None


def min_max_normalization(array):
    global global_min_value
    global global_max_value

    if global_max_value is None or np.max(array) > global_max_value:
        global_max_value = np.max(array)

    if global_min_value is None or np.min(array) < global_min_value:
        global_min_value = np.min(array)

    normalized_array = (array - global_min_value) / (global_max_value - global_min_value)

    return normalized_array


global_max_value_disc = None
global_min_value_disc = None


def min_max_normalization_disc(array):
    global global_min_value_disc
    global global_max_value_disc

    if global_max_value_disc is None or np.max(array) > global_max_value_disc:
        global_max_value_disc = np.max(array)

    if global_min_value_disc is None or np.min(array) < global_min_value_disc:
        global_min_value_disc = np.min(array)

    normalized_array = (array - global_min_value_disc) / (global_max_value_disc - global_min_value_disc)

    return normalized_array


def get_agency_dict(agency_file):
    agency_dict = pd.read_csv(agency_file, header=None, index_col=0)
    agency_dict.squeeze().to_dict()

    agency_dict = agency_dict[1]
    agency_dict_main = {}
    for i, key in enumerate(agency_dict.keys()):
        # print(key)
        if i == 0:
            continue
        agency_dict_main[key[:-1]] = agency_dict[key]

    ag_dict = {'0': [], '1': [], '2': []}
    dict_src = {'agency_neg': '0', 'agency_pos': '1', 'agency_equal': '2'}

    for key in agency_dict_main.keys():
        if (agency_dict_main[key] == agency_dict_main[key]):
            ag_dict[dict_src[agency_dict_main[key]]].append(key)

    return ag_dict


def get_opt_sent(sents, metadata):
    min_score = 10000
    ind = 0
    meta_array = np.array(metadata)

    ind = np.argmin(meta_array[:, 1, ...])
    val = np.min(meta_array[:, 1, ...])
    sent_best = sents[ind].split()
    return " ".join(sent_best[1:-1]), meta_data[ind][-4], ind


def tokenize_batch(batch):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]


def untokenize_batch(batch):
    return [tokenizer.convert_ids_to_tokens(list(sent.to('cpu').numpy())) for sent in batch]


def detokenize(sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent


CLS = "[CLS]"
SEP = "[SEP]"
MASK = "[MASK]"
mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]


# mr_id = 2720 #tokenizer.convert_tokens_to_ids("mr")[0]
# ms_id = 5796 #tokenizer.convert_tokens_to_ids("ms")[0]


def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
    """Generate a word from from out[gen_idx]

    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k
    """
    logits = out[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)
    return idx.tolist() if return_list else idx


def get_init_text(seed_text, max_len, batch_size=1, rand_init=False):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [[CLS] + seed_text + [SEP] for _ in range(batch_size)]  # TODO

    return tokenize_batch(batch)


def printer(sent, should_detokenize=True):
    if should_detokenize:
        sent = detokenize(sent)[1:-1]
    print(" ".join(sent))


def to_file(sents, file):
    with open(file, "a") as f:
        f.write("\n".join(sents) + "\n")


def lexicon_dec(batch, dict_ag, target):
    sents = untokenize_batch(batch)
    cnt = [0] * batch.shape[0]
    for verb in dict_ag[str(target)]:
        for i, sent in enumerate(sents):
            if verb in sent:
                cnt[i] += 1

    # print(np.array(cnt))
    return np.array(cnt)


# Generation modes as functions
import math
import time


#   return  score

def tokens_to_sentences(tokenized_sentences):
    sentences = []

    for tokenized_sentence in tokenized_sentences:
        # Remove [CLS] and [SEP] tokens
        sentence_tokens = [token for token in tokenized_sentence if token not in ['[CLS]', '[SEP]']]
        # Join the remaining tokens to form a sentence
        sentence = ' '.join(sentence_tokens)
        # Append the reconstructed sentence to the list
        sentences.append(sentence)

    return sentences


def perplexity_score(batch, beta=1):
    start_time_per_bert = time.time()
    untoken = untokenize_batch(batch)
    sentences = tokens_to_sentences(untoken)

    per = []
    for sentence in sentences:
        print(sentence)

        perpl = model_perp.compute_sentence(sentence)
        per.append(perpl)

    per = min_max_normalization(per)
    print("Finished Perp score in %.3fs" % (time.time() - start_time_per_bert))
    print(per)

    return [perplexity * beta for perplexity in per], [perplexity * beta for perplexity in per]


def energy_score_mlm(batch, beta=1):
    seq_len = len(batch[0]) - 2
    posns = [i + 1 for i in range(seq_len)]

    norm_score = [0.0] * batch.shape[0]
    raw_score = [0.0] * batch.shape[0]
    for posn in posns:
        old_wrd = batch[:, posn].clone()

        batch[:, posn] = mask_id

        output = model(batch)['logits'][:, posn, :]

        norm_output = output.log_softmax(dim=-1)
        for i in range(batch.shape[0]):
            raw_score[i] += output[i, old_wrd[i]].item()
            norm_score[i] += norm_output[i, old_wrd[i]].item()

        batch[:, posn] = old_wrd
    return [-1.0 * raw_s * beta for raw_s in raw_score], [-1.0 * norm_s * beta for norm_s in norm_score]


def energy_score_mlm2(batch, beta=1):
    seq_len = batch.size(1) - 2
    posns = torch.arange(1, seq_len + 1)

    norm_score = torch.zeros(batch.size(0))
    raw_score = torch.zeros(batch.size(0))

    for posn in posns:
        old_wrd = batch[:, posn].clone()

        # Mask the position
        batch[:, posn] = mask_id

        # Get the model logits for the masked position
        output = model(batch)['logits'][:, posn, :]

        # Calculate raw and normalized scores
        raw_score += output.gather(1, old_wrd.unsqueeze(1)).squeeze(1)
        norm_output = output.log_softmax(dim=-1)
        norm_score += norm_output.gather(1, old_wrd.unsqueeze(1)).squeeze(1)

        # Restore the original values
        batch[:, posn] = old_wrd

    return [-raw_s * beta for raw_s in raw_score], [-norm_s * beta for norm_s in norm_score]


def energy_score_disc(batch, model_disc, tokenizer_disc, sentiment=0, alpha=0):
    # encoded_input = tokenizer(text, return_tensors='pt').to(device)
    # tokens = encoded_input['input_ids'][0]
    seq_len = len(batch[0]) - 2
    posns = [i + 1 for i in range(seq_len)]
    # random.shuffle(posns)
    norm_score = np.array([0.0] * batch.shape[0])
    raw_score = np.array([0.0] * batch.shape[0])

    output = model_disc(batch)['logits']
    pred = np.argmax(np.array(output.log_softmax(dim=-1).cpu().detach()), axis=-1)

    # print(output.shape)
    classes = output.shape[-1]
    for i in range(classes):
        if i == sentiment:
            raw_score += np.array(output[:, i].cpu().detach())
            norm_output = output.log_softmax(dim=-1)
            norm_score += np.array(norm_output[:, i].cpu().detach())

    raw_score = min_max_normalization_disc(raw_score)
    return [-1.0 * raw_s * alpha for raw_s in raw_score], [-1.0 * norm_s * alpha for norm_s in norm_score], pred


EOT_TOKEN = '<|endoftext|>'


def perplexity_fudge(sentences):
    # calculate perplexity
    gpt_model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to(device)
    gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    with torch.no_grad():
        ppl = []
        sos_token = gpt_tokenizer.decode([0])
        sentences = untokenize_batch(sentences)
        for sentence in sentences:
            # Preprocess each input sentence
            sentence = sentence.lower()
            sentence = tokenizer.tokenize(sentence)
            sentence = tokenizer.convert_tokens_to_ids(sentence)
            sentence = tokenizer.decode(sentence)
            sentence = sentence.replace(' \' ', '\'')
            sentence = sentence.replace(' - ', '-')
            sentence = sentence.replace(' .', '.')
            sentence = sentence.replace(' ,', ',')

            # Encode the sentence and calculate perplexity
            full_tensor_input = gpt_tokenizer.encode(sos_token + sentence.replace(EOT_TOKEN, ' ').strip(),
                                                     return_tensors='pt').to(device)
            full_loss = gpt_model(full_tensor_input, labels=full_tensor_input)[0].mean()
            ppl.append(torch.exp(full_loss).flatten().cpu().item())

    return np.mean(ppl), np.std(ppl), np.median(ppl)


def parallel_sequential_generation(
        seed_text,
        dict_ag,
        seed_text_src,
        model_disc,
        tokenizer_disc,
        sentiment=0,
        batch_size=10,
        max_len=15,
        top_k=0,
        temperature=1,
        max_iter=300,
        burnin=200,
        cuda=False,
        print_every=10,
        verbose=True,
        args=args
):
    """Generate for one random position at a timestep

    args:
        - burnin: during burn-in period, sample from full distribution; afterwards take argmax
    """
    #     seed_len = len(seed_text)

    iteration_times, old_energy_times, mlm_times, disc_times, sampling_times, new_energy_times, meta_times = ([] for i
                                                                                                              in
                                                                                                              range(7))
    parallel_sequential_starttime = time.time()
    batch = torch.tensor(get_init_text(seed_text, max_len, batch_size)).to(device)

    diff = len(seed_text) - len(seed_text_src) if len(seed_text) > len(seed_text_src) else 0

    # print(len(seed_text),len(seed_text_src))
    batch_original = torch.tensor(
        get_init_text(seed_text_src[:len(seed_text)] + ['.'] * (diff), max_len, batch_size)).to(
        device)  # batch.detach().clone()
    print(batch.shape, batch_original.shape)
    seq_len = batch.shape[-1] - 2
    posns = [pos for pos, tok in enumerate(batch[0]) if tok == mask_id]  # [i+1 for i ,y in enumerate(range(seq_len)) ]
    # posns =  [i for i, y in enumerate(batch[0])  if (y == mask_id or y == mr_id or y==ms_id)] #TODO [i for i, y in enumerate(batch[0]) if y == mask_id]
    # print(batch)
    # print(mask_pos)

    full_meta_data = [[] for i in range(batch_size)]
    meta_data = []
    selected_batch = batch
    word_changed_bool = False

    global_max_value = None
    global_min_value = None

    global_max_value_disc = None
    global_min_value_disc = None
    for iteration in range(max_iter):
        start_time_iteration = time.time()
        if (args.shuffle_positions):
            random.shuffle(posns)
        if not (args.block):
            nmasks = 1
        else:
            nmasks = random.randint(max(1, math.ceil(seq_len / 2) - 3), min(seq_len - 1, math.ceil(seq_len / 2) + 3))

        groups = [posns[i:i + nmasks] for i in range(0, len(posns), nmasks)]
        if (args.shuffle_positions):
            random.shuffle(groups)
            # kk = mask_pos[np.random.randint(0, len(mask_pos))]
        for positions in groups:

            if args.degenerate:
                # for jj in range(batch_size):
                #     batch[jj][kk] = mask_id
                # inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
                # out = model(inp)
                # topk = top_k if (ii >= burnin) else 0
                # idxs = generate_step(
                #     out,
                #     gen_idx=kk,
                #     top_k=topk if (ii >= burnin) else 0,
                #     temperature=temperature,
                #     sample=(ii < burnin),
                # )
                # for jj in range(batch_size):
                #     batch[jj][kk] = idxs[jj]
                # r_score, norm_score = np.array(energy_score(batch))
                # for i in range(batch_size):
                #     meta_data[i].append( (ii,kk,r_score[i],norm_score[i]) )
                # old_e = np.array(enorm_score(batch))
                # print(kk)
                # old_e = np.array(enorm_score(batch))
                old_r, old_norm = np.array(energy_score_mlm(batch))
                # print(nmasks, positions, groups)
                old_wrd = batch[:, positions].detach().clone()

                batch[:, positions] = mask_id
                # print(old_wrd.shape)
                ##here
                # output = model(batch)[:,kk,:].softmax(dim=-1)
                output = (model(batch)[:, positions, :] / temperature)
                output[:, :, mask_id] = -10000000000.0
                output = output.softmax(dim=-1)

                # print(output.shape)
                qxbx = np.array([1.0] * batch_size)
                qxxb = np.array([1.0] * batch_size)
                # for i,posn in enumerate(positions):

                d = Categorical(output)
                new_wrd = d.sample()
                # print(new_wrd.shape)
                # n_flag = ~(old_wrd[i] == new_wrd) #TODO
                n_flag = np.array([0] * batch_size)
                msk_change = [False] * batch_size

                for ii in range(len(positions)):
                    for jj in range(batch_size):
                        # print("shape",output[:,ii,old_wrd[jj,ii]].cpu().shape)
                        qxxb[jj] *= output[jj, ii, old_wrd[jj, ii]].cpu()
                        #    qxxb.append(output[jj,old_wrd[i][jj]].item())
                        qxbx[jj] *= output[jj, ii, new_wrd[jj, ii]].cpu()
                        if not (old_wrd[jj, ii].item() == new_wrd[jj, ii].item()):
                            n_flag[jj] = 1
                        if (old_wrd[jj, ii].item() == mask_id):
                            msk_change[jj] = True

                batch[:, positions] = new_wrd
                new_r, new_norm = np.array(energy_score_mlm(batch))

                # mask_id == np.array(old_wrd.cpu())

                # print(msk_change)
                axbx = np.where(msk_change, 1.0, np.minimum(1.0, np.divide(
                    np.multiply(np.exp(old_r - new_r), np.array(qxxb)), np.array(qxbx))))

                # print(axbx.shape)

                acc = torch.ones(axbx.shape)  # torch.squeeze(torch.bernoulli(torch.Tensor([axbx])))

                batch[:, positions] = torch.where(acc.unsqueeze(1).repeat(1, len(positions)).to(device) > 0.0,
                                                  batch[:, positions], old_wrd)

                r_score = np.squeeze(np.where(acc > 0.0, new_r, old_r))
                norm_score = np.squeeze(np.where(acc > 0.0, new_norm, old_norm))

                acc = np.array(acc.cpu()) * np.array(n_flag)

                # for i in range(batch_size):
                #    meta_data[i].append( (ii,positions,r_score[i],norm_score[i],qxxb[i],qxbx[i],axbx[i],acc[i].item()) )

                # print(meta_data)
                # exit(0)


            else:

                start_time_old_energy = time.time()
                if 1 == 1:
                    # print(kk)
                    # old_e = np.array(enorm_score(batch))
                    distance = np.sum(1 - np.array((batch == batch_original).detach().cpu()) * 1, axis=-1)
                    bleurt_score = np.zeros(distance.shape)
                    if args.gamma:
                        bleurt_score = np.array(model_bleurt(batch_original, batch)[0].squeeze().detach().cpu())

                    lexicon_score = 0
                    if args.theta:
                        lexicon_score = lexicon_dec(batch, dict_ag, sentiment)
                    start_time_disc = time.time()

                    if iteration == 0 or untokenize_batch(selected_batch) == []:
                        disc_1, disc_2, disc_preds = energy_score_disc(batch,
                                                                       model_disc=model_disc,
                                                                       tokenizer_disc=tokenizer_disc,
                                                                       sentiment=sentiment,
                                                                       alpha=args.alpha)
                        # print(disc_1.shape)
                        print(disc_1)
                    else:

                        disc_1_part, disc_2_part, disc_preds_part = energy_score_disc(selected_batch,
                                                                                      model_disc=model_disc,
                                                                                      tokenizer_disc=tokenizer_disc,
                                                                                      sentiment=sentiment,
                                                                                      alpha=args.alpha)
                        #
                        # disc_1_test, disc_2_test, disc_preds_test = energy_score_disc(batch, model_disc=model_disc,
                        #                                                               tokenizer_disc=tokenizer_disc,
                        #                                                               sentiment=sentiment, alpha=args.alpha)
                        # disc_1 = np.put_along_axis(disc_1, row_indices[:, None], disc_1_part, axis=0)

                        # disc_1 = [disc_1[pos] = value for pos,value in zip(row_indices, disc_1_part)]
                        print("Partvalues")
                        print(disc_1_part)
                        disc_1 = np.array(disc_1)
                        disc_1_part = np.array(disc_1_part)
                        disc_1[row_indices] = disc_1_part
                        print(disc_1.shape)
                        # disc_1 = [disc_1_part.pop(0) if i in row_indices else val for i, val in
                        #               enumerate(disc_1)]
                        # disc_2 = np.put_along_axis(disc_2, row_indices[:, None], disc_2_part, axis=0)
                        # disc_preds = np.put_along_axis(disc_preds_new, row_indices[:, None], disc_preds, axis=0)

                        # print("Testvalues")
                        # print(disc_1_test)
                        print("Values")
                        print(disc_1)
                    print("Finished disc score in %.3fs" % (time.time() - start_time_disc))
                    disc_times.append(time.time() - start_time_disc)
                    disc_times_set.append(time.time() - start_time_disc)

                    start_time_perplexity = time.time()
                    # mean,std, median = perplexity_fudge(batch)
                    # print(median)
                    # print("Finished perplexity score in %.3fs" % (time.time() - start_time_perplexity))
                    start_time_mlm = time.time()
                    start_time_mlm_orig = time.time()
                    mlm_1, mlm_2 = np.array(energy_score_mlm(batch, args.beta))
                    print("Finished Original mlm score in %.3fs" % (time.time() - start_time_mlm_orig))

                    print("Original")
                    print(mlm_1)

                    old_r, old_norm = np.array(perplexity_score(batch, args.beta)) + np.array([disc_1, disc_2])
                    print("Finished mlm score in %.3fs" % (time.time() - start_time_mlm))
                    mlm_times.append(time.time() - start_time_mlm)
                    mlm_times_set.append(time.time() - start_time_mlm)
                    old_r += args.delta * distance
                    old_r -= args.gamma * bleurt_score
                    old_r -= args.theta * lexicon_score

                old_wrd = batch[:, positions].detach().clone()
                print("Old word")
                print(untokenize_batch(old_wrd))
                batch[:, positions] = mask_id

                # output = model(batch)[:,kk,:].softmax(dim=-1)
                output = (model(batch)['logits'][:, positions, :] / temperature)
                output[:, :, mask_id] = -10000000000.0
                output = output.softmax(dim=-1)
                # print(output)

                qxbx = np.array([1.0] * batch_size)
                qxxb = np.array([1.0] * batch_size)

                print("Finished old energy score calculation in %.3fs" % (time.time() - start_time_old_energy))
                old_energy_times.append(time.time() - start_time_old_energy)
                old_energy_times_set.append(time.time() - start_time_old_energy)

                start_time_sampling = time.time()

                d = Categorical(output)
                new_wrd = d.sample()
                print("New Word")
                print(untokenize_batch(new_wrd))
                # print(untokenize_batch(new_wrd))
                # print(new_wrd.shape)
                # n_flag = ~(old_wrd[i] == new_wrd) #TODO
                n_flag = np.array([0] * batch_size)
                msk_change = [False] * batch_size

                for ii in range(len(positions)):
                    for jj in range(batch_size):
                        # print("shape",output[:,ii,old_wrd[jj,ii]].cpu().shape)
                        qxxb[jj] *= output[jj, ii, old_wrd[jj, ii]].cpu()
                        #    qxxb.append(output[jj,old_wrd[i][jj]].item())
                        qxbx[jj] *= output[jj, ii, new_wrd[jj, ii]].cpu()
                        if not (old_wrd[jj, ii].item() == new_wrd[jj, ii].item()):
                            n_flag[jj] = 1
                        if (old_wrd[jj, ii].item() == mask_id):
                            msk_change[jj] = True

                batch[:, positions] = new_wrd

                print("Finished sampling in %.3fs" % (time.time() - start_time_sampling))
                sampling_times.append(time.time() - start_time_sampling)
                sampling_times_set.append(time.time() - start_time_sampling)
                start_time_new_energy = time.time()

                distance_new = np.sum(1 - np.array((batch == batch_original).detach().cpu()) * 1, axis=-1)
                bleurt_new = 0
                lexicon_new = 0

                if args.gamma:
                    bleurt_new = np.array(model_bleurt(batch_original, batch)[0].squeeze().detach().cpu())

                if args.theta:
                    lexicon_new = lexicon_dec(batch, dict_ag=dict_ag, target=sentiment)
                # print("new dist\n",distance)
                disc_1, disc_2, disc_preds_new = energy_score_disc(batch, model_disc=model_disc,
                                                                   tokenizer_disc=tokenizer_disc, sentiment=sentiment,
                                                                   alpha=args.alpha)
                # print("Disc Energy")
                # print(disc_1)
                new_r, new_norm = np.array(perplexity_score(batch, beta=args.beta)) + np.array([disc_1, disc_2])
                new_r += args.delta * distance_new
                new_r -= args.gamma * bleurt_new
                new_r -= args.theta * lexicon_new

                # mask_id == np.array(old_wrd.cpu())

                # print(msk_change)
                axbx = np.where(msk_change, 1.0, np.minimum(1.0, np.divide(
                    np.multiply(np.exp(old_r - new_r), np.array(qxxb)), np.array(qxbx))))

                acc = torch.squeeze(torch.bernoulli(torch.Tensor([axbx])))

                # word_changed_bool = (acc.unsqueeze(1).repeat(1, len(positions)).to(device) > 0.0).all().item()
                change_bool = (acc.unsqueeze(1).repeat(1, len(positions)).to(device) > 0.0).type(torch.bool)
                batch[:, positions] = torch.where(acc.unsqueeze(1).repeat(1, len(positions)).to(device) > 0.0,
                                                  batch[:, positions], old_wrd)

                print(change_bool)
                row_indices, col_indices = np.where(change_bool == False)
                print(row_indices)
                selected_batch = batch[row_indices, :]
                # print("Ergebnis")
                # print(untokenize_batch(batch))
                # print(untokenize_batch(selected_batch))
                print("Finished new energy score calculation in %.3fs" % (time.time() - start_time_new_energy))
                new_energy_times.append(time.time() - start_time_new_energy)
                new_energy_times_set.append(time.time() - start_time_new_energy)

                start_time_meta_preparation = time.time()
                r_score = np.squeeze(np.where(acc > 0.0, new_r, old_r))
                norm_score = np.squeeze(np.where(acc > 0.0, new_norm, old_norm))
                disc_preds = np.squeeze(np.where(acc > 0.0, disc_preds_new, disc_preds))
                distance = np.squeeze(np.where(acc > 0.0, distance_new, distance))
                bleurt = np.squeeze(np.where(acc > 0.0, bleurt_new, bleurt_score))
                lexicon = np.squeeze(np.where(acc > 0.0, lexicon_new, lexicon_score))

                acc = np.array(acc.cpu()) * np.array(n_flag)

                for i in range(batch_size):
                    full_meta_data[i].append((sentiment, r_score[i], norm_score[i], qxxb[i], qxbx[i], axbx[i],
                                              acc[i].item(), disc_preds[i], distance[i], bleurt[i], lexicon[i]))

                print("Finsihed metadata preparation in %.3fs" % (time.time() - start_time_meta_preparation))
                meta_times.append(time.time() - start_time_meta_preparation)
                meta_times_set.append(time.time() - start_time_meta_preparation)

        if verbose and np.mod(ii + 1, print_every) == 0:
            for_print = tokenizer.convert_ids_to_tokens(batch[0])
            for_print = for_print[: kk + 1] + ["(*)"] + for_print[kk + 1:]
            print("iter", ii + 1, " ".join(for_print))

        print("Finished iteration %d in %.3fs" % (iteration, time.time() - start_time_iteration))
        iteration_times.append(time.time() - start_time_iteration)
        iteration_times_set.append(time.time() - start_time_iteration)

    for i in range(batch_size):
        meta_data.append((sentiment, r_score[i], norm_score[i], qxxb[i], qxbx[i], axbx[i], acc[i].item(), disc_preds[i],
                          distance[i], bleurt[i]))

    print("Finished parallel_sequential method in %.3fs" % (time.time() - parallel_sequential_starttime))
    # print(iteration_times)
    print("Iteration: Avg(%f), Var(%f), " % (np.mean(iteration_times), np.var(iteration_times)))
    print("Disc Score: Avg(%f), Var(%f), " % (np.mean(disc_times), np.var(disc_times)))
    print("MLM Score: Avg(%f), Var(%f), " % (np.mean(mlm_times), np.var(mlm_times)))
    print("Old Full Energy Score: Avg(%f), Var(%f), " % (np.mean(old_energy_times), np.var(old_energy_times)))
    print("Sampling: Avg(%f), Var(%f), " % (np.mean(sampling_times), np.var(sampling_times)))
    print("New Energy Score: Avg(%f), Var(%f), " % (np.mean(new_energy_times), np.var(new_energy_times)))
    print("Metadata: Avg(%f), Var(%f), " % (np.mean(meta_times), np.var(meta_times)))
    # print(untokenize_batch(batch))
    return untokenize_batch(batch), meta_data, full_meta_data


times = []
iteration_times_set, old_energy_times_set, mlm_times_set, disc_times_set, sampling_times_set, new_energy_times_set, meta_times_set = (
[] for i in range(7))


def generate(
        n_samples,
        dict_ag,
        model_disc,
        tokenizer_disc,
        sentiment,
        seed_text="[CLS]",
        seed_src="[CLS]",
        batch_size=10,
        max_len=25,
        top_k=100,
        temperature=1.0,
        burnin=200,
        max_iter=500,
        cuda=False,
        print_every=1,
        args=args
):
    # main generation function to call
    sentences = []

    n_batches = math.ceil(n_samples / batch_size)
    start_time = time.time()
    for batch_n in range(n_batches):

        batch, metadata, full_metadata = parallel_sequential_generation(
            seed_text,
            dict_ag,
            seed_src,
            model_disc=model_disc,
            tokenizer_disc=tokenizer_disc,
            batch_size=batch_size,
            sentiment=sentiment,
            max_len=max_len,
            top_k=top_k,
            temperature=temperature,
            burnin=burnin,
            max_iter=max_iter,
            cuda=cuda,
            verbose=False,
        )

        if (batch_n + 1) % print_every == 0:
            print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
            times.append(time.time() - start_time)
            print(times)
            print("Average computation time: %.3fs " % np.mean(times))

            print(
                "Iteration Complete: Avg(%f), Var(%f), " % (np.mean(iteration_times_set), np.var(iteration_times_set)))
            print("Disc Score Complete: Avg(%f), Var(%f), " % (np.mean(disc_times_set), np.var(disc_times_set)))
            print("MLM Score Complete: Avg(%f), Var(%f), " % (np.mean(mlm_times_set), np.var(mlm_times_set)))
            print("Old Full Energy Score Complete: Avg(%f), Var(%f), " % (
            np.mean(old_energy_times_set), np.var(old_energy_times_set)))
            print("Sampling Complete: Avg(%f), Var(%f), " % (np.mean(sampling_times_set), np.var(sampling_times_set)))
            print("New Energy Score Complete: Avg(%f), Var(%f), " % (
            np.mean(new_energy_times_set), np.var(new_energy_times_set)))
            print("Metadata Complete: Avg(%f), Var(%f), " % (np.mean(meta_times_set), np.var(meta_times_set)))
            start_time = time.time()

        sentences += batch
        print(sentences)
    return sentences, metadata, full_metadata


# Choose the prefix context

seeds = [
    "[CLS] mr",  # TODO
    "[CLS] ms",  # TODO
]

import secrets

#
degenerate = args.degenerate
#
top_k = args.top_k  # 40 #not used
# leed_out_len = 5  # max_len, not used
burnin = args.burnin  # 250 #not used
temperature = args.temperature
###########


dirname = args.out_path
n_samples = args.n_samples
batch_size = args.batch_size
max_iter = args.max_iter
max_len = 1  # this is dummy!!
########

tokenizer_disc = BertTokenizer.from_pretrained(
    args.disc_dir)  # finiteautomata/bertweet-base-sentiment-analysis ("textattack/bert-base-uncased-imdb")
model_disc = AutoModelForSequenceClassification.from_pretrained(args.disc_dir).to(
    device)  # ("textattack/bert-base-uncased-imdb").to(device)

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

if args.degenerate:
    folder_name = "mask_degenerate_topk_{}_burnin_{}_disc_{}_data_{}_max_iter_{}_temp_{}_shuffle_{}_block_{}_alpha_{}_beta_{}_delta_{}_gamma_{}_theta_{}_date_{}".format(
        top_k, burnin, args.disc_name, args.data_name, max_iter, temperature, args.shuffle_positions, args.block,
        args.alpha, args.beta, args.delta, args.gamma, args.theta, dt_string)
else:
    folder_name = "mask_disc_{}_data_{}_max_iter_{}_date_{}".format(args.disc_name, args.data_name, max_iter,
                                                                    temperature, args.shuffle_positions, args.block,
                                                                    args.alpha, args.beta, args.delta, args.gamma,
                                                                    args.theta, dt_string)

directory = "{}/{}".format(dirname, folder_name)
if not os.path.exists(directory):
    os.mkdir(directory)

dirname = directory
data_dir = args.data_path
data_dir_src = args.src_path
attr_dir = args.attr_path

dict_ag = get_agency_dict(args.dict_path)

with open(f"{dirname}/samples.txt", "w") as f, open(f"{dirname}/opt_samples.txt", "w") as optimal_f, open(
        f"{dirname}/opt_cls.txt", "w") as optimal_class, open(f"{dirname}/opt_meta.txt", "w") as opt_meta_file, open(
        f"{dirname}/metadata.txt", "w") as f_meta, open(f"{data_dir}", "r") as data_file, open(f"{data_dir_src}",
                                                                                               "r") as src_file, open(
        f"{attr_dir}", "r") as attr_file:
    for i, (line, line_src, src) in enumerate(zip(data_file, src_file, attr_file)):
        seed_text = line[:-1]
        seed_src = line_src[:-1]
        sentiment = int(1 - int(src[:-1]))
        print(seed_text)
        seed_text = tokenizer.tokenize(seed_text)
        seed_text_src = tokenizer.tokenize(seed_src)
        print(seed_text)
        torch.cuda.empty_cache()
        bert_sents, meta_data, full_meta_data = generate(
            n_samples,
            dict_ag,
            model_disc=model_disc,
            tokenizer_disc=tokenizer_disc,
            sentiment=sentiment,
            seed_text=seed_text,
            seed_src=seed_text_src,
            batch_size=batch_size,
            max_len=max_len,
            top_k=top_k,
            temperature=temperature,
            burnin=burnin,
            max_iter=max_iter,
            cuda=cuda,
            args=args
        )

        sents = list(map(lambda x: " ".join(detokenize(x)), bert_sents))

        f.write("\n".join(sents) + "\n")
        f.flush()

        # meta_data_str = [str(l) for l in meta_data]

        # f_meta.write("\n".join(meta_data_str)+"\n")
        # f_meta.flush()

        full_meta_data_str = [str(l) for l in full_meta_data]
        f_meta.write("\n".join(full_meta_data_str) + "\n")
        f_meta.flush

        opt_sent, opt_cls, ind = get_opt_sent(sents, meta_data)
        optimal_f.write(opt_sent + "\n")
        optimal_f.flush()

        opt_meta_str = str(full_meta_data[ind])
        opt_meta_file.write(opt_meta_str + "\n")
        opt_meta_file.flush()

        optimal_class.write(str(opt_cls) + "\n")
        optimal_class.flush()
