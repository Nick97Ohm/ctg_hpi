import os
import math
import time
import numpy as np
import torch
import argparse
import random
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, BertTokenizer, \
    BertModel
from torch.distributions.categorical import Categorical
from datetime import datetime
from pyplexity import PerplexityModel


################

def parse_arguments():

    parser = argparse.ArgumentParser(description="style transfer")
    parser.add_argument("--normalize", type= bool, default=False)
    parser.add_argument("--single", type=bool, default=False)
    parser.add_argument("--fluency", type=str, default="MLM")
    parser.add_argument("--max_iter", type=int, help="number of changes to make in the gibbs chain", default=100)
    parser.add_argument("--n_samples", type=int, help="number of changes to make in the gibbs chain", default=20)
    parser.add_argument("--batch_size", type=int, help="number of changes to make in the gibbs chain", default=20)
    parser.add_argument("--temperature", type=float, help="number of changes to make in the gibbs chain", default=1.0)
    parser.add_argument("--shuffle_positions", action='store_true')

    parser.add_argument("--data_path", type=str, help="dir", default='./data/yelp')
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


    args = parser.parse_args()


    return args

def load_models():
    # Load pre-trained model (weights)
    model_version = args.model_path  # os.environ["MODEL_PATH"]
    model = AutoModelForMaskedLM.from_pretrained(model_version)
    model.eval()

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = AutoTokenizer.from_pretrained(args.tok_path)


    tokenizer_disc = BertTokenizer.from_pretrained(args.disc_dir)
    model_disc = AutoModelForSequenceClassification.from_pretrained(args.disc_dir)
    model_perp = PerplexityModel.from_str("trigrams-bnc")

    return model, tokenizer, tokenizer_disc, model_disc, model_perp

def prepare_input(line, sentiment, f_meta, f):
    seed_text = line[:-1]
    print(seed_text)
    seed_text = tokenizer.tokenize(seed_text)
    print(seed_text)
    bert_sents, meta_data, full_meta_data = generate(
        n_samples,
        model_disc=model_disc,
        tokenizer_disc=tokenizer_disc,
        sentiment=sentiment,
        seed_text=seed_text,
        batch_size=batch_size,
        max_len=max_len,
        temperature=temperature,
        max_iter=args.max_iter,
        args2=args
    )

    sents = list(map(lambda x: " ".join(detokenize(x)), bert_sents))

    f.write("\n".join(sents) + "\n")
    f.flush()

    full_meta_data_str = [str(l) for l in full_meta_data]
    f_meta.write("\n".join(full_meta_data_str) + "\n")
    f_meta.flush

    opt_sent, opt_cls, ind = get_opt_sent(sents, meta_data)

    return opt_sent, opt_cls, full_meta_data[ind]




def min_max_normalization(array, global_max_value=None, global_min_value=None):
    # Calculate global min and max values if not provided
    if global_min_value is None or np.min(array) < global_min_value:
        global_min_value = np.min(array)
    if global_max_value is None or np.max(array) > global_max_value:
        global_max_value = np.max(array)

    if global_min_value == global_max_value:
        normalized_array = np.zeros_like(array)
    else:
        normalized_array = (array - global_min_value) / (global_max_value - global_min_value)

    return normalized_array, global_max_value, global_min_value



def get_opt_sent(sents, metadata):
    meta_array = np.array(metadata)

    ind = np.argmin(meta_array[:, 1, ...])
    sent_best = sents[ind].split()
    return " ".join(sent_best[1:-1]), metadata[ind][-3], ind

##########################################  TOKEN STUFF  ###############################################################
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

# Convert tokenized sentences back to words
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


##########################################  DON'T UNDERSTAND ###########################################################
def get_init_text(seed_text, max_len, batch_size=1, rand_init=False):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [[CLS] + seed_text + [SEP] for _ in range(batch_size)]  # TODO

    return tokenize_batch(batch)

##########################################  ENERGY SCORES  #############################################################

def energy_score_mlm(batch, beta=1):

    seq_len = len(batch[0]) - 2
    posns = [i + 1 for i in range(seq_len)]

    raw_score = [0.0] * batch.shape[0]
    for posn in posns:
        old_wrd = batch[:, posn].clone()
        batch[:, posn] = mask_id
        output = model(batch)['logits'][:, posn, :]
        for i in range(batch.shape[0]):
            raw_score[i] += output[i, old_wrd[i]].item()

        batch[:, posn] = old_wrd
    return [-1.0 * raw_s * beta for raw_s in raw_score]

def energy_score_disc(batch, model_disc, sentiment=0, alpha=0):

    raw_score = np.array([0.0] * batch.shape[0])

    output = model_disc(batch)['logits']
    pred = np.argmax(np.array(output.log_softmax(dim=-1).cpu().detach()), axis=-1)

    classes = output.shape[-1]
    for i in range(classes):
        #if i == sentiment:
        raw_score += np.array(output[:, i].cpu().detach())

    return [-1.0 * raw_s * alpha for raw_s in raw_score], pred

def perplexity_score(batch, beta = 1):

    start_time_per_bert = time.time()
    untoken = untokenize_batch(batch)
    sentences = tokens_to_sentences(untoken)

    per = []
    for sentence in sentences:
        perpl = model_perp.compute_sentence(sentence)
        per.append(perpl)

    #print("Finished Perp score in %.3fs" % (time.time() - start_time_per_bert))
    #print(per)


    return [perplexity * beta for perplexity in per]


def fluency_energy(batch, beta):
    if args.fluency == "pyplexity":
        return perplexity_score(batch, beta)
    elif args.fluency == "mlm":
        return energy_score_mlm(batch, beta)


def parallel_sequential_generation(
        seed_text,
        model_disc,
        args3,
        sentiment=0,
        batch_size=10,
        max_len=15,
        temperature=1,
        max_iter=300

):
    """Generate for one random position at a timestep """

    batch = torch.tensor(get_init_text(seed_text, max_len, batch_size))
    batch_original = batch.detach().clone()

    seq_len = batch.shape[-1] - 2
    posns = [i + 1 for i in range(seq_len)]


    full_meta_data = [[] for i in range(batch_size)]
    meta_data = []

    global_max_value = None
    global_min_value = None

    global_max_value_disc = None
    global_min_value_disc = None
    for iteration in range(max_iter):
        print("Iteration {}".format(iteration))
        if (args3.shuffle_positions):
            random.shuffle(posns)

        nmasks = 1
        groups = [posns[i:i + nmasks] for i in range(0, len(posns), nmasks)]
        if (args3.shuffle_positions):
            random.shuffle(groups)
        for positions in groups:

            distance = np.sum(1 - np.array((batch == batch_original).detach().cpu()) * 1, axis=-1)
            disc_score, disc_preds = energy_score_disc(batch, model_disc=model_disc, sentiment=sentiment, alpha=args3.alpha)
            # start_time_mlm = time.time()
            # if iteration > 0:
            #     old_r, old_norm = np.array(perplexity_score(batch,args3.beta))+np.array([disc_1,disc_2])
            # else:
            #     old_r, old_norm = np.array([[1.0]*batch_size, [1.0]*batch_size])+np.array([disc_1,disc_2])

            fluency_score = fluency_energy(batch, args3.beta)
            if args.normalize:
                fluency_score, global_max_value, global_min_value = min_max_normalization(fluency_score, global_max_value, global_min_value)

                disc_score, global_max_value_disc, global_min_value_disc = min_max_normalization(disc_score, global_max_value_disc, global_min_value_disc)
                print(global_max_value)
                print(global_min_value)
                print("Fluency Norm")
                print(fluency_score)
            old_r = np.array(fluency_score) + np.array(disc_score)
            old_r += args3.delta * distance
            old_wrd = batch[:, positions].detach().clone()

            batch[:, positions] = mask_id

            output = (model(batch)['logits'][:, positions, :] / temperature)
            output[:, :, mask_id] = -10000000000.0
            output = output.softmax(dim=-1)

            qxbx = np.array([1.0] * batch_size)
            qxxb = np.array([1.0] * batch_size)

            d = Categorical(output)
            new_wrd = d.sample()
            print(untokenize_batch(new_wrd))
            n_flag = np.array([0] * batch_size)
            msk_change = [False] * batch_size

            for ii in range(len(positions)):
                for jj in range(batch_size):

                    qxxb[jj] *= output[jj, ii, old_wrd[jj, ii]].cpu()
                    qxbx[jj] *= output[jj, ii, new_wrd[jj, ii]].cpu()

                    if not (old_wrd[jj, ii].item() == new_wrd[jj, ii].item()):
                        n_flag[jj] = 1
                    if (old_wrd[jj, ii].item() == mask_id):
                        msk_change[jj] = True

            batch[:, positions] = new_wrd

            distance_new = np.sum(1 - np.array((batch == batch_original).detach().cpu()) * 1, axis=-1)
            disc_1_new, disc_preds_new = energy_score_disc(batch, model_disc=model_disc, sentiment=sentiment, alpha=args3.alpha)
            fluency_1_new = fluency_energy(batch, args3.beta)
            if args.normalize:
                fluency_1_new, global_max_value, global_min_value = min_max_normalization(fluency_1_new, global_max_value, global_min_value)
                disc_1_new, global_max_value_disc, global_min_value_disc = min_max_normalization(disc_1_new, global_max_value_disc, global_min_value_disc)
            new_r = np.array(fluency_1_new) + np.array(disc_1_new)
            new_r += args3.delta * distance_new


            axbx = np.where(msk_change, 1.0, np.minimum(1.0, np.divide(np.multiply(np.exp(old_r - new_r), np.array(qxxb)), np.array(qxbx))))
            acc = torch.squeeze(torch.bernoulli(torch.Tensor([axbx])))
            batch[:, positions] = torch.where(acc.unsqueeze(1).repeat(1, len(positions)) > 0.0,
                                              batch[:, positions], old_wrd)

            r_score = np.squeeze(np.where(acc > 0.0, new_r, old_r))
            disc_preds = np.squeeze(np.where(acc > 0.0, disc_preds_new, disc_preds))
            distance = np.squeeze(np.where(acc > 0.0, distance_new, distance))


            acc = np.array(acc.cpu()) * np.array(n_flag)

            for i in range(batch_size):
                full_meta_data[i].append((sentiment, r_score[i], qxxb[i], qxbx[i], axbx[i],
                                          acc[i].item(), disc_preds[i], distance[i]))

    print(untokenize_batch(batch))

    for i in range(batch_size):
        meta_data.append((sentiment, r_score[i], qxxb[i], qxbx[i], axbx[i], acc[i].item(), disc_preds[i],
                          distance[i]))

    return untokenize_batch(batch), meta_data, full_meta_data


def generate(
        n_samples,
        model_disc,
        tokenizer_disc,
        args2,
        sentiment,
        seed_text="[CLS]",
        batch_size=10,
        max_len=25,
        temperature=1.0,
        max_iter=500,
        print_every=1

):

    # main generation function to call
    sentences = []
    n_batches = math.ceil(n_samples / batch_size)
    start_time = time.time()
    for batch_n in range(n_batches):
        batch, metadata, full_metadata = parallel_sequential_generation(
            seed_text,
            model_disc=model_disc,
            batch_size=batch_size,
            sentiment=sentiment,
            max_len=max_len,
            temperature=temperature,
            max_iter=max_iter,
            args3= args2
        )

        if (batch_n + 1) % print_every == 0:
            print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
            start_time = time.time()

        sentences += batch
    return sentences, metadata, full_metadata




def generate_samples(input=""):
    global args
    with open(f"{dirname}/samples.txt", "w") as f, \
         open(f"{dirname}/opt_samples.txt", "w") as optimal_f, \
         open(f"{dirname}/opt_cls.txt", "w") as optimal_class, \
         open(f"{dirname}/opt_meta.txt","w") as opt_meta_file, \
         open(f"{dirname}/metadata.txt", "w") as f_meta:

        if args.single:
            sentiment = 1
            prepare_input(input, sentiment, f_meta, f)


        else:
            with open(f"{data_dir}", "r") as data_file, \
                 open(f"{attr_dir}", "r") as attr_file:

                for i, (line, src) in enumerate(zip(data_file, attr_file)):
                    sentiment = int(1 - int(src[:-1]))
                    opt_sent, opt_cls, full_meta_data_row = prepare_input(line, sentiment, f_meta, f)

                    optimal_f.write(opt_sent + "\n")
                    optimal_f.flush()

                    opt_meta_str = str(full_meta_data_row)
                    opt_meta_file.write(opt_meta_str + "\n")
                    opt_meta_file.flush()

                    optimal_class.write(str(opt_cls) + "\n")
                    optimal_class.flush()


if __name__ == "__main__":

    args = parse_arguments()
    model, tokenizer, tokenizer_disc, model_disc, model_perp = load_models()



    CLS = "[CLS]"
    SEP = "[SEP]"
    MASK = "[MASK]"
    mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
    sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
    cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]

    #

    temperature = args.temperature
    dirname = args.out_path
    n_samples = args.n_samples
    batch_size = args.batch_size


    max_len = 1  # this is dummy!!
    ########


    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    folder_name = "disc_{}_data_{}_max_iter_{}_date_{}".format(args.disc_name, args.data_name, args.max_iter, dt_string)

    directory = "{}/{}".format(dirname, folder_name)
    if not os.path.exists(directory):
        os.mkdir(directory)

    dirname = directory
    data_dir = args.data_path
    attr_dir = args.attr_path

    if args.single:
        while True:
            user_input = input("Geben Sie einen Satz ein (oder 'exit' zum Beenden): ")

            if user_input.lower() == 'exit':
                break

            generate_samples(user_input)
    else:

        generate_samples()


