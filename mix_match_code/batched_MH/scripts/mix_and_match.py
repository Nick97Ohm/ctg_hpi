import os
import time
import numpy as np
import torch
import argparse
import random
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)
from torch.distributions.categorical import Categorical
from datetime import datetime
from pyplexity import PerplexityModel


# ---------------------------------------------------------------------------------------------------------------------
# 1. Parse args
# ---------------------------------------------------------------------------------------------------------------------
def parse_arguments():
    """Gets the arguments from the shell script sample_batched.sh

    Returns:
        ArgumentParser: Object for parsing command line strings into Python objects.
    """
    parser = argparse.ArgumentParser(description="style transfer")
    parser.add_argument("--normalize", type=bool, default=False)
    parser.add_argument("--single", type=bool, default=False)
    parser.add_argument("--fluency", type=str, default="MLM")
    parser.add_argument("--max_iter", type=int, help="number of changes to make in the gibbs chain", default=100)
    parser.add_argument(
        "--batch_size",
        type=int,
        help="number of changes to make in the gibbs chain",
        default=20,
    )
    parser.add_argument("--shuffle_positions", action='store_true')

    parser.add_argument("--data_path", type=str, help="dir", default='./data/yelp')
    parser.add_argument("--attr_path", type=str, help="dir", default='./data/yelp')
    parser.add_argument("--data_name", type=str, help="dir", default='yelp')
    parser.add_argument("--out_path", type=str, help="dir", default='./batched')
    parser.add_argument(
        "--model_path", type=str, help="dir", default="bert-base-uncased"
    )
    parser.add_argument("--tok_path", type=str, help="dir", default="bert-base-uncased")
    # disc
    parser.add_argument("--disc_name", type=str, help="disc dir", default='imdb')
    parser.add_argument("--disc_dir", type=str, help="disc dir", default='textattack/bert-base-uncased-imdb')
    # hyper params
    parser.add_argument("--alpha", type=float, help="knob", default=1)  # disc
    parser.add_argument("--beta", type=float, help="knob", default=1)
    parser.add_argument("--delta", type=float, help="knob", default=1)  # hamming

    args = parser.parse_args()

    return args


# End step 1 ---------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# 2. Load models
# ---------------------------------------------------------------------------------------------------------------------
def load_models():
    """Loads the necessary models. These are the Bertmodel for proposal generation,
       a tokenizer to split the sentences into a list, the sentiment classifier
       and the model for perplexity calculation from pyplexity

    Returns:
        AutoModelForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification, PerplexityModel
    """
    # Bert model
    model_version = args.model_path
    model = AutoModelForMaskedLM.from_pretrained(model_version)
    model.eval()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tok_path)

    # Sentiment classifier
    model_disc = AutoModelForSequenceClassification.from_pretrained(args.disc_dir)

    # Pyplexity language model
    model_perp = PerplexityModel.from_str("trigrams-bnc")

    return model, tokenizer, model_disc, model_perp


# End step 2 ---------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# 3. Create folders for output
# ---------------------------------------------------------------------------------------------------------------------
def create_output_folder():
    """Creates the folder where the output files are being generated

    Returns:
        string: the folder path
    """
    # get time to add it into the folder name
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    folder_name = "disc_{}data_{}max_iter_{}date_{}".format(
        args.disc_name, args.data_name, args.max_iter, dt_string
    )

    # full path
    directory = "{}/{}".format(args.out_path, folder_name)

    # create folder if it doesn't exist already
    if not os.path.exists(directory):
        os.mkdir(directory)

    return directory


# End step 3 ---------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# 4. Main execution
# ----------------------------------------------------------------------------------------------------------------------
def generate_samples(input=""):
    """Based on the --single argument here will be decided where it will get it's sentence from.
       If --single = true then from the input mask
       If --single = false or not declared then it will iteratively work through each sentencre from the dataset
       Then follows the main execution for this sentence
       At the end the output files will be written into the generated output folder 

    Args:
        input (str): The sentence from the input mask when --single = true
    """
    global args

    # Determine if a single input or a all data will be executed
    if args.single:
        labeled_data = zip([input], [0])

    else:
        data_file = open(f"{args.data_path}", "r")
        attr_file = open(f"{args.attr_path}", "r")
        labeled_data = zip(data_file, attr_file)

    # Execute main process for all lines
    for line, src in labeled_data:
        if isinstance(src, int):  # Check if src is an integer
            sentiment = 1 - src
        else:
            sentiment = int(1 - int(src[:-1]))
        # 4.1 Tokenize samples
        # --------------------------------------------------------------------
        seed_text = tokenizer.tokenize(line[:-1])
        # --------------------------------------------------------------------

        # 4.2 Create batch of sentences
        # --------------------------------------------------------------------
        batch = torch.tensor(tokenize_batch([[CLS] + seed_text + [SEP] for _ in range(args.batch_size)]))
        # --------------------------------------------------------------------

        # 4.3 Execute Mix and match
        # --------------------------------------------------------------------
        start_time = time.time()
        bert_sents, meta_data, full_meta_data = parallel_sequential_generation(
            batch=batch,
            model_disc=model_disc,
            batch_size=args.batch_size,
            sentiment=sentiment,
            max_iter=args.max_iter,
            args3=args,
        )
        print("Finished batch in %.3fs" % (time.time() - start_time))
        # --------------------------------------------------------------------

        # 4.4 Detokenize
        # --------------------------------------------------------------------
        sents = list(map(lambda x: " ".join(detokenize(x)), bert_sents))
        # --------------------------------------------------------------------

        # 4.5 Determine optimal sentence
        # --------------------------------------------------------------------
        opt_sent, ind = get_opt_sent(sents, meta_data)
        # --------------------------------------------------------------------

        # 4.6 Update output files
        # --------------------------------------------------------------------
        generate_output(sents=sents, full_meta_data=full_meta_data, opt_sent=opt_sent, ind=ind)
        # --------------------------------------------------------------------


def min_max_normalization(array, global_max_value=None, global_min_value=None):
    """Min-max normalization method. Takes the biggest and slowest value from a set and normalizes the values
       based on them.
       Will only be executed when --normalize = true

    Args:
        array (type): description
        global_max_value (float, optional): The currently highest value for this attribute. Defaults to None.
        global_min_value (float, optional): The currently lowest value for this attribute. Defaults to None.

    Returns:
        ndarray, flaot, float: Array with normalized values together with new highest and lowest value
    """
    # needs to be executed when it is the first iteration or the min/max value needs to be updated
    if global_min_value is None or np.min(array) < global_min_value:
        global_min_value = np.min(array)
    if global_max_value is None or np.max(array) > global_max_value:
        global_max_value = np.max(array)

    if global_min_value == global_max_value:
        normalized_array = np.zeros_like(array)
    else:
        # actual min-max-normalization
        normalized_array = (array - global_min_value) / (
                global_max_value - global_min_value
        )

    return normalized_array, global_max_value, global_min_value


def tokenize_batch(batch):
    """Uses the loaded tokenizer to assign numerical values to each token.
       This is needed to perform calculations with these tokens.

    Args:
        batch (torch.Tensor): batch of string-tokenized sentences

    Returns:
        torch.Tensor: batch of tokenized sentences with numerical values
    """
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]


def untokenize_batch(batch):
    """Converts the numerical values from the tokenized sentences back to their original string representation

    Args:
        batch (torch.Tensor): batch of tokenized sentences with numerical values

    Returns:
        torch.Tensor: batch of string-tokenized sentences
    """
    return [tokenizer.convert_ids_to_tokens(list(sent.to("cpu").numpy())) for sent in batch]


def tokens_to_sentences(tokenized_sentences):
    """Forms the numerical representations of a batch of tokenized sentences back to their string representation.
       This method is needed to calculate the perplexity with pyplexity
    Args:
        tokenized_sentences (torch.Tensor): batch of tokenized sentences with numerical values

    Returns:
        list: List of string sentences
    """
    sentences = []

    for tokenized_sentence in tokenized_sentences:
        # Remove [CLS] and [SEP] tokens
        sentence_tokens = [
            token for token in tokenized_sentence if token not in ["[CLS]", "[SEP]"]
        ]
        # Join the remaining tokens to form a sentence
        sentence = " ".join(sentence_tokens)
        # Append the reconstructed sentence to the list
        sentences.append(sentence)

    return sentences


def energy_score_mlm(batch, beta=1):
    """ The original method to calculate the fluency energy score.
        Iteratively masks each token and sums the raw logits retrieved by BERT

    Args:
        batch (torch.Tensor): batch of tokenized sentences with numerical values
        beta (float, optional): Weight in calculating the overall energy. Defaults to 1.

    Returns:
        list: List of fluency energy scores for each sentence of the batch
    """
    seq_len = len(batch[0]) - 2
    posns = [i + 1 for i in range(seq_len)]

    raw_score = [0.0] * batch.shape[0]
    # Iteratively mask each position and retrieve raw logits for this sentence from Bert
    for posn in posns:
        old_wrd = batch[:, posn].clone()
        batch[:, posn] = mask_id
        output = model(batch)["logits"][:, posn, :]
        for i in range(batch.shape[0]):
            raw_score[i] += output[i, old_wrd[i]].item()

        batch[:, posn] = old_wrd
    return [-1.0 * raw_s * beta for raw_s in raw_score]


def energy_score_disc(batch, model_disc, sentiment=0, alpha=0):
    """Calculation of positive sentiment energy score

    Args:
        batch (torch.Tensor): batch of tokenized sentences with numerical values
        model_disc (AutoModelForSequenceClassification): sentiment classifier
        sentiment (int, optional): The actual labeled sentiment for this sentence. Defaults to 0.
        alpha (int, optional): Weight in calculating the overall energy. Defaults to 0.

    Returns:
        list, ndarray[int]: List of attribute discriminator energy scores for each sentence of the batch,
                            Array of the predicted sentiment for each sentence
    """
    raw_score = np.array([0.0] * batch.shape[0])

    # Retrieve raw logits from sentiment classifier model
    output = model_disc(batch)["logits"]

    # Make predictions
    pred = np.argmax(np.array(output.log_softmax(dim=-1).cpu().detach()), axis=-1)

    classes = output.shape[-1]
    for i in range(classes):
        if i == sentiment:
            raw_score += np.array(output[:, i].cpu().detach())

    return [-1.0 * raw_s * alpha for raw_s in raw_score], pred


def perplexity_score(batch, beta=1):
    """The new method to calculate the fluency energy score using the perplexity retrieved from pyplexity

    Args:
        batch (torch.Tensor): batch of tokenized sentences with numerical values
        beta (int, optional): Weight in calculating the overall energy. Defaults to 1.

    Returns:
        list: List of the new perplexity energy scores for each sentence of the batch
    """
    # this package works with sentences as strings -> batch needs to be converted back
    untoken = untokenize_batch(batch)
    sentences = tokens_to_sentences(untoken)

    per = []
    # evaluate perplexity for each sentence
    for sentence in sentences:
        perpl = model_perp.compute_sentence(sentence)
        per.append(perpl)


    return [perplexity * beta for perplexity in per]


def fluency_energy(batch, beta):
    """Depending on the attribute --fluency calculates one of the methods to calculate a fluency energy score

    Args:
        batch (torch.Tensor): batch of tokenized sentences with numerical values
        beta (int, optional): Weight in calculating the overall energy.

    Returns:
        list: List of the relevant fluency energy scores for each sentence of the batch
    """
    if args.fluency == "pyplexity":
        return perplexity_score(batch, beta)
    elif args.fluency == "mlm":
        return energy_score_mlm(batch, beta)


def parallel_sequential_generation(
        batch: torch.Tensor, model_disc, args3, sentiment=0, batch_size=10, max_iter=300
):
    """Main execution of mix-and-match method.

    Args:
        batch (torch.Tensor): batch of tokenized sentences with numerical values
        model_disc (AutoModelForSequenceClassification): sentiment classifier
        args3 (ArgumentParser): Object for parsing command line strings into Python objects.
        sentiment (int, optional): The actual labeled sentiment for this sentence. Defaults to 0.
        batch_size (int, optional): The amount of how many sentences will be generated in each batch. Defaults to 10.
        max_iter (int, optional): How many iterations will be performed. Defaults to 300.

    Returns:
        torch.Tensor, list, list: batch of string-tokenized sentences, metadata for each masked position step,
                                  metadata for each iteration step
    """

    batch_original = batch.detach().clone()

    seq_len = batch.shape[-1] - 2
    posns = [i + 1 for i in range(seq_len)]

    full_meta_data = [[] for i in range(batch_size)]
    meta_data = []

    # Create min and max values for each attribute for min-max-normalization
    global_max_value = None
    global_min_value = None

    global_max_value_disc = None
    global_min_value_disc = None
    for iteration in range(max_iter):
        print("Iteration {}".format(iteration))

        # if true then the masked position changes randomly without order
        if args3.shuffle_positions:
            random.shuffle(posns)

        nmasks = 1
        groups = [posns[i: i + nmasks] for i in range(0, len(posns), nmasks)]
        if args3.shuffle_positions:
            random.shuffle(groups)
        for positions in groups:
            # Energy scores of the current sentence
            # Hamming-Distance
            distance = np.sum(1 - np.array((batch == batch_original).detach().cpu()) * 1, axis=-1)
            # Sentiment classifier energy score
            disc_score, disc_preds = energy_score_disc(
                batch, model_disc=model_disc, sentiment=sentiment, alpha=args3.alpha
            )

            # Fluency energy score
            fluency_score = fluency_energy(batch, args3.beta)


            #Normalization of energy values from the current sentence
            if args.normalize:
                (
                    fluency_score,
                    global_max_value,
                    global_min_value,
                ) = min_max_normalization(fluency_score, global_max_value, global_min_value
                )

                (
                    disc_score,
                    global_max_value_disc,
                    global_min_value_disc,
                ) = min_max_normalization(disc_score, global_max_value_disc, global_min_value_disc
                )

            # Summing up to overall energy score
            old_r = np.array(fluency_score) + np.array(disc_score)
            old_r += args3.delta * distance
            old_wrd = batch[:, positions].detach().clone()

            # Masking a token
            batch[:, positions] = mask_id


            #Sampling of new word with Bert
            output = model(batch)["logits"][:, positions, :]
            output[:, :, mask_id] = -10000000000.0
            output = output.softmax(dim=-1)

            qxbx = np.array([1.0] * batch_size)
            qxxb = np.array([1.0] * batch_size)

            d = Categorical(output)
            new_wrd = d.sample()
            # Print out newly proposed tokens
            print(untokenize_batch(new_wrd))
            n_flag = np.array([0] * batch_size)
            msk_change = [False] * batch_size

            for ii in range(len(positions)):
                for jj in range(batch_size):
                    qxxb[jj] *= output[jj, ii, old_wrd[jj, ii]].cpu()
                    qxbx[jj] *= output[jj, ii, new_wrd[jj, ii]].cpu()

                    if not (old_wrd[jj, ii].item() == new_wrd[jj, ii].item()):
                        n_flag[jj] = 1
                    if old_wrd[jj, ii].item() == mask_id:
                        msk_change[jj] = True

            batch[:, positions] = new_wrd
            # Energy scores of the sentence
            # Hamming-Distance
            distance_new = np.sum(
                1 - np.array((batch == batch_original).detach().cpu()) * 1, axis=-1)
            # Sentiment classifier energy score
            disc_1_new, disc_preds_new = energy_score_disc(
                batch, model_disc=model_disc, sentiment=sentiment, alpha=args3.alpha
            )
            # Fluency energy score
            fluency_1_new = fluency_energy(batch, args3.beta)

            # Normalization of energy values from the new sentence
            if args.normalize:
                (
                    fluency_1_new,
                    global_max_value,
                    global_min_value,
                ) = min_max_normalization(fluency_1_new, global_max_value, global_min_value)

                (
                    disc_1_new,
                    global_max_value_disc,
                    global_min_value_disc,
                ) = min_max_normalization(disc_1_new, global_max_value_disc, global_min_value_disc
                )
            new_r = np.array(fluency_1_new) + np.array(disc_1_new)
            new_r += args3.delta * distance_new


            # Metropolis-Hastings-Correction Step
            # Calculate acceptance probability
            axbx = np.where(msk_change,1.0,np.minimum(1.0,np.divide(
                        np.multiply(np.exp(old_r - new_r), np.array(qxxb)),np.array(qxbx),),),)

            # Transform probabilities to decisions
            acc = torch.squeeze(torch.bernoulli(torch.Tensor([axbx])))

            # Apply correction or rejection
            batch[:, positions] = torch.where(
                acc.unsqueeze(1).repeat(1, len(positions)) > 0.0,
                batch[:, positions],
                old_wrd,
            )

            # Prepare values for meta datas
            r_score = np.squeeze(np.where(acc > 0.0, new_r, old_r))
            disc_preds = np.squeeze(np.where(acc > 0.0, disc_preds_new, disc_preds))
            distance = np.squeeze(np.where(acc > 0.0, distance_new, distance))

            acc = np.array(acc.cpu()) * np.array(n_flag)

            # Prepare full meta data
            for i in range(batch_size):
                full_meta_data[i].append(
                    (
                        sentiment,
                        r_score[i],
                        qxxb[i],
                        qxbx[i],
                        axbx[i],
                        acc[i].item(),
                        disc_preds[i],
                        distance[i],
                    )
                )

    # Print out new generated sentences this iteration
    print(untokenize_batch(batch))


    # Prepare meta data
    for i in range(batch_size):
        meta_data.append(
            (
                sentiment,
                r_score[i],
                qxxb[i],
                qxbx[i],
                axbx[i],
                acc[i].item(),
                disc_preds[i],
                distance[i],
            )
        )

    return untokenize_batch(batch), meta_data, full_meta_data


def detokenize(sent):
    """Forms the numerical representation of a tokenized sentence back to it's string representation

    Args:
        sent (torch.Tensor): tokenized sentence with numerical values

    Returns:
        string: sentence
    """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent


def get_opt_sent(sents, metadata):
    """Determines the sentence with the lowest energy score out af all final sentences from the batch

    Args:
        sents (torch.Tensor): batch of string-tokenized sentences
        metadata (list): list with all important values retrieved during the mix-and-match method

    Returns:
        string, ind: Optimal sentence, Index of sentence with lowest energy inside batch
    """
    meta_array = np.array(metadata)

    ind = np.argmin(meta_array[:, 1, ...])
    sent_best = sents[ind].split()
    return " ".join(sent_best[1:-1]), ind


def generate_output(sents: list, full_meta_data: list, opt_sent: str, ind: int):
    """Creates the output-files metadata.txt, opt_meta.txt, samples.txt and opt_samples.txt in the output-folder

    Args:
        sents (list): list of the finally generated batch of sentences
        full_meta_data (list): metadata for each masked position step
        opt_sent (string): best generated sentence
        ind (int): index of the best generated sentence from the whole batch
    """
    with open(f"{dirname}/samples.txt", "a") as f, open(
            f"{dirname}/opt_samples.txt", "a"
    ) as optimal_f, open(f"{dirname}/opt_meta.txt", "a") as opt_meta_file, open(
        f"{dirname}/metadata.txt", "a"
    ) as f_meta:
        # samples.txt file with the final batches for each sentence
        f.write("\n".join(sents) + "\n")
        f.flush()

        # metadata.txt file with metadata for each masked position step
        full_meta_data_str = [str(l) for l in full_meta_data]
        f_meta.write("\n".join(full_meta_data_str) + "\n")
        f_meta.flush

        # opt_samples.txt file with the best sentences from each batch
        optimal_f.write(opt_sent + "\n")
        optimal_f.flush()

        # opt_meta.txt file with the metadata for the best sentence from each batch
        opt_meta_str = str(full_meta_data[ind])
        opt_meta_file.write(opt_meta_str + "\n")
        opt_meta_file.flush()


# End step 4 ---------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """Main Code execution. This code runs when mix_and_match.py is executed
    """

    # 1. Parse arguments
    args = parse_arguments()

    # 2. Load models
    # --------------------------------------------------------------------
    model, tokenizer, model_disc, model_perp = load_models()
    CLS = "[CLS]"
    SEP = "[SEP]"
    MASK = "[MASK]"
    mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
    sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
    cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]
    # --------------------------------------------------------------------

    # 3. Create output folders
    # --------------------------------------------------------------------
    dirname = create_output_folder()
    # --------------------------------------------------------------------

    # 4. Main execution
    # --------------------------------------------------------------------

    if args.single:
        while True:
            user_input = input("Geben Sie einen Satz ein (oder 'exit' zum Beenden): ")
            if user_input.lower() == "exit":
                break
            else: generate_samples(user_input)
    else: generate_samples()
    # --------------------------------------------------------------------