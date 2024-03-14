import argparse
import datetime
import math
import os
import time
from sample_batched_input_improved import parallel_sequential_generation, detokenize, get_opt_sent

def generate_samples(dirname: str, data_dir: str, attr_dir: str, input: str | None = None):
    """"""
    with open(f"{dirname}/samples.txt", "w") as f, \
         open(f"{dirname}/opt_samples.txt", "w") as optimal_f, \
         open(f"{dirname}/opt_cls.txt", "w") as optimal_class, \
         open(f"{dirname}/opt_meta.txt","w") as opt_meta_file, \
         open(f"{dirname}/metadata.txt", "w") as f_meta:

        if input:
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


def prepare_input(line, sentiment, f_meta, f):
    """"""
    seed_text = line[:-1]
    print(seed_text)
    seed_text = tokenizer.tokenize(seed_text)
    print(seed_text)
    bert_sents = []
    start_time = time.time()
    batch, meta_data, full_meta_data = parallel_sequential_generation(
        seed_text,
        model_disc=model_disc,
        batch_size=batch_size,
        sentiment=sentiment,
        max_len=max_len,
        temperature=temperature,
        max_iter=args.max_iter,
        args3= args
    )


    print("Finished batch in %.3fs" % (time.time() - start_time))
    start_time = time.time()

    bert_sents += batch

    sents = list(map(lambda x: " ".join(detokenize(x)), bert_sents))

    f.write("\n".join(sents) + "\n")
    f.flush()

    full_meta_data_str = [str(l) for l in full_meta_data]
    f_meta.write("\n".join(full_meta_data_str) + "\n")
    f_meta.flush

    opt_sent, opt_cls, ind = get_opt_sent(sents, meta_data)

    return opt_sent, opt_cls, full_meta_data[ind]


if _name_ == "_main_":
    parser = argparse.ArgumentParser(description="Generate samples based on user input sentences.")
    args = main()

    model, tokenizer, tokenizer_disc, model_disc, model_perp = load_models()



    CLS = "[CLS]"
    SEP = "[SEP]"
    MASK = "[MASK]"
    mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
    sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
    cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]

    #
    degenerate = args.degenerate
    temperature = args.temperature
    dirname = args.out_path
    n_samples = args.n_samples
    batch_size = args.batch_size


    max_len = 1  # this is dummy!!
    ########


    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    folder_name = "disc_{}data{}max_iter{}date{}".format(args.disc_name, args.data_name, args.max_itedt_string)

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