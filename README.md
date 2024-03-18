# Mix and Match

## File Structure

This repository contains three main folders:

- **data:** Contains the Yelp dataset used in this work.
- **mix_match_code:** Inside mix_match_code are two additional folders: batched_MH, responsible for the computation of the actual mix-and-match method, and get_metrics for the evaluation of the results. Inside each of those folders are two files for running the code. One bash file for running the respective code with specific arguments and the python file that will be executed. batched_MH additionally contains a folder for the generated output named output_samples.
- **sample_generations:** Contains generated samples, that where used in the report or by the referenced paper
## Shell Script Execution

To start the mix-and-match method, execute the following command:
```bash
bash ./mix_match_code/batched_MH/scripts/sample_batched.sh 
```


This section describes the bash file that is used to execute the Python code. The arguments passed to the script control the mode and parameters used for generating sentences.
* `--single (true/false)`: If `true`, the code prompts for a user-input sentence instead of processing a dataset. Defaults to `false` (dataset processing).
* `--normalize (true/false)`: When `true`, energy scores are normalized during the mix-and-match process (as explained in Section 3.4). Defaults to `false` (unnormalized scores).
* `--fluency (mlm/pyplexity)`: Selects the method for calculating the fluency energy score: `mlm` or `pyplexity`.
* `--max_iter (integer)`: Sets the maximum number of iterations for the sentence generation process.
* `--alpha (float)`: Coefficient of the attribute model, controlling its influence on generated sentences.
* `--beta (float)`: Coefficient of the fluency model, controlling its influence on generated sentences.
* `--delta (float)`: Coefficient of the faithfulness model, controlling its influence on generated sentences.
* `--disc_dir (path)`: Path to the directory containing the sentiment classification model.
* `--data_path (path)`: Path to the dataset of sentences for sentiment transfer.
* `--out_path (path)`: Path to the directory where the generated positive sentiment sentences will be stored.

**Example Usage:**

```bash
python ./mix_match_code/batched_MH/scripts/sample_batched_input_improved.py \
--single true \
--normalize true \
--fluency pyplexity \
--max_iter 4 \
--alpha 4 \
--beta 1 \
--delta 2 \
--disc_dir [YOUR PATH]/mix_match_chkpts/mix_match_chkpts/yelp_cls_2/models/checkpoint-100 \
--data_path ./data/test_li_100.txt \
--attr_path ./data/test_li.attr \
--out_path ./mix_match_code/batched_MH/output_samples/yelp \
```
**Important:**
Replace [YOUR PATH] in --disc_dir with the actual path to your sentiment classification model directory.
The model can be downloaded here: https://zenodo.org/records/5855005

## Evaluation
To evaluate the results based on the metrics that have been presented in this work, another shell script has to be executed using the following command:

```bash
bash ./mix_match_code/get_metrics/get_metrics.sh
```

In the shell script itself, the attribute `--checkpoint_dir` has to be modified with the path of the created output folder and `--clsf_name` with the path where the model for sentiment classification is located. The attributes `--attr_file`, which leads to the file with the labeled sentiments, and `--text_file`, the Yelp dataset of sentences, can be kept as they are. `--ext_clsf` tells the code that the evaluation should also use an external classifier.

```bash
python -Wignore ./mix_match_code/get_metrics/metric.py \
--checkpoint_dir ./mix_match_code/batched_MH/output_samples/yelp/[YOUR OUTPUT FOLDER] \
--attr_file ./data/yelp/test_li.attr \
--text_file ./data/yelp/test_li_100.txt \
--ref_file ./data/yelp/test_li_reference.txt \
--clsf_name [YOUR PATH]/mix_match_chkpts/mix_match_chkpts/yelp_cls_2/models/checkpoint-100 \
--ext_clsf \
```

After the execution, the evaluated metrics will be printed in the format:

[Int. Classifier Accuracy, Ext. Classifier Accuracy, Hamming-Distance, GPT2-Perplexity, Bert-Score]


Additionally, the result is stored in a `metrics.txt` file in the output folder, which has been used for the `--checkpoint_dir` attribute.

