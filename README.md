# EMNLP20: Centering-based Neural Coherence Modeling with Hierarchical Discourse Segments
### [Sungho Jeon](https://sdeva14.github.io/) and [Michael Strube](https://www.h-its.org/people/prof-dr-michael-strube/)
#### [NLP Lab, Heidelberg Institute for Theoretical Studies (HITS)](https://www.h-its.org/research/nlp/people/)

This project contains a python implementation for the EMNLP20 paper whose title is "Centering-based Neural Coherence Modeling with Hierarchical Discourse Segments".
<!-- 
## Updates (2021.01.09.)
If you downloaded it before 2021.01.09. please update to a newer version. Previously, the codes used in development were uploaded, and it causes performance degradation. We also add a new option to decide an encoding type of texts, whether encoding a document at once or sentences individually for a structure-aware transformer input. Regarding this, please see our COLING20 paper, "Incremental Neural Lexical Coherence Modeling".
 -->
## Requirements

#### Conda environment
We recommend using Conda environment for a setup. It is easy to build an environment by the provided environment file. It is also possible to setup an environment manually by the information in "spec-file.txt". 

Our environment file is built based on CUDA9 driver and corresponding libraries, thus an environment should be managed by the target GPU environment. Otherwise, GPU flag should be disabled as a library. For the variation of XLNet, we use Transformers library implemented by Huggingface (Wolf et al, 2019).

    conda create --name py3_torch_cuda9 --file spec-file.txt
    source activate py3_torch_cuda9
    pip install transformers==2.4.1
    pip install stanza==1.0.1

#### Dataset and materials
Datasets cannot be attached in the submission due to license problems, hence it should be downloaded from the given links.

- Dataset: The location of the target dataset should be configured in "build_config.py" with "--data_dir" option. The TOEFL dataset is available according to the link in the original paper (Blanchard et al. 2013) with LDC license. The index of the CV partition in TOEFL is attached as a file, whose name is ids_toefl_cv5.tar.gz. The NYT dataset can be downloaded with LDC license. We partition the dataset following previous work (Ferracaneet al. 2019), and attach our pre-processing script for this ("pp_nyt.py").

TOEFL dataset link: https://catalog.ldc.upenn.edu/LDC2014T06

NYT dataset link: https://catalog.ldc.upenn.edu/LDC2008T19

NYT partition link: https://github.com/elisaF/structured

For baselines which do not use a pretrained language model, we use the 100-dimensional pretrained embedding model on Google News, Glove (Pennington, Socher, and Manning 2014). We use the 50-dimensional embeddings on NYT. For our model and baselines employing the pretrained language model, we use the pretrained model "XLNet-base".

Glove link: https://nlp.stanford.edu/projects/glove/

XLNet link: https://github.com/huggingface/transformers/

## Using code to generate segmentations:
(Stephen Wan)

Note: this arcane incantation is just due to my current minimal amendments to the original Joen EMNLP2 codebase where 
logging of data is just done on the training data subset (and I've also co-opted the Joen's "TOEFL" data schema for re-use)

1. Prepare your data in 3 csv files: train.csv, test.csv, valid.csv

(FYI: scripts for PDTB2 prepared in https://github.com/sineswan/ConnRel/blob/main/data_wrapper/create_csv_version_pdtb.py)

```text
Cols: ['essay_id', 'prompt', 'native_lang', 'essay_score', 'essay']
essay_id is the doc_id for your data set (e.g., PDTB2 id: section \d\d + filenum \d\d
prompt: set to 1
native_lang: set to "ENG" (probably ignored)
essay_score: set to 1
essay: the raw text
```

2. repeat the following step 3 times, where you change the file pointer for the train_fold_0.csv (or simply copy/rename files) 
to each of the 3 original csv files.

```shell

#activate environment
conda activate sungho_emnlp20

#change to the data directory 
cd <proj_dir>/dataset/toefl/cv

#copy the CSV files to here
cp <csv_dir>/*.csv .

#change pointer so that train set is pointed to by the train_fold_0.csv file
ln -s train.csv train_fold_0.csv

#run code
#change back to the main directory
python main_segmenter.py --essay_prompt_id_train 1 --essay_prompt_id_test 1 --target_model cent_hds --num_fold 1 --cur_fold 0 --max_epoch 1 --batch_size 1 --gen_log 1

#change pointer so that dev set is pointed to by the train_fold_0.csv file
ln -s dev.csv train_fold_0.csv

# ...rerun python code "main_segmenter.py"

#change pointer so that test set is pointed to by the train_fold_0.csv file
ln -s text.csv train_fold_0.csv

# ...rerun python code "main_segmenter.py"
```

3. Collect the output files

Go to the logs directory and inspect the 3 log subdirectories created as part of the three executions of main_segmenter.py
#concatenate the results (manually) into a single file  

```shell

rm segments.dat
cat <dir1>/log_cp_toefl_1.log >> segments.dat   #where this run was for training data
cat <dir2>/log_cp_toefl_1.log >> segments.dat   #where this run was for validation data
cat <dir3>/log_cp_toefl_1.log >> segments.dat   #where this run was for testing data


rm sentences.dat
cat <dir1>/log_textpro_toefl_1.text.log >> sentences.dat   #where this run was for training data
cat <dir2>/log_textpro_toefl_1.text.log >> sentences.dat   #where this run was for validation data
cat <dir3>/log_textpro_toefl_1.text.log >> sentences.dat   #where this run was for testing data


```
e.g. 

## Run Models
#### Basic run
A basic run is performed by "main.py" with configuration options by providing in terminal or modifying "build_config.py" file.
Detail information about the configuration can be found in the "build_config.py"

	Examples for execution (assume that a data path is given in build_config.py).

    For TOEFL) python main.py --essay_prompt_id_train 1 --essay_prompt_id_test 1 --target_model cent_hds

    For NYT) python main.py --target_model cent_hds

#### The list of models in this framework
	conll17: The automated essay scoring model in Dong et al. (2017)
	emnlp18: The coherence model in Mesgar and Strube (2018)
	latent_doc_stru: The latent learning model in Liu and Lapata (2018)
	dis_avg: The first baseline which averages representations
	dis_tt: The second baseline which combines the averaged XLNet and the tree transformer
	cent_hds: Our model which approximates Centering theory

#### Pre-defined configuration
For convenient reproductions, we provide pre-defined configurations, configurations for RNN-based models (e.g., "toefl_build_config.py") and configurations for XLNet-based models (e.g., "toefl_xlnet_build_config.py") for the two datasets.
The location of the dataset and pretrained embedding layer should be managed properly in "build_config.py".

Note that additional parameters for baseline models should be configured as target models as described in the literatures

## Acknowledge
This implementation was possible thanks to many shared implementations. We describe an original source link at the first line of codes if we use theirs.
