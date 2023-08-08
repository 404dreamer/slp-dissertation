# Online Implementation for Efficient Training of Diffusion Language Models

This code is based on the implementation of: [**Latent Diffusion for Language Generation**](https://arxiv.org/abs/2212.09462). 


## Training Track
All models involved in the thesis can be tracked through [wandb](https://wandb.ai/404dreamer/denoising_diffusion). The default random seed is 42. 



## Environment
A suitable environment can be created with the following commands. 
```bash
conda env create -f environment.yml
python -m spacy download en_core_web_sm
```

## Datasets

The dataset files for the E2E and ROCStories datasets are included in the `datasets/` directory and do not require any additional processing.

## Annealing Functions
We provide three kinds of annealing functions mentioned in `./diffusion/denoising_diffusion.py`: 
  
**total_step**: total number of training steps (forward-backward steps)  
**timesteps**: entire diffusion range during training   
**curr_steps**: current training step  




```python
def linear_T_schedule(total_step, timesteps, curr_step):
    modified_steps = int(total_step / 2)
    if curr_step >= modified_steps:
        return timesteps

    min_T = int(timesteps / 5)
    modified_T = int(min_T + (curr_step / modified_steps) * (timesteps - min_T))
    modified_T = min(modified_T, timesteps)

    return modified_T


def polynomial_T_schedule(total_step, timesteps, curr_step, power=2):
    modified_steps = int(total_step / 2)
    if curr_step >= modified_steps:
        return timesteps

    min_T = int(timesteps / 5)
    modified_T = int(min_T + ((curr_step / modified_steps) ** power) * (timesteps - min_T))
    modified_T = min(modified_T, timesteps)

    return modified_T


def cosine_T_schedule(total_step, timesteps, curr_step):
    modified_steps = int(total_step / 2)
    if curr_step >= modified_steps:
        return timesteps

    min_T = int(timesteps / 5)
    cosine_scale = 0.5 * (1 + np.cos(np.pi * (curr_step / modified_steps)))
    modified_T = int(min_T + (1 - cosine_scale) * (timesteps - min_T))

    return modified_T

```



## Training

We provide scripts to train the diffusion models for each dataset with our default hyperparameters. Train a model with the command   
  
**dataset_name**: Dataset to train models. `{roc, e2e}`  
**T_schedule**: Annealing function. `{normal, cosine, linear, polynomial}`, where the 'normal' refers to train without annealing diffusion ranges.  
**polynomial_T_power**: Power of the polynomial annealing function.  
**num_samples**: Number of samples to generate for evaluation.  
**save_and_sample_every**: For a specific number of training steps, evaluating the generated samples.   
**optimizer**: The choice of optimizers. `{adamW, lion}`

```bash
python train_text_diffusion.py  --dataset_name e2e \ 
                                --adam_weight_decay 0.01 \
                                --learning_rate 1e-4 \
                                --num_train_steps 80000 \
                                --train_batch_size 64 \
                                --tx_dim 768 \
                                --tx_depth 12 \
                                --objective pred_x0 \
                                --T_schedule polynomial \
                                --polynomial_T_power 2\
                                --enc_dec_model facebook/bart-base \
                                --num_samples 1000 \
                                --self_condition \
                                --normalize_latent \
                                --optimizer adamW \
                                --scale_shift \
                                --loss_type l1 \
                                --random_seed 41 \
                                --save_and_sample_every 5000 \
                                --wandb_name e2e_polynomial_seed_41 \
                                --beta_schedule linear
``` 

## Evaluation (from the origial implementation)
To evaluate a trained model on the validation set, see the `scripts/diffusion/eval_text_diffusion.sh` script for an example. The `--resume_dir` argument should be updated with the path of a trained model. 


Different sampling configurations can be explored by changing the `{num_samples, sampling_timesteps, ddim_sampling_eta}` arguments. We utilize 1,000 random samples for computing the metrics in our work. Note that MAUVE scores computed with different numbers of samples are not directly comparable (see [here](https://github.com/krishnap25/mauve) for more information about MAUVE scores).

To evaluate a trained model on the test set with 5 random seeds, see the `scripts/diffusion/test_eval_text_diffusion.sh` script for an example. The only difference is that the `eval_test` flag is used instead of the `eval` flag. The `--resume_dir` argument will need to be updated as before.


## Acknowledgement
This work built upon excellent open-source implementations from [LD4LG](https://github.com/justinlovelace/latent-diffusion-for-language).