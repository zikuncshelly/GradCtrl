# Exploring Gradient-based Multi-directional Controls in GANs

> Zikun Chen<sup>1</sup>, Ruowei Jiang<sup>1</sup>, Brendan Duke<sup>1,2</sup>, Han Zhao<sup>3</sup>, Parham Aarabi<sup>1,2</sup><br>
> <sup>1</sup>ModiFace Inc., <sup>2</sup>University of Toronto, <sup>3</sup>University of Illinois at Urbana-Champaign<br>
> https://arxiv.org/abs/2209.00698
>
> <p align="justify"><b>Abstract:</b> <i>Generative Adversarial Networks (GANs) have been widely applied in modeling diverse image distributions. However, despite its impressive applications, the structure of the latent space in GANs largely remains as a black-box, leaving its controllable generation an open problem, especially when spurious correlations between different semantic attributes exist in the image distributions. To address this problem, previous methods typically learn linear directions or individual channels that control semantic attributes in the image space. However, they often suffer from imperfect disentanglement, or are unable to obtain multi-directional controls. In this work, in light of the above challenges, we propose a novel approach that discovers nonlinear controls, which enables multi-directional manipulation as well as effective disentanglement, based on gradient information in the learned GAN latent space. More specifically, we first learn interpolation directions by following the gradients from classification networks trained separately on the attributes, and then navigate the latent space by exclusively controlling channels activated for the target attribute in the learned directions. Empirically, with small training data, our approach is able to gain fine-grained controls over a diverse set of bi-directional and multi-directional attributes, and we showcase its ability to achieve disentanglement significantly better than state-of-the-art methods both qualitatively and quantitatively.
.</i></p>

![image](./images/overview.png)

**Figure:** *Our method's semantic interpolation in a nonlinear vector field.
Directions corresponding to the female semantic in GAN latent space (left) are determined by differentiating a simple neural network classifier (middle).
Disentanglement is achieved by masking channels of the female semantic gradient vector that are highly salient for non-target attributes such as glasses (right).*

![image](./images/more_ffhq1.png)
![image](./images/more_ffhq2.png)
![image](./images/more_cars.png)
**Figure:** *Binary attribute editing results for StyleGAN2-FFHQ, LSUN-Car.*
![image](./images/multiclass.png)
**Figure:** *Multi-attribute editing results for StyleGAN2-LSUN-Car, LSUN-Cat.*

## Manipulation Demo for StyleGAN2-FFHQ
We use the official StyleGAN2 repo (https://github.com/NVlabs/stylegan2) under `stylegan2`.
```
python manipulate.py \
--model_dir model_ffhq \
--attribute $ATTRIBUTE_NAME \
--out_dir out_ffhq
```
$ATTRIBUTE_NAME can be eyeglasses/age/gender/smile.

We also provide pretrained models for LSUN cats and cars with $ATTRIBUTE_NAME = color (for cats) or sky_blue/grassground/hubcap_dense (for cars). Specify `--model cat/car` and change the corresponding directories & attribute names to obtain the editing results.

## Usage
We use the official StyleGAN2 model trained on FFHQ dataset as an example.

### Prepare data

* Sample latent codes from $W$ space and the corresponding images with `python sample.py --out_dir $GAN_OUTPUT_DIRECTORY` 
* Use your own image classifier to obtain semantic scores for each attribute `$ATTRIBUTE_NAME` in all images, and save the outputs as `$GAN_OUTPUT_DIRECTORY/logits/logits_$ATTRIBUTE_NAME.pkl`. The outputs should be of shape ($NUM_SAMPLES, $NUM_CLASSES), e.g., for binary classification, $NUM_CLASSES = 1.
* Run `python prepare_data.py --data_dir $GAN_OUTPUT_DIRECTORY --out_dir $DATA_OUTPUT_DIRECTORY --attributes $ATTRIBUTE_NAME_1,ATTRIBUTE_NAME_2,...` to generate the training/validation sets for our model. 

### Train the model
* `python train.py --attributes $ATTRIBUTE_NAME_1,ATTRIBUTE_NAME_2,... --data_dir $DATA_OUTPUT_DIRECTORY --out_dir $MODEL_DIRECTORY` to learn editing directions for the specified attributes. 

<!-- For non-binary attributes, specify the number of classes with `--multiclass_classes`. -->

### Manipulate latent semantics
```bash
python manipulate.py 
 --input_dir $INPUT_DIRECTORY \
 --model_dir $MODEL_DIRECTORY \
 --attribute $ATTRIBUTE_NAME \
 --out_dir $EDIT_OUTPUT_DIRECTORY
 ```
To set the disentanglement parameters manually, specify `--exclude $ATTR1,ATTR2... --top_channels $ATTR1_CHANNEL,ATTR2_CHANNEL...`. 
<!-- To edit multiclass attribute, specify  -->
