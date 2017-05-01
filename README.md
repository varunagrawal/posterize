# posterize
Semantic Style Transfer

Semantic Segmentation - [Sharpmask](https://github.com/facebookresearch/deepmask)
Style Transfer - [Texture Nets](https://github.com/DmitryUlyanov/texture_nets)


## How To

### Setup the Project

Clone the repository and then clone my updated version of [deepmask](https://github.com/varunagrawal/deepmask) and [texture_nets](https://github.com/DmitryUlyanov/texture_nets) repositories and set the `$POSTERIZE_SEGMENTATION` and `$POSTERIZE_STYLER` variables to the locations of the repos.

I don't use submodules since this is research and new things keep coming in daily.


### Train the Texture Net

```
th $POSTERIZE_STYLER/train.lua -data coco  -style_image <path-to-style-image> -style_size 600 -image_size 512 -model johnson -batch_size 2 -learning_rate 1e-2 -style_weight 10 -style_layers relu1_2,relu2_2,relu3_2,relu4_2 -content_layers relu4_2 -num_iterations 3000
```


### Stylize the Image

```
th $POSTERIZE_STYLER/test.lua -input_image input.jpg -model_t7 data/checkpoints/model_3000.t7
```

### Compute the Segmentation Masks

This saves the masks and generates a file called `num_masks` which has the number of masks generated.

We use the Sharpmask algorithm.

```shell 
th $POSTERIZE_SEGMENTATION/computeProposals.lua pretrained/sharpmask -img data/steven.jpg
```

### Generate the Foreground and Background images.

```shell
python3 get_img_segmentations.py input.jpg <num_masks>
```


## Key Objective

Get the style transfer to work especially well on face images and self-portraits (selfies).

I use the LFW dataset to train the Texture Net hoping that it learns a face prior.