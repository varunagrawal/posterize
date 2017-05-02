# convert /home/varun/projects/posterize/input.jpg -resize 512x512! /home/varun/projects/posterize/input.jpg

cd /home/varun/projects/deepmask
th computeProposals.lua pretrained/sharpmask -img ../posterize/input.jpg
mv mask_*.jpg ../posterize/masks/
cp num_masks ../posterize/masks/

# cd /home/varun/projects/posterize
# python2 generate_face_mask_segmentations.py input.jpg

# Texture Net
cd ../texture_nets
th test.lua -input_image ../posterize/input.jpg -model_t7 data/checkpoints/frida_model_30_3000.t7
cp stylized.jpg ../posterize/

# Adaptive IN
# cd /home/varun/projects/AdaIN-style
# th test.lua -content /home/varun/projects/posterize/input.jpg -style /home/varun/projects/posterize/styles/frida_kahlo.jpg
# cp output/input_stylized_frida_kahlo.jpg ../posterize/stylized.jpg

cd /home/varun/projects/posterize
python3 get_img_segmentations.py input.jpg `cat masks/num_masks`
python2 blend.py `cat masks/num_masks`
