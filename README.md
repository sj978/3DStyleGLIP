# 3DStyleGLIP: Part-Tailored Text-Guided 3D Neural Stylization
![teaser](./result_images/1_teaser/teaser.png)
**3DStyleGLIP** is a method specifically designed for text-driven, part-tailored 3D stylization. Given a 3D mesh and a text prompt, 3DStyleGLIP utilizes the vision-language embedding space of the Grounded Language-Image Pre-training (GLIP) model to localize individual parts of the 3D mesh and modify their appearance to match the styles specified in the text prompt. 3DStyleGLIP effectively integrates part localization and stylization guidance within GLIP’s shared embedding space through an end-to-end process, enabled by part-level style loss and two complementary learning techniques.

## Instruction
### Coming Soon

## Neural Stylization
### SVBRDF & lighting based method
![L_results](./result_images/2_L_v_results/L_results.png)

### Vertex displacement & color based method
![D_results](./result_images/3_D_v_results/D_results.png)


## Several Results
### Artistic bags
![artistic_bags](./result_images/2_L_v_results/artistic_bags.png)

### Complex objects

### limitations

## Ablation Study
![D_ablation](./result_images/4_ablation/D_ablation.png)
![L_ablation](./result_images/4_ablation/L_ablation.png)
