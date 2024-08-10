# 3DStyleGLIP: Part-Tailored Text-Guided 3D Neural Stylization
![teaser](./result_images/1. teaser/teaser.png)
**3DStyleGLIP** is a method specifically designed for text-driven, part-tailored 3D stylization. Given a 3D mesh and a text prompt, 3DStyleGLIP utilizes the vision-language embedding space of the Grounded Language-Image Pre-training (GLIP) model to localize individual parts of the 3D mesh and modify their appearance to match the styles specified in the text prompt. 3DStyleGLIP effectively integrates part localization and stylization guidance within GLIP’s shared embedding space through an end-to-end process, enabled by part-level style loss and two complementary learning techniques.

## Instruction
### Installation
### System Requirements
### Run Demo

## Neural Stylization
![L_results](./result_images/2. L_v_results/L_results.png)
![artistic_bags](./result_images/2. L_v_results/artistic_bags.png)
![D_results](./result_images/3. D_v_results/D_results.png)

## Ablation Study
![D_ablation](./result_images/4. ablation (L&D)/D_ablation.png)
![L_ablation](./result_images/4. ablation (L&D)/L_ablation.png)
