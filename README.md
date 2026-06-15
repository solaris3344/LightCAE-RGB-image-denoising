[model_zoo](model_zoo)
--------
- download link [https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D](https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D)
 ------------------------------------------------------------------------------------------
 Download SCUNet models
```python
python main_download_pretrained_models.py --models "SCUNet" --model_dir "model_zoo"
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Testing (kodak24,cbsd68,McM)
----------------------------------------------------------------------------------------------------------------------------
dncnn: [main_test_dncnn.py](main_test_dncnn.py) |```dncnn_15.pth, dncnn_25.pth, dncnn_50.pth, dncnn_gray_blind.pth, dncnn_color_blind.pth, dncnn3.pth```|
test: python main_test_dncnn.py --model_name dncnn_color_blind --testset_name kodak24 --noise_level_img 25
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ircnn: [main_test_ircnn_denoiser.py](main_test_ircnn_denoiser.py) | ```ircnn_gray.pth, ircnn_color.pth```| 
    noise_level_img = 25             
    model_name = 'ircnn_color'        
    testset_name = 'kodak24'
test: python main_test_ircnn_denoiser.py
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ffdnet: [main_test_ffdnet.py](main_test_ffdnet.py) | ```ffdnet_gray.pth, ffdnet_color.pth, ffdnet_gray_clip.pth, ffdnet_color_clip.pth```|
    noise_level_img = 25                 
    noise_level_model = noise_level_img  
    model_name = 'ffdnet_color'          
    testset_name = 'kodak24'
test: python main_test_ffdnet.py
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
swinir: [main_test_swinir.py](main_test_swinir.py) |
test: python main_test_swinir.py --task color_dn --noise 25 --folder_gt testsets/kodak24 --model_path model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
scunet: [main_test_scunet_color_gaussian.py] |
color images;
test: python main_test_scunet_color_gaussian.py --model_name scunet_color_25 --noise_level_img 25 --testset_name kodak24
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ADNet: [color/test_c.py] |
color images;
test: python test_c.py --num_of_layers 17 --logdir c25 --test_data kodak24 --test_noiseL 25 
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DRUNet: [main_dpir_denoising.py] 
    noise_level_img = 25                # set AWGN noise level for noisy image
    noise_level_model = noise_level_img  # set noise level for model
    model_name = 'drunet_color'           # set denoiser model, 'drunet_gray' | 'drunet_color'
    testset_name = 'McM'               # set test set,  'McM' | 'cbsd68' | 'KODAK24'
test:  python main_dpir_denoising.py
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

[testsets](testsets)
-----------
- [cbsd68](https://github.com/cszn/FFDNet/tree/master/testsets)
- [kodak24](https://github.com/cszn/FFDNet/tree/master/testsets)
-[MacMaster](https://github.com/cszn/FFDNet/tree/master/testsets)->McM

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Performance test(RGB): python flops_all_models2.py
Output;
========================================================================
OK LightCAE v2.2
OK DnCNN 
OK IRCNN 
OK FFDNet 
OK ADNet 
OK DRUNet (KAIR)
OK SCUNet
OK SwinIR

========================================================================
  Model                            Params      MACs    GFLOPs     Time
  --------------------------------------------------------------------
  LightCAE  (Ours)                 1.501M    24.41G    48.81G     8.3ms <--
  DnCNN                            0.558M    36.72G    73.43G    10.1ms
  IRCNN                            0.188M    12.39G    24.78G     3.5ms
  FFDNet                           0.497M     8.17G    16.34G     2.1ms
  ADNet                            0.522M    34.30G    68.60G     9.1ms (update->Params=1.5M)
  DRUNet                          32.654M   143.61G   287.22G    27.8ms
  SCUNet                          17.946M    79.93G   159.85G    62.7ms
  SwinIR                          11.504M   752.13G  1504.27G   640.1ms
========================================================================


SIDD test
Deamnet(Note : change Benchmark_test.py -->Benchmark_test_updated.py),https://github.com/chaoren88/DeamNet/blob/main
python Benchmark_test_updated.py --Type SIDD --data_folder "...path\SIDD_Val_PNG\Noisy" --out_folder "...path\SIDD_Val_PNG\Deam_Denoised"

Ablation models;
https://drive.google.com/file/d/1TpNKx0p5RVcIjTWbJ1wtLvQd6vGBwp6S/view?usp=drive_link
https://drive.google.com/file/d/1q3DZvaUgfoEIqPVsBwo2GL-e8vUZEMNz/view?usp=drive_link


RGB ours models;
https://drive.google.com/file/d/1q3DZvaUgfoEIqPVsBwo2GL-e8vUZEMNz/view?usp=drive_link    (sigma=15/255)
https://drive.google.com/file/d/1q3DZvaUgfoEIqPVsBwo2GL-e8vUZEMNz/view?usp=drive_link    (sigma=25/255)
https://drive.google.com/file/d/1q3DZvaUgfoEIqPVsBwo2GL-e8vUZEMNz/view?usp=drive_link    (sigma=50/255)

SIDD ours test datas;
https://drive.google.com/file/d/1GcOSXBfcYQfRG63VBvESlD22TcYJXDfz/view?usp=drive_link


SIDD general test datas;
https://drive.google.com/file/d/1qoazB2neUIPI6K6UGxIVro2xorwKLIUH/view?usp=drive_link

DOI: 10.5281/zenodo.20693416
URL: https://doi.org/10.5281/zenodo.20693416
Github Link: https://github.com/solaris3344/LightCAE-RGB-image-denoising/releases/tag/v1.0.0






