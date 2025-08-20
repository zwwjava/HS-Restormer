
# Stamp Removal and Extraction Based on High-Pass Spatial Transposition Self-Attention

<hr />

> **Abstract:** *In banking, securities, and commercial sectors, stamp removal and extraction serve as critical prerequisites for both document OCR and stamp OCR systems. Existing models, however, often fail to deliver satisfactory performance when processing document images with complex backgrounds and overlapping stamps. To address these challenges, we propose HS-Restormer, a novel model designed for joint stamp removal and extraction. Regardless of whether it is for the stamp extraction task or the stamp removal task, the high-pass spatial enhancement technology of the model helps distinguish the foreground and background information of the stamp. The multi-layer feature fusion technology proposed in this paper can efficiently integrate high-dimensional and low-dimensional features, thereby ensuring the recovery of pixel details in stamp extraction and removal tasks. In addition to evaluations specific to stamps, we further validate HS-Restormer on public benchmarks including the watermark dataset CLWD and deraining datasets Rain100H/Rain100L, where it achieves state-of-the-art (SOTA) performance across all tasks.* 
<hr />



## Installation

See [INSTALL.md](INSTALL.md) for the installation of dependencies required to run HS-Restormer.



## Demo

To test the pre-trained HS-Restormer models
```
python demo.py --task Task_Name --input_dir path_to_images --result_dir save_images_here
```



## Results
Experiments are performed for different stamp image processing tasks

<details>
<summary><strong>stamp removal</strong> (click to expand) </summary>
  
![image](https://github.com/zwwjava/HS-Restormer/blob/main/image/stamp_removal.png)
</details>

<details>
<summary><strong>stamp removal</strong> (click to expand) </summary>

![image](https://github.com/zwwjava/HS-Restormer/blob/main/image/stamp_extraction.png)
</details>



