# Pro4S
Pro4S is a novel multimodal model that unifies qualitative and quantitative protein solubility prediction. By integrating protein language models, structural data, and surface descriptors via contrastive learning, it achieves superior accuracy and robustness. Pro4S outperforms existing methods and effectively screens *de novo* proteins, significantly increasing experimental expression success to accelerate protein engineering.   


![image](https://github.com/TEKHOO/Pro4S/blob/main/Pro4S.png)
## Dataset
The dataset can be found in `/Pro4S/dataset`


## Software Prerequisites
* [ESM2-3B](https://github.com/facebookresearch/esm) - We use esm2_t36_3B_UR50D to get sequence embeddings.
* [H5py](https://docs.h5py.org/en/stable/quick.html#quick) ≥ 3.11.0
* [Pytorch](https://pytorch.org/) ≥ 1.12.0
* [Pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning) ≥ 2.1
* [Torch-geometric](https://github.com/pyg-team/pytorch_geometric) ≥ 2.5.3
* [Torchmetrics](https://lightning.ai/docs/torchmetrics/stable/) ≥ 0.9.3
* [Masif](https://github.com/LPDI-EPFL/masif) - We use Masif to get protein surface features.

    To replicate the Conda environment used in this project, follow the steps below:

    Run the following code:
    ```bash
    
    conda env create -f environment.yml
    conda activate pro4s

    ```
    Our CUDA version is ≥ 12.2. If your CUDA version does not match and causes the YML file to be unusable, please modify the YML file or adjust your CUDA version.
   
    Additionally, it is necessary to install an environment under conda that can successfully run [Masif](https://github.com/LPDI-EPFL/masif), as instructed by [Masif](https://github.com/LPDI-EPFL/masif), and name it `masif`.
    
## Make predictions
1. **Download Model files**  

    Model can be downloaded from [huggingface](https://huggingface.co/qj666/Pro4S/tree/main).  

2. **Input Files**
   
    Place the PDB files in `/Pro4S/test/pdb`.

    ⚠️ It is recommended that the PDB file contain only the single protein chain to be predicted, and it should be named Chain A.
   
    If you have a solubility label file for the protein, place it as `/Pro4S/test/test.txt`. If not, the model will default the true label to 0 in the output file.

3. **Calculate Protein Feature**  
   
    Run `masif.sh`:

    ⚠️ If the PDB file name contains special characters, we will automatically exclude it to prevent errors in Masif.
    
    ```bash
    cd Pro4S/scripts/
    conda activate masif
    bash masif.sh
    ```
    
    Run `seq_structure_feature.sh`:
    
    ```bash
    conda deactivate
    conda activate pro4s
    bash seq_structure_feature.sh
    ```

4. **Run the Model**  
   
    By default, we use the finetune model from the article for testing. If you need to change it, modify the `-ck_point` . Available options are `classification`, `regression`, and `finetune`.
    
    Run `test.sh`:
    
    ```bash
    conda activate pro4s
    python test.py \
    -ck_point finetune
    ```

6. **View the Output**

    Check the output results in `Pro4S/test/result/test.xlsx`, which provides the two outputs: `Prediction` and `Label` values.

    ```bash
    cd Pro4S/test/result/
    ```

## Reference
Jie Qian, *et. al.* Pro4S: prediction of protein solubility by fusing sequence, structure, and surface, *submitted*  
