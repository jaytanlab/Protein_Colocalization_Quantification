# Protein Colocalization Quantification

X Journal: Protein (Puncta/Diffuse Signal) Colocalization Quantification

#### Environment

1. Install [Anaconda](https://www.anaconda.com/download#Downloads):

   ```
   wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
   chmod +x Anaconda3-2023.03-1-Linux-x86_64.sh
   ./Anaconda3-2023.03-1-Linux-x86_64.sh
   ```

2. Install Python with Anaconda (there should be a default "base" environment, but a separate environment is recommended):

```
conda create -n coloc python=3.8
conda activate coloc
```

#### Data preparation

Folder structure

```
|--Data
    |---original
    	|Control_0.png
    	|Control_1.png
    	|Control_2.png
    	|...
    	|Treated_0.png
    	|Treated_1.png
    	|Treated_2.png
    	|...
    |---painted
    	|Control_0.png
    	|Control_1.png
    	|Control_2.png
    	|...
    	|Treated_0.png
    	|Treated_1.png
    	|Treated_2.png
    	|...
    |---results     Note: this folder will be generated automatically by the program
    |process.py
```

Note:

1. You need to manually create the "painted" folder, and segment the cells.
2. We recommend .png images because it does not lose information. Don't use .jpg.

#### Quantification

For single channel total intensity: (for the green channel; cells segmentation are labeled in red)

```
python process_total_intensity.py --mode single_total
```

For fraction of protein A (intensity) with protein B: (e.g. **Fraction of STING intensity on GM130**)

(In the code, A is the "second channel", B is the "first channel"; cells segmentation are labeled in white)

```
python process_fraction_of_protein_A_with_protein_B.py --mode coloc_total
```

For fraction of protein A (area) with protein B puncta: (e.g. **Fraction of GM130 positive for STING**)

(In the code, A is the "second channel", B is the "first channel"; cells segmentation are labeled in white)

```
python process_fraction_of_protein_A_with_protein_B_puncta.py --mode coloc_total
```

#### Extra code

```
python process_pixels_between_two_points.py
python process_cell_death_by_DAPI.py
python Calculate_S_Curve_from_cell_death.py
```

#### Citation

If you find this toolbox helpful, please cite the following paper:

```
To be added
```

Any questions regarding reproduction or intermediate results, please contact Haoxiang Yang (yanghaoxiang7@gmail.com) and Jay Tan (jay.tan@pit.edu) while posting a github issue.

