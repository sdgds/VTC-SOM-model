ReadMe

Purpose: Code needed to reproduce the results within the " A biologically inspired computational model of human ventral temporal cortex" paper.

The Python code used to build the VTC-SOM model and reproduce the results in Figs. 2 to 9 is provided in the /code/ folder. The stimulus sets, neuroimaging data, and physiological data used for model training and testing, and required to reproduce the results in Figs. 2 to 9 are stored in the /data/ folder. 

1. The “1.VTC-SOM model.py” is used to build the VTC-SOM model and reproduce Figure 2a.

2. The “2.Simulated_regions.py” is used to reproduce Figure 2b (maps in the artificial VTC-SOM). The outputted nii files for all category-selective regions can be visualized by using the wb_view software (https://humanconnectome.org/software/get-connectome-workbench), which can display these regions on the human brain template.

3. The “3.Simulated_maps.py” is used to reproduce Figure 3. Similar to that in 2, the outputted nii files for all abstract functional maps can be visualized on the human brain template by using the wb_view software.

4. The “4.RSM_and_multisigma.py” is used to reproduce Figure 5 and 7-9.

5. The “5.V1-SOM.py” is used to reproduce Figure 6. 

All figures produced in each step by this code is put into the /results/ folder.

* Note that the reproduced results might be slightly different from those in the manuscript, due to possible differences among computing environments, such as calculation precision, or random factors in the algorithm. However, it would not violate our conclusions. In addition, depending on the performance of computer configuration, it may take a long time (1-3 days) to reproduce all results.
* Due to the amount of data required in the code, we have centralised the data in 'Releases' of GitHub.
