# Student Grade Predictor

-   Authors: Shrijaa Venkatasubramanian Subashini, Rahiq Raees, Jiro Amato, Christine Chow

This is a data analysis project for DSCI 522 (Data Science workflows); a course in the Master of Data Science program at the University of British Columbia.

## About

!!! pull from Summary section in report to populate and include dataset info

## Report

The final report can be found in the notebooks folder.

## Usage

#### Initial Setup (one-time)

1.  Clone the repository and navigate to the project root directory
2.  Create the conda environment:

``` bash
conda env create --file environment.yml
```

#### Running the Analysis

1.  Activate the environment by using the environment name defined in YAML file

``` bash
conda activate student_grade_predictor
```

2.  Launch JupyterLab:

``` bash
jupyter lab
```

3.  Open `notebooks/student_grade_predictor.ipynb` in Jupyter Lab

4.  Under the "Kernel" menu click "Restart Kernel and Run All Cells..." to execute analysis and generate the final report.

## Dependencies

-   `conda` (version 23.9.0 or higher)
-   `jupyterlab` (version 4.0.0 or higher)
-   `nb_conda_kernels` (version 2.3.1 or higher)
-   Python and packages listed in [`environment.yml`](environment.yml)

## License

This project utilizes the Student Performance Dataset from UCI Machine Learning Repository, which is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.

The Student Grade Predictor report contained herein is licensed under the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License. See the LICENSE.md file for more information. If re-using/re-mixing please provide attribution and link to this repository.

The software code contained within this repository is licensed under the MIT license. See the LICENSE.md file for more information.

## References

Cortez, P. (2008). Student Performance \[Dataset\]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T.

Cortez, P. & Silva, A. (2008). Using data mining to predict secondary school student performance. EUROSIS. https://doi.org/10.24432/C5TG7T.

Ma, Y., Liu, B., Wong, C., Yu, P., & Lee, S. (2000). Targeting the right students using data mining. Proceedings of the sixth ACM SIGKDD international conference on Knowledge discovery and data mining, 457-464. https://doi.org/10.1145/347090.347184.

Pritchard, M. & Wilson, G. (2003). Using Emotional and Social Factors to Predict Student Success. Journal of College Student Development, 44, 18-28. https://doi.org/10.1353/csd.2003.0008.

Johora, F. T., Hasan, M. N., Rajbongshi, A., Ashrafuzzaman, M., & Akter, F. (2025). An explainable AI-based approach for predicting undergraduate students academic performance. Array, 26, 100384. https://doi.org/10.1016/j.array.2025.100384.

### Project Acknowledgment

This project is a demonstration for educational purposes, and the structure and workflow was adapted from the "Breast Cancer Predictor" project by Tiffany Timbers, Melissa Lee, Joel Ostblom & Weilin Han.