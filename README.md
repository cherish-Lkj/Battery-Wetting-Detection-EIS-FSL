EIS-Based Wetting Defect Detection in Lithium-Ion Batteries

This repository contains the official MATLAB implementation and dataset for the paper:
"Detecting electrolyte wetting defects in lithium-ion batteries based on electrochemical impedance spectroscopy" Published in Cell Reports Physical Science (2026).

1. 1.Overview
2. This project proposes a data-driven quality monitoring framework for the electrolyte injection and wetting process in battery manufacturing. It integrates Physics-Informed Feature Engineering with a Few-Shot Learning (FSL) algorithm to identify wetting defects using only a limited number of labeled samples.
3.
4. 2. Requirements
      Software: MATLAB R2022b or later.
      Toolboxes:Statistics and Machine Learning Toolbox.
      Signal Processing Toolbox.

   3. Dataset Structure
      The provided dataset consists of 100 pouch cell samples (2 Ah)
      The raw impedance data is provided in Excel format (.xlsx), named 1#.xlsx to 100#.xlsx.
      Samples 1# – 50#: Normal wetting cells (Labels: 0).
      Samples 51# – 100#: Defective wetting cells (Labels: 1).
      Data Format: Each file contains frequency ($Hz$), $Z_{real}$ ($\Omega$), and $Z_{imag}$ ($\Omega$).
