# Weld Defect Segmentation

This project aims to detect weld defects using the Segformer model for semantic segmentation. The model is trained to identify and segment defects in weld images, facilitating automated quality control in welding processes.

## Table of Contents

- [Overview](#overview)
- [Setup and Installation](#setup-and-installation)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Visualization](#visualization)
- [Acknowledgements](#acknowledgements)

## Overview

The project utilizes the Segformer model, a state-of-the-art transformer-based architecture for semantic segmentation, to detect and segment defects in weld images. The model is trained and evaluated on a custom dataset of weld images and corresponding defect masks.

## Setup and Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AbdulazizAnorboev/weld-defect-segmentation.git
   cd weld-defect-segmentation
   ```
   
2. Create a virtual environment and activate it:
   
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```


## Dataset

The dataset consists of weld images and corresponding segmentation masks indicating defect regions. The data is split into training, validation, and test sets. The dataset should be organized as follows:

```
data/
├── train/
│   ├── images/
│   └── masks/
├── valid/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

