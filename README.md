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

1. **Clone the repository**:
   ```sh
   git clone https://github.com/AbdulazizAnorboev/weld-defect-segmentation.git
   cd weld-defect-segmentation

Create a virtual environment and activate it:

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate


