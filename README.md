# Waste Classification App

A Streamlit-based web application for classifying waste materials using deep learning. This application helps users identify different types of waste materials through image classification.

## Dataset Requirements

This project uses four Kaggle datasets that need to be downloaded separately:

1. **Waste Classification data**
   - Source: [Kaggle Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data/data)
   - Contains: 22,500 images of organic and recyclable waste products
   - Download and place in: `/data/waste-classification`

2. **Recyclable and Household Waste Classification**
   - Source: [Kaggle Dataset](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification/data)
   - Contains: 15,000 images (mix of studio and real-world photographs)
   - Download and place in: `/data/recyclable-household`

3. **Garbage dataset**
   - Source: [Kaggle Dataset](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
   - Contains: 19,762 images across 10 garbage categories
   - Download and place in: `/data/garbage`

4. **RealWaste Image Classification**
   - Source: [Kaggle Dataset](https://www.kaggle.com/datasets/joebeachcapital/realwaste)
   - Contains: 4,752 images across 9 material types
   - Download and place in: `/data/realwaste`

## Setup Instructions

1. Clone this repository:
```bash
git clone [your-repository-url]
cd waste-classifier
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the datasets from Kaggle and organize them in the following structure:
```
waste-classifier/
├── data/
│   ├── waste-classification/
│   ├── recyclable-household/
│   ├── garbage/
│   └── realwaste/
```

5. Run the Streamlit app:
```bash
streamlit run app.py
```

## Project Structure

```
waste-classifier/
├── data/               # Dataset directories (not included in git)
├── models/            # Trained model files (not included in git)
├── mlruns/           # MLflow tracking (not included in git)
├── app.py            # Main Streamlit application
├── waste_classifier.py # Model architecture and training utilities
├── train.py          # Script to train the model from scratch
├── resume_training.py         # Script to resume training from a checkpoint
└── requirements.txt  # Project dependencies
```

## Model Training

The project includes three main Python files for model development and training:

1. **waste_classifier.py**
   - Contains the model architecture
   - Includes data preprocessing utilities
   - Defines training and validation functions
   - Implements custom loss functions and metrics

2. **train.py**
   - Script to train the model from scratch
   - Handles dataset loading and preprocessing
   - Implements training loop with logging
   - Saves model checkpoints and training history

3. **resume.py**
   - Allows resuming training from a saved checkpoint
   - Useful for continuing interrupted training sessions
   - Maintains training history and metrics

To train the model from scratch:
```bash
python train.py
```

To resume training from a checkpoint:
```bash
#Ensure you have added the relative path to last trained model
python resume_training.py 
```

## Features

- Image upload and classification
- Real-time waste material identification
- Support for multiple waste categories
- User-friendly interface

## Coming Soon!

- Implementation of EfficientNet model for improved classification accuracy
- Additional waste categories
- Batch processing capabilities
