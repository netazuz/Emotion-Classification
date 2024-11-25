
# **Emotion Classification**

An advanced machine learning project focused on the classification of emotions based on audio and/or textual input. This repository is designed for researchers, developers, and enthusiasts interested in Speech Emotion Recognition (SER) and text-based sentiment analysis. The project explores feature extraction, data preprocessing, and machine learning models to accurately identify human emotions.

---

## **Features**

- **Audio-Based Emotion Recognition**: Extracts features such as Mel-Frequency Cepstral Coefficients (MFCCs) and pitch to classify emotions from speech.
- **Text-Based Sentiment Analysis**: Processes textual data to identify emotional sentiment using Natural Language Processing (NLP).
- **Multimodal Approach**: Combines audio and text features for enhanced accuracy.
- **Feature Importance Analysis**: Visualizes and evaluates the most impactful features contributing to the classification results.
- **Modular Design**: Extensible pipeline for custom datasets and models.

---

## **Technologies Used**

- **Python**: Primary programming language.
- **Machine Learning Libraries**:
  - `scikit-learn` for preprocessing and model training.
  - `librosa` for audio feature extraction.
  - `pandas` and `numpy` for data manipulation.
- **Deep Learning Frameworks** (optional):
  - `TensorFlow` or `PyTorch` for advanced model implementations.
- **Data Visualization**:
  - `matplotlib` and `seaborn` for feature importance plots and analysis.

---

## **Setup and Installation**

1. Clone the repository:

   ```bash
   git clone https://github.com/netazuz/Emotion-Classification.git
   cd Emotion-Classification
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download or prepare your dataset:
   - Add your audio/text dataset in the `data/` directory.
   - Ensure the file structure matches the required format (e.g., labels, features).

---

## **Usage**

### **1. Preprocessing Data**

Preprocess the audio/text dataset for feature extraction:

```bash
python preprocess.py
```

### **2. Train the Model**

Train the emotion classification model:

```bash
python train.py --model "svm" --dataset "data/emotions.csv"
```

Available models include:

- SVM (Support Vector Machines)
- Random Forest
- Neural Networks

### **3. Evaluate the Model**

Evaluate the performance of the trained model on a test dataset:

```bash
python evaluate.py --model "svm"
```

### **4. Visualize Feature Importance**

Analyze and visualize the key features influencing classification:

```bash
python visualize_features.py
```

---

## **Folder Structure**

```
Emotion-Classification/
├── data/                   # Dataset files (audio and text)
├── models/                 # Saved trained models
├── scripts/                # Core scripts (preprocessing, training, evaluation)
├── notebooks/              # Jupyter notebooks for experimentation
├── utils/                  # Utility functions and helper modules
├── results/                # Output files (e.g., predictions, visualizations)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## **Example Dataset**

- **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song ([Link](https://zenodo.org/record/1188976)).
- **TESS**: Toronto Emotional Speech Set ([Link](https://tspace.library.utoronto.ca/handle/1807/24487)).
- **CREMA**:  Crowd-sourced Emotional Multimodal Actors Dataset ([Link](https://github.com/CheyneyComputerScience/CREMA-D)).
- **SAVEE**: Surrey Audio-Visual Expressed Emotion ([Link](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee))

---

## **Contributing**

Contributions are welcome! If you'd like to:

- Report bugs
- Suggest improvements
- Add new features

Feel free to open an issue or submit a pull request.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgements**

- Inspired by state-of-the-art research in emotion recognition.
- Uses datasets like RAVDESS and TESS for development and testing.
- Libraries like `librosa` and `scikit-learn` power the feature extraction and model training pipeline.

---

## **Contact**

For questions or collaborations:

- **Netanel Mazuz**  
- LinkedIn: [linkedin.com/in/netanel-mazuz](https://www.linkedin.com/in/netanel-mazuz/)  
- Email: [netazuz@gmail.com](mailto:netazuz@gmail.com)
