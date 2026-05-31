## Setup & Installation

### 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector

### 2. Install dependencies
pip install -r requirements.txt

### 3. Download the dataset
Download True.csv and Fake.csv from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
Place both files inside a `dataset/` folder.

### 4. Train the model (this generates model.pkl automatically)
python train_model.py

### 5. Run the web app
streamlit run app_v2.py
