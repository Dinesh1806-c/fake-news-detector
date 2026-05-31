## Setup & Installation

### 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector

### 2. Install dependencies
pip install -r requirements.txt

### 3. Add the dataset
- Download True.csv and Fake.csv from Kaggle:
  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- Place both files inside the `dataset/` folder

### 4. Train the model (this generates model.pkl and vectorizer.pkl)
python train_model.py

### 5. Run the web app
streamlit run app.py
