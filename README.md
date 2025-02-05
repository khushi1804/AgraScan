AgraScan - Plant Disease Detection System

AgraScan is a deep learning-based web application that detects diseases in bell pepper, potato, and tomato leaves using a Convolutional Neural Network (CNN) model. The system also suggests appropriate pesticides to mitigate the detected disease.

Features

🌿 Plant Disease Detection: Upload an image of a plant leaf, and the model will classify the disease.

🔬 CNN-Based Classification: Uses a deep learning model trained on a plant disease dataset.

🏷 Pesticide Recommendation: Provides suitable pesticides based on the detected disease.

🌐 Web-Based Interface: Deployed using Streamlit for ease of use.

📊 User-Friendly UI: Simple interface for farmers and researchers to diagnose plant health.

Tech Stack

Frontend: Streamlit

Backend: Python, TensorFlow/Keras, OpenCV

Deployment: Streamlit Cloud

Installation & Setup

Prerequisites

Ensure you have the following installed:

Python (>= 3.7)

pip

Steps to Run Locally

1️⃣ Clone the Repository

git clone https://github.com/khushi1804/AgraScan.git
cd AgraScan

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Streamlit App

streamlit run app.py

4️⃣ Open in Browser

Visit http://localhost:8501 to use the application.

Dataset

You can download the plant disease dataset from this Google Drive link https://drive.google.com/drive/folders/13Eax7FRWZkc228uLRAwMXa1HfGysqaUN?usp=sharing

Deployment

The application is deployed using Streamlit Cloud.

Contributing

We welcome contributions! To contribute:

Fork the repository.

Create a new branch: git checkout -b feature-branch.

Make your changes and commit: git commit -m "Added new feature".

Push the branch: git push origin feature-branch.

Open a pull request.

License

This project is licensed under the MIT License.

Contact

For queries or collaboration, feel free to reach out:
📧 Email: khushi18042003@gmail.com🔗 GitHub: khushi1804

