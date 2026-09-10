# Used Car Price Prediction ML App

A production-deployed machine learning application that predicts the selling price of a used car based on vehicle details such as brand/model, manufacturing year, kilometers driven, fuel type, seller type, transmission, and ownership history.

The project covers the complete journey from **data preprocessing and model training to REST API development, Dockerization, and cloud deployment**.

## 🚀 Live Demo

**Try the deployed application:**

[Used Car Price Prediction App](https://car-price-ml-app.vercel.app/?utm_source=chatgpt.com)

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │      User / UI      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   React + Vite      │
                    │      Frontend       │
                    │      Vercel         │
                    └──────────┬──────────┘
                               │
                         HTTP POST
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │       Backend       │
                    │       Render        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Scikit-learn      │
                    │   ML Pipeline       │
                    │                     │
                    │ Preprocessing       │
                    │ Feature Engineering │
                    │ Extra Trees Model   │
                    │ Log Target          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Predicted Car Price │
                    └─────────────────────┘
```

## ✨ Features

* Used car price prediction
* Real-world used-car dataset
* Automated categorical feature encoding
* Feature engineering
* Car age calculation
* Brand extraction from car name
* Log transformation of target variable
* Extra Trees regression model
* Model evaluation using R² and MAE
* FastAPI REST API
* CORS configuration
* Dockerized backend
* React + Vite frontend
* Vercel frontend deployment
* Render backend deployment
* Production frontend/backend integration

## 🧠 Machine Learning

### Problem

This project solves a **supervised regression problem**.

Given information about a used car, the model predicts its expected selling price.

### Input Features

The model uses:

* `name`
* `year`
* `km_driven`
* `fuel`
* `seller_type`
* `transmission`
* `owner`

Additional engineered features:

* `brand`
* `car_age`

### Target

```text
selling_price
```

The target is transformed using:

```python
np.log1p(selling_price)
```

during training and converted back using:

```python
np.expm1(prediction)
```

This helps the model handle the highly skewed distribution of used-car prices.

## 📊 Model Performance

Several model configurations were experimentally evaluated.

### Baseline

The initial Random Forest model achieved approximately:

```text
R² ≈ 0.57
```

### Feature Engineering + Model Experiments

Multiple combinations were tested, including:

* Random Forest + normal target
* Random Forest + log target
* Extra Trees + normal target
* Extra Trees + log target

The best configuration achieved:

```text
Model : Extra Trees + Log Target
R²    : 0.7307
MAE   : ₹90,480.87
```

### Model Selection

A later hyperparameter-tuning experiment produced a higher cross-validation score but a lower hold-out test score:

```text
CV R²    : 0.7922
Test R²  : 0.6996
```

Therefore, the tuned model was **not promoted**.

The model with the best hold-out test performance (`R² = 0.7307`) was retained as the production candidate.

This demonstrates an important ML engineering principle:

> A model should not be selected only because it has a higher cross-validation score. Its performance on unseen hold-out data must also be considered.

## 📁 Project Structure

```text
car-price-ml-app/
│
├── data/
│   └── raw/
│       └── car_data.csv
│
├── model/
│   └── car_price_model.pkl
│
├── backend/
│   ├── train.py
│   └── main.py
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Abhijeetb0/car-price-ml-app.git
cd car-price-ml-app
```

### 2. Create Python virtual environment

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

### 3. Install backend dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

From the project root:

```bash
python backend/train.py
```

The trained model will be saved to:

```text
model/car_price_model.pkl
```

### 5. Start the FastAPI backend

```bash
uvicorn backend.main:app --reload
```

Backend will be available at:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

## 🎨 Frontend Setup

Open another terminal:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Frontend will normally run at:

```text
http://localhost:5173
```

For local development, the frontend uses:

```text
http://127.0.0.1:8000
```

as the default backend API URL.

## 🔌 API

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy"
}
```

### Price Prediction

```http
POST /predict
```

Example request:

```json
{
  "name": "Maruti Swift Dzire VDI",
  "year": 2015,
  "km_driven": 50000,
  "fuel": "Diesel",
  "seller_type": "Individual",
  "transmission": "Manual",
  "owner": "First Owner"
}
```

Example response:

```json
{
  "predicted_price": 425000.0,
  "currency": "INR"
}
```

## 🐳 Docker

Build the backend image:

```bash
docker build -t car-price-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 car-price-api
```

The API will then be available at:

```text
http://localhost:8000
```

## ☁️ Deployment

### Frontend

The React/Vite frontend is deployed using **Vercel**.

Production URL:

[https://car-price-ml-app.vercel.app/](https://car-price-ml-app.vercel.app/?utm_source=chatgpt.com)

The frontend communicates with the backend using the:

```text
VITE_API_URL
```

environment variable.

### Backend

The FastAPI backend is deployed using **Render**.

The backend URL is:

```text
https://car-price-ml-app-nu4f.onrender.com
```

The backend uses:

```text
FRONTEND_URL
```

to configure CORS and allow requests from the deployed frontend.

## 🔐 Environment Variables

### Frontend

```text
VITE_API_URL=https://car-price-ml-app-nu4f.onrender.com
```

### Backend

```text
FRONTEND_URL=https://car-price-ml-app.vercel.app
```

## 🛠️ Tech Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

### Backend

* FastAPI
* Uvicorn
* Pydantic

### Frontend

* React
* Vite
* JavaScript
* CSS

### DevOps / Deployment

* Git
* GitHub
* Docker
* Vercel
* Render

## 📌 Current Limitations

The current dataset contains a relatively limited set of vehicle attributes.

Features such as:

* engine capacity
* mileage
* maximum power
* torque
* number of seats

could potentially improve prediction performance.

The current model is therefore a strong learning/portfolio implementation rather than a production-grade automotive valuation system.

## 🔮 Future Improvements

The project is intended to evolve into a complete MLOps pipeline.

Planned improvements include:

* Richer vehicle dataset
* Advanced feature engineering
* Model experiment tracking
* Automated model evaluation
* Model versioning
* Model registry
* Data validation
* CI/CD using GitHub Actions
* Automated model deployment
* Model monitoring
* Data drift detection
* Model performance monitoring
* Automated retraining
* Production model promotion/rejection gates
* Experiment tracking with tools such as MLflow
* Data/model artifact management

### Planned MLOps Flow

```text
Raw Data
   ↓
Data Validation
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Experiment Tracking
   ↓
Model Evaluation
   ↓
Quality Gate
   ↓
Model Registry
   ↓
Deployment
   ↓
Monitoring
   ↓
Drift Detection
   ↓
Retraining
   └──────────────→ New Model
```

## 🎯 Project Goal

The goal of this project is not only to build a machine learning model, but to progressively transform a simple ML application into a **complete production-oriented ML/MLOps system**.

The current version represents the first stage:

```text
Dataset
   ↓
ML Model
   ↓
FastAPI
   ↓
Docker
   ↓
Cloud Deployment
   ↓
Live Web Application
```

Future stages will add the complete MLOps lifecycle around this deployed model.
