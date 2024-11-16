# Sentiment Analysis MLOps Project 🚀

--- 

## Table of Contents
- [Introduction](#-Introduction)
- [Deployment Pipelines](#-Deployment-Pipelines)
- [Tech Stack](#-Tech-Stack)
- [Local Setup](#-Local-Setup-and-Installation)
- [Zenml Integration](#-Zenml-Integration)

## Introduction
Transforming sentiment analysis into a fully automated, production-ready pipeline with cutting-edge **MLOps tools**. By integrating powerful tools like **ZenML** for seamless pipeline management, **MLflow** for streamlined model deployment and experiment tracking, **PostgreSQL** for robust data ingestion, and **Streamlit** for an interactive user interface, this project ensures efficiency and scalability.

**It features:**

- A **Continuous Deployment Pipeline** that trains and deploys the model automatically.
- An **Inference Pipeline** that enables real-time sentiment predictions through a user-friendly Streamlit interface.

***Explore how MLOps transforms sentiment analysis into a fully automated and production-ready solution!***

## Deployment Pipelines
### - **Continuous Deployment Pipeline**
This pipeline is here to make your life easier! 🔄 It automatically handles the deployment of your best-performing model, ensuring the entire process — from training to serving — is smooth and efficient. By continuously checking model performance, it deploys only the
top-performing versions, keeping your production environment optimized, scalable, and ready to handle real-world data seamlessly.

![Continuous Deployment Pipeline](assets/continuous_deployment_pipeline.png)


### - **Inference Pipeline**
The inference pipeline is all about making predictions effortless! 🧠 It loads the deployed model service from **MLflow** and processes new input data to generate predictions seamlessly. Designed for real-time predictions, this pipeline ensures your production system is always ready to deliver accurate results quickly and efficiently.

![Inference Pipeline](assets/Inference_pipeline.png)


## Teck Stack
- **Streamlit**: Powers the front end, offering an intuitive and interactive user interface.
- **ZenML**: Manages MLOps pipelines for seamless integration and automation.
- **MLflow**: Handles experiment tracking and deploys the trained models effortlessly.
- **PostgreSQL**: Ensures robust and efficient data ingestion and management.
- **Docker**: Provides a consistent and scalable environment for pipeline execution.

### PostgreSQL 
![customer_reviews-pgadmin](assets/customer_reviews_table_pg.png)

### Using MLFlow for Model Deployer & Experiment Tracker with ZenML
![Zenml, MLFlow-Model Deployer-Experiment_tracker](assets/zenml-mlflow.png)


### Streamlit
![Streamlit app](assets/streamlit_app.png)

## Local Setup
1. **Clone the Repository**:
```bash
git clone https://github.com/karthikponna/sentiment_analysis_MLOps.git
cd sentiment_analysis_MLOps
```

2. **Set Up a Virtual Environment**:
```bash
# For macOS and Linux:
python3 -m venv venv

# For Windows:
python -m venv venv
```

3. **Activate the Virtual Environment**:
```bash
# For macOS and Linux:
source venv/bin/activate

# For Windows:
.\venv\Scripts\activate
```

4. **Install Required Dependencies**:
```bash
pip install -r requirements.txt
```

## Zenml Integration
1. Install ZenML - https://docs.zenml.io/getting-started/installation 

2. Install some integrations using ZenML:
```bash
zenml integration install mlflow -y
```

3. Register mlflow in the stack:
```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register local-mlflow-stack -a default -o default -d mlflow -e mlflow_tracker --set
```

