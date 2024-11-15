import json
import numpy as np
import streamlit as st
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main as run_main
from pipelines.utils import get_data_for_test


def main():
    st.set_page_config(page_title="Customer Sentiment Analysis", page_icon="😊", layout="centered")
    st.title("📊 Customer Review Sentiment Analysis")
    
    # Introduction and description
    st.write(
        """
        **Welcome!** This app analyzes customer reviews and predicts their sentiment (Positive or Negative).
        """
    )
    
    # Text input box for the user to enter a review
    review_text = st.text_area("📝 Enter a your review:", height=150, placeholder="Type your review here...")
    
    # Predict button
    if st.button("Predict"):
        if review_text:
            # Load the prediction service
            service = prediction_service_loader(
                pipeline_name="continuous_deployment_pipeline",
                pipeline_step_name="mlflow_model_deployer_step",
                running=False,
            )

            # If no service is found, run the pipeline to create a service
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            run_main()

            # Process and predict sentiment
        
        data = get_data_for_test()
        json_list = json.loads(data)
        input_data = np.array(json_list["data"])
        prediction = service.predict(input_data)
                
        if isinstance(prediction, np.ndarray):
            prediction = prediction[0]

        # Display the result
        sentiment = "😊 Positive" if prediction == 1 else "😞 Negative"
        st.success(f"The sentiment of the review is: **{sentiment}**")

    else:
        st.warning("Please enter a review to analyze.")
            

if __name__ == "__main__":
    main()
