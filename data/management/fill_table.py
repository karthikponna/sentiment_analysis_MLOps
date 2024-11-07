import sys
import os
import pandas as pd
from indexing import CustomerReviews, Session, engine


#Ensure the engine connects to the database
try:
    with engine.connect() as connection:
        result = connection.execute("SELECT 1")
        print("Database connection is working:", result.fetchone())
except Exception as e:
    print(f"Error connecting to database: {e}")

with Session.begin() as db:
    data = pd.read_csv(
        "/home/karthikponna/karthik/sentiment_analysis_mlops_project_1/sentiment_analysis_MLOps/data/Reviews.csv",
        dtype={
        "ProductId": str,
        "UserId": str,
        "HelpfulnessNumerator": "Int64",
        "HelpfulnessDenominator": "Int64",
        "Score": "Int64",
        "Time": "Int64",
        "Text": str,
        }
    )
    print(data.head(10))

    
    for index, row in data.iterrows():
        try:
            customer_reviews = CustomerReviews(
                product_id=row["ProductId"],
                user_id=row["UserId"],
                helpfulness_numerator=row["HelpfulnessNumerator"],
                helpfulness_denominator=row["HelpfulnessDenominator"],
                score=row["Score"],
                time=row["Time"],
                review_text=row["Text"],
            )
            db.add(customer_reviews)
        except Exception as e:
            print(f"Error adding row {index}: {e}")


    db.commit()  # Commit once after all entries are added
    
