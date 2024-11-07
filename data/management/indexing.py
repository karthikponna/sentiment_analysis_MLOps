import os
from dotenv import load_dotenv
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Integer,
    Numeric,
    Sequence,
    SmallInteger,
    String,
    Text,
    create_engine,
)

load_dotenv()


engine = create_engine(
    os.getenv("DB_URL"),
    echo=True,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=-1,
    pool_recycle=3600,
    pool_pre_ping=True,
    connect_args={
        "connect_timeout": 60,
        "keepalives":1,
        "keepalives_idle":30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    },
)

Session = sessionmaker(bind=engine)
connection = engine.connect()
connection.close()

Base = declarative_base()

class CustomerReviews(Base):
    __tablename__ = "customer_reviews"
    id = Column(Integer, Sequence("product_reviews_id_seq"), primary_key=True)
    product_id = Column(String)
    user_id = Column(String)
    helpfulness_numerator = Column(SmallInteger)
    helpfulness_denominator = Column(SmallInteger)
    score = Column(Integer)
    time = Column(Integer)
    review_text = Column(Text)

