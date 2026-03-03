from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from app.database import Base

class Generation(Base):
    __tablename__ = "generations"

    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(String, nullable=False)
    generated_text = Column(String, nullable=False)
    temperature = Column(Float, nullable=False)
    top_p = Column(Float, nullable=False)
    max_new_tokens = Column(Integer, nullable=False)
    response_time_ms = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())