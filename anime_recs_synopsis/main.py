from typing import Optional
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from recommender import AnimeRecommender

app = FastAPI()
model = AnimeRecommender()

class RecommendByTitle(BaseModel):
    title: Optional[str] = None
    measure: Optional[str] = 'similarity'
    amount: Optional[int] = 100

class RecommendByText(BaseModel):
    text: Optional[str] = None
    genres: Optional[str] = None
    rating: Optional[str] = None
    measure: Optional[str] = 'similarity'
    amount: Optional[int] = 100

@app.post('/api/recommend_by_title/')
def synopsis_recommend_by_title(data: RecommendByTitle):
    return model.recommend(
        title=data.title, 
        text=None, 
        genres=None, 
        rating=None, 
        measure=data.measure, 
        k=data.amount)

@app.post('/api/recommend_by_text/')
def synopsis_recommend_by_synopsis(data: RecommendByText):
    return model.recommend(
        title=None,
        text=data.text,
        genres=data.genres,
        rating=data.rating,
        measure=data.measure,
        k=data.amount
    )

if __name__ == "__main__":
    uvicorn.run("fastapi_code:app")