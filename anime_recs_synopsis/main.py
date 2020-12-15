from typing import Optional
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from recommender import AnimeRecommender

app = FastAPI()
model = AnimeRecommender()

class RecommendBy(BaseModel):
    title: Optional[str] = None
    synopsis: Optional[str] = None
    synopsis_genres: Optional[str] = None
    measure: Optional[str] = 'similarity'
    amount: Optional[int] = 100

@app.post('/api/recommend_by_topic/')
def recommend(data: RecommendBy):
    print(data)
    print(data.title)
    
    recs = model.recommend(title=data.title, synopsis=data.synopsis, synopsis_genres=data.synopsis_genres, measure=data.measure, k=data.amount)

    return recs

if __name__ == "__main__":
    uvicorn.run("fastapi_code:app")