from fastapi import FastAPI, HTTPException
from models import CompanyRequest,NewsResponse,TextToSpeechRequest
from gtts import gTTS
from googletrans import Translator
import uvicorn
import time
from utils import (
    fetch_news_articles, 
    perform_sentiment_analysis, 
    generate_comparative_analysis, 
    extract_topics,
    summarize_article,
    generate_hindi_speech
)
import json
import os


AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

translator = Translator()

app = FastAPI(title="News Sentiment API")



@app.get("/")
def read_root():
    return {"message": "News Sentiment Analysis API", "status": "active"}

@app.post("/analyze", response_model=NewsResponse)
async def analyze_news(request: CompanyRequest):
    try:
        # Fetch news articles
        articles = fetch_news_articles(request.company_name, request.article_count)
        
        if not articles or len(articles) == 0:
            raise HTTPException(status_code=404, detail=f"No news articles found for {request.company_name}")
        
        # Process each article
        processed_articles = []
        for article in articles:
            # Summarize the article
            summary = summarize_article(article["content"])
            
            # Analyze sentiment
            sentiment = perform_sentiment_analysis(article["content"])
            
            # Extract topics
            topics = extract_topics(article["content"])
            
            processed_articles.append({
                "title": article["title"],
                "summary": summary,
                "sentiment": sentiment,
                "topics": topics,
                "url": article["url"]
            })
        
        # Generate comparative analysis
        comparative_analysis = generate_comparative_analysis(processed_articles)
        
        # Determine overall sentiment
        positive_count = comparative_analysis["sentiment_distribution"]["positive"]
        negative_count = comparative_analysis["sentiment_distribution"]["negative"]
        neutral_count = comparative_analysis["sentiment_distribution"]["neutral"]
        
        if positive_count > negative_count and positive_count > neutral_count:
            final_sentiment = f"{request.company_name}'s latest news coverage is mostly positive. Potential stock growth expected."
        elif negative_count > positive_count and negative_count > neutral_count:
            final_sentiment = f"{request.company_name}'s latest news coverage is mostly negative. Potential stock decline expected."
        else:
            final_sentiment = f"{request.company_name}'s latest news coverage is mostly neutral. No significant stock movement expected."
        
        # Prepare response data
        response_data = {
            "company": request.company_name,
            "articles": processed_articles,
            "comparative_sentiment_score": comparative_analysis,
            "final_sentiment_analysis": final_sentiment,
        }
        
        return response_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-speech")
async def generate_speech(request: TextToSpeechRequest):
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text is required for TTS")

        # Translate English text to Hindi
        translated_text = translator.translate(text, src="en", dest="hi").text

        # Generate filename with timestamp
        filename = f"speech_{int(time.time())}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)

        # Generate Hindi speech using gTTS
        tts = gTTS(text=translated_text, lang="hi")
        tts.save(filepath)

        return {"audio_path": filepath}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    uvicorn.run("api:app", port=8000, reload=True)