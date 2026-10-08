from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from schemas import LoanApplication
from model import CreditScoringModel
from database import StatsStore

app = FastAPI(title="Online Loan Decision API", version="1.0.0")
templates = Jinja2Templates(directory="templates")

model = CreditScoringModel()
stats = StatsStore()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Главная страница с формой и статистикой."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "stats": stats.get_stats(),
    })


@app.post("/api/predict")
async def predict(application: LoanApplication):
    """
    Основной эндпоинт принятия решения.
    Принимает данные заявителя, возвращает решение и объяснение.
    """
    result = model.predict(application)
    stats.record(application, result)
    return result


@app.get("/api/stats")
async def get_stats():
    """Возвращает агрегированную статистику одобрений."""
    return stats.get_stats()


@app.get("/api/health")
async def health():
    return {"status": "ok"}