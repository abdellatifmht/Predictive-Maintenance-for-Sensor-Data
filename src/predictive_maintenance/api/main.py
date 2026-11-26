from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from .inference import model_service
from .schemas import HealthResponse, PredictRequest, PredictResponse

logger.info("Starting predictive_maintenance_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_service.load()
    except FileNotFoundError as exc:
        logger.warning("Model unavailable; train the pipeline before serving: {}", exc)
    yield


app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        """API for predicting the Remaining Useful Life (RUL) of engines based on sensor readings."""
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    """Root endpoint to check if the API is running."""
    return {"message": "Predictive Maintenance API is running."}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check the health of the API and the model."""
    if not model_service.is_ready():
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    info = model_service.best_model_info
    return HealthResponse(
        status="ok",
        model_name=model_service.model_name,
        model_type=model_service.model_type,
        min_readings_required=model_service.min_readings_required,
        test_RMSE=info["test_RMSE"],
        test_MAE=info["test_MAE"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict the Remaining Useful Life (RUL) and class based on sensor readings."""
    if not model_service.is_ready():
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    try:
        readings_dicts = [r.model_dump(exclude_none=True) for r in request.readings]
        result = model_service.predict(readings_dicts, unit=request.unit)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    logger.info(
        f"Prediction — unit={request.unit} model={result['model_name']}"
        f" RUL={result['predicted_rul']:.2f} class={result['predicted_class']} latency={result['latency_ms']:.2f}"
    )

    return PredictResponse(**result)


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle unexpected exceptions and return a JSON response."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )
