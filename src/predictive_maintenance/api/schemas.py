from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class SensorReading(BaseModel):
    """Une lecture brute pour un seul cycle (un instant du moteur)."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    os_1: float
    os_2: float
    os_3: float | None = None
    sensor_1: float | None = None
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float | None = None
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float | None = None
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float | None = None
    sensor_17: float
    sensor_18: float | None = None
    sensor_19: float | None = None
    sensor_20: float
    sensor_21: float


class PredictRequest(BaseModel):
    """Requête de prédiction : une fenêtre de lectures pour un moteur."""

    model_config = ConfigDict(extra="forbid")

    unit: str | None = Field(
        default=None,
        max_length=200,
        validation_alias=AliasChoices("unit", "unit_id"),
        description="Engine identifier (optional)",
    )
    readings: list[SensorReading] = Field(..., min_length=1, max_length=10000)


class PredictResponse(BaseModel):
    unit: str | None = None
    predicted_rul: float
    predicted_class: str = Field(..., description="healthy | anomaly | imminent_failure")
    model_name: str
    n_readings_received: int
    n_readings_used: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_type: str = Field(..., description="'classical' ou 'lstm'")
    min_readings_required: int
    test_RMSE: float
    test_MAE: float
