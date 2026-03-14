from pydantic import BaseModel


class PolarSleepJSON(BaseModel):
    """Polar sleep session from /v3/users/sleep."""

    polar_user: str | None = None
    date: str | None = None
    sleep_start_time: str | None = None
    sleep_end_time: str | None = None
    device_id: str | None = None

    total_sleep_minutes: int | None = None
    total_interruption_duration_minutes: int | None = None
    total_intermission_duration_minutes: int | None = None
    sleep_charge: int | None = None
    sleep_score: int | None = None
    sleep_rating: str | None = None

    # HR / HRV / SpO2
    heart_rate_avg: int | None = None
    beat_to_beat_avg: int | None = None
    heart_rate_variability_avg: int | None = None
    breathing_rate_avg: float | None = None

    # Stage durations in seconds
    light_sleep: int | None = None
    deep_sleep: int | None = None
    rem_sleep: int | None = None
    unrecognized_sleep_stage: int | None = None

    sleep_cycles: int | None = None
    group: str | None = None


class PolarContinuousHRSampleJSON(BaseModel):
    """Single HR sample within a continuous-heart-rate response."""

    date_time: str
    heart_rate: int


class PolarContinuousHRJSON(BaseModel):
    """Polar continuous heart rate from /v3/users/continuous-heart-rate/{date}."""

    polar_user: str | None = None
    date: str | None = None
    heart_rate_samples: list[PolarContinuousHRSampleJSON] = []


class PolarNightlyRechargeJSON(BaseModel):
    """Polar nightly recharge (recovery) from /v3/users/nightly-recharge."""

    polar_user: str | None = None
    date: str | None = None
    heart_rate_avg: int | None = None
    beat_to_beat_avg: int | None = None
    beat_to_beat_sdnn: int | None = None
    heart_rate_variability_avg: int | None = None
    breathing_rate_avg: float | None = None
    nightly_recharge_status: int | None = None
    ans_charge: float | None = None
    ans_rate: float | None = None
    sleep_charge: int | None = None
    sleep_score: int | None = None
    sleep_rating: str | None = None


class PolarActivityJSON(BaseModel):
    """Polar daily activity summary from /v3/users/activities."""

    polar_user: str | None = None
    id: str | None = None
    date: str | None = None
    duration: str | None = None
    active_calories: int | None = None
    calories: int | None = None
    active_steps: int | None = None
    steps: int | None = None
    distance: float | None = None
    active_time: str | None = None
    sleep_time: str | None = None


class PolarActivitySampleJSON(BaseModel):
    """Single sample within an activity-samples response."""

    recording_rate: int | None = None
    sample_type: str | None = None
    data: str | None = None


class PolarActivitySamplesJSON(BaseModel):
    """Polar activity samples from /v3/users/activities/samples."""

    polar_user: str | None = None
    date: str | None = None
    samples: list[PolarActivitySampleJSON] = []
