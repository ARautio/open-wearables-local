"""Polar 247 data implementation for sleep, recovery, and activity samples.

Syncs data from Polar AccessLink API v3 endpoints:
- /v3/users/sleep                          → EventRecord (category=sleep) + SleepDetails + HealthScore
- /v3/users/nightly-recharge               → DataPointSeries (recovery_score, HRV) + HealthScore
- /v3/users/continuous-heart-rate/{date}   → DataPointSeries (heart_rate)
- /v3/users/activities                     → DataPointSeries (steps, energy, exercise_time)
"""

import re
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from app.config import settings
from app.database import DbSession
from app.models import DataPointSeries, EventRecord
from app.repositories import EventRecordRepository, UserConnectionRepository
from app.repositories.data_point_series_repository import DataPointSeriesRepository
from app.repositories.data_source_repository import DataSourceRepository
from app.schemas.enums import HealthScoreCategory, ProviderName, SeriesType
from app.schemas.model_crud.activities import (
    EventRecordCreate,
    EventRecordDetailCreate,
    HealthScoreCreate,
    TimeSeriesSampleCreate,
)
from app.services.event_record_service import event_record_service
from app.services.health_score_service import health_score_service
from app.services.providers.api_client import make_authenticated_request
from app.services.providers.templates.base_247_data import Base247DataTemplate
from app.services.providers.templates.base_oauth import BaseOAuthTemplate
from app.utils.dates import parse_datetime_or_default, parse_iso_datetime
from app.utils.structured_logging import log_structured

# ---------------------------------------------------------------------------
# Series type mappings
# ---------------------------------------------------------------------------
_NIGHTLY_RECHARGE_METRICS: list[tuple[str, SeriesType]] = [
    ("ans_charge", SeriesType.recovery_score),
    ("heart_rate_variability_avg", SeriesType.heart_rate_variability_rmssd),
]

_ACTIVITY_SERIES_MAP: dict[str, SeriesType] = {
    "steps": SeriesType.steps,
    "active_calories": SeriesType.energy,
    "distance": SeriesType.distance_walking_running,
    "active_time_minutes": SeriesType.exercise_time,
}


def _parse_iso_duration_minutes(duration_str: str | None) -> float | None:
    """Parse ISO 8601 duration string (e.g. 'PT2H31M4S') to total minutes."""
    if not duration_str:
        return None
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_str)
    if not m:
        return None
    hours, minutes, seconds = (int(g or 0) for g in m.groups())
    total = hours * 60 + minutes + seconds / 60
    return total if total > 0 else None


class Polar247Data(Base247DataTemplate):
    """Polar implementation for 247 data (sleep, recovery, activity)."""

    def __init__(
        self,
        provider_name: str,
        api_base_url: str,
        oauth: BaseOAuthTemplate,
    ) -> None:
        super().__init__(provider_name, api_base_url, oauth)
        self.event_record_repo = EventRecordRepository(EventRecord)
        self.data_source_repo = DataSourceRepository()
        self.connection_repo = UserConnectionRepository()
        self.data_point_repo = DataPointSeriesRepository(DataPointSeries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_api_request(
        self,
        db: DbSession,
        user_id: UUID,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make authenticated request to Polar API."""
        return make_authenticated_request(
            db=db,
            user_id=user_id,
            connection_repo=self.connection_repo,
            oauth=self.oauth,
            api_base_url=self.api_base_url,
            provider_name=self.provider_name,
            endpoint=endpoint,
            method="GET",
            params=params,
        )

    def _fetch_date_range(
        self,
        db: DbSession,
        user_id: UUID,
        endpoint: str,
        start_date: datetime,
        end_date: datetime,
        chunk_days: int = 28,
    ) -> list[dict[str, Any]]:
        """Fetch data with from/to ISO date params, chunked to respect Polar's 28-day limit."""
        all_data: list[dict[str, Any]] = []
        current_start = start_date

        while current_start < end_date:
            current_end = min(current_start + timedelta(days=chunk_days), end_date)
            params = {
                "from": current_start.strftime("%Y-%m-%d"),
                "to": current_end.strftime("%Y-%m-%d"),
            }
            try:
                response = self._make_api_request(db, user_id, endpoint, params=params)
                if isinstance(response, list):
                    all_data.extend(response)
                elif isinstance(response, dict):
                    # Polar wraps results in endpoint-specific keys (e.g. "nights",
                    # "recharge_dates", "activity_summaries"). Pick the first list value.
                    for value in response.values():
                        if isinstance(value, list):
                            all_data.extend(value)
                            break
            except Exception as e:
                log_structured(
                    self.logger,
                    "warning",
                    f"Error fetching {endpoint}: {e}",
                    provider=self.provider_name,
                    task="fetch_date_range",
                )
            current_start = current_end

        return all_data

    # ------------------------------------------------------------------
    # Sleep — /v3/users/sleep
    # ------------------------------------------------------------------

    def get_sleep_data(
        self,
        db: DbSession,
        user_id: UUID,
        start_time: datetime,
        end_time: datetime,
    ) -> list[dict[str, Any]]:
        """Fetch sleep data from Polar API."""
        return self._fetch_date_range(db, user_id, "/v3/users/sleep", start_time, end_time)

    def _normalize_sleep_health_score(
        self,
        normalized_sleep: dict[str, Any],
        user_id: UUID,
    ) -> HealthScoreCreate | None:
        """Extract sleep health score from a normalized Polar sleep record."""
        sleep_score = normalized_sleep.get("efficiency_percent")
        if sleep_score is None:
            return None

        start_time_str = normalized_sleep.get("start_time")
        if not start_time_str:
            return None

        recorded_at = parse_iso_datetime(start_time_str)
        if not recorded_at:
            return None

        return HealthScoreCreate(
            id=uuid4(),
            user_id=user_id,
            provider=ProviderName.POLAR,
            category=HealthScoreCategory.SLEEP,
            value=sleep_score,
            recorded_at=recorded_at,
        )

    def normalize_sleep(  # type: ignore[override]
        self,
        raw_sleep: dict[str, Any],
        user_id: UUID,
    ) -> tuple[dict[str, Any], HealthScoreCreate | None]:
        """Normalize a single Polar sleep entry to our internal dict format."""
        sleep_start = raw_sleep.get("sleep_start_time")
        sleep_end = raw_sleep.get("sleep_end_time")

        # Stage durations — Polar provides these in seconds
        total_sleep_min = int(raw_sleep.get("total_sleep_minutes") or 0)
        light_sleep_sec = int(raw_sleep.get("light_sleep") or 0)
        deep_sleep_sec = int(raw_sleep.get("deep_sleep") or 0)
        rem_sleep_sec = int(raw_sleep.get("rem_sleep") or 0)

        # Time in bed = calendar span from start to end (more reliable than total_sleep_minutes)
        start_dt = parse_iso_datetime(sleep_start) if sleep_start else None
        end_dt = parse_iso_datetime(sleep_end) if sleep_end else None
        if start_dt and end_dt:
            time_in_bed_seconds = int((end_dt - start_dt).total_seconds())
        else:
            time_in_bed_seconds = total_sleep_min * 60  # fallback when timestamps missing

        # Actual sleep = sum of scored stages (excludes interruptions/awake time)
        total_sleep_seconds = total_sleep_min * 60
        awake_seconds = max(0, time_in_bed_seconds - light_sleep_sec - deep_sleep_sec - rem_sleep_sec)

        normalized = {
            "id": uuid4(),
            "user_id": user_id,
            "provider": self.provider_name,
            "date": raw_sleep.get("date"),
            "start_time": sleep_start,
            "end_time": sleep_end,
            "duration_seconds": time_in_bed_seconds,
            "time_in_bed_seconds": time_in_bed_seconds,
            "total_sleep_seconds": total_sleep_seconds,
            "efficiency_percent": raw_sleep.get("sleep_score"),
            "is_nap": False,
            "stages": {
                "deep_seconds": deep_sleep_sec,
                "light_seconds": light_sleep_sec,
                "rem_seconds": rem_sleep_sec,
                "awake_seconds": awake_seconds,
            },
            "avg_heart_rate_bpm": raw_sleep.get("heart_rate_avg"),
            "avg_hrv_ms": raw_sleep.get("heart_rate_variability_avg"),
            "polar_sleep_date": raw_sleep.get("date"),
        }
        return normalized, self._normalize_sleep_health_score(normalized, user_id)

    def save_sleep_data(
        self,
        db: DbSession,
        user_id: UUID,
        normalized_sleep: dict[str, Any],
    ) -> None:
        """Save normalized sleep data as EventRecord + SleepDetails."""
        sleep_id: UUID = normalized_sleep["id"]

        start_dt = parse_iso_datetime(normalized_sleep.get("start_time"))
        end_dt = parse_iso_datetime(normalized_sleep.get("end_time"))

        if not start_dt or not end_dt:
            log_structured(
                self.logger,
                "warning",
                f"Skipping sleep record {sleep_id}: missing start/end time",
                provider=self.provider_name,
                task="save_sleep_data",
            )
            return

        record = EventRecordCreate(
            id=sleep_id,
            category="sleep",
            type="sleep_session",
            source_name="Polar",
            device_model=None,
            duration_seconds=normalized_sleep.get("duration_seconds"),
            start_datetime=start_dt,
            end_datetime=end_dt,
            external_id=normalized_sleep.get("polar_sleep_date"),
            source=self.provider_name,
            user_id=user_id,
        )

        stages = normalized_sleep.get("stages", {})
        # Actual sleep = total_sleep_seconds from API; fall back to sum of stages
        total_sleep_seconds = normalized_sleep.get("total_sleep_seconds") or (
            stages.get("deep_seconds", 0) + stages.get("light_seconds", 0) + stages.get("rem_seconds", 0)
        )
        time_in_bed_seconds = normalized_sleep.get("time_in_bed_seconds", 0)

        detail = EventRecordDetailCreate(
            record_id=sleep_id,
            sleep_total_duration_minutes=total_sleep_seconds // 60,
            sleep_time_in_bed_minutes=time_in_bed_seconds // 60,
            sleep_efficiency_score=Decimal(str(normalized_sleep["efficiency_percent"]))
            if normalized_sleep.get("efficiency_percent") is not None
            else None,
            sleep_deep_minutes=stages.get("deep_seconds", 0) // 60,
            sleep_light_minutes=stages.get("light_seconds", 0) // 60,
            sleep_rem_minutes=stages.get("rem_seconds", 0) // 60,
            sleep_awake_minutes=stages.get("awake_seconds", 0) // 60,
            is_nap=False,
        )

        try:
            event_record_service.create_or_merge_sleep(db, user_id, record, detail, settings.sleep_end_gap_minutes)
        except Exception as e:
            log_structured(
                self.logger,
                "error",
                f"Error saving sleep record {sleep_id}: {e}",
                provider=self.provider_name,
                task="save_sleep_data",
            )

    def load_and_save_sleep(
        self,
        db: DbSession,
        user_id: UUID,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        """Load sleep data from API and save to database."""
        raw_data = self.get_sleep_data(db, user_id, start_time, end_time)
        count = 0
        health_scores: list[HealthScoreCreate] = []

        for item in raw_data:
            try:
                normalized, health_score = self.normalize_sleep(item, user_id)
                self.save_sleep_data(db, user_id, normalized)
                count += 1
                if health_score:
                    health_scores.append(health_score)
            except Exception as e:
                log_structured(
                    self.logger,
                    "warning",
                    f"Failed to save sleep data: {e}",
                    provider=self.provider_name,
                    task="load_and_save_sleep",
                )

        if health_scores:
            health_score_service.bulk_create(db, health_scores)
            db.commit()

        return count

    # ------------------------------------------------------------------
    # Nightly Recharge (Recovery) — /v3/users/nightly-recharge
    # ------------------------------------------------------------------

    def get_recovery_data(
        self,
        db: DbSession,
        user_id: UUID,
        start_time: datetime,
        end_time: datetime,
    ) -> list[dict[str, Any]]:
        """Fetch nightly recharge data from Polar API."""
        return self._fetch_date_range(db, user_id, "/v3/users/nightly-recharge", start_time, end_time)

    def _normalize_recovery_health_score(
        self,
        normalized_recovery: dict[str, Any],
        user_id: UUID,
    ) -> HealthScoreCreate | None:
        """Extract recovery health score from a normalized Polar nightly recharge record."""
        ans_charge = normalized_recovery.get("ans_charge")
        if ans_charge is None:
            return None

        date_str = normalized_recovery.get("date")
        if not date_str:
            return None

        recorded_at = parse_iso_datetime(date_str) if isinstance(date_str, str) else date_str
        if not recorded_at:
            return None

        return HealthScoreCreate(
            id=uuid4(),
            user_id=user_id,
            provider=ProviderName.POLAR,
            category=HealthScoreCategory.RECOVERY,
            value=ans_charge,
            recorded_at=recorded_at,
        )

    def normalize_recovery(  # type: ignore[override]
        self,
        raw_recovery: dict[str, Any],
        user_id: UUID,
    ) -> tuple[dict[str, Any], HealthScoreCreate | None]:
        """Normalize Polar nightly recharge to our schema.

        ANS charge (0.0–4.0) is scaled to 0-100 to match recovery_score convention.
        """
        date = raw_recovery.get("date")
        ans_charge_raw = raw_recovery.get("ans_charge")
        # Scale ANS charge from 0-4 to 0-100
        ans_charge_scaled = float(ans_charge_raw) / 4.0 * 100 if ans_charge_raw is not None else None

        normalized = {
            "user_id": user_id,
            "provider": self.provider_name,
            "date": date,
            "ans_charge": ans_charge_scaled,
            "heart_rate_variability_avg": raw_recovery.get("heart_rate_variability_avg"),
        }
        return normalized, self._normalize_recovery_health_score(normalized, user_id)

    def save_recovery_data(
        self,
        db: DbSession,
        user_id: UUID,
        normalized_recovery: dict[str, Any],
    ) -> int:
        """Save normalized nightly recharge data as DataPointSeries records."""
        if not normalized_recovery:
            return 0

        date_str = normalized_recovery.get("date")
        if not date_str:
            return 0

        recorded_at = parse_iso_datetime(date_str) if isinstance(date_str, str) else date_str
        if not recorded_at:
            return 0

        samples: list[TimeSeriesSampleCreate] = []
        for field_name, series_type in _NIGHTLY_RECHARGE_METRICS:
            value = normalized_recovery.get(field_name)
            if value is None:
                continue
            samples.append(
                TimeSeriesSampleCreate(
                    id=uuid4(),
                    user_id=user_id,
                    source=self.provider_name,
                    recorded_at=recorded_at,
                    value=Decimal(str(value)),
                    series_type=series_type,
                )
            )

        if samples:
            self.data_point_repo.bulk_create(db, samples)

        return len(samples)

    def load_and_save_recovery(
        self,
        db: DbSession,
        user_id: UUID,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        """Load nightly recharge data from API and save to database."""
        raw_data = self.get_recovery_data(db, user_id, start_time, end_time)
        total_count = 0
        health_scores: list[HealthScoreCreate] = []

        for item in raw_data:
            try:
                normalized, health_score = self.normalize_recovery(item, user_id)
                if normalized:
                    total_count += self.save_recovery_data(db, user_id, normalized)
                    if health_score:
                        health_scores.append(health_score)
            except Exception as e:
                log_structured(
                    self.logger,
                    "warning",
                    f"Failed to save recovery data: {e}",
                    provider=self.provider_name,
                    task="load_and_save_recovery",
                )

        if health_scores:
            health_score_service.bulk_create(db, health_scores)
            db.commit()

        return total_count

    # ------------------------------------------------------------------
    # Continuous Heart Rate — /v3/users/continuous-heart-rate/{date}
    # ------------------------------------------------------------------

    def get_continuous_heart_rate(
        self,
        db: DbSession,
        user_id: UUID,
        date: str,
    ) -> list[dict[str, Any]]:
        """Fetch continuous heart rate samples for a single date.

        Returns list of dicts with 'heart_rate' and 'recorded_at' (full datetime).
        Polar returns samples with only a time component ('sample_time': 'HH:MM:SS'),
        so we combine the requested date with each sample_time to build a full timestamp.
        """
        endpoint = f"/v3/users/continuous-heart-rate/{date}"
        try:
            response = self._make_api_request(db, user_id, endpoint)
            if not isinstance(response, dict):
                return []
            raw_samples = response.get("heart_rate_samples", [])
            # Combine date with sample_time → full ISO datetime string
            enriched = []
            for s in raw_samples:
                sample_time = s.get("sample_time")
                hr = s.get("heart_rate")
                if sample_time and hr is not None:
                    enriched.append({"recorded_at": f"{date}T{sample_time}", "heart_rate": hr})
            return enriched
        except Exception as e:
            log_structured(
                self.logger,
                "warning",
                f"Error fetching continuous HR for {date}: {e}",
                provider=self.provider_name,
                task="get_continuous_heart_rate",
            )
        return []

    def save_continuous_heart_rate(
        self,
        db: DbSession,
        user_id: UUID,
        hr_samples: list[dict[str, Any]],
    ) -> int:
        """Save continuous HR samples as DataPointSeries records (bulk)."""
        all_samples: list[TimeSeriesSampleCreate] = []

        for sample in hr_samples:
            recorded_at_str = sample.get("recorded_at")
            hr = sample.get("heart_rate")
            if not recorded_at_str or hr is None:
                continue

            recorded_at = parse_iso_datetime(recorded_at_str)
            if not recorded_at:
                continue

            all_samples.append(
                TimeSeriesSampleCreate(
                    id=uuid4(),
                    user_id=user_id,
                    source=self.provider_name,
                    recorded_at=recorded_at,
                    value=Decimal(str(hr)),
                    series_type=SeriesType.heart_rate,
                )
            )

        if all_samples:
            self.data_point_repo.bulk_create(db, all_samples)

        return len(all_samples)

    def load_and_save_continuous_heart_rate(
        self,
        db: DbSession,
        user_id: UUID,
        start_time: datetime,
        end_time: datetime,
    ) -> int:
        """Load continuous HR for each day in range and save to database."""
        total_count = 0
        current = start_time
        while current <= end_time:
            date_str = current.strftime("%Y-%m-%d")
            try:
                samples = self.get_continuous_heart_rate(db, user_id, date_str)
                total_count += self.save_continuous_heart_rate(db, user_id, samples)
            except Exception as e:
                log_structured(
                    self.logger,
                    "warning",
                    f"Failed to sync continuous HR for {date_str}: {e}",
                    provider=self.provider_name,
                    task="load_and_save_continuous_heart_rate",
                )
            current += timedelta(days=1)
        return total_count

    # ------------------------------------------------------------------
    # Activity Samples — required by base, no-op for Polar
    # HR is handled via continuous-heart-rate; no unified samples endpoint.
    # ------------------------------------------------------------------

    def get_activity_samples(
        self,
        db: DbSession,
        user_id: UUID,
        start_time: datetime,
        end_time: datetime,
    ) -> list[dict[str, Any]]:
        """Polar does not expose a unified activity-samples endpoint."""
        return []

    def normalize_activity_samples(
        self,
        raw_samples: list[dict[str, Any]],
        user_id: UUID,
    ) -> dict[str, list[dict[str, Any]]]:
        return {}

    # ------------------------------------------------------------------
    # Daily Activity Statistics — /v3/users/activities
    # ------------------------------------------------------------------

    def get_daily_activity_statistics(
        self,
        db: DbSession,
        user_id: UUID,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict[str, Any]]:
        """Fetch daily activity summaries from Polar API."""
        return self._fetch_date_range(db, user_id, "/v3/users/activities", start_date, end_date)

    def normalize_daily_activity(
        self,
        raw_stats: dict[str, Any],
        user_id: UUID,
    ) -> dict[str, Any]:
        """Normalize Polar daily activity to our schema.

        Polar returns `start_time` (ISO-8601) instead of a bare `date` field.
        Steps are in `steps`; calories in `calories` (total) and `active_calories`.
        `distance` is in meters.
        """
        recorded_at = raw_stats.get("start_time") or raw_stats.get("date")

        return {
            "user_id": user_id,
            "provider": self.provider_name,
            "date": recorded_at,
            "steps": raw_stats.get("steps"),
            "active_calories": raw_stats.get("active_calories"),
            "calories": raw_stats.get("calories"),
            "distance": raw_stats.get("distance"),
            "active_time_minutes": _parse_iso_duration_minutes(raw_stats.get("active_time")),
        }

    def save_daily_activity_statistics(
        self,
        db: DbSession,
        user_id: UUID,
        normalized_stats: list[dict[str, Any]],
    ) -> int:
        """Save daily activity statistics as DataPointSeries (bulk)."""
        all_samples: list[TimeSeriesSampleCreate] = []

        for stat in normalized_stats:
            date_str = stat.get("date")
            if not date_str:
                continue

            recorded_at = parse_iso_datetime(date_str)
            if not recorded_at:
                continue

            for field_name, series_type in _ACTIVITY_SERIES_MAP.items():
                value = stat.get(field_name)
                if value is None:
                    continue
                all_samples.append(
                    TimeSeriesSampleCreate(
                        id=uuid4(),
                        user_id=user_id,
                        source=self.provider_name,
                        recorded_at=recorded_at,
                        value=Decimal(str(value)),
                        series_type=series_type,
                    )
                )

        if all_samples:
            self.data_point_repo.bulk_create(db, all_samples)

        return len(all_samples)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def load_and_save_all(
        self,
        db: DbSession,
        user_id: UUID,
        start_time: datetime | str | None = None,
        end_time: datetime | str | None = None,
        is_first_sync: bool = False,
    ) -> dict[str, int]:
        """Load all Polar health data types and save to database."""
        now = datetime.now()
        end_dt = parse_datetime_or_default(end_time, now)
        start_dt = parse_datetime_or_default(start_time, end_dt - timedelta(days=28))

        results: dict[str, int] = {
            "sleep_sessions_synced": 0,
            "recovery_samples_synced": 0,
            "continuous_hr_samples_synced": 0,
            "daily_activity_synced": 0,
        }

        # 1. Sleep → EventRecord + SleepDetails + HealthScore
        try:
            results["sleep_sessions_synced"] = self.load_and_save_sleep(db, user_id, start_dt, end_dt)
        except Exception as e:
            log_structured(
                self.logger,
                "error",
                f"Failed to sync sleep data: {e}",
                provider=self.provider_name,
                task="load_and_save_all",
            )

        # 2. Nightly Recharge → DataPointSeries (recovery_score, HRV) + HealthScore
        try:
            results["recovery_samples_synced"] = self.load_and_save_recovery(db, user_id, start_dt, end_dt)
        except Exception as e:
            log_structured(
                self.logger,
                "error",
                f"Failed to sync nightly recharge data: {e}",
                provider=self.provider_name,
                task="load_and_save_all",
            )

        # 3. Continuous HR → DataPointSeries (heart_rate)
        try:
            results["continuous_hr_samples_synced"] = self.load_and_save_continuous_heart_rate(
                db, user_id, start_dt, end_dt
            )
        except Exception as e:
            log_structured(
                self.logger,
                "error",
                f"Failed to sync continuous HR data: {e}",
                provider=self.provider_name,
                task="load_and_save_all",
            )

        # 4. Daily activities → DataPointSeries (steps, energy)
        try:
            raw_daily = self.get_daily_activity_statistics(db, user_id, start_dt, end_dt)
            normalized_daily = [self.normalize_daily_activity(item, user_id) for item in raw_daily]
            results["daily_activity_synced"] = self.save_daily_activity_statistics(db, user_id, normalized_daily)
        except Exception as e:
            log_structured(
                self.logger,
                "error",
                f"Failed to sync daily activity data: {e}",
                provider=self.provider_name,
                task="load_and_save_all",
            )

        return results
