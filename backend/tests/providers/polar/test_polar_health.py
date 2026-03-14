"""
Tests for Polar247Data — normalization and save logic.

Tests cover:
- Sleep normalization (stage durations, start/end times, efficiency)
- Sleep health score extraction
- Nightly recharge normalization (ANS charge scaling, HRV)
- Recovery health score extraction
- Continuous HR sample saving
- Daily activity normalization (steps, calories)
- load_and_save_all orchestration (with mocked API calls)
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.services.providers.polar.data_247 import Polar247Data
from app.services.providers.polar.oauth import PolarOAuth


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def polar_health(db) -> Polar247Data:
    """Create Polar247Data instance for testing."""
    oauth = PolarOAuth(
        user_repo=MagicMock(),
        connection_repo=MagicMock(),
        provider_name="polar",
        api_base_url="https://www.polaraccesslink.com",
    )
    return Polar247Data(
        provider_name="polar",
        api_base_url="https://www.polaraccesslink.com",
        oauth=oauth,
    )


@pytest.fixture
def sample_sleep_raw() -> dict:
    return {
        "date": "2024-01-15",
        "sleep_start_time": "2024-01-14T23:00:00",
        "sleep_end_time": "2024-01-15T07:00:00",
        "total_sleep_minutes": 420,
        "light_sleep": 9000,
        "deep_sleep": 5400,
        "rem_sleep": 6000,
        "sleep_score": 82,
        "heart_rate_avg": 52,
        "heart_rate_variability_avg": 45,
    }


@pytest.fixture
def sample_recovery_raw() -> dict:
    return {
        "date": "2024-01-15",
        "ans_charge": 3.2,
        "heart_rate_variability_avg": 48,
        "heart_rate_avg": 50,
    }


@pytest.fixture
def sample_activity_raw() -> dict:
    return {
        "date": "2024-01-15",
        "steps": 8500,
        "active_calories": 650,
        "calories": 2200,
        "distance": 6800.0,
        "active_time": "PT2H31M4S",
    }


# ---------------------------------------------------------------------------
# Sleep normalization
# ---------------------------------------------------------------------------


class TestPolarHealthSleepNormalization:
    def test_normalize_sleep_basic_fields(self, polar_health: Polar247Data, sample_sleep_raw: dict) -> None:
        user_id = uuid4()
        result, _ = polar_health.normalize_sleep(sample_sleep_raw, user_id)

        assert result["user_id"] == user_id
        assert result["provider"] == "polar"
        assert result["start_time"] == "2024-01-14T23:00:00"
        assert result["end_time"] == "2024-01-15T07:00:00"
        # duration_seconds = time in bed (end - start = 8h = 28800s), not total_sleep_minutes
        assert result["duration_seconds"] == 8 * 3600
        assert result["time_in_bed_seconds"] == 8 * 3600
        assert result["total_sleep_seconds"] == 420 * 60
        assert result["efficiency_percent"] == 82
        assert result["is_nap"] is False

    def test_normalize_sleep_stage_durations(self, polar_health: Polar247Data, sample_sleep_raw: dict) -> None:
        user_id = uuid4()
        result, _ = polar_health.normalize_sleep(sample_sleep_raw, user_id)

        stages = result["stages"]
        assert stages["light_seconds"] == 9000
        assert stages["deep_seconds"] == 5400
        assert stages["rem_seconds"] == 6000
        # awake = time_in_bed - (light + deep + rem)
        expected_awake = max(0, 8 * 3600 - 9000 - 5400 - 6000)
        assert stages["awake_seconds"] == expected_awake

    def test_normalize_sleep_hr_and_hrv(self, polar_health: Polar247Data, sample_sleep_raw: dict) -> None:
        user_id = uuid4()
        result, _ = polar_health.normalize_sleep(sample_sleep_raw, user_id)

        assert result["avg_heart_rate_bpm"] == 52
        assert result["avg_hrv_ms"] == 45

    def test_normalize_sleep_missing_stages(self, polar_health: Polar247Data) -> None:
        user_id = uuid4()
        raw = {
            "date": "2024-01-15",
            "sleep_start_time": "2024-01-14T23:00:00",
            "sleep_end_time": "2024-01-15T07:00:00",
            "total_sleep_minutes": 60,
        }
        result, _ = polar_health.normalize_sleep(raw, user_id)

        stages = result["stages"]
        assert stages["deep_seconds"] == 0
        assert stages["light_seconds"] == 0
        assert stages["rem_seconds"] == 0
        # awake = time_in_bed (8h) since no stage data
        assert stages["awake_seconds"] == 8 * 3600

    def test_normalize_sleep_missing_start_end_times(self, polar_health: Polar247Data) -> None:
        user_id = uuid4()
        raw = {"date": "2024-01-15", "total_sleep_minutes": 420}
        result, _ = polar_health.normalize_sleep(raw, user_id)

        assert result["start_time"] is None
        assert result["end_time"] is None

    def test_normalize_sleep_returns_health_score(self, polar_health: Polar247Data, sample_sleep_raw: dict) -> None:
        user_id = uuid4()
        _, health_score = polar_health.normalize_sleep(sample_sleep_raw, user_id)

        assert health_score is not None
        assert health_score.value == 82
        assert health_score.category == "sleep"
        assert health_score.provider == "polar"

    def test_normalize_sleep_no_health_score_without_score(self, polar_health: Polar247Data) -> None:
        user_id = uuid4()
        raw = {
            "date": "2024-01-15",
            "sleep_start_time": "2024-01-14T23:00:00",
            "sleep_end_time": "2024-01-15T07:00:00",
            "total_sleep_minutes": 420,
        }
        _, health_score = polar_health.normalize_sleep(raw, user_id)

        assert health_score is None

    @patch("app.services.providers.polar.data_247.event_record_service")
    def test_save_sleep_skips_missing_times(self, mock_service: MagicMock, polar_health: Polar247Data, db) -> None:
        user_id = uuid4()
        normalized = {
            "id": uuid4(),
            "start_time": None,
            "end_time": None,
            "duration_seconds": 3600,
            "efficiency_percent": None,
            "is_nap": False,
            "stages": {},
            "polar_sleep_date": "2024-01-15",
        }
        polar_health.save_sleep_data(db, user_id, normalized)
        mock_service.create_or_merge_sleep.assert_not_called()

    @patch("app.services.providers.polar.data_247.event_record_service")
    def test_save_sleep_creates_record_and_detail(
        self, mock_service: MagicMock, polar_health: Polar247Data, db, sample_sleep_raw: dict
    ) -> None:
        user_id = uuid4()
        normalized, _ = polar_health.normalize_sleep(sample_sleep_raw, user_id)
        polar_health.save_sleep_data(db, user_id, normalized)

        mock_service.create_or_merge_sleep.assert_called_once()


# ---------------------------------------------------------------------------
# Recovery (Nightly Recharge) normalization
# ---------------------------------------------------------------------------


class TestPolarHealthRecoveryNormalization:
    def test_normalize_recovery_scales_ans_charge(
        self, polar_health: Polar247Data, sample_recovery_raw: dict
    ) -> None:
        user_id = uuid4()
        result, _ = polar_health.normalize_recovery(sample_recovery_raw, user_id)

        # 3.2 / 4.0 * 100 = 80.0
        assert result["ans_charge"] == pytest.approx(80.0)

    def test_normalize_recovery_hrv(self, polar_health: Polar247Data, sample_recovery_raw: dict) -> None:
        user_id = uuid4()
        result, _ = polar_health.normalize_recovery(sample_recovery_raw, user_id)

        assert result["heart_rate_variability_avg"] == 48

    def test_normalize_recovery_missing_ans_charge(self, polar_health: Polar247Data) -> None:
        user_id = uuid4()
        raw = {"date": "2024-01-15", "heart_rate_variability_avg": 40}
        result, _ = polar_health.normalize_recovery(raw, user_id)

        assert result["ans_charge"] is None

    def test_normalize_recovery_returns_health_score(
        self, polar_health: Polar247Data, sample_recovery_raw: dict
    ) -> None:
        user_id = uuid4()
        _, health_score = polar_health.normalize_recovery(sample_recovery_raw, user_id)

        assert health_score is not None
        assert health_score.value == pytest.approx(80.0)
        assert health_score.category == "recovery"
        assert health_score.provider == "polar"

    def test_normalize_recovery_no_health_score_without_ans_charge(self, polar_health: Polar247Data) -> None:
        user_id = uuid4()
        raw = {"date": "2024-01-15", "heart_rate_variability_avg": 40}
        _, health_score = polar_health.normalize_recovery(raw, user_id)

        assert health_score is None

    def test_save_recovery_skips_missing_date(self, polar_health: Polar247Data, db) -> None:
        user_id = uuid4()
        count = polar_health.save_recovery_data(db, user_id, {"ans_charge": 80.0})
        assert count == 0

    @patch.object(Polar247Data, "data_point_repo")
    def test_save_recovery_creates_series_samples(
        self, mock_repo: MagicMock, polar_health: Polar247Data, db, sample_recovery_raw: dict
    ) -> None:
        user_id = uuid4()
        normalized, _ = polar_health.normalize_recovery(sample_recovery_raw, user_id)
        count = polar_health.save_recovery_data(db, user_id, normalized)

        # Should save: recovery_score (ans_charge) + HRV = 2 samples
        assert count == 2


# ---------------------------------------------------------------------------
# Continuous Heart Rate
# ---------------------------------------------------------------------------


class TestPolarHealthContinuousHR:
    def test_save_continuous_hr_bulk_creates(self, polar_health: Polar247Data, db) -> None:
        user_id = uuid4()
        samples = [
            {"recorded_at": "2024-01-15T10:00:00", "heart_rate": 65},
            {"recorded_at": "2024-01-15T10:01:00", "heart_rate": 67},
        ]
        with patch.object(polar_health.data_point_repo, "bulk_create") as mock_bulk:
            count = polar_health.save_continuous_heart_rate(db, user_id, samples)

        assert count == 2
        mock_bulk.assert_called_once()
        created = mock_bulk.call_args[0][1]
        assert len(created) == 2
        assert all(s.value in (Decimal("65"), Decimal("67")) for s in created)

    def test_save_continuous_hr_skips_invalid_timestamps(self, polar_health: Polar247Data, db) -> None:
        user_id = uuid4()
        samples = [
            {"recorded_at": None, "heart_rate": 65},
            {"recorded_at": "not-a-date", "heart_rate": 70},
        ]
        with patch.object(polar_health.data_point_repo, "bulk_create") as mock_bulk:
            count = polar_health.save_continuous_heart_rate(db, user_id, samples)

        assert count == 0
        mock_bulk.assert_not_called()

    def test_save_continuous_hr_empty(self, polar_health: Polar247Data, db) -> None:
        user_id = uuid4()
        with patch.object(polar_health.data_point_repo, "bulk_create") as mock_bulk:
            count = polar_health.save_continuous_heart_rate(db, user_id, [])

        assert count == 0
        mock_bulk.assert_not_called()


# ---------------------------------------------------------------------------
# Daily Activity Statistics
# ---------------------------------------------------------------------------


class TestPolarHealthDailyActivity:
    def test_normalize_daily_activity_steps_and_calories(
        self, polar_health: Polar247Data, sample_activity_raw: dict
    ) -> None:
        user_id = uuid4()
        result = polar_health.normalize_daily_activity(sample_activity_raw, user_id)

        assert result["steps"] == 8500
        assert result["active_calories"] == 650
        assert result["date"] == "2024-01-15"
        assert result["distance"] == 6800.0

    def test_normalize_daily_activity_active_time_parsed(
        self, polar_health: Polar247Data, sample_activity_raw: dict
    ) -> None:
        user_id = uuid4()
        result = polar_health.normalize_daily_activity(sample_activity_raw, user_id)

        # "PT2H31M4S" = 2*60 + 31 + 4/60 ≈ 151.067 minutes
        assert result["active_time_minutes"] == pytest.approx(2 * 60 + 31 + 4 / 60, abs=0.01)

    def test_normalize_daily_activity_active_time_missing(self, polar_health: Polar247Data) -> None:
        user_id = uuid4()
        raw = {"date": "2024-01-15", "steps": 5000}
        result = polar_health.normalize_daily_activity(raw, user_id)
        assert result["active_time_minutes"] is None

    def test_normalize_daily_activity_falls_back_to_steps(self, polar_health: Polar247Data) -> None:
        user_id = uuid4()
        raw = {"date": "2024-01-15", "steps": 5000}
        result = polar_health.normalize_daily_activity(raw, user_id)
        assert result["steps"] == 5000

    def test_save_daily_activity_bulk_creates(
        self, polar_health: Polar247Data, db, sample_activity_raw: dict
    ) -> None:
        user_id = uuid4()
        normalized = [polar_health.normalize_daily_activity(sample_activity_raw, user_id)]

        with patch.object(polar_health.data_point_repo, "bulk_create") as mock_bulk:
            count = polar_health.save_daily_activity_statistics(db, user_id, normalized)

        # steps + active_calories + distance + active_time_minutes = 4 samples
        assert count == 4
        mock_bulk.assert_called_once()

    def test_save_daily_activity_skips_missing_date(self, polar_health: Polar247Data, db) -> None:
        user_id = uuid4()
        normalized = [{"steps": 5000, "active_calories": 300}]  # no date

        with patch.object(polar_health.data_point_repo, "bulk_create") as mock_bulk:
            count = polar_health.save_daily_activity_statistics(db, user_id, normalized)

        assert count == 0
        mock_bulk.assert_not_called()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class TestPolarHealthLoadAndSaveAll:
    @patch.object(Polar247Data, "load_and_save_sleep", return_value=2)
    @patch.object(Polar247Data, "load_and_save_recovery", return_value=3)
    @patch.object(Polar247Data, "load_and_save_continuous_heart_rate", return_value=100)
    @patch.object(Polar247Data, "get_daily_activity_statistics", return_value=[])
    def test_load_and_save_all_aggregates_results(
        self,
        mock_daily: MagicMock,
        mock_hr: MagicMock,
        mock_recovery: MagicMock,
        mock_sleep: MagicMock,
        polar_health: Polar247Data,
        db,
    ) -> None:
        user_id = uuid4()
        results = polar_health.load_and_save_all(db, user_id)

        assert results["sleep_sessions_synced"] == 2
        assert results["recovery_samples_synced"] == 3
        assert results["continuous_hr_samples_synced"] == 100
        assert results["daily_activity_synced"] == 0

    @patch.object(Polar247Data, "load_and_save_sleep", side_effect=RuntimeError("API error"))
    @patch.object(Polar247Data, "load_and_save_recovery", return_value=1)
    @patch.object(Polar247Data, "load_and_save_continuous_heart_rate", return_value=0)
    @patch.object(Polar247Data, "get_daily_activity_statistics", return_value=[])
    def test_load_and_save_all_continues_on_error(
        self,
        mock_daily: MagicMock,
        mock_hr: MagicMock,
        mock_recovery: MagicMock,
        mock_sleep: MagicMock,
        polar_health: Polar247Data,
        db,
    ) -> None:
        user_id = uuid4()
        results = polar_health.load_and_save_all(db, user_id)

        # Sleep failed but other data types still synced
        assert results["sleep_sessions_synced"] == 0
        assert results["recovery_samples_synced"] == 1

    def test_load_and_save_all_uses_default_date_range(self, polar_health: Polar247Data, db) -> None:
        user_id = uuid4()

        with (
            patch.object(Polar247Data, "load_and_save_sleep", return_value=0) as mock_sleep,
            patch.object(Polar247Data, "load_and_save_recovery", return_value=0),
            patch.object(Polar247Data, "load_and_save_continuous_heart_rate", return_value=0),
            patch.object(Polar247Data, "get_daily_activity_statistics", return_value=[]),
        ):
            polar_health.load_and_save_all(db, user_id)

            call_args = mock_sleep.call_args
            start_dt, end_dt = call_args[0][2], call_args[0][3]
            assert isinstance(start_dt, datetime)
            assert isinstance(end_dt, datetime)
            # Default range is ~28 days
            delta = end_dt - start_dt
            assert 27 <= delta.days <= 29
