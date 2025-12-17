"""
EmberVLM Carbon Tracking Utilities
CodeCarbon integration for environmental impact monitoring.
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
import threading
import time

logger = logging.getLogger(__name__)

# Try to import codecarbon
try:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    EmissionsTracker = None
    OfflineEmissionsTracker = None


@dataclass
class EmissionsSnapshot:
    """Snapshot of emissions data."""
    timestamp: str
    step: int
    emissions_kg: float
    energy_kwh: float
    duration_hours: float
    cpu_power_w: float
    gpu_power_w: float
    ram_power_w: float

    def to_dict(self) -> Dict:
        return asdict(self)


class CarbonTracker:
    """
    Carbon emissions tracker for training.
    Integrates with CodeCarbon for real-time monitoring.
    """

    def __init__(
        self,
        project_name: str = "embervlm",
        output_dir: str = "carbon_logs",
        log_level: str = "warning",
        max_budget_kg_co2: float = 50.0,
        alert_threshold_per_hour: float = 5.0,
        offline_mode: bool = False,
        country_iso_code: str = "USA"
    ):
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_budget = max_budget_kg_co2
        self.alert_threshold = alert_threshold_per_hour

        self.tracker = None
        self.is_tracking = False
        self.snapshots: list = []
        self.start_time = None
        self.current_step = 0

        # Alert callbacks
        self.alert_callbacks: list = []

        # Initialize tracker
        if CODECARBON_AVAILABLE:
            try:
                if offline_mode:
                    self.tracker = OfflineEmissionsTracker(
                        project_name=project_name,
                        output_dir=str(self.output_dir),
                        log_level=log_level,
                        country_iso_code=country_iso_code,
                        tracking_mode="process"
                    )
                else:
                    self.tracker = EmissionsTracker(
                        project_name=project_name,
                        output_dir=str(self.output_dir),
                        log_level=log_level,
                        tracking_mode="process"
                    )
                logger.info("CodeCarbon tracker initialized")
            except Exception as e:
                logger.warning(f"Could not initialize CodeCarbon: {e}")
                self.tracker = None
        else:
            logger.warning("CodeCarbon not available. Install with: pip install codecarbon")

    def start(self):
        """Start emissions tracking."""
        if self.tracker:
            self.tracker.start()
            self.is_tracking = True
            self.start_time = datetime.now()
            logger.info("Started carbon tracking")

    def stop(self) -> float:
        """Stop tracking and return total emissions in kg CO2."""
        if self.tracker and self.is_tracking:
            emissions = self.tracker.stop()
            self.is_tracking = False

            # Log final summary
            self._save_summary()

            logger.info(f"Total emissions: {emissions:.4f} kg CO2")
            return emissions
        return 0.0

    def update(self, step: int):
        """Update tracking with current step."""
        self.current_step = step

        if not self.tracker or not self.is_tracking:
            return

        try:
            # Get current emissions
            emissions_data = self._get_emissions_data()

            if emissions_data:
                snapshot = EmissionsSnapshot(
                    timestamp=datetime.now().isoformat(),
                    step=step,
                    **emissions_data
                )
                self.snapshots.append(snapshot)

                # Check budget and alerts
                self._check_budget(emissions_data['emissions_kg'])
                self._check_rate_alert(emissions_data)

        except Exception as e:
            logger.debug(f"Could not update emissions: {e}")

    def _get_emissions_data(self) -> Optional[Dict]:
        """Get current emissions data from tracker."""
        if not self.tracker:
            return None

        try:
            # Access tracker's internal data
            emissions = self.tracker._total_emissions or 0.0
            energy = self.tracker._total_energy or 0.0

            duration = 0.0
            if self.start_time:
                duration = (datetime.now() - self.start_time).total_seconds() / 3600

            # Power estimates (simplified)
            cpu_power = getattr(self.tracker, '_cpu_power', 0.0) or 0.0
            gpu_power = getattr(self.tracker, '_gpu_power', 0.0) or 0.0
            ram_power = getattr(self.tracker, '_ram_power', 0.0) or 0.0

            return {
                'emissions_kg': emissions,
                'energy_kwh': energy,
                'duration_hours': duration,
                'cpu_power_w': cpu_power,
                'gpu_power_w': gpu_power,
                'ram_power_w': ram_power
            }
        except Exception:
            return None

    def _check_budget(self, current_emissions: float):
        """Check if emissions exceed budget."""
        if current_emissions > self.max_budget:
            message = f"CARBON BUDGET EXCEEDED: {current_emissions:.2f} kg > {self.max_budget} kg"
            logger.warning(message)
            self._trigger_alerts(message, "budget_exceeded")

    def _check_rate_alert(self, emissions_data: Dict):
        """Check if emissions rate is too high."""
        duration = emissions_data['duration_hours']
        if duration > 0:
            rate = emissions_data['emissions_kg'] / duration
            if rate > self.alert_threshold:
                message = f"HIGH EMISSION RATE: {rate:.2f} kg/hour > {self.alert_threshold} kg/hour"
                logger.warning(message)
                self._trigger_alerts(message, "high_rate")

    def _trigger_alerts(self, message: str, alert_type: str):
        """Trigger registered alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(message, alert_type)
            except Exception as e:
                logger.debug(f"Alert callback failed: {e}")

    def register_alert_callback(self, callback: Callable[[str, str], None]):
        """Register a callback for carbon alerts."""
        self.alert_callbacks.append(callback)

    def _save_summary(self):
        """Save emissions summary to file."""
        summary_path = self.output_dir / f"{self.project_name}_summary.json"

        if not self.snapshots:
            return

        summary = {
            'project_name': self.project_name,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': datetime.now().isoformat(),
            'total_steps': self.current_step,
            'final_emissions_kg': self.snapshots[-1].emissions_kg if self.snapshots else 0,
            'final_energy_kwh': self.snapshots[-1].energy_kwh if self.snapshots else 0,
            'snapshots': [s.to_dict() for s in self.snapshots]
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved emissions summary to {summary_path}")

    def get_current_emissions(self) -> float:
        """Get current total emissions in kg CO2."""
        if self.snapshots:
            return self.snapshots[-1].emissions_kg
        return 0.0

    def get_emissions_per_step(self) -> float:
        """Calculate average emissions per training step."""
        if self.current_step > 0 and self.snapshots:
            return self.snapshots[-1].emissions_kg / self.current_step
        return 0.0

    def get_carbon_equivalent(self) -> Dict[str, float]:
        """Convert emissions to real-world equivalents."""
        emissions = self.get_current_emissions()

        return {
            'kg_co2': emissions,
            'car_km': emissions / 0.12,  # ~0.12 kg CO2 per km for average car
            'smartphone_charges': emissions / 0.0085,  # ~8.5g CO2 per charge
            'tree_hours': emissions / 0.001,  # Tree absorbs ~1g CO2 per hour
            'streaming_hours': emissions / 0.036  # ~36g CO2 per hour of streaming
        }

    def log_to_wandb(self, step: int):
        """Log emissions data to WandB."""
        try:
            import wandb

            emissions_data = self._get_emissions_data()
            if emissions_data:
                wandb.log({
                    'carbon/emissions_kg': emissions_data['emissions_kg'],
                    'carbon/energy_kwh': emissions_data['energy_kwh'],
                    'carbon/duration_hours': emissions_data['duration_hours'],
                    'carbon/gpu_power_w': emissions_data['gpu_power_w'],
                    'carbon/emissions_per_step': self.get_emissions_per_step()
                }, step=step)

        except ImportError:
            pass


class DynamicBatchSizer:
    """
    Dynamically adjust batch size based on carbon emission rate.
    Reduces batch size when emissions are too high.
    """

    def __init__(
        self,
        carbon_tracker: CarbonTracker,
        initial_batch_size: int,
        min_batch_size: int = 1,
        max_batch_size: int = 256,
        target_emissions_per_step: float = 0.001  # kg CO2 per step
    ):
        self.tracker = carbon_tracker
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_emissions = target_emissions_per_step

    def update(self) -> int:
        """Update and return recommended batch size."""
        emissions_per_step = self.tracker.get_emissions_per_step()

        if emissions_per_step <= 0:
            return self.current_batch_size

        # Adjust batch size based on emissions
        if emissions_per_step > self.target_emissions * 1.5:
            # Too high, reduce batch size
            self.current_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
            logger.info(f"Reduced batch size to {self.current_batch_size} due to high emissions")

        elif emissions_per_step < self.target_emissions * 0.5:
            # Low emissions, can increase
            self.current_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * 1.2)
            )

        return self.current_batch_size


def create_carbon_tracker(
    project_name: str = "embervlm",
    max_budget_kg: float = 50.0,
    **kwargs
) -> CarbonTracker:
    """Factory function for carbon tracker."""
    return CarbonTracker(
        project_name=project_name,
        max_budget_kg_co2=max_budget_kg,
        **kwargs
    )


if __name__ == "__main__":
    # Test carbon tracking
    print("Testing Carbon Tracking Module...")

    tracker = CarbonTracker(
        project_name="test_project",
        output_dir="test_carbon",
        max_budget_kg_co2=1.0,
        offline_mode=True
    )

    # Test tracking
    tracker.start()

    # Simulate training steps
    for step in range(10):
        time.sleep(0.1)
        tracker.update(step)

    emissions = tracker.stop()
    print(f"Test emissions: {emissions:.6f} kg CO2")

    # Test equivalents
    equivalents = tracker.get_carbon_equivalent()
    print(f"Equivalents: {equivalents}")

    print("Carbon tracking tests complete!")

