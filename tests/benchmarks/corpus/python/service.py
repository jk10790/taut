import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Reservation:
    """A single seat reservation.

    Attributes:
        reservation_id: The unique identifier for this reservation.
        passenger_name: Full legal name as printed on the travel document.
        seat: Seat assignment in row-letter form, e.g. "14C".
        confirmed: Whether payment has cleared.
    """
    reservation_id: str
    passenger_name: str
    seat: Optional[str] = None
    confirmed: bool = False


class ReservationService:
    """Coordinates seat reservations against the inventory backend.

    This service is deliberately synchronous; the calling layer is responsible
    for running it in a thread pool when used from async code.
    """

    def __init__(self, backend, max_retries: int = 3) -> None:
        """Initialise the service.

        Args:
            backend: Any object exposing reserve() and release().
            max_retries: How many times to retry a transient backend failure.
        """
        self.backend = backend
        self.max_retries = max_retries

    def reserve(self, passenger_name: str, seat: str) -> Reservation:
        """Reserve a seat for a passenger.

        Args:
            passenger_name: Full legal name.
            seat: Requested seat identifier.

        Returns:
            The created Reservation.

        Raises:
            ValueError: If the seat identifier is malformed.
        """
        if not seat or len(seat) < 2:
            raise ValueError("seat identifier must be at least two characters")
        # Attempt the reservation, retrying transient failures.
        for attempt in range(self.max_retries):
            try:
                record = self.backend.reserve(passenger_name, seat)
                return Reservation(
                    reservation_id=record["id"],
                    passenger_name=passenger_name,
                    seat=seat,
                    confirmed=record.get("confirmed", False),
                )
            except TimeoutError:
                logger.warning("backend timeout on attempt %d", attempt)
        raise RuntimeError("could not reserve seat")

    def release(self, reservation_id: str) -> bool:
        """Release a previously held reservation.

        Args:
            reservation_id: Identifier returned by reserve().

        Returns:
            True if the seat was released.
        """
        # The backend is idempotent, so a double release is harmless.
        return bool(self.backend.release(reservation_id))
