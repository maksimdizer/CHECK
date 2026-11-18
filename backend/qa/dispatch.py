import logging
from datetime import timedelta
from time import sleep

from sqlalchemy import select

from .db import Outbox, Session, now
from .jobs import evaluate_run

logger = logging.getLogger(__name__)


def publish():
    with Session.begin() as session:
        row = session.scalar(
            select(Outbox).where(Outbox.available_at <= now())
            .order_by(Outbox.available_at).limit(1).with_for_update(skip_locked=True)
        )
        if row is None:
            return False
        evaluate_run.apply_async(args=(row.run_id,), retry=False, confirm_timeout=5)
        # Keep the row until completion so expired worker leases can be recovered.
        row.available_at = now() + timedelta(minutes=1)
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    while True:
        try:
            if not publish():
                sleep(0.7)
        except Exception as exc:
            logger.warning("Outbox publish failed (%s)", type(exc).__name__)
            sleep(3)
