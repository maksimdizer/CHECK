import logging
import os
from datetime import timedelta

from celery import Celery
from sqlalchemy import select

from .agent import AgentProvider
from .cloud import CloudChatProvider
from .db import Outbox, Revision, Run, Session, new_id, now
from .evaluation import OllamaProvider, evaluate
from .model_settings import cloud_key
from .settings import ENGINE_VERSION, PROMPT_VERSION, QA_TASK_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)
celery = Celery("qa", broker=os.getenv("BROKER_URL", "amqp://qa:qa-local@localhost:5672//"))
celery.conf.update(
    task_acks_late=True, task_reject_on_worker_lost=True, task_ignore_result=True,
    task_time_limit=QA_TASK_TIMEOUT_SECONDS, worker_prefetch_multiplier=1, broker_connection_retry_on_startup=True,
    broker_transport_options={"confirm_publish": True},
    broker_connection_timeout=5, worker_enable_remote_control=False,
    worker_cancel_long_running_tasks_on_connection_loss=True,
)
MAX_ATTEMPTS = 3


@celery.task(name="qa.evaluate")
def evaluate_run(run_id):
    with Session.begin() as session:
        run = session.scalar(select(Run).where(Run.id == run_id).with_for_update())
        if not run or run.status in {"complete", "failed"} or (run.lease_until and run.lease_until > now()):
            return
        if run.attempts >= MAX_ATTEMPTS:
            run.status, run.error, run.finished_at = "failed", "The evaluation timed out.", now()
            run.lease = run.lease_until = None
            session.delete(session.get(Outbox, run.id))
            return
        lease = new_id()
        run.status, run.lease = "running", lease
        run.lease_until = now() + timedelta(seconds=QA_TASK_TIMEOUT_SECONDS + 60)
        run.attempts += 1
        config, data, org = run.config, session.get(Revision, run.revision_id).data, run.org

    result, error, provider = None, None, None
    try:
        if config["engine_version"] != ENGINE_VERSION or config["prompt_version"] not in (4, 5, 6, PROMPT_VERSION):
            raise ValueError("Incompatible job version")
        if config["model"]:
            options = {"num_predict": config.get("num_predict", 1200), "prompt_version": config["prompt_version"]}
            cloud = config.get("cloud_provider")
            if cloud:
                with Session() as session:
                    key = cloud_key(session, org, cloud)
                if not key:
                    raise ValueError("No API key is configured for the cloud provider")
                provider = CloudChatProvider(config["model"], provider=cloud, api_key=key, **options)
            elif config.get("agent_mode"):
                provider = AgentProvider(config["model"], quality=config.get("agent_quality", False),
                                         agent_version=config.get("agent_version", 6),
                                         context_tokens=config.get("context_tokens"),
                                         reasoning_tokens=config.get("reasoning_tokens"), **options)
            else:
                provider = OllamaProvider(config["model"], think=config.get("think"),
                                          context_tokens=config.get("context_tokens"),
                                          reasoning_tokens=config.get("reasoning_tokens"), **options)
        result = evaluate(data, config, provider)
    except Exception as exc:
        logger.warning("Evaluation %s failed (%s)", run_id, type(exc).__name__)
        error = "The evaluation failed. Check model availability and settings."
    finally:
        if provider:
            provider.release_cache()

    with Session.begin() as session:
        run = session.scalar(select(Run).where(Run.id == run_id).with_for_update())
        if run.lease != lease:
            return
        run.lease = run.lease_until = None
        outbox = session.get(Outbox, run_id)
        if error and run.attempts < MAX_ATTEMPTS:
            run.status = "queued"
            outbox.available_at = now() + timedelta(seconds=2 ** run.attempts)
        else:
            run.status = "failed" if error else "complete"
            run.result, run.error, run.finished_at = result, error, now()
            session.delete(outbox)
