from collections import defaultdict
from copy import deepcopy
from hashlib import sha256

from sqlalchemy import select, text

from .db import Scorecard
from .settings import CARD, LEGACY_CARD_FINGERPRINTS
from .timing import METRIC_NAMES


class CardConflict(ValueError):
    pass


def active_card(session, org):
    saved = session.scalar(select(Scorecard).where(Scorecard.org == org).order_by(Scorecard.version.desc()).limit(1))
    card = english_card(saved.data if saved else CARD)
    if "timing_targets" not in card:
        targets = deepcopy(CARD["timing_targets"])
        first_reply = next((item for item in card["criteria"] if item["id"] == "first_response"), None)
        if first_reply:
            first_target = {"channel_seconds": first_reply.get("channel_threshold_seconds") or {}}
            if first_reply.get("threshold_seconds") is not None:
                first_target["default_seconds"] = first_reply["threshold_seconds"]
            targets["first_response"] = first_target
            card["criteria"] = [item for item in card["criteria"] if item["id"] != "first_response"]
            group_id = first_reply["group_id"]
            if not any(item["group_id"] == group_id and item["weight"] > 0 for item in card["criteria"]):
                for group in card.get("groups") or []:
                    if group["id"] == group_id and "weight" in group:
                        group["weight"] = 0
        card["timing_targets"] = targets
    return card


def english_card(snapshot):
    card = deepcopy(snapshot)
    if card["id"] != CARD["id"]:
        return card

    def translate(item, legacy, english, fields):
        for field in fields:
            value = item.get(field)
            if isinstance(value, str) and field in english and sha256(value.encode()).hexdigest() == legacy.get(field):
                item[field] = english[field]

    translate(card, LEGACY_CARD_FINGERPRINTS, CARD, ("name",))
    for section, fields in (("groups", ("name",)), ("criteria", ("name", "description", "applicability"))):
        originals = LEGACY_CARD_FINGERPRINTS[section]
        translations = {item["id"]: item for item in CARD[section]}
        if section == "criteria":
            translations["first_response"] = {"name": METRIC_NAMES["first_response"]}
        for item in card.get(section) or []:
            if item["id"] not in originals or item["id"] not in translations:
                continue
            legacy, english = originals[item["id"]], translations[item["id"]]
            translate(item, legacy, english, fields)
            if "rubric" in item:
                translate(item["rubric"], legacy.get("rubric", {}), english.get("rubric", {}), item["rubric"])
    return card


def save_card(session, org, payload):
    # Serialize publications per organization and reject stale editors.
    session.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"), {"key": "scorecard:" + org})
    current = active_card(session, org)
    if payload.base_version != current["version"]:
        raise CardConflict("The scorecard has changed. Refresh the page and apply your edits to the latest version.")
    criterion_weights = defaultdict(float)
    for criterion in payload.criteria:
        if criterion.id != "first_response":
            criterion_weights[criterion.group_id] += criterion.weight
    previous_criterion_weights = defaultdict(float)
    for criterion in current["criteria"]:
        previous_criterion_weights[criterion.get("group_id")] += criterion["weight"]
    previous_weights = {
        group["id"]: group.get("weight", previous_criterion_weights[group["id"]])
        for group in current.get("groups", [])
    }
    preserve_flat_weights = (
        payload.timing_targets is not None
        and any(group.get("weight") is None for group in current.get("groups", []))
        and all(group.weight is None for group in payload.groups)
    )
    groups = []
    for group in payload.groups:
        weight = group.weight
        if preserve_flat_weights:
            weight = criterion_weights[group.id]
        elif weight is None:
            weight = previous_weights.get(group.id, criterion_weights[group.id])
        if weight > 0 and not criterion_weights[group.id]:
            raise ValueError(f"Add a criterion with positive weight to “{group.name}” or set the group weight to 0")
        groups.append({"id": group.id, "name": group.name, "weight": weight})
    if not any(group["weight"] > 0 for group in groups):
        raise ValueError("Give at least one group a positive weight")
    builtins = {item["id"]: item for item in CARD["criteria"] if item["evaluator_type"] == "code"}
    old = {item["id"]: item for item in current["criteria"]}
    criteria = []
    for definition in payload.criteria:
        item = definition.model_dump(exclude_none=True)
        if item["id"] == "first_response":
            continue
        if item["evaluator_type"] == "code" or item["id"] in builtins:
            native = builtins.get(item["id"])
            if not native or item["evaluator_type"] != "code":
                raise ValueError("A computed check must use a supported built-in criterion")
            # Labels, weights, group and review policy are editable. Wording and
            # rubrics of code rules must continue to describe what the code does.
            for key in ("description", "applicability", "rubric", "required_evidence", "category"):
                if key in native:
                    item[key] = deepcopy(native[key])
        before = old.get(item["id"])
        comparable = lambda value: {k: v for k, v in value.items() if k != "version"}
        item["version"] = (before["version"] + (comparable(item) != comparable(before))) if before else 1
        criteria.append(item)
    targets = current["timing_targets"]
    if payload.timing_targets is not None:
        targets.update({metric: target.model_dump(exclude_none=True)
                        for metric, target in payload.timing_targets.items()})
    if preserve_flat_weights:
        groups = [{"id": group["id"], "name": group["name"]} for group in groups]
    card = {"id": current["id"], "version": current["version"] + 1, "name": payload.name,
            "groups": groups, "criteria": criteria, "timing_targets": targets}
    session.add(Scorecard(org=org, version=card["version"], data=card))
    return card
