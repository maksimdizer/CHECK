"""Calendar-time measurements kept separate from quality criteria."""

METRIC_NAMES = {
    "first_response": "First response time",
    "time_to_close": "Time to close",
}


def evaluate_timing(ticket, scope, targets, settings):
    customer = next((m for m in ticket.messages if m.public and m.role == "customer"), None)
    first_reply = next((m for m in targets if customer and m.at >= customer.at), None)
    closed = ticket.closed_at or next((e.at for e in reversed(ticket.events) if e.kind == "ticket_closed"), None)
    opened = ticket.created_at or (customer.at if customer and ticket.messages_complete else None)
    elapsed = {
        "first_response": (first_reply.at - customer.at).total_seconds()
        if ticket.messages_complete and first_reply else None,
        "time_to_close": (closed - opened).total_seconds()
        if ticket.status == "closed" and opened and closed and closed >= opened else None,
    }
    results = []
    for metric, name in METRIC_NAMES.items():
        config = settings.get(metric, {})
        target = config.get("channel_seconds", {}).get(ticket.channel, config.get("default_seconds"))
        actual = elapsed[metric] if scope.kind == "conversation" else None
        if scope.kind != "conversation":
            status = "not_applicable"
        elif actual is None:
            status = "unavailable"
        elif target is None:
            status = "no_target"
        else:
            status = "late" if actual > target else "met"
        results.append({
            "id": metric, "name": name, "status": status,
            "actual_seconds": actual, "target_seconds": target,
            "delay_seconds": max(0, actual - target) if actual is not None and target is not None else None,
        })
    return results
