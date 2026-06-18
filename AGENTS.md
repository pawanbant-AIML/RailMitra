# RailYatra Engineering Instructions

## General Rules

* Analyze before editing.
* Make the smallest correct change possible.
* Preserve working functionality.
* Prefer patches over rewrites.
* Explain every modification.

## Architecture Rules

* Backend: FastAPI
* Frontend: React/Vite
* Preserve existing API contracts unless explicitly changing them.
* Avoid introducing breaking changes.

## Booking Rules

Booking must be form-driven.

Chat may:

* detect booking intent
* extract slots
* prefill data

Chat must not:

* silently create bookings
* skip validation

Required booking fields:

* source
* destination
* travel_date
* travel_class
* passenger_count
* confirmed_train

Optional fields:

* berth_preference
* time_preference
* quota
* budget

## Session Memory Rules

Only reuse memory for obvious follow-ups:

Examples:

* "book the first one"
* "what about fare?"
* "that train"
* "same route tomorrow"

Do not reuse stale routes for new searches.

## LLM Rules

Always log:

* llm_used
* llm_failed
* fallback_used

Never fail silently.

## API Rules

If frontend needs to perform a UI action:

Return structured payloads.

Example:

{
"action": "OPEN_BOOKING_DRAWER",
"booking_draft": {}
}

Avoid relying on chat text parsing for UI behavior.

## Change Management

Before coding:

1. Analyze
2. Identify affected files
3. Explain risks
4. Propose plan
5. Wait for approval

Then implement in phases.
