import json
import logging
import os
from typing import Any, Dict

import azure.functions as func
from azure.core.exceptions import HttpResponseError
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Use colocated indexer (QuestionnaireIndexerHttp/indexer.py)
from . import indexer as local_indexer


def _json_response(status: int, body: Dict[str, Any]) -> func.HttpResponse:
    return func.HttpResponse(
        body=json.dumps(body, ensure_ascii=False),
        mimetype="application/json",
        status_code=status,
    )


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=10))
def _invoke_indexer(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not local_indexer:
        raise RuntimeError("indexer module not available in function app.")

    app_id = payload.get("appId")
    container = payload.get("container") or os.getenv("BLOB_CONTAINER")

    if not app_id:
        raise ValueError("Missing required field: appId")
    if not container:
        raise ValueError("Missing required field: container")

    # Always index the entire container
    result = local_indexer.index_container(app_id=app_id, container=container)
    return {"mode": "container", **(result or {})}


async def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("HTTP indexer function triggered")

    try:
        # Generate a simple correlation id for this request
        import uuid
        request_id = str(uuid.uuid4())

        if req.headers.get("Content-Type", "").startswith("application/json"):
            payload = req.get_json()
        else:
            try:
                payload = json.loads(req.get_body() or b"{}")
            except Exception:  # noqa: BLE001
                payload = {}

        if not isinstance(payload, dict):
            return _json_response(400, {"error": "Invalid JSON payload", "requestId": request_id})

        trimmed = {k: payload.get(k) for k in ("appId", "container")}
        logging.info("Payload received (%s): %s", request_id, trimmed)

        result = _invoke_indexer(payload)
        return _json_response(200, {
            "status": "ok",
            "requestId": request_id,
            "input": trimmed,
            "result": result,
        })

    except (ValueError, RuntimeError) as e:
        logging.error("Bad request: %s", e)
        return _json_response(400, {"status": "error", "message": str(e), "requestId": request_id})
    except HttpResponseError as e:
        logging.exception("Azure SDK error: %s", e)
        return _json_response(502, {"status": "error", "message": str(e), "requestId": request_id})
    except Exception as e:  # pylint: disable=broad-except
        logging.exception("Unhandled error: %s", e)
        return _json_response(500, {"status": "error", "message": str(e), "requestId": request_id})
