import time

from collections import deque

from config.config import (
    MAX_REQUESTS_PER_MINUTE,
    MAX_REQUESTS_PER_SESSION,
    BLOCK_DURATION,
)


# ------------------------------
# CONTRÔLE DES REQUÊTES
# ------------------------------

def check_request_limit(
    request_times: deque,
    request_count: int,
    blocked_until: float,
) -> tuple[bool, str | None, int, float]:
    """Vérifie les limites de requêtes de la session."""

    now = time.time()

    # Vérification d'un éventuel blocage temporaire
    if now < blocked_until:

        remaining = int(blocked_until - now) + 1

        return (
            False,
            f"Trop de requêtes. Veuillez patienter {remaining} seconde(s).",
            request_count,
            blocked_until,
        )

    # Suppression des requêtes datant de plus d'une minute
    while (
        request_times
        and now - request_times[0] > 60
    ):
        request_times.popleft()

    # Vérification du nombre maximum de requêtes par session
    if request_count >= MAX_REQUESTS_PER_SESSION:

        return (
            False,
            "La limite de requêtes pour cette session a été atteinte.",
            request_count,
            blocked_until,
        )

    # Vérification du nombre maximum de requêtes par minute
    if len(request_times) >= MAX_REQUESTS_PER_MINUTE:

        blocked_until = now + BLOCK_DURATION

        return (
            False,
            "Trop de requêtes rapprochées. Veuillez patienter avant de continuer.",
            request_count,
            blocked_until,
        )

    # nregistrement de la requête autorisée
    request_times.append(now)
    request_count += 1

    return (
        True,
        None,
        request_count,
        blocked_until,
    )