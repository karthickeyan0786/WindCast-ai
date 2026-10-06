# WindCast AI — Inference API container
#
# WHY containerize: the API depends on an exact combination of Python,
# scikit-learn, xgboost, and the trained model binary. "Works on my
# machine" fails the moment a teammate or a cloud VM has a different
# library version — a stale scikit-learn can even fail to *unpickle* a
# model trained on a newer version. Docker packages the interpreter, every
# dependency, and the code into one immutable image, so the exact same
# artifact that passed testing is what runs in production. It also makes
# horizontal scaling trivial: `docker run` the same image N times behind
# a load balancer.

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (separate layer) so Docker's build cache
# skips this slow step whenever only application code changes.
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source, trained model, and API code
COPY src/ ./src/
COPY models/ ./models/
COPY api/ ./api/
COPY data/ ./data/

EXPOSE 8000

# --host 0.0.0.0 is required so the server is reachable from OUTSIDE the
# container, not just from localhost inside it.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
