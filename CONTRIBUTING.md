# Contributing to CHECK

Use GitHub Issues to report problems or propose changes. Include the CHECK version, reproduction steps, and a synthetic ticket when the issue depends on conversation content. Keep credentials, customer data, model weights, and local benchmark exports out of issues and commits.

## Build the web application

Use Node.js 22 and run:

```sh
cd frontend
npm ci
npm run build
```

## Run the backend tests

Use Python 3.12 and an isolated PostgreSQL test database. The tests create tables and write evaluation data. Run these commands from the repository root:

```sh
docker run --rm -d --name check-test-db -p 127.0.0.1:55432:5432 \
  -e POSTGRES_PASSWORD=check_test -e POSTGRES_DB=check_test postgres:17-alpine
docker exec check-test-db pg_isready -U postgres -d check_test
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r backend/requirements.txt
cd backend
DATABASE_URL=postgresql+psycopg://postgres:check_test@127.0.0.1:55432/check_test \
  python -m unittest discover -s tests
docker stop check-test-db
```

Wait for PostgreSQL to become ready before running the tests. Model and Zendesk requests are mocked by the test suite. The quick start in the [README](README.md) runs the full Compose installation.

Keep user-facing text, model instructions, examples, and documentation in English. Imported source material and quotes can retain their original language.
