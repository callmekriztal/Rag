# Setup

## Create project environment

```bash
uv venv
source .venv/bin/activate
```

## Install dependencies

For a fresh project:

```bash
uv add -r requirements.txt
```

For an existing cloned repository:

```bash
uv sync
```

## Configure environment

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

## Run application

```bash
uv run python app.py
```

