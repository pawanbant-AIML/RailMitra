# Rail Mitra (AI Train Ticket Assistant)

A natural‑language assistant for Indian Railways, built with real open data.

## Quick Start

1. `docker compose up --build`
2. Open http://localhost:5173

## Enabling the LLM Agent (Optional)

By default, Rail Mitra uses a robust local NLP (regex/fuzzy-matching) engine to understand queries, extract slots, and handle bookings. This means the app works perfectly out-of-the-box.

However, the architecture supports a powerful LLM Agent pipeline (using Hugging Face models) for complex reasoning. To enable it:

1. Sign up at [Hugging Face](https://huggingface.co/) and generate an Access Token.
2. In the `backend/` folder, copy `.env.example` to `.env`.
3. Set the variables:
   ```env
   HUGGINGFACE_API_KEY=your_actual_token_here
   HUGGINGFACEHUB_API_TOKEN=your_actual_token_here
   ```
4. Restart the backend. The app will automatically detect the token and route complex queries through the LLM!

See full documentation in `docs/` and below.