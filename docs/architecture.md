# System Architecture

```mermaid
graph LR
    FE[React + Vite] -->|REST| API[FastAPI]
    subgraph Backend
        API --> DB[(PostgreSQL)]
        API --> NLP[NLP Service (spaCy + sklearn)]
        API --> MockBooking[Mock Booking Table]
    end