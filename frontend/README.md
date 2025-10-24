# Frontend (Next.js)

Prerequisites
- Node.js >= 18.17 (Node 20 LTS recommended)
- npm >= 9 (npm 10 recommended)

Environment
- Create a .env.local file in this folder (or at the repo root .env) and set:
  NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/v1

Install
```bash
npm install
```

Run dev server
```bash
npm run dev
# opens http://localhost:3000
```

Build & start
```bash
npm run build
npm start
```

Notes
- The UI expects the FastAPI backend to be running at NEXT_PUBLIC_API_BASE_URL.
- If using Docker Compose for the backend, the default is http://localhost:8000/v1
- Tailwind CSS v4 is used (no tailwind.config needed by default).
