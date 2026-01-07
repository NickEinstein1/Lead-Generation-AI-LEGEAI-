# LEGEAI - AI-Powered Lead Generation Platform

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15.5.5-black)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-19-61dafb)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)](https://www.typescriptlang.org/)

**Revolutionize Your Leads with Intelligent Automation**

---

## Overview

LEGEAI is an enterprise-grade AI-powered lead generation and management platform designed to maximize sales efficiency and customer engagement through intelligent automation and advanced analytics.

### Key Features

- **AI-Powered Lead Scoring** - Deep learning models for intelligent lead qualification across multiple insurance types
- **Real-Time Analytics** - Comprehensive dashboards with actionable insights and performance metrics
- **Marketing Automation** - Automated campaigns, customer segmentation, and behavioral triggers
- **Multi-Channel Communications** - Integrated email, SMS, and call management
- **Document Management** - E-signature integration with DocuSeal for seamless contract processing
- **Meeting Scheduler** - Built-in scheduling with Zoom, Google Meet, and Teams integration
- **Insurance Product Management** - Specialized modules for Auto, Home, Health, and Life insurance
- **Enterprise Security** - JWT authentication with role-based access control

---

## Technology Stack

**Backend:**
- FastAPI 0.104.1 - High-performance async web framework
- Python 3.9+ - Core programming language
- TensorFlow & PyTorch - Deep learning frameworks
- SQLAlchemy - Database ORM
- PostgreSQL - Primary database
- Redis - Caching and session management

**Frontend:**
- Next.js 15.5.5 - React framework with App Router
- React 19 - UI library
- TypeScript 5.0 - Type-safe development
- Tailwind CSS - Modern styling framework

**Infrastructure:**
- Docker - Containerization
- Alembic - Database migrations
- Prometheus - Metrics and monitoring

---

## Quick Start

### Prerequisites

- **Python 3.9+** - [Download Python](https://www.python.org/downloads/)
- **Node.js 18.17.0+** - [Download Node.js](https://nodejs.org/)
- **PostgreSQL** (optional) - For database features
- **Redis** (optional) - For caching

### Installation

#### Option 1: Automated Setup (Recommended)

**Windows:**
```powershell
.\setup.ps1
```

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

#### Option 2: Manual Setup

**1. Clone the repository:**
```bash
git clone <repository-url>
cd Lead-Generation-AI-LEGEAI-
```

**2. Set up the backend:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**3. Set up the frontend:**
```bash
cd frontend
npm install
cd ..
```

**4. Configure environment variables:**
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
```

**5. Train ML models (optional):**
```bash
# Windows:
.\train_all_models.ps1

# macOS/Linux:
chmod +x train_all_models.sh
./train_all_models.sh
```

### Running the Application

**Start the backend:**
```bash
# Windows:
.\run_backend.ps1

# macOS/Linux:
python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

**Start the frontend:**
```bash
cd frontend
npm run dev
```

**Access the application:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

---

## Usage

### Default Credentials

For development and testing:
- **Username:** `demo@example.com`
- **Password:** `demo123`

### Main Features

1. **Dashboard** - View key metrics, lead pipeline, and performance analytics
2. **Lead Management** - Create, score, and manage leads across insurance products
3. **Marketing Campaigns** - Create and manage automated marketing campaigns
4. **Communications** - Send emails, SMS, and manage calls
5. **Document Management** - Upload, sign, and manage documents
6. **Reports** - Generate comprehensive reports and analytics
7. **Scheduler** - Schedule meetings and appointments

---

## Documentation

For detailed documentation, see the [docs](./docs/) folder:

- [Setup Guide](./docs/SETUP_GUIDE.md) - Comprehensive setup instructions
- [Webpack Build Fix](./docs/WEBPACK_BUILD_ERRORS_FIX.md) - Troubleshooting build issues

---

## License

This project is proprietary software. All rights reserved.

---

**Made with Purpose by the LEGEAI Team**

**Revolutionize Your Leads with Intelligent Automation**

