# 🚀 LEGEAI - AI-Powered Lead Generation Platform

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15.5.5-black)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-19-61dafb)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)](https://www.typescriptlang.org/)

> **Revolutionize Your Leads with Intelligent Automation**

---

## 📖 Overview

**LEGEAI** is an enterprise-grade AI-powered lead generation and management platform for the insurance industry. Built with cutting-edge technology, it combines advanced machine learning, real-time analytics, and intelligent automation to maximize sales efficiency and customer engagement.

### ✨ Key Features

- 🎯 **AI-Powered Lead Scoring** - Deep learning models for intelligent lead qualification
- 📊 **Real-Time Analytics** - Comprehensive dashboards with actionable insights
- 🤖 **Marketing Automation** - Automated campaigns, segments, and triggers
- 📞 **Multi-Channel Communications** - Email, SMS, and call management
- 📄 **Document Management** - E-signature integration with DocuSeal
- 🔗 **Meeting Scheduler** - Zoom, Google Meet, and Teams integration
- 🔒 **Enterprise Security** - JWT authentication and role-based access control
- 📈 **Performance Monitoring** - Real-time metrics and KPI tracking

---

## 🏗️ Architecture

### Technology Stack

**Backend:**
- FastAPI 0.104.1 - High-performance async web framework
- Python 3.9+ - Core language
- Deep Learning - TensorFlow, PyTorch, Neural Networks, NLP, Autoencoders, Transformer-based models.
- SQLAlchemy - ORM for database operations
- Pydantic - Data validation
- JWT - Secure authentication
- Prometheus - Metrics collection

**Frontend:**
- Next.js 15.5.5 - React framework with App Router
- React 19 - UI library
- TypeScript 5.0 - Type-safe development
- Tailwind CSS - Utility-first styling
- Axios - HTTP client

**Infrastructure:**
- PostgreSQL - Primary database
- Redis - Caching and sessions
- Docker - Containerization
- Alembic - Database migrations

### Project Structure

```
LEAGAI/
├── backend/                          # Backend application
│   ├── api/                          # FastAPI endpoints (23+ modules)
│   │   ├── main.py                   # Application entry point
│   │   ├── auth_api.py               # Authentication endpoints
│   │   ├── leads_api.py              # Lead management
│   │   ├── dashboard_api.py          # Dashboard data
│   │   ├── analytics_api.py          # Analytics endpoints
│   │   └── ...
│   ├── models/                       # Data models (15+ models)
│   ├── database/                     # Database layer
│   ├── security/                     # Authentication & authorization
│   ├── ai_sales_automation/          # Sales automation engines
│   ├── advanced_ml/                  # Machine learning models
│   ├── analytics/                    # Analytics engine
│   ├── compliance/                   # Compliance management
│   └── monitoring/                   # Performance monitoring
│
├── frontend/                         # Next.js frontend
│   ├── src/
│   │   ├── app/                      # App Router pages
│   │   │   ├── page.tsx              # Home page
│   │   │   ├── login/page.tsx        # Login page
│   │   │   ├── register/page.tsx     # Registration page
│   │   │   └── dashboard/            # Dashboard pages (27+ sub-pages)
│   │   ├── components/               # Reusable React components
│   │   ├── lib/                      # Utilities and helpers
│   │   └── styles/                   # Global styles
│   ├── public/                       # Static assets
│   └── package.json                  # Dependencies
│
├── docs/                             # Documentation
├── requirements.txt                  # Python dependencies
├── run_backend.ps1                   # Backend startup script
└── README.md                         # This file
```

---

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```powershell
.\setup.ps1
```

### Option 2: Manual Setup

**See the complete installation guide:** 📘 **[SETUP_GUIDE.md](./SETUP_GUIDE.md)**

The setup guide includes:
- System requirements and prerequisites
- Step-by-step installation for all platforms
- Database configuration options
- Troubleshooting common issues
- Production deployment instructions

### Quick Manual Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 2. Start backend (Terminal 1)
.\run_backend.ps1  # Windows
# OR
python -m uvicorn backend.api.main:app --reload  # macOS/Linux

# 3. Start frontend (Terminal 2)
cd frontend
npm run dev
```

**Access the application:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/v1/docs

### 🔑 Test Accounts

| Role | Email | Password |
|------|-------|----------|
| Admin | admin@legeai.dev | AdminPass123! |
| Manager | manager@legeai.dev | ManagerPass123! |
| Agent | agent1@legeai.dev | AgentPass123! |

**Full credentials:** See [CREDENTIALS_REFERENCE.txt](./CREDENTIALS_REFERENCE.txt)

---

## 📚 Documentation

### 📘 Setup & Installation
- **[SETUP_GUIDE.md](./SETUP_GUIDE.md)** - Complete installation and setup guide
  - System requirements
  - Prerequisites installation (Python, Node.js, PostgreSQL, Redis)
  - Backend and frontend setup
  - Database configuration
  - Running the application
  - Troubleshooting
  - Production deployment

### 🔧 Technical Documentation
- **[DEEP_LEARNING_SUMMARY.md](./DEEP_LEARNING_SUMMARY.md)** - AI/ML models documentation
- **[DOCUMENT_MANAGEMENT_SYSTEM.md](./DOCUMENT_MANAGEMENT_SYSTEM.md)** - Document management features
- **[docs/Insurance_Lead_Scoring_Models_Documentation.md](./docs/Insurance_Lead_Scoring_Models_Documentation.md)** - Lead scoring models
- **[docs/Life_Insurance_Policy_Types.md](./docs/Life_Insurance_Policy_Types.md)** - Life insurance policy types

### 🐛 Implementation & Fixes
- **[DATABASE_READINESS_STATUS_UPDATED.md](./DATABASE_READINESS_STATUS_UPDATED.md)** - Database migration status
- **[LIFE_INSURANCE_GET_QUOTE_FIX.md](./LIFE_INSURANCE_GET_QUOTE_FIX.md)** - Quote functionality implementation
- **[REPORT_GENERATION_FIX.md](./REPORT_GENERATION_FIX.md)** - Report generation fixes

### 🌐 API Documentation

**Interactive API Docs:**
- **Swagger UI**: http://localhost:8000/v1/docs
- **ReDoc**: http://localhost:8000/v1/redoc
- **OpenAPI Schema**: http://localhost:8000/v1/openapi.json

**Key Endpoints:**
- Authentication: `/v1/auth/*`
- Leads: `/v1/leads/*`
- Customers: `/v1/customers/*`
- Policies: `/v1/policies/*`
- Claims: `/v1/claims/*`
- Communications: `/v1/communications/*`
- Marketing: `/v1/marketing/*`
- Analytics: `/v1/analytics/*`

---

## 🎨 Features

### 🏠 Landing Page
- Futuristic cyberpunk design with animated particle background
- Glassmorphism effects and gradient text
- Features showcase, pricing tiers, and call-to-action

### 🔐 Authentication
- JWT-based secure authentication
- Role-based access control (Admin, Manager, Agent, Viewer, User)
- Password hashing with bcrypt

### 📊 Dashboard (27+ Pages)
- **Main Dashboard** - KPIs, sales pipeline, lead scoring metrics
- **Leads** - New, Qualified, Contacted leads with AI scoring
- **Customers** - Active/Inactive customer management
- **Policies** - Auto, Home, Life, Health insurance policies
- **Claims** - Pending, Approved, Rejected claims tracking
- **Documents** - E-signature integration with DocuSeal
- **Communications** - Email, SMS, Calls, Campaigns
- **Marketing Automation** - Campaigns, Segments, Templates, Triggers
- **Meeting Scheduler** - Zoom, Google Meet, Teams integration
- **Reports** - Sales, Pipeline, Performance analytics
- **Settings** - Profile, Team, Integrations, Notifications

### 🤖 AI/ML Features
- **Lead Scoring Models** - XGBoost, LightGBM, CatBoost ensemble
- **Deep Learning** - PyTorch neural networks for advanced scoring
- **Real-Time Predictions** - Instant lead qualification
- **Multi-Product Support** - Auto, Home, Life, Health insurance

### 📱 Responsive Design
- Mobile, tablet, and desktop optimized
- Futuristic UI with cyberpunk aesthetic
- Interactive sidebar with keyboard shortcuts (Ctrl+B)

---

## 🔒 Security

- ✅ JWT token-based authentication
- ✅ Password hashing with bcrypt
- ✅ Role-based access control (5 roles)
- ✅ CORS protection
- ✅ Input validation with Pydantic
- ✅ SQL injection prevention
- ✅ Rate limiting (configurable)

---

## 🗄️ Database

### Models (15+ Tables)
- User, Lead, Customer, Policy, Claim
- Document, Communication, Session
- Score, Analytics, Marketing Campaign
- Meeting, Report, Notification

### Migrations
- Alembic for schema management
- Automated migration scripts
- See [DATABASE_READINESS_STATUS_UPDATED.md](./DATABASE_READINESS_STATUS_UPDATED.md)

---

## 🧪 Testing

```bash
# Frontend tests
cd frontend
npm test

# Backend tests
pytest backend/

# API testing
# Use Swagger UI at http://localhost:8000/v1/docs
```

---

## 📈 Performance & Monitoring

- ⚡ Async/await for non-blocking operations
- 🔄 Redis caching for frequently accessed data
- 📊 Prometheus metrics at `/v1/metrics`
- 🎯 Real-time dashboard updates
- 🚀 Horizontal scaling support

---

## 🛠️ Troubleshooting

**For detailed troubleshooting, see:** 📘 **[SETUP_GUIDE.md - Troubleshooting Section](./SETUP_GUIDE.md#troubleshooting)**

### Quick Fixes

**Backend port 8000 in use:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:8000 | xargs kill -9
```

**Frontend port 3000 in use:**
```bash
npm run dev -- -p 3001
```

**Database connection errors:**
```bash
# Use in-memory fallback
set USE_DB=false  # Windows
export USE_DB=false  # macOS/Linux
```

**Module not found errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt
cd frontend && npm install
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Support

- 📘 **Documentation**: See markdown files in root directory
- 🌐 **API Docs**: http://localhost:8000/v1/docs
- 🐛 **Issues**: GitHub Issues
- 📧 **Email**: support@legeai.dev

---

## ✅ Status

**Production Ready** - All features implemented and tested

- **Version**: 1.0.0
- **Last Updated**: November 2024
- **Maintainer**: LEGEAI Team

---

## 📄 License

This project is proprietary software. All rights reserved.

---

<div align="center">

**Made with ❤️ by the LEGEAI Team**

🚀 **Revolutionize Your Leads with Intelligent Automation** 🚀

</div>

