# LEAGAI - Insurance Lead Generation AI Platform

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15.5.5-black)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-19-61dafb)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)](https://www.typescriptlang.org/)


##  Overview

**LEAGAI** is an enterprise-grade Lead Generation AI Platform designed to revolutionize insurance sales through intelligent automation, AI-powered lead scoring, and real-time analytics. The platform combines advanced machine learning models with a modern, intuitive user interface to maximize sales efficiency and customer engagement.

### Key Capabilities

- 🤖 **AI-Powered Lead Scoring** - Intelligent lead qualification using deep learning models
- 📊 **Real-Time Analytics** - Comprehensive dashboards with actionable insights
- 🔄 **Automated Workflows** - Intelligent lead nurturing and follow-up automation
- 💬 **Multi-Channel Communications** - Email, SMS, and call management
- 📋 **Document Management** - E-signature integration with DocuSeal
- 🎯 **Sales Pipeline Management** - Track leads through 5-stage conversion funnel
- 🔐 **Enterprise Security** - JWT authentication, role-based access control
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

##  Quick Start

### Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- npm or yarn
- PostgreSQL (optional - uses in-memory fallback)
- Redis (optional - uses in-memory fallback)

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/leagai.git
   cd leagai
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. **Start the backend**
   ```powershell
   .\run_backend.ps1
   ```
   Backend runs on: `http://localhost:8000`

5. **Start the frontend** (in a new terminal)
   ```bash
   cd frontend
   npm run dev
   ```
   Frontend runs on: `http://localhost:3000`

### Test Accounts

Pre-configured test accounts are available:

| Username | Email | Password | Role |
|----------|-------|----------|------|
| admin | admin@leagai.dev | AdminPass123! | Admin |
| manager | manager@leagai.dev | ManagerPass123! | Manager |
| agent1 | agent1@leagai.dev | AgentPass123! | Agent |
| agent2 | agent2@leagai.dev | AgentPass456! | Agent |
| viewer | viewer@leagai.dev | ViewerPass123! | Viewer |

---

## 📚 API Documentation

### Interactive API Docs

- **Swagger UI**: http://localhost:8000/v1/docs
- **ReDoc**: http://localhost:8000/v1/redoc
- **OpenAPI Schema**: http://localhost:8000/v1/openapi.json

### Key Endpoints

**Authentication**
- `POST /v1/auth/register` - Register new user
- `POST /v1/auth/login` - User login
- `POST /v1/auth/logout` - User logout

**Leads**
- `GET /v1/leads` - List all leads
- `POST /v1/leads` - Create new lead
- `GET /v1/leads/{id}` - Get lead details
- `PUT /v1/leads/{id}` - Update lead

**Dashboard**
- `GET /v1/dashboard/metrics` - Get dashboard metrics
- `GET /v1/dashboard/pipeline` - Get sales pipeline data

**Analytics**
- `GET /v1/analytics/reports` - Get analytics reports
- `GET /v1/analytics/performance` - Get performance metrics

---

## Dashboard Features

### Main Dashboard
- Key performance indicators (KPIs)
- Sales pipeline visualization
- Lead scoring metrics
- Insurance product performance
- Conversion funnel analysis

### Sidebar Navigation (8 Sections)

1. **Leads** - New, Qualified, Contacted leads
2. **Customers** - Active, Inactive customers
3. **Policies** - Auto, Home, Life, Health insurance
4. **Claims** - Pending, Approved, Rejected claims
5. **Documents** - Pending signatures, Signed docs, Templates
6. **Communications** - Emails, SMS, Calls, Campaigns
7. **Reports** - Sales, Pipeline, Performance analytics
8. **Settings** - Profile, Team, Integrations, Notifications

### Total Pages: 27+ Sub-Pages

---

## Security Features

- **JWT Authentication** - Secure token-based authentication
- **Password Hashing** - bcrypt with salt
- **Role-Based Access Control** - 5 user roles (Admin, Manager, Agent, Viewer, User)
- **CORS Protection** - Cross-origin request handling
- **Input Validation** - Pydantic models for all inputs
- **SQL Injection Prevention** - Parameterized queries
- **Rate Limiting** - API rate limiting (configurable)

---

## Database Models

- **User** - User accounts and profiles
- **Lead** - Lead information and scoring
- **Customer** - Customer details
- **Policy** - Insurance policies
- **Claim** - Insurance claims
- **Document** - Document management
- **Communication** - Email, SMS, call logs
- **Session** - User sessions
- **Score** - Lead scoring data
- **Analytics** - Analytics data

---

## Testing & Quality

### Run Frontend Tests
```bash
cd frontend
npm test
```

### Run Backend Tests
```bash
pytest backend/
```

### API Testing
Use the interactive Swagger UI at `http://localhost:8000/v1/docs`

---

## 📈 Performance & Monitoring

- **Prometheus Metrics** - Available at `/v1/metrics`
- **Real-Time Monitoring** - Dashboard metrics updated in real-time
- **Performance Optimization** - Async/await for non-blocking operations
- **Caching** - Redis caching for frequently accessed data
- **Load Balancing** - Horizontal scaling support

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 🆘 Support & Troubleshooting

### Backend Issues

**Port 8000 already in use:**
```bash
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Database connection errors:**
- Set `USE_DB=false` to use in-memory fallback
- Check PostgreSQL is running
- Verify connection string in `.env`

### Frontend Issues

**Dependencies not installing:**
```bash
rm -rf node_modules package-lock.json
npm install
```

**Port 3000 already in use:**
```bash
npm run dev -- -p 3001
```

---

## 📞 Contact & Support

- **Documentation**: See `/docs` folder
- **API Docs**: http://localhost:8000/v1/docs
- **Issues**: GitHub Issues
- **Email**: support@leagai.dev

---

## 🎉 Status

✅ **Production Ready** - All features implemented and tested

**Last Updated**: October 2025  
**Version**: 1.0.0  
**Maintainer**: LEAGAI Team

---

**Made with ❤️ by the LEAGAI Team**

