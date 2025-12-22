# Frontend and Backend part of the project

## Environment configuration

### Frontend (`frontend/.env`)
```env
REACT_APP_BACKEND_URL=http://localhost:8000
WDS_SOCKET_PORT=8000
ENABLE_HEALTH_CHECK=false
```

### Backend (`backend/.env`)
```env
MONGO_URL="mongodb://localhost:27017"
DB_NAME="test_database"
CORS_ORIGINS="*"
```