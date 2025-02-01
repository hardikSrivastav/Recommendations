# Music Recommender System Implementation Plan

## Overview
This document outlines the step-by-step implementation plan for the music recommendation system, building upon existing frontend components and model architecture.

## Implementation Timeline

### Phase 1: Infrastructure Setup (Week 1)

#### Initial Project Structure
```bash
music-recommender/
├── app/
│   ├── models/         # ML models and predictors
│   ├── routes/         # API endpoints
│   ├── services/       # Business logic
│   └── utils/          # Helper functions
├── config/             # Configuration files
├── tests/
│   ├── unit/          # Unit tests
│   └── integration/   # Integration tests
└── data/              # Data storage
```

#### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

#### Required Dependencies
```text
# requirements.txt
torch==2.0.1
flask==2.3.3
sqlalchemy==2.0.20
pymongo==4.5.0
redis==4.6.0
pandas==2.1.0
numpy==1.24.3
pytest==7.4.2
```

### Phase 2: Core Model Integration (Week 1-2)

#### Model Structure
```python
# app/models/base.py
class BasePredictor:
    async def predict(self, user_id: str) -> Dict[str, float]:
        raise NotImplementedError

# app/models/recommender.py
class RecommenderSystem(BasePredictor):
    def __init__(self, config: Dict[str, Any]):
        self.model = self.load_model(config)
        self.encoders = self.load_encoders()

# app/models/demographic_predictor.py
class DemographicPredictor(BasePredictor):
    async def predict(self, user_id: str) -> Dict[str, float]:
        demographics = await self.get_user_demographics(user_id)
        return self.generate_demographic_predictions(demographics)
```

### Phase 3: Data Layer Setup (Week 2)

#### PostgreSQL Setup
```python
# app/database/postgres.py
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class UserDemographics(Base):
    __tablename__ = 'user_demographics'
    
    user_id = Column(String, primary_key=True)
    age = Column(Integer)
    gender = Column(String)
    location = Column(String)
    occupation = Column(String)

def init_postgres():
    engine = create_engine(config.POSTGRES_URL)
    Base.metadata.create_all(engine)
    return engine
```

#### MongoDB Setup
```python
# app/database/mongo.py
from pymongo import MongoClient, ASCENDING

def init_mongo():
    client = MongoClient(config.MONGODB_URL)
    db = client[config.MONGODB_DB]
    
    # Create indexes
    db.listening_history.create_index([("user_id", ASCENDING)])
    db.predictions.create_index([
        ("user_id", ASCENDING),
        ("timestamp", ASCENDING)
    ])
    
    return db
```

#### Redis Setup
```python
# app/database/redis.py
import redis

def init_redis():
    return redis.Redis.from_url(
        config.REDIS_URL,
        decode_responses=True
    )
```

### Phase 4: Service Layer Implementation (Week 3)

#### Demographics Service
```python
# app/services/demographics.py
class DemographicsService:
    def __init__(self, db_session):
        self.db = db_session
        
    async def validate_demographics(self, data: Dict) -> bool:
        validators = {
            'age': lambda x: 13 <= x <= 100,
            'gender': lambda x: x in ['M', 'F', 'NB', 'O'],
            'location': lambda x: len(x) == 2,  # Country code
            'occupation': lambda x: x in VALID_OCCUPATIONS
        }
        return all(
            validators[field](value) 
            for field, value in data.items()
        )
        
    async def store_demographics(self, user_id: str, data: Dict):
        if not await self.validate_demographics(data):
            raise ValidationError("Invalid demographics data")
            
        user_demo = UserDemographics(
            user_id=user_id,
            **data
        )
        self.db.add(user_demo)
        await self.db.commit()
```

#### History Service
```python
# app/services/history.py
class HistoryService:
    def __init__(self, mongodb, redis):
        self.db = mongodb
        self.cache = redis
        self.HISTORY_LIMIT = 50
        self.WARNING_LIMIT = 45
        
    async def add_song(self, user_id: str, song_id: str):
        current_count = await self.get_history_count(user_id)
        
        if current_count >= self.HISTORY_LIMIT:
            raise LimitError("History limit reached")
            
        await self.db.listening_history.insert_one({
            "user_id": user_id,
            "song_id": song_id,
            "timestamp": datetime.utcnow()
        })
        
        if current_count + 1 >= self.WARNING_LIMIT:
            return {
                "warning": f"{self.HISTORY_LIMIT - current_count - 1} slots remaining"
            }
```

### Phase 5: Backend Integration (Week 3-4)

#### API Routes
```python
# app/routes/api.py
@app.route('/api/demographics', methods=['POST'])
async def handle_demographics():
    user_id = get_user_id()
    data = request.json
    
    try:
        await demographics_service.store_demographics(user_id, data)
        await model_service.update_user_context(user_id)
        return jsonify({"status": "success"})
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/history/add', methods=['POST'])
async def add_to_history():
    user_id = get_user_id()
    song_id = request.json['song_id']
    
    try:
        result = await history_service.add_song(user_id, song_id)
        return jsonify(result)
    except LimitError as e:
        return jsonify({"error": str(e)}), 400
```

### Phase 6: Frontend-Backend Connection (Week 4)

#### API Service Layer
```typescript
// frontend/services/api.ts
class APIService {
    async submitDemographics(data: Demographics): Promise<void> {
        try {
            await axios.post('/api/demographics', data);
        } catch (error) {
            handleAPIError(error);
        }
    }
    
    async searchSongs(query: string): Promise<SearchResult> {
        try {
            const response = await axios.get(
                '/api/search',
                { params: { q: query } }
            );
            return response.data;
        } catch (error) {
            handleAPIError(error);
            return { results: [], recommendation: null };
        }
    }
}
```

### Phase 7: Recommendation Engine Integration (Week 5)

#### Ensemble System
```python
# app/models/ensemble.py
class EnsembleRecommender:
    def __init__(self, predictors: Dict[str, BasePredictor]):
        self.predictors = predictors
        self.weight_calculator = WeightCalculator()
        
    async def get_recommendations(
        self,
        user_id: str,
        n: int = 5
    ) -> List[Dict]:
        predictions = {}
        for name, predictor in self.predictors.items():
            try:
                preds = await predictor.predict(user_id)
                predictions[name] = preds
            except Exception as e:
                logger.error(f"Predictor {name} failed: {str(e)}")
                
        weights = await self.weight_calculator.compute_weights(
            predictions,
            await self.get_user_context(user_id)
        )
        
        return self.blend_predictions(predictions, weights, n)
```

### Phase 8: Testing and Refinement (Week 6)

#### Unit Tests
```python
# tests/unit/test_demographics.py
def test_demographics_validation():
    service = DemographicsService(db_session)
    
    valid_data = {
        "age": 25,
        "gender": "M",
        "location": "US",
        "occupation": "Engineer"
    }
    assert service.validate_demographics(valid_data)
    
    invalid_data = {
        "age": 10,  # Too young
        "gender": "Invalid",
        "location": "USA",  # Should be 2 letters
        "occupation": "Invalid"
    }
    assert not service.validate_demographics(invalid_data)
```

#### Integration Tests
```python
# tests/integration/test_recommendations.py
async def test_recommendation_flow():
    # Setup
    user_id = "test_user"
    await demographics_service.store_demographics(
        user_id,
        valid_demographics
    )
    
    # Add some songs
    for song_id in test_songs[:5]:
        await history_service.add_song(user_id, song_id)
    
    # Get recommendations
    recs = await recommendation_service.get_recommendations(user_id)
    
    assert len(recs) == 5
    assert all('confidence' in rec for rec in recs)
```

## Key Considerations

### 1. Database Migration Strategy
- Create clear migration scripts
- Plan for data backups
- Consider rollback procedures

### 2. Testing Strategy
- Unit tests for each component
- Integration tests for flows
- Mock external services

### 3. Monitoring Setup
- Basic logging configuration
- Performance metrics tracking
- Error monitoring

### 4. Deployment Considerations
- Environment configuration
- Service dependencies
- Scaling strategy

## Next Steps

1. Begin with database setup and basic infrastructure
2. Implement core model integration
3. Build service layer components
4. Connect frontend and backend
5. Add comprehensive testing
6. Deploy monitoring solutions

