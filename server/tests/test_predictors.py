import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from application.models.predictors import (
    DemographicPredictor,
    PopularityPredictor,
    DemographicInput,
    PopularityInput
)

@pytest.fixture
def sample_demographics():
    """Create sample demographics data."""
    return pd.DataFrame({
        'user_id': ['user_1', 'user_2', 'user_3', 'user_4'],
        'age_group_encoded': [0, 1, 0, 2],  # 0: 18-25, 1: 26-35, 2: 36-50
        'gender_encoded': [0, 0, 1, 1],     # 0: M, 1: F
        'location_encoded': [0, 0, 1, 2],   # 0: US, 1: UK, 2: CA
        'occupation_encoded': [0, 1, 0, 2]  # 0: Student, 1: Professional, 2: Other
    })

@pytest.fixture
def sample_listening_history():
    """Create sample listening history."""
    now = datetime.now()
    return pd.DataFrame({
        'user_id': ['user_1', 'user_1', 'user_2', 'user_3'],
        'song_id': ['song_1', 'song_2', 'song_1', 'song_3'],
        'timestamp': [
            now - timedelta(days=1),
            now - timedelta(days=2),
            now - timedelta(days=3),
            now - timedelta(days=4)
        ]
    })

class TestDemographicPredictor:
    @pytest.fixture
    def predictor(self):
        return DemographicPredictor(similarity_threshold=0.6)
        
    def test_demographic_similarity(self, predictor):
        """Test demographic similarity calculation."""
        user1 = {
            'user_id': 'user_1',
            'age_group_encoded': 0,
            'gender_encoded': 0,
            'location_encoded': 0,
            'occupation_encoded': 0
        }
        
        # Test identical demographics
        similarity = predictor._calculate_demographic_similarity(user1, user1)
        assert similarity == pytest.approx(1.0)
        
        # Test partially similar demographics
        user2 = {
            'user_id': 'user_2',
            'age_group_encoded': 1,  # Different age group
            'gender_encoded': 0,     # Same gender
            'location_encoded': 0,   # Same location
            'occupation_encoded': 1   # Different occupation
        }
        similarity = predictor._calculate_demographic_similarity(user1, user2)
        assert 0.0 < similarity < 1.0
        
        # Test completely different demographics
        user3 = {
            'user_id': 'user_3',
            'age_group_encoded': 2,
            'gender_encoded': 1,
            'location_encoded': 1,
            'occupation_encoded': 2
        }
        similarity = predictor._calculate_demographic_similarity(user1, user3)
        assert similarity < predictor.similarity_threshold
        
    def test_find_similar_users(self, predictor, sample_demographics):
        """Test finding similar users."""
        user_demographics = {
            'user_id': 'user_1',
            'age_group_encoded': 0,
            'gender_encoded': 0,
            'location_encoded': 0,
            'occupation_encoded': 0
        }
        
        similar_users = predictor._find_similar_users(
            user_demographics,
            sample_demographics
        )
        
        assert not similar_users.empty
        assert 'user_id' in similar_users.columns
        assert 'similarity' in similar_users.columns
        assert all(similar_users['similarity'] >= predictor.similarity_threshold)
        
    @pytest.mark.asyncio
    async def test_predict(self, predictor, sample_demographics, sample_listening_history):
        """Test the full prediction pipeline."""
        input_data = DemographicInput(
            user_id='user_1',
            demographics={
                'user_id': 'user_1',
                'age_group_encoded': 0,
                'gender_encoded': 0,
                'location_encoded': 0,
                'occupation_encoded': 0
            },
            timestamp=datetime.now()
        )
        
        predictions = await predictor.predict(
            input_data,
            sample_listening_history,
            sample_demographics
        )
        
        assert isinstance(predictions, dict)
        assert all(isinstance(score, float) for score in predictions.values())
        assert all(0 <= score <= 1 for score in predictions.values())

class TestPopularityPredictor:
    @pytest.fixture
    def predictor(self):
        return PopularityPredictor(max_age_days=7)
        
    def test_time_weight(self, predictor):
        """Test time-based weight calculation."""
        # Recent interaction should have high weight
        recent_weight = predictor._calculate_time_weight(1)
        assert recent_weight > 0.9
        
        # Old interaction should have low weight
        old_weight = predictor._calculate_time_weight(7)
        assert old_weight < recent_weight
        
        # Older interactions should have decreasing weights
        weights = [
            predictor._calculate_time_weight(days)
            for days in range(1, 8)
        ]
        assert all(w1 > w2 for w1, w2 in zip(weights, weights[1:]))
        
    def test_popularity_scores(self, predictor, sample_listening_history):
        """Test popularity score calculation."""
        scores = predictor._calculate_popularity_scores(
            sample_listening_history,
            datetime.now()
        )
        
        assert isinstance(scores, dict)
        assert all(isinstance(score, float) for score in scores.values())
        assert all(0 <= score <= 1 for score in scores.values())
        
        # Most recent interaction should have highest score
        recent_song = sample_listening_history.iloc[0]['song_id']
        assert scores[recent_song] == pytest.approx(1.0)
        
    def test_demographic_filter(self, predictor, sample_demographics, sample_listening_history):
        """Test demographic filtering of popularity scores."""
        base_scores = {'song_1': 1.0, 'song_2': 0.8, 'song_3': 0.6}
        user_context = {
            'demographics': {
                'user_id': 'user_1',
                'age_group_encoded': 0,
                'gender_encoded': 0,
                'location_encoded': 0,
                'occupation_encoded': 0
            }
        }
        
        filtered_scores = predictor._apply_demographic_filter(
            base_scores,
            user_context,
            sample_listening_history,
            sample_demographics
        )
        
        assert isinstance(filtered_scores, dict)
        assert all(isinstance(score, float) for score in filtered_scores.values())
        assert all(0 <= score <= 1.2 for score in filtered_scores.values())
        
    @pytest.mark.asyncio
    async def test_predict(self, predictor, sample_listening_history, sample_demographics):
        """Test the full prediction pipeline."""
        input_data = PopularityInput(
            time_window=timedelta(days=7),
            user_context={
                'demographics': {
                    'user_id': 'user_1',
                    'age_group_encoded': 0,
                    'gender_encoded': 0,
                    'location_encoded': 0,
                    'occupation_encoded': 0
                }
            }
        )
        
        predictions = await predictor.predict(
            input_data,
            sample_listening_history,
            sample_demographics
        )
        
        assert isinstance(predictions, dict)
        assert all(isinstance(score, float) for score in predictions.values())
        assert all(0 <= score <= 1.2 for score in predictions.values()) 