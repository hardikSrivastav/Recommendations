import unittest
import torch
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import asyncio
import pytest
from application.models.model import (
    ConfidenceCalculator,
    WeightedEnsembleRecommender,
    MusicRecommender,
    RecommenderSystem
)

class TestConfidenceCalculator(unittest.TestCase):
    def setUp(self):
        self.calculator = ConfidenceCalculator()
        
    def test_history_factor(self):
        """Test history factor calculation with different history lengths."""
        # Test with no history
        self.assertAlmostEqual(
            self.calculator._calculate_history_factor(0),
            0.0
        )
        
        # Test with medium history
        medium_factor = self.calculator._calculate_history_factor(25)
        self.assertTrue(0 < medium_factor < 1)
        
        # Test with large history
        large_factor = self.calculator._calculate_history_factor(100)
        self.assertAlmostEqual(large_factor, 1.0)
        
    def test_embedding_factor(self):
        """Test embedding factor calculation."""
        # Test with identical embeddings (should give 1.0)
        predictions = {'1': 0.8}
        user_embedding = torch.tensor([1.0, 0.0, 0.0])  # Unit vector
        song_embeddings = {
            '1': torch.tensor([1.0, 0.0, 0.0])  # Same unit vector
        }
        context = {
            'user_embedding': user_embedding,
            'song_embeddings': song_embeddings
        }
        
        factor = self.calculator._calculate_embedding_factor(predictions, context)
        self.assertAlmostEqual(factor, 1.0, places=6)
        
        # Test with orthogonal embeddings (should give 0.5)
        song_embeddings['1'] = torch.tensor([0.0, 1.0, 0.0])  # Perpendicular unit vector
        context['song_embeddings'] = song_embeddings
        
        factor = self.calculator._calculate_embedding_factor(predictions, context)
        self.assertAlmostEqual(factor, 0.5, places=6)
        
        # Test with opposite embeddings (should give 0.0)
        song_embeddings['1'] = torch.tensor([-1.0, 0.0, 0.0])  # Opposite unit vector
        context['song_embeddings'] = song_embeddings
        
        factor = self.calculator._calculate_embedding_factor(predictions, context)
        self.assertAlmostEqual(factor, 0.0, places=6)
        
        # Test with random embeddings (should be between 0 and 1)
        predictions = {'1': 0.8, '2': 0.6}
        user_embedding = torch.randn(50)  # 50-dim embedding
        song_embeddings = {
            '1': torch.randn(50),
            '2': torch.randn(50)
        }
        context = {
            'user_embedding': user_embedding,
            'song_embeddings': song_embeddings
        }
        
        factor = self.calculator._calculate_embedding_factor(predictions, context)
        self.assertTrue(0 <= factor <= 1)
        
        # Test with missing embeddings
        empty_context = {}
        default_factor = self.calculator._calculate_embedding_factor(predictions, empty_context)
        self.assertEqual(default_factor, 0.5)
        
    def test_diversity_factor(self):
        """Test diversity factor calculation."""
        # Test with identical predictions (low diversity)
        uniform_preds = {'1': 0.8, '2': 0.8, '3': 0.8}
        uniform_factor = self.calculator._calculate_diversity_factor(uniform_preds)
        self.assertAlmostEqual(uniform_factor, 1.0)  # Low diversity -> high confidence
        
        # Test with diverse predictions
        diverse_preds = {'1': 0.2, '2': 0.5, '3': 0.8}
        diverse_factor = self.calculator._calculate_diversity_factor(diverse_preds)
        self.assertTrue(diverse_factor < uniform_factor)  # Higher diversity -> lower confidence

class TestWeightedEnsembleRecommender(unittest.TestCase):
    def setUp(self):
        self.ensemble = WeightedEnsembleRecommender()
        
    def test_blend_predictions(self):
        """Test prediction blending with multiple sources."""
        predictions = {
            'model': {'1': 0.8, '2': 0.6},
            'demographic': {'1': 0.7, '2': 0.5}
        }
        confidence_scores = {
            'model': 0.8,
            'demographic': 0.6
        }
        
        blended = self.ensemble._blend_predictions(predictions, confidence_scores)
        
        # Check structure
        self.assertTrue(all(
            key in blended['1'] 
            for key in ['score', 'confidence', 'source_weights']
        ))
        
        # Check weights sum to 1
        self.assertAlmostEqual(
            sum(blended['1']['source_weights'].values()),
            1.0
        )
        
    @pytest.mark.asyncio
    @patch('application.models.model.ConfidenceCalculator')
    async def test_get_recommendations(self, mock_calculator):
        """Test the full recommendation pipeline."""
        # Mock confidence calculator
        mock_calculator.return_value.calculate_confidence.return_value = 0.8
        
        # Mock predictor
        mock_predictor = Mock()
        mock_predictor.predict.return_value = {'1': 0.8, '2': 0.6}
        
        self.ensemble.prediction_sources['model'] = mock_predictor
        
        # Test recommendations
        user_context = {'history_length': 10}
        recommendations = await self.ensemble.get_recommendations('user_1', user_context, n=2)
        
        self.assertEqual(len(recommendations), 2)
        self.assertTrue(all(
            key in recommendations[0]
            for key in ['song_id', 'score', 'confidence', 'source_weights']
        ))

class TestMusicRecommender(unittest.TestCase):
    def setUp(self):
        self.model = MusicRecommender(
            num_users=100,
            num_songs=1000,
            embedding_dim=50,
            metadata_dim=10,
            demographic_dim=4
        )
        
    def test_forward_pass(self):
        """Test the forward pass of the model."""
        batch_size = 32
        
        # Create dummy inputs
        user_input = torch.randint(0, 100, (batch_size,))
        song_input = torch.randint(0, 1000, (batch_size,))
        metadata_input = torch.rand((batch_size, 10))
        demographic_input = torch.rand((batch_size, 4))
        
        # Test forward pass
        score, confidence = self.model(
            user_input,
            song_input,
            metadata_input,
            demographic_input
        )
        
        # Check shapes
        self.assertEqual(score.shape, (batch_size,))
        self.assertEqual(confidence.shape, (batch_size,))
        
        # Check value ranges
        self.assertTrue(torch.all(score >= 0) and torch.all(score <= 1))
        self.assertTrue(torch.all(confidence >= 0) and torch.all(confidence <= 1))
        
    def test_embeddings(self):
        """Test embedding layer outputs."""
        user_input = torch.tensor([1, 2, 3])
        song_input = torch.tensor([1, 2, 3])
        
        user_emb = self.model.get_user_embedding(user_input)
        song_emb = self.model.get_song_embedding(song_input)
        
        self.assertEqual(user_emb.shape, (3, 50))
        self.assertEqual(song_emb.shape, (3, 50))

class TestRecommenderSystem(unittest.TestCase):
    def setUp(self):
        self.system = RecommenderSystem()
        
        # Create dummy data
        self.demographics_df = pd.DataFrame({
            'user_id': [f'user_{i}' for i in range(10)],
            'age': np.random.randint(18, 70, 10),
            'gender': np.random.choice(['M', 'F'], 10),
            'location': np.random.choice(['US', 'UK', 'CA'], 10),
            'occupation': np.random.choice(['Student', 'Professional'], 10)
        })
        
        self.listening_history = pd.DataFrame({
            'user_id': [f'user_{i}' for i in range(10)],
            'song_id': [f'song_{i}' for i in range(10)],
            'timestamp': pd.date_range(start='2024-01-01', periods=10)
        })
        
        self.song_metadata = pd.DataFrame({
            'song_id': [f'song_{i}' for i in range(10)],
            'tags': ['rock,pop'] * 10,
            'genres': ['rock'] * 10,
            'track_title': [f'Track {i}' for i in range(10)],
            'artist_name': [f'Artist {i}' for i in range(10)]
        })
        
    def test_encode_demographics(self):
        """Test demographic encoding."""
        encoded_df = self.system.encode_demographics(self.demographics_df)
        
        self.assertTrue('age_group_encoded' in encoded_df.columns)
        self.assertTrue('gender_encoded' in encoded_df.columns)
        self.assertTrue(encoded_df['age_group_encoded'].dtype in [np.int32, np.int64])
        
    def test_get_user_context(self):
        """Test user context generation."""
        # First encode demographics
        encoded_demographics = self.system.encode_demographics(self.demographics_df)
        
        # Get context
        context = self.system.get_user_context('user_0', encoded_demographics)
        
        self.assertEqual(context['user_id'], 'user_0')
        self.assertIn('demographics', context)
        self.assertIn('song_embeddings', context)
        
    @pytest.mark.asyncio
    @patch('application.models.model.WeightedEnsembleRecommender')
    async def test_predict_next_songs(self, mock_ensemble):
        """Test the prediction pipeline."""
        # Train the model first
        self.system.fit(
            self.listening_history,
            self.song_metadata,
            self.demographics_df,
            epochs=1
        )
        
        # Mock ensemble recommendations
        mock_ensemble.return_value.get_recommendations.return_value = [{
            'song_id': '0',
            'score': 0.8,
            'confidence': 0.7,
            'source_weights': {'model': 1.0}
        }]
        
        # Test predictions
        recommendations = await self.system.predict_next_songs(
            'user_0',
            self.song_metadata,
            self.demographics_df,
            n=1
        )
        
        self.assertEqual(len(recommendations), 1)
        self.assertIn('song', recommendations[0])
        self.assertIn('score', recommendations[0])
        self.assertIn('confidence', recommendations[0])
        
    def test_combined_loss(self):
        """Test the combined loss function."""
        predictions = torch.tensor([0.8, 0.2, 0.6])
        confidence = torch.tensor([0.9, 0.8, 0.7])
        labels = torch.tensor([1.0, 0.0, 1.0])
        
        loss = self.system._calculate_combined_loss(predictions, confidence, labels)
        
        self.assertTrue(isinstance(loss.item(), float))
        self.assertTrue(loss.item() > 0)

def run_async_test(coro):
    """Helper function to run async tests."""
    return asyncio.get_event_loop().run_until_complete(coro)

if __name__ == '__main__':
    unittest.main() 