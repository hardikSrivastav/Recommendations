import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from unittest.mock import Mock, patch
import torch
from application.models.continuous_learning import (
    TrainingConfig,
    UserFeedback,
    FeedbackCollector,
    ContinuousLearningManager
)

@pytest.fixture
def sample_feedback():
    """Create sample user feedback."""
    now = datetime.now()
    return [
        UserFeedback(
            user_id='user_1',
            song_id='song_1',
            interaction_type='play',
            timestamp=now - timedelta(minutes=5),
            context={'session_id': '123'}
        ),
        UserFeedback(
            user_id='user_2',
            song_id='song_2',
            interaction_type='like',
            timestamp=now - timedelta(minutes=10),
            context={'session_id': '124'}
        ),
        UserFeedback(
            user_id='user_1',
            song_id='song_3',
            interaction_type='skip',
            timestamp=now - timedelta(minutes=15),
            context={'session_id': '123'}
        )
    ]

class TestFeedbackCollector:
    @pytest.fixture
    def collector(self):
        return FeedbackCollector(max_size=1000)
        
    def test_add_feedback(self, collector, sample_feedback):
        """Test adding feedback to the collector."""
        for feedback in sample_feedback:
            collector.add_feedback(feedback)
            
        assert len(collector.feedback_queue) == len(sample_feedback)
        
    @pytest.mark.asyncio
    async def test_collect_recent_feedback(self, collector, sample_feedback):
        """Test collecting recent feedback."""
        # Add feedback
        for feedback in sample_feedback:
            collector.add_feedback(feedback)
            
        # Collect recent feedback
        recent = await collector.collect_recent_feedback()
        
        assert len(recent) == len(sample_feedback)
        assert all(isinstance(f, UserFeedback) for f in recent)
        
    def test_convert_to_training_data(self, collector, sample_feedback):
        """Test converting feedback to training data."""
        df = collector._convert_to_training_data(sample_feedback)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_feedback)
        assert all(col in df.columns for col in ['user_id', 'song_id', 'label'])
        assert all(df['label'].between(0, 1))

class TestContinuousLearningManager:
    @pytest.fixture
    def config(self):
        return TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            min_feedback_samples=1,  # Lower threshold for testing
            training_interval=timedelta(seconds=1),
            max_queue_size=1000
        )
        
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        # Create a mock parameter that is actually a tensor
        param = torch.nn.Parameter(torch.randn(1))
        model.parameters = Mock(return_value=[param])
        
        # Set up the model's forward pass
        output = torch.tensor([0.5, 0.5, 0.5])
        model.forward = Mock(return_value=output)
        model.train = Mock(return_value=None)
        
        # Set up loss calculation
        criterion = Mock()
        criterion.return_value = torch.tensor(0.5, requires_grad=True)
        model.criterion = criterion
        
        return model
        
    @pytest.fixture
    def manager(self, mock_model, config):
        manager = ContinuousLearningManager(mock_model, config)
        # Ensure the manager starts in training mode
        manager.is_training = True
        return manager
        
    def test_initialization(self, manager, mock_model, config):
        """Test manager initialization."""
        assert manager.model == mock_model
        assert manager.config == config
        assert isinstance(manager.feedback_collector, FeedbackCollector)
        
    @pytest.mark.asyncio
    async def test_add_and_process_feedback(self, manager, sample_feedback):
        """Test adding and processing feedback."""
        # Add feedback
        for feedback in sample_feedback:
            manager.add_feedback(feedback)
            
        # Get feedback from collector
        recent = await manager.feedback_collector.collect_recent_feedback()
        
        assert len(recent) == len(sample_feedback)
        assert all(isinstance(f, UserFeedback) for f in recent)
        
    @pytest.mark.asyncio
    async def test_training_loop_with_feedback(self, manager, sample_feedback):
        """Test the training loop processes feedback correctly."""
        # Add feedback
        for feedback in sample_feedback:
            manager.add_feedback(feedback)
            
        # Convert feedback to training data and add to queue
        training_data = pd.DataFrame({
            'user_id': ['user_1', 'user_2', 'user_1'],
            'song_id': ['song_1', 'song_2', 'song_3'],
            'user_encoded': [1, 2, 1],
            'song_encoded': [1, 2, 3],
            'label': [1.0, 1.0, 0.0],
            'timestamp': [f.timestamp for f in sample_feedback],
            'session_id': ['123', '124', '123']
        })
        await manager.training_queue.put(training_data)
        
        # Set up expected model behavior
        manager.model.forward.return_value = torch.tensor([0.5, 0.5, 0.5])
        manager.model.criterion.return_value = torch.tensor(0.5, requires_grad=True)
        
        # Start training loop but don't wait indefinitely
        try:
            await asyncio.wait_for(
                manager.training_loop(),
                timeout=2.0
            )
        except asyncio.TimeoutError:
            pass  # Expected timeout
            
        # Check that training was attempted
        assert manager.model.forward.call_count > 0
        assert manager.model.criterion.call_count > 0
        
    @pytest.mark.asyncio
    async def test_train_increment(self, manager):
        """Test incremental training."""
        # Create sample training data with encoded columns
        training_data = pd.DataFrame({
            'user_id': ['user_1', 'user_2', 'user_3'],
            'song_id': ['song_1', 'song_2', 'song_3'],
            'user_encoded': [1, 2, 3],
            'song_encoded': [1, 2, 3],
            'label': [1.0, 0.0, 1.0]
        })
        
        # Set up expected model behavior
        manager.model.forward.return_value = torch.tensor([0.5, 0.5, 0.5])
        manager.model.criterion.return_value = torch.tensor(0.5, requires_grad=True)
        
        # Perform training
        await manager._train_increment(training_data)
        
        # Check that the model was called with correct inputs
        assert manager.model.forward.call_count > 0
        assert manager.model.criterion.call_count > 0 