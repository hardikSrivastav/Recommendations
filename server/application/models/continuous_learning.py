from typing import Dict, Any, List
import asyncio
import logging
from datetime import datetime, timedelta
import torch
import pandas as pd
from dataclasses import dataclass
from collections import deque

@dataclass
class TrainingConfig:
    """Configuration for continuous learning."""
    batch_size: int = 32
    learning_rate: float = 0.001
    min_feedback_samples: int = 100
    training_interval: timedelta = timedelta(hours=1)
    max_queue_size: int = 10000

@dataclass
class UserFeedback:
    """Structure for user feedback data."""
    user_id: str
    song_id: str
    interaction_type: str  # 'play', 'skip', 'like', 'dislike'
    timestamp: datetime
    context: Dict[str, Any]

class FeedbackCollector:
    """Collects and processes user feedback."""
    
    def __init__(self, max_size: int = 10000):
        self.feedback_queue = deque(maxlen=max_size)
        self.last_collection_time = None  # Initialize to None
        
    def add_feedback(self, feedback: UserFeedback):
        """Add new feedback to the queue."""
        self.feedback_queue.append(feedback)
        
    async def collect_recent_feedback(self) -> List[UserFeedback]:
        """Collect feedback since last collection."""
        current_time = datetime.now()
        
        # If this is the first collection, return all feedback
        if self.last_collection_time is None:
            recent_feedback = list(self.feedback_queue)
        else:
            recent_feedback = [
                f for f in self.feedback_queue
                if f.timestamp > self.last_collection_time
            ]
            
        self.last_collection_time = current_time
        return recent_feedback
        
    def _convert_to_training_data(
        self,
        feedback_list: List[UserFeedback]
    ) -> pd.DataFrame:
        """Convert feedback to training data format."""
        training_data = []
        
        # Create user and song ID mappings
        unique_users = list(set(f.user_id for f in feedback_list))
        unique_songs = list(set(f.song_id for f in feedback_list))
        user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        song_to_idx = {sid: i for i, sid in enumerate(unique_songs)}
        
        for feedback in feedback_list:
            # Convert interaction type to label
            label = {
                'play': 1.0,
                'like': 1.0,
                'dislike': 0.0,
                'skip': 0.0
            }.get(feedback.interaction_type, 0.5)
            
            # Add encoded IDs
            training_data.append({
                'user_id': feedback.user_id,
                'song_id': feedback.song_id,
                'user_encoded': user_to_idx[feedback.user_id],
                'song_encoded': song_to_idx[feedback.song_id],
                'label': label,
                'timestamp': feedback.timestamp,
                **feedback.context
            })
            
        return pd.DataFrame(training_data)

class ContinuousLearningManager:
    """
    Manages continuous model improvement using real-time user feedback
    and background training.
    """
    def __init__(self, model, config: TrainingConfig):
        self.model = model
        self.config = config
        self.training_queue = asyncio.Queue()
        self.feedback_collector = FeedbackCollector(max_size=config.max_queue_size)
        self.is_training = False
        
    async def start(self):
        """Start the continuous learning process."""
        await asyncio.gather(
            self.collect_feedback_loop(),
            self.training_loop()
        )
        
    async def collect_feedback_loop(self):
        """Continuously collect and process user feedback."""
        while True:
            try:
                feedback = await self.feedback_collector.collect_recent_feedback()
                if len(feedback) >= self.config.min_feedback_samples:
                    # Convert feedback to training data
                    training_data = self.feedback_collector._convert_to_training_data(feedback)
                    await self.training_queue.put(training_data)
                    
            except Exception as e:
                logging.error(f"Error collecting feedback: {str(e)}")
                
            await asyncio.sleep(60)  # Check every minute
            
    async def training_loop(self):
        """Background training loop."""
        while True:
            try:
                if not self.training_queue.empty():
                    self.is_training = True
                    training_data = await self.training_queue.get()
                    
                    # Perform incremental training
                    await self._train_increment(training_data)
                    
                    self.training_queue.task_done()
                    self.is_training = False
                    
            except Exception as e:
                logging.error(f"Error in training loop: {str(e)}")
                self.is_training = False
                
            await asyncio.sleep(
                self.config.training_interval.total_seconds()
            )
            
    async def _train_increment(self, training_data: pd.DataFrame):
        """Perform incremental training on new data."""
        try:
            # Set model to training mode
            self.model.train()
            
            # Prepare data
            users = torch.tensor(training_data['user_encoded'].values)
            songs = torch.tensor(training_data['song_encoded'].values)
            labels = torch.tensor(training_data['label'].values)
            
            # Create optimizer for incremental training
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
            
            # Forward pass
            predictions = self.model.forward(users, songs)  # Use forward explicitly
            loss = self.model.criterion(predictions, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            logging.info(
                f"Completed incremental training on {len(training_data)} samples"
            )
            
        except Exception as e:
            logging.error(f"Error in incremental training: {str(e)}")
            raise
            
    def add_feedback(self, feedback: UserFeedback):
        """Add new user feedback."""
        self.feedback_collector.add_feedback(feedback) 