from flask import Blueprint, jsonify, current_app
from application.models.continuous_learning import ContinuousLearningManager, TrainingConfig
from application.models.recommendation_service import RecommendationService
import logging

# Configure blueprint
training_bp = Blueprint('training', __name__)

def init_services(testing=False):
    """Initialize services if needed."""
    pass  # Services are now initialized in create_app

@training_bp.before_request
def ensure_services():
    """Ensure services are initialized before each request."""
    global continuous_learning_manager, recommendation_service, training_config
    if not hasattr(current_app, 'continuous_learning_manager'):
        raise RuntimeError("Application not properly initialized")
    continuous_learning_manager = current_app.continuous_learning_manager
    recommendation_service = current_app.recommendation_service
    training_config = continuous_learning_manager.config

@training_bp.route('/status', methods=['GET'])
async def get_training_status():
    """Get the current status of model training."""
    try:
        status = {
            'is_training': continuous_learning_manager.is_training,
            'queue_size': continuous_learning_manager.training_queue.qsize(),
            'last_training_time': await continuous_learning_manager.get_last_training_time()
        }
        return jsonify(status), 200
    except Exception as e:
        logging.error(f"Error getting training status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/trigger', methods=['POST'])
async def trigger_training():
    """Manually trigger model training if conditions are met."""
    try:
        logging.info("Received request to trigger training")
        
        if continuous_learning_manager.is_training:
            logging.info("Training already in progress")
            return jsonify({
                'status': 'error',
                'message': 'Training is already in progress'
            }), 409

        should_train = await continuous_learning_manager.should_train()
        if not should_train:
            logging.info("Training conditions not met - see previous logs for details")
            return jsonify({
                'status': 'error',
                'message': 'Training conditions not met'
            }), 400

        # Queue training cycle
        logging.info("Training conditions met, executing training cycle")
        await continuous_learning_manager.execute_training_cycle()
        
        return jsonify({
            'status': 'success',
            'message': 'Training cycle queued successfully'
        }), 202
    except Exception as e:
        logging.error(f"Error triggering training: {str(e)}")
        return jsonify({'error': str(e)}), 500

@training_bp.route('/config', methods=['GET'])
async def get_training_config():
    """Get current training configuration."""
    try:
        config = {
            'batch_size': training_config.batch_size,
            'learning_rate': training_config.learning_rate,
            'min_feedback_samples': training_config.min_feedback_samples,
            'training_interval': training_config.training_interval.total_seconds(),
            'max_queue_size': training_config.max_queue_size
        }
        return jsonify(config), 200
    except Exception as e:
        logging.error(f"Error getting training config: {str(e)}")
        return jsonify({'error': str(e)}), 500