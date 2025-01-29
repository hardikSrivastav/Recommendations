from flask import Blueprint, request, jsonify
from application.models.integrator import IntegratedMusicRecommender
from application.models.fma_dataset_processor import FMADatasetProcessor
import pandas as pd

recommendation_bp = Blueprint('recommendation', __name__)
recommender = IntegratedMusicRecommender()
fma_processor = FMADatasetProcessor()

@recommendation_bp.route('/<user_id>', methods=['GET'])
def get_recommendations(user_id):
    """Get personalized song recommendations for a user"""
    try:
        # Get number of recommendations requested (default 5)
        n_recommendations = int(request.args.get('n', 5))
        
        # Get the dataset
        track_data = fma_processor.process_dataset()
        
        # Get demographics data for the user (implement retrieval logic)
        # For now using dummy data matching the format in integrator.py
        demographics_df = pd.DataFrame([{
            'user_id': user_id,
            'age': 25,
            'gender': 'M',
            'location': 'US',
            'occupation': 'Student'
        }])
        
        # Get recommendations
        recommendations = recommender.predict_next_songs(
            user_id=user_id,
            track_data=track_data,
            demographics_df=demographics_df,
            n=n_recommendations
        )
        
        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations
        }), 200
        
    except ValueError as e:
        return jsonify({
            'error': 'Invalid request',
            'details': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@recommendation_bp.route('/batch', methods=['POST'])
def batch_recommendations():
    """Get recommendations for multiple users"""
    data = request.get_json()
    
    # Validate request
    if not isinstance(data.get('user_ids', []), list):
        return jsonify({
            'error': 'Invalid request format',
            'details': 'user_ids must be a list'
        }), 400
    
    try:
        n_recommendations = int(data.get('n', 5))
        track_data = fma_processor.process_dataset()
        
        # Get recommendations for each user
        results = {}
        for user_id in data['user_ids']:
            # Get demographics for user (implement retrieval logic)
            demographics_df = pd.DataFrame([{
                'user_id': user_id,
                'age': 25,
                'gender': 'M',
                'location': 'US',
                'occupation': 'Student'
            }])
            
            recommendations = recommender.predict_next_songs(
                user_id=user_id,
                track_data=track_data,
                demographics_df=demographics_df,
                n=n_recommendations
            )
            
            results[user_id] = recommendations
            
        return jsonify(results), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500 