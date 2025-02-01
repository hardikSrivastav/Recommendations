import pytest
from flask import jsonify
from application.utils.token_required import token_required
from application.routes.user_routes import user_bp

def test_missing_token(client):
    """Test request without token returns 401."""
    response = client.get('/api/users/profile/123')
    assert response.status_code == 401
    assert b'Token is missing' in response.data

def test_invalid_token_format(client, invalid_token):
    """Test request with invalid token format returns 401."""
    headers = {'Authorization': f'Bearer {invalid_token}'}
    response = client.get('/api/users/profile/123', headers=headers)
    assert response.status_code == 401
    assert b'Invalid token' in response.data

def test_expired_token(client, expired_token):
    """Test request with expired token returns 401."""
    headers = {'Authorization': f'Bearer {expired_token}'}
    response = client.get('/api/users/profile/123', headers=headers)
    assert response.status_code == 401
    assert b'Invalid token' in response.data

def test_valid_token(client, auth_headers, test_user):
    """Test request with valid token succeeds."""
    response = client.get(f'/api/users/profile/{test_user["_id"]}', headers=auth_headers)
    assert response.status_code == 200

def test_token_wrong_user(client, auth_headers):
    """Test accessing another user's data returns 403."""
    response = client.get('/api/users/profile/wrong_user_id', headers=auth_headers)
    assert response.status_code == 403
    assert b'Unauthorized' in response.data

def test_malformed_token(client):
    """Test malformed token returns 401."""
    headers = {'Authorization': 'not_a_bearer_token'}
    response = client.get('/api/users/profile/123', headers=headers)
    assert response.status_code == 401
    assert b'Invalid token' in response.data

def test_token_without_bearer(client, auth_token):
    """Test token without 'Bearer' prefix returns 401."""
    headers = {'Authorization': auth_token}
    response = client.get('/api/users/profile/123', headers=headers)
    assert response.status_code == 401
    assert b'Invalid token' in response.data 