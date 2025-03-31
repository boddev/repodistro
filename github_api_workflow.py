import requests
import jwt
import time
from github import Github

# Replace these with your own values
APP_ID = 'your_app_id'
INSTALLATION_ID = 'your_installation_id'
PRIVATE_KEY = 'your_private_key'
TEMPLATE_OWNER = 'template_owner'
TEMPLATE_REPO = 'template_repo'
NEW_REPO_NAME = 'new_repo_name'
NEW_FILE_PATH = 'path/to/new_file.txt'
NEW_FILE_CONTENT = 'This is the content of the new file.'

# Generate a JWT for authentication
def generate_jwt(app_id, private_key):
    payload = {
        'iat': int(time.time()),
        'exp': int(time.time()) + (10 * 60),
        'iss': app_id
    }
    return jwt.encode(payload, private_key, algorithm='RS256')

# Get an installation access token
def get_installation_access_token(jwt_token, installation_id):
    headers = {
        'Authorization': f'Bearer {jwt_token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    response = requests.post(f'https://api.github.com/app/installations/{installation_id}/access_tokens', headers=headers)
    response_data = response.json()
    return response_data['token']

# Create a new repository from a template
def create_repo_from_template(token, template_owner, template_repo, new_repo_name):
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.baptiste-preview+json'
    }
    data = {
        'owner': 'your_github_username',
        'name': new_repo_name,
        'description': 'This is a new repository created from a template',
        'private': False
    }
    response = requests.post(f'https://api.github.com/repos/{template_owner}/{template_repo}/generate', headers=headers, json=data)
    return response.json()

# Add or update a file in the repository
def add_or_update_file(token, repo_name, file_path, file_content):
    g = Github(token)
    repo = g.get_repo(f'your_github_username/{repo_name}')
    try:
        contents = repo.get_contents(file_path)
        repo.update_file(contents.path, "Updating file", file_content, contents.sha)
    except:
        repo.create_file(file_path, "Creating new file", file_content)

# Main workflow
jwt_token = generate_jwt(APP_ID, PRIVATE_KEY)
installation_token = get_installation_access_token(jwt_token, INSTALLATION_ID)
create_repo_from_template(installation_token, TEMPLATE_OWNER, TEMPLATE_REPO, NEW_REPO_NAME)
add_or_update_file(installation_token, NEW_REPO_NAME, NEW_FILE_PATH, NEW_FILE_CONTENT)
