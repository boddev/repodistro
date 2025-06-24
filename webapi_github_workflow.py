from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import OAuth2AuthorizationCodeBearer
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import requests
import time
import logging
from github import Github
import os
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="GitHub Integration API for Power Platform",
    description="Secure GitHub repository management API with Power Platform integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# GitHub OAuth Configuration
CLIENT_ID = os.getenv('GITHUB_CLIENT_ID', 'Ov23li8CQAivGoBQeDol')
CLIENT_SECRET = os.getenv('GITHUB_CLIENT_SECRET', '2ce7252b751c6e1094f6d525a2513a4742fe684b')

# Use YOUR API's callback endpoint - not Power Platform's
REDIRECT_URI = os.getenv('REDIRECT_URI', 'http://localhost:8000/callback')  # Your API endpoint

# GitHub OAuth URLs
AUTHORIZE_URL = 'https://github.com/login/oauth/authorize'
TOKEN_URL = 'https://github.com/login/oauth/access_token'
API_URL = 'https://api.github.com/'

# CORS Configuration for Power Platform
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.flow.microsoft.com",
        "https://*.powerapps.com",
        "https://*.dynamics.com",
        "https://connectorcreator.azurewebsites.net", 
        "http://localhost:*",
        "http://127.0.0.1:*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Session middleware
app.add_middleware(SessionMiddleware, secret_key=CLIENT_SECRET)

# OAuth2 scheme for GitHub
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=AUTHORIZE_URL,
    tokenUrl=TOKEN_URL,
    scopes={
        "repo": "Repository access",
        "user": "User information"
    }
)

# Pydantic models for request/response validation
class ModifyRepoRequest(BaseModel):
    """Request model for modifying repository files"""
    repo_name: str = Field(..., description="Name of the repository to modify")
    file_path: str = Field(..., description="Path to the file in the repository")
    file_content: str = Field(..., description="Content to write to the file")

class CloneTemplateRequest(BaseModel):
    """Request model for cloning template repositories"""
    template_owner: str = Field(..., description="Owner of the template repository")
    template_repo: str = Field(..., description="Name of the template repository")
    new_repo_name: str = Field(..., description="Name for the new repository")

class StandardResponse(BaseModel):
    """Standard API response format"""
    success: bool = Field(..., description="Indicates if the operation was successful")
    message: str = Field(..., description="Human-readable message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error_code: Optional[str] = Field(None, description="Error code if applicable")

# Middleware for request/response logging
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.4f}s")
    
    # Add API versioning headers
    response.headers["X-API-Version"] = "1.0.0"
    response.headers["X-Response-Time"] = str(process_time)
    
    return response

async def get_github_user(request: Request):
    """
    Validate GitHub token and return user information
    
    Args:
        request: FastAPI request object
        
    Returns:
        GitHub user information
        
    Raises:
        HTTPException: If token is invalid or missing
    """
    github_token = request.session.get('github_token')
    if not github_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="GitHub authentication required. Please visit /login"
        )
    
    try:
        g = Github(github_token)
        user = g.get_user()
        return {
            "login": user.login,
            "name": user.name,
            "id": user.id,
            "avatar_url": user.avatar_url
        }
    except Exception as e:
        logger.error(f"GitHub authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid GitHub token. Please re-authenticate"
        )

@app.get("/", response_model=StandardResponse)
@limiter.limit("100/minute")
async def home(request: Request):
    """
    Home endpoint providing API information
    
    Returns:
        StandardResponse: Welcome message and API information
    """
    return StandardResponse(
        success=True,
        message="GitHub Integration API for Power Platform",
        data={
            "version": "1.0.0",
            "documentation": "/docs",
            "authentication": "/login",
            "endpoints": {
                "login": "GET /login - Initiate GitHub OAuth",
                "callback": "GET /callback - OAuth callback handler",
                "modify_repo": "POST /modify_repo - Modify repository files",
                "clone_template": "POST /clone_template - Clone template repository",
                "get_schema": "GET /get_schema - Retrieve API schema",
                "health": "GET /health - Health check"
            }
        }
    )

@app.get("/login")
@limiter.limit("10/minute")
async def login(request: Request):
    """
    Initiate GitHub OAuth2 authentication flow
    Always uses YOUR API's callback endpoint
    """
    # Store the original requester info for later redirect
    power_platform_callback = request.query_params.get('callback_url')
    if power_platform_callback:
        request.session['power_platform_callback'] = power_platform_callback
    
    # Fix URL encoding - use proper URL encoding
    encoded_redirect_uri = urllib.parse.quote(REDIRECT_URI, safe='')
    # Request repo and user scopes (repo includes public_repo permissions)
    auth_url = f"{AUTHORIZE_URL}?scope=repo%20user&client_id={CLIENT_ID}&redirect_uri={encoded_redirect_uri}"
    
    logger.info(f"Initiating GitHub OAuth authentication with API callback: {REDIRECT_URI}")
    return RedirectResponse(auth_url)

@app.get("/callback")
@limiter.limit("10/minute")
async def callback(request: Request):
    """
    Handle OAuth2 callback from GitHub
    This is YOUR API endpoint that GitHub calls back to
    """
    code = request.query_params.get('code')
    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authorization code not provided"
        )
    

    
    debug = request.query_params.get('debug')
    if debug and debug.lower() == 'true':
        logger.info(f"Debug mode enabled for callback with code: {code}")
        return JSONResponse(
            content={
                "message": "Debug mode is active",
                "code": code,
                "redirect_uri": REDIRECT_URI,
                "client_id": CLIENT_ID
            },
            status_code=status.HTTP_200_OK
        )
    
    try:
        # Exchange code for token using YOUR callback URI
        token = await get_access_token(code, REDIRECT_URI)
        request.session['github_token'] = token
        
        # Get user info to verify token
        user_info = await get_github_user(request)
        
        logger.info(f"User authenticated successfully: {user_info['login']}")
        
        # Check if this came from Power Platform and redirect back
        power_platform_callback = request.session.get('power_platform_callback')
        if power_platform_callback:
            # Clean up session
            del request.session['power_platform_callback']
            
            # Redirect back to Power Platform with success indicator
            callback_url = f"{power_platform_callback}?auth_success=true&user={user_info['login']}"
            return RedirectResponse(callback_url)
        
        # Direct API access - return JSON response
        return StandardResponse(
            success=True,
            message="GitHub authentication successful",
            data={
                "authenticated": True,
                "user": user_info
            }
        )
        
    except Exception as e:
        logger.error(f"Authentication callback error: {str(e)}")
        
        # If from Power Platform, redirect back with error
        power_platform_callback = request.session.get('power_platform_callback')
        if power_platform_callback:
            callback_url = f"{power_platform_callback}?auth_success=false&error={str(e)}"
            return RedirectResponse(callback_url)
        
        # Direct API access - return error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GitHub authentication failed: {str(e)}"
        )
    
@app.get("/get_auth_token")
@limiter.limit("10/minute")
async def get_auth_token(request: Request):
    code = request.query_params.get('code')
    try:
        # Exchange code for token using YOUR callback URI
        token = await get_access_token(code, REDIRECT_URI)
        request.session['github_token'] = token
        
        # Get user info to verify token
        user_info = await get_github_user(request)
        
        logger.info(f"User authenticated successfully: {user_info['login']}")
        
        # Check if this came from Power Platform and redirect back
        power_platform_callback = request.session.get('power_platform_callback')
        if power_platform_callback:
            # Clean up session
            del request.session['power_platform_callback']
            
            # Redirect back to Power Platform with success indicator
            callback_url = f"{power_platform_callback}?auth_success=true&user={user_info['login']}"
            return RedirectResponse(callback_url)
        
        # Direct API access - return JSON response
        return StandardResponse(
            success=True,
            message="GitHub authentication successful",
            data={
                "authenticated": True,
                "user": user_info
            }
        )
        
    except Exception as e:
        logger.error(f"Authentication callback error: {str(e)}")
        
        # If from Power Platform, redirect back with error
        power_platform_callback = request.session.get('power_platform_callback')
        if power_platform_callback:
            callback_url = f"{power_platform_callback}?auth_success=false&error={str(e)}"
            return RedirectResponse(callback_url)
        
        # Direct API access - return error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GitHub authentication failed: {str(e)}"
        )

async def get_access_token(code: str, redirect_uri: str) -> str:
    """
    Exchange authorization code for GitHub access token
    Always uses YOUR API's callback URI
    """
    data = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': code,
        'redirect_uri': redirect_uri
    }
    
    # Log the request data (without exposing the secret)
    log_data = data.copy()
    log_data['client_secret'] = f"{CLIENT_SECRET[:8]}..." if CLIENT_SECRET else "NOT_SET"
    logger.info(f"Token exchange request data: {log_data}")
    
    try:
        logger.info(f"Making POST request to: {TOKEN_URL}")
        response = requests.post(
            TOKEN_URL, 
            headers={'Accept': 'application/json'}, 
            data=data,
            timeout=30
        )
        
        # Log the response details
        logger.info(f"Token exchange response status: {response.status_code}")
        logger.info(f"Token exchange response headers: {dict(response.headers)}")
        logger.info(f"Token exchange response text: {response.text}")
        
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
            
        response_data = response.json()
        logger.info(f"Parsed response data keys: {list(response_data.keys())}")
        
        if 'error' in response_data:
            error_msg = response_data.get('error_description', response_data.get('error', 'Unknown error'))
            logger.error(f"GitHub OAuth error in response: {error_msg}")
            raise Exception(f"GitHub OAuth error: {error_msg}")
        
        if 'access_token' not in response_data:
            logger.error(f"No access token in response. Response data: {response_data}")
            raise Exception(f"No access token in response: {response_data}")
        
        logger.info("Access token successfully obtained")
        return response_data['access_token']
        
    except requests.RequestException as e:
        logger.error(f"Network error during token exchange: {str(e)}")
        raise Exception(f"Network error during token exchange: {str(e)}")
    except Exception as e:
        logger.error(f"Token exchange failed with exception: {str(e)}")
        raise

# Add endpoint for Power Platform to check auth status
@app.get("/auth/check")
@limiter.limit("100/minute")
async def check_auth_for_power_platform(request: Request, session_id: Optional[str] = None):
    """
    Check if user is authenticated - designed for Power Platform polling
    """
    try:
        user_info = await get_github_user(request)
        return StandardResponse(
            success=True,
            message="User is authenticated",
            data={
                "authenticated": True,
                "user": user_info,
                "github_token_exists": True
            }
        )
    except HTTPException:
        return StandardResponse(
            success=False,
            message="User is not authenticated",
            data={
                "authenticated": False,
                "login_url": f"/login?callback_url=https://your-power-platform-callback-url.com",
                "github_token_exists": False
            }
        )

@app.post("/modify_repo", response_model=StandardResponse)
@limiter.limit("30/minute")
async def modify_repo(request: Request, repo_request: ModifyRepoRequest):
    """
    Modify or create a file in a GitHub repository
    
    Args:
        request: FastAPI request object
        repo_request: Repository modification request
        
    Returns:
        StandardResponse: Operation result
        
    Raises:
        HTTPException: If operation fails
    """
    try:
        # Check authentication
        user_info = await get_github_user(request)
        
        github_token = request.session['github_token']
        g = Github(github_token)
        user = g.get_user()
        
        repo_full_name = f"{user.login}/{repo_request.repo_name}"
        repo = g.get_repo(repo_full_name)
        
        # Try to update existing file, create if not exists
        operation = "created"
        try:
            contents = repo.get_contents(repo_request.file_path)
            repo.update_file(
                contents.path,
                "Update file via Power Platform API",
                repo_request.file_content,
                contents.sha
            )
            operation = "updated"
        except Exception:
            repo.create_file(
                repo_request.file_path,
                "Create file via Power Platform API",
                repo_request.file_content
            )
        
        logger.info(f"File {operation}: {repo_request.file_path} in {repo_full_name} by {user_info['login']}")
        
        return StandardResponse(
            success=True,
            message=f"File {operation} successfully",
            data={
                "repository": repo_full_name,
                "file_path": repo_request.file_path,
                "operation": operation,
                "user": user_info['login']
            }
        )
        
    except HTTPException:
        # Re-raise authentication errors
        raise
    except Exception as e:
        logger.error(f"Modify repo error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to modify repository: {str(e)}"
        )

async def modify_repo_internal(request: Request):
    """Internal method for handling modify repo after authentication"""
    # This would be called from callback if edit flag is set
    # Implementation depends on how you want to handle the edit flow
    return StandardResponse(
        success=True,
        message="Edit operation completed after authentication"
    )

@app.post("/clone_template", response_model=StandardResponse)
@limiter.limit("10/minute")
async def clone_template(request: Request, clone_request: Optional[CloneTemplateRequest] = None):
    """
    Create a new repository from a template
    
    Args:
        request: FastAPI request object
        clone_request: Template cloning request (optional for backward compatibility)
        
    Returns:
        StandardResponse: Clone operation result
        
    Raises:
        HTTPException: If cloning fails
    """
    try:
        # Check authentication
        user_info = await get_github_user(request)
        github_token = request.session['github_token']
        
        # Handle backward compatibility with hardcoded values
        if clone_request is None:
            template_owner = 'boddev'
            template_repo = 'GraphConnectorApiTemplate'
            new_repo_owner = user_info.get('login', 'default_owner')
            new_repo_name = 'myGcTemplate'
        else:
            template_owner = clone_request.template_owner
            template_repo = clone_request.template_repo
            new_repo_owner = user_info.get('login', 'default_owner')
            new_repo_name = clone_request.new_repo_name
        
        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        # First, validate the template repository exists and is a template
        template_url = f"https://api.github.com/repos/{template_owner}/{template_repo}"
        template_response = requests.get(template_url, headers=headers)
        
        if template_response.status_code != 200:
            logger.error(f"Template repository not found: {template_owner}/{template_repo}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template repository {template_owner}/{template_repo} not found or not accessible"
            )
        
        template_data = template_response.json()
        
        # Check if it's actually a template repository
        if not template_data.get('is_template', False):
            logger.error(f"Repository is not a template: {template_owner}/{template_repo}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Repository {template_owner}/{template_repo} is not configured as a template"
            )
        
        # Check if repository with new name already exists
        existing_repo_url = f"https://api.github.com/repos/{new_repo_owner}/{new_repo_name}"
        existing_response = requests.get(existing_repo_url, headers=headers)
        
        if existing_response.status_code == 200:
            logger.error(f"Repository already exists: {new_repo_owner}/{new_repo_name}")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Repository {new_repo_owner}/{new_repo_name} already exists"
            )
        
        payload = {
            "owner": new_repo_owner,
            "name": new_repo_name,
            "description": "Repository created from template via Power Platform API",
            "include_all_branches": False,
            "private": False
        }
        
        # Log the request for debugging
        logger.info(f"Creating repository from template: {template_owner}/{template_repo} -> {new_repo_owner}/{new_repo_name}")
        logger.info(f"Request payload: {payload}")
        
        url = f"https://api.github.com/repos/{template_owner}/{template_repo}/generate"
        response = requests.post(url=url, headers=headers, json=payload)
        
        # Log detailed response for debugging
        logger.info(f"GitHub API response status: {response.status_code}")
        logger.info(f"GitHub API response headers: {dict(response.headers)}")
        logger.info(f"GitHub API response body: {response.text}")
        
        if response.status_code == 201:
            repo_data = response.json()
            logger.info(f"Repository cloned: {new_repo_name} by {user_info['login']}")
            
            return StandardResponse(
                success=True,
                message="Repository created successfully from template",
                data={
                    "new_repository": f"{new_repo_owner}/{new_repo_name}",
                    "template": f"{template_owner}/{template_repo}",
                    "clone_url": repo_data.get("clone_url"),
                    "html_url": repo_data.get("html_url"),
                    "user": user_info['login']
                }
            )
        elif response.status_code == 403:
            error_data = response.json() if response.content else {}
            error_message = error_data.get('message', 'Forbidden')
            
            # Common 403 reasons
            if 'rate limit' in error_message.lower():
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="GitHub API rate limit exceeded. Please try again later."
                )
            elif 'permission' in error_message.lower() or 'access' in error_message.lower():
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions to create repository from template. Token may need 'public_repo' scope. Error: {error_message}"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"GitHub API access forbidden: {error_message}"
                )
        else:
            error_data = response.json() if response.content else {}
            logger.error(f"GitHub API error: {response.status_code} - {error_data}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"GitHub API error: {error_data.get('message', 'Unknown error')}"
            )
            
    except HTTPException:
        # Re-raise authentication and GitHub API errors
        raise
    except Exception as e:
        logger.error(f"Clone template error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clone template: {str(e)}"
        )

@app.get("/get_schema", response_model=StandardResponse)
@limiter.limit("60/minute")
async def get_endpoint_schema(request: Request, url: str):
    """
    Retrieve schema from an external API endpoint
    
    Args:
        request: FastAPI request object
        url: URL of the endpoint to analyze
        
    Returns:
        StandardResponse: Schema information
        
    Raises:
        HTTPException: If schema retrieval fails
    """
    try:
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid URL format. Must start with http:// or https://"
            )
        
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            logger.info(f"Schema retrieved from: {url}")
            
            # Try to parse as JSON, fallback to text
            try:
                schema_data = response.json()
            except:
                schema_data = response.text
            
            return StandardResponse(
                success=True,
                message="Schema retrieved successfully",
                data={
                    "url": url,
                    "schema": schema_data,
                    "content_type": response.headers.get("content-type"),
                    "status_code": response.status_code
                }
            )
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to retrieve schema from {url}. Status: {response.status_code}"
            )
            
    except requests.RequestException as e:
        logger.error(f"Schema retrieval error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Network error retrieving schema: {str(e)}"
        )

@app.get("/health", response_model=StandardResponse)
@limiter.limit("200/minute")
async def health_check(request: Request):
    """
    Health check endpoint for monitoring
    
    Returns:
        StandardResponse: API health status
    """
    return StandardResponse(
        success=True,
        message="API is healthy",
        data={
            "status": "healthy",
            "timestamp": int(time.time()),
            "version": "1.0.0",
            "github_oauth_configured": bool(CLIENT_ID and CLIENT_SECRET)
        }
    )

@app.get("/auth/status", response_model=StandardResponse)
@limiter.limit("60/minute")
async def auth_status(request: Request):
    """
    Check authentication status
    
    Returns:
        StandardResponse: Authentication status
    """
    try:
        user_info = await get_github_user(request)
        return StandardResponse(
            success=True,
            message="User is authenticated",
            data={
                "authenticated": True,
                "user": user_info
            }
        )
    except HTTPException:
        return StandardResponse(
            success=False,
            message="User is not authenticated",
            data={
                "authenticated": False,
                "login_url": "/login"
            }
        )

# Debug endpoints
@app.get("/debug/oauth-url")
async def get_oauth_url(request: Request):
    """Debug endpoint to see the exact OAuth URL being generated"""
    encoded_redirect_uri = urllib.parse.quote(REDIRECT_URI, safe='')
    auth_url = f"{AUTHORIZE_URL}?scope=repo%20user&client_id={CLIENT_ID}&redirect_uri={encoded_redirect_uri}"
    
    return {
        "oauth_url": auth_url,
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "encoded_redirect_uri": encoded_redirect_uri,
        "scopes": "repo user",
        "note": "repo scope includes public_repo permissions"
    }

@app.get("/debug/token-exchange")
async def debug_token_exchange(code: str):
    """Debug endpoint to see what GitHub returns during token exchange"""
    data = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': code,
        'redirect_uri': REDIRECT_URI
    }
    
    try:
        response = requests.post(
            TOKEN_URL, 
            headers={'Accept': 'application/json'}, 
            data=data
        )
        
        return {
            "status_code": response.status_code,
            "response_data": response.json(),
            "request_data": {
                "client_id": CLIENT_ID,
                "redirect_uri": REDIRECT_URI,
                "code_length": len(code)
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/token-permissions")
@limiter.limit("30/minute")
async def debug_token_permissions(request: Request):
    """Debug endpoint to check token permissions"""
    try:
        user_info = await get_github_user(request)
        github_token = request.session['github_token']
        
        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json"
        }
        
        # Check token scopes using the correct endpoint
        user_response = requests.get("https://api.github.com/user", headers=headers)
        
        # Also check the applications endpoint to see granted scopes
        app_response = requests.get("https://api.github.com/applications/grants", headers=headers)
        
        # Try to access a repo to test repo scope
        repos_response = requests.get("https://api.github.com/user/repos", headers=headers)
        
        return {
            "user": user_info,
            "token_scopes_header": user_response.headers.get('X-OAuth-Scopes', 'No scopes header found'),
            "user_endpoint_status": user_response.status_code,
            "repos_endpoint_status": repos_response.status_code,
            "repos_access": repos_response.status_code == 200,
            "rate_limit_remaining": user_response.headers.get('X-RateLimit-Remaining'),
            "rate_limit_limit": user_response.headers.get('X-RateLimit-Limit'),
            "all_user_headers": dict(user_response.headers),
            "token_preview": f"{github_token[:10]}..." if github_token else "No token",
            "can_access_repos": repos_response.status_code == 200,
            "apps_endpoint_status": app_response.status_code if app_response else "Failed"
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/oauth-flow")
async def debug_oauth_flow(request: Request):
    """Complete OAuth flow debugging"""
    
    # Check what's in the session
    session_data = {
        "github_token_exists": 'github_token' in request.session,
        "github_token_length": len(request.session.get('github_token', '')) if 'github_token' in request.session else 0,
        "session_keys": list(request.session.keys())
    }
    
    # Generate OAuth URL
    encoded_redirect_uri = urllib.parse.quote(REDIRECT_URI, safe='')
    auth_url = f"{AUTHORIZE_URL}?scope=repo%20user&client_id={CLIENT_ID}&redirect_uri={encoded_redirect_uri}"
    
    # Test if we can make GitHub API calls
    github_api_test = {"status": "no_token"}
    if 'github_token' in request.session:
        try:
            headers = {
                "Authorization": f"Bearer {request.session['github_token']}",
                "Accept": "application/vnd.github+json"
            }
            test_response = requests.get("https://api.github.com/user", headers=headers)
            github_api_test = {
                "status": "success" if test_response.status_code == 200 else "failed",
                "status_code": test_response.status_code,
                "response_headers": dict(test_response.headers),
                "has_scopes_header": 'X-OAuth-Scopes' in test_response.headers
            }
        except Exception as e:
            github_api_test = {"status": "error", "error": str(e)}
    
    return {
        "oauth_config": {
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "auth_url": auth_url,
            "scopes_requested": "repo user"
        },
        "session_info": session_data,
        "github_api_test": github_api_test
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=StandardResponse(
            success=False,
            message=exc.detail,
            error_code=str(exc.status_code)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=StandardResponse(
            success=False,
            message="Internal server error",
            error_code="500"
        ).dict()
    )

@app.get("/logout")
@limiter.limit("30/minute")
async def logout(request: Request):
    """
    Clear authentication session
    """
    request.session.clear()
    
    return StandardResponse(
        success=True,
        message="Successfully logged out",
        data={
            "authenticated": False,
            "login_url": "/login"
        }
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
