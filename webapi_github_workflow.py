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
REDIRECT_URI = os.getenv('REDIRECT_URI', 'https://{environment-name}.flow.microsoft.com/manage/oauthresponse')

# GitHub OAuth URLs
AUTHORIZE_URL = 'https://github.com/login/oauth/authorize'
TOKEN_URL = 'https://github.com/login/oauth/access_token'
API_URL = 'https://api.github.com'

# CORS Configuration for Power Platform
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.flow.microsoft.com",
        "https://*.powerapps.com",
        "https://*.dynamics.com",
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
    new_repo_owner: str = Field(..., description="Owner for the new repository")
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

@app.get("/login", response_model=StandardResponse)
@limiter.limit("10/minute")
async def login(request: Request):
    """
    Initiate GitHub OAuth2 authentication flow
    
    Returns:
        RedirectResponse: Redirect to GitHub OAuth login
    """
    auth_url = f"{AUTHORIZE_URL}?scope=repo,user&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"
    
    logger.info("Initiating GitHub OAuth authentication")
    return RedirectResponse(auth_url)

@app.get("/callback", response_model=StandardResponse)
@limiter.limit("10/minute")
async def callback(request: Request):
    """
    Handle OAuth2 callback from GitHub
    
    Args:
        request: FastAPI request object containing authorization code
        
    Returns:
        StandardResponse: Authentication result
        
    Raises:
        HTTPException: If authentication fails
    """
    code = request.query_params.get('code')
    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authorization code not provided"
        )
    
    try:
        token = await get_access_token(code)
        request.session['github_token'] = token
        
        # Get user info to verify token
        user_info = await get_github_user(request)
        
        logger.info(f"User authenticated successfully: {user_info['login']}")
        
        # Handle edit flow if needed
        try:
            edit = request.session.get('edit', False)
            if edit:
                logger.info('Processing edit request after authentication')
                request.session['edit'] = False
                return await modify_repo_internal(request)
        except Exception as e:
            logger.warning(f"Edit flow error: {str(e)}")
        
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GitHub authentication failed: {str(e)}"
        )

async def get_access_token(code: str) -> str:
    """
    Exchange authorization code for GitHub access token
    
    Args:
        code: Authorization code from GitHub
        
    Returns:
        str: GitHub access token
        
    Raises:
        Exception: If token exchange fails
    """
    data = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': code,
        'redirect_uri': REDIRECT_URI
    }
    
    response = requests.post(
        TOKEN_URL, 
        headers={'Accept': 'application/json'}, 
        data=data
    )
    response_data = response.json()
    
    if 'access_token' not in response_data:
        error_msg = response_data.get('error_description', 'Unknown error')
        raise Exception(f"Failed to obtain access token: {error_msg}")
    
    return response_data['access_token']

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
            new_repo_owner = 'bodoutlook'
            new_repo_name = 'myGcTemplate'
        else:
            template_owner = clone_request.template_owner
            template_repo = clone_request.template_repo
            new_repo_owner = clone_request.new_repo_owner
            new_repo_name = clone_request.new_repo_name
        
        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        payload = {
            "owner": new_repo_owner,
            "name": new_repo_name,
            "description": "Repository created from template via Power Platform API",
            "include_all_branches": False,
            "private": False
        }
        
        url = f"{API_URL}/repos/{template_owner}/{template_repo}/generate"
        response = requests.post(url=url, headers=headers, json=payload)
        
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
        else:
            error_data = response.json()
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

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
