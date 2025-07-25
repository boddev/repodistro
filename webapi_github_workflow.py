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

# Application Insights imports
try:
    from opencensus.ext.azure.log_exporter import AzureLogHandler
    from opencensus.ext.azure.trace_exporter import AzureExporter
    from opencensus.trace.samplers import ProbabilitySampler
    from opencensus.trace import config_integration
    AZURE_LOGGING_AVAILABLE = True
except ImportError:
    AZURE_LOGGING_AVAILABLE = False
    print("Warning: Azure logging packages not available. Install 'opencensus-ext-azure' for Application Insights integration.")

# Configure logging with Application Insights
def setup_logging():
    """Configure logging with Application Insights integration"""
    
    # Basic logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Add Application Insights handler if available and configured
    app_insights_connection_string = os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING',"InstrumentationKey=af576058-8de0-413d-bf88-8c3ddc51c6df;IngestionEndpoint=https://northcentralus-0.in.applicationinsights.azure.com/;LiveEndpoint=https://northcentralus.livediagnostics.monitor.azure.com/;ApplicationId=31c112fd-22ef-4e2b-8083-bf9ce2e82915")
    
    if app_insights_connection_string and AZURE_LOGGING_AVAILABLE:
        try:
            # Create Azure Log Handler
            azure_handler = AzureLogHandler(connection_string=app_insights_connection_string)
            azure_handler.setLevel(logging.INFO)
            
            # Add custom properties to all logs
            def add_custom_properties(envelope):
                envelope.tags['ai.cloud.role'] = 'github-integration-api'
                envelope.tags['ai.cloud.roleInstance'] = 'power-platform-connector'
                envelope.tags['ai.application.ver'] = '1.0.0'
                return True
            
            azure_handler.add_telemetry_processor(add_custom_properties)
            
            # Add the handler to the logger
            logger.addHandler(azure_handler)
            
            # Configure trace integration for requests
            config_integration.trace_integrations(['requests'])
            
            logger.info("✅ Application Insights logging configured successfully")
            logger.info(f"Connection string configured: {app_insights_connection_string[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Failed to configure Application Insights logging: {str(e)}")
            logger.warning("Falling back to console logging only")
            
    elif not app_insights_connection_string:
        logger.warning("⚠️ APPLICATIONINSIGHTS_CONNECTION_STRING environment variable not set")
        logger.info("Logs will only be sent to console/stdout")
        
    elif not AZURE_LOGGING_AVAILABLE:
        logger.warning("⚠️ Azure logging packages not installed")
        logger.info("Install 'opencensus-ext-azure' to enable Application Insights logging")
    
    return logger

# Initialize logging
logger = setup_logging()

# Structured logging helper for better Application Insights integration
def log_structured_event(event_type: str, level: str = "INFO", **kwargs):
    """
    Log structured events that will appear nicely in Application Insights
    
    Args:
        event_type: Type of event (e.g., 'oauth_login', 'token_exchange', 'api_call')
        level: Log level (INFO, WARNING, ERROR)
        **kwargs: Additional properties to include in the log
    """
    log_data = {
        'event_type': event_type,
        'timestamp': time.time(),
        **kwargs
    }
    
    # Remove None values and sensitive data
    filtered_data = {}
    for key, value in log_data.items():
        if value is not None:
            # Mask sensitive data
            if 'token' in key.lower() and isinstance(value, str) and len(value) > 10:
                filtered_data[key] = f"{value[:10]}***"
            elif 'secret' in key.lower():
                filtered_data[key] = "***MASKED***"
            else:
                filtered_data[key] = value
    
    message = f"[{event_type}] Event logged"
    
    if level.upper() == "ERROR":
        logger.error(message, extra={'custom_dimensions': filtered_data})
    elif level.upper() == "WARNING":
        logger.warning(message, extra={'custom_dimensions': filtered_data})
    else:
        logger.info(message, extra={'custom_dimensions': filtered_data})

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
CLIENT_ID = os.getenv('GITHUB_CLIENT_ID')
CLIENT_SECRET = os.getenv('GITHUB_CLIENT_SECRET')

# Use YOUR API's callback endpoint - not Power Platform's
REDIRECT_URI_API = 'https://connectorcreator-fyduenajachkcxax.northcentralus-01.azurewebsites.net/callback'
REDIRECT_URI_POWER_PLATFORM = 'https://global.consent.azure-apim.net/redirect'

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

async def get_github_token_from_request(request: Request) -> str:
    """
    Get GitHub token from either session (for direct API calls) or Authorization header (for Power Platform)
    """
    # First try to get from Authorization header (Power Platform style)
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return auth_header[7:]  # Remove 'Bearer ' prefix
    
    # Fallback to session (for direct API calls)
    github_token = request.session.get('github_token')
    if github_token:
        return github_token
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="GitHub token required. Provide via Authorization header or authenticate via /login"
    )

async def get_github_user_flexible(request: Request):
    """
    Updated version that works with both session and header tokens
    """
    github_token = await get_github_token_from_request(request)
    
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
            detail="Invalid GitHub token"
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
    Chooses redirect URI based on source
    """
    logger.info("=== LOGIN ENDPOINT START ===")
    
    # Log structured event for Application Insights
    log_structured_event(
        "oauth_login_initiated",
        request_url=str(request.url),
        request_method=request.method,
        client_ip=request.client.host if request.client else 'Unknown',
        headers=dict(request.headers),
        query_params=dict(request.query_params)
    )
    
    # Check if request is from Power Platform
    user_agent = request.headers.get('user-agent', '').lower()
    source = request.query_params.get('source', 'api')
    
    # Determine redirect URI logic
    is_powerplatform_ua = 'powerplatform' in user_agent
    is_microsoft_ua = 'microsoft' in user_agent
    is_powerplatform_source = source == 'powerplatform'
    
    log_structured_event(
        "oauth_redirect_decision",
        user_agent=user_agent,
        source=source,
        is_powerplatform_ua=is_powerplatform_ua,
        is_microsoft_ua=is_microsoft_ua,
        is_powerplatform_source=is_powerplatform_source
    )
    
    # Use Power Platform redirect for Power Platform requests
    if source == 'powerplatform' or 'powerplatform' in user_agent or 'microsoft' in user_agent:
        redirect_uri = REDIRECT_URI_POWER_PLATFORM
        redirect_type = "power_platform"
    else:
        redirect_uri = REDIRECT_URI_API
        redirect_type = "direct_api"
    
    log_structured_event(
        "oauth_redirect_selected",
        redirect_uri=redirect_uri,
        redirect_type=redirect_type
    )
    
    # Store the original requester info for later redirect
    power_platform_callback = request.query_params.get('callback_url')
    
    if power_platform_callback:
        try:
            request.session['power_platform_callback'] = power_platform_callback
            log_structured_event("session_data_stored", data_type="power_platform_callback")
        except Exception as e:
            log_structured_event("session_storage_failed", "ERROR", 
                                data_type="power_platform_callback", 
                                error=str(e))
    
    # Store which redirect URI we're using
    try:
        request.session['oauth_redirect_uri'] = redirect_uri
        log_structured_event("session_data_stored", data_type="oauth_redirect_uri")
    except Exception as e:
        log_structured_event("session_storage_failed", "ERROR", 
                            data_type="oauth_redirect_uri", 
                            error=str(e))
    
    # Build OAuth URL
    try:
        encoded_redirect_uri = urllib.parse.quote(redirect_uri, safe='')
        auth_url = f"{AUTHORIZE_URL}?scope=repo%20user&client_id={CLIENT_ID}&redirect_uri={encoded_redirect_uri}"
        
        log_structured_event(
            "oauth_url_generated",
            original_redirect_uri=redirect_uri,
            encoded_redirect_uri=encoded_redirect_uri,
            client_id=CLIENT_ID,
            scopes="repo user"
        )
        
    except Exception as e:
        log_structured_event("oauth_url_generation_failed", "ERROR", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build OAuth URL: {str(e)}"
        )
    
    log_structured_event("oauth_redirect_initiated", redirect_uri=redirect_uri)
    logger.info("=== LOGIN ENDPOINT END - REDIRECTING TO GITHUB ===")
    
    try:
        return RedirectResponse(auth_url)
    except Exception as e:
        log_structured_event("redirect_response_failed", "ERROR", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to redirect to GitHub: {str(e)}"
        )

@app.get("/callback")
@limiter.limit("10/minute")
async def callback(request: Request):
    """
    Handle OAuth2 callback from GitHub
    This is YOUR API endpoint that GitHub calls back to
    """
    logger.info("=== CALLBACK ENDPOINT START ===")
    
    # Log structured event for Application Insights
    log_structured_event(
        "oauth_callback_received",
        callback_url=str(request.url),
        request_method=request.method,
        client_ip=request.client.host if request.client else 'Unknown',
        query_params=dict(request.query_params)
    )
    
    # Get and validate authorization code
    code = request.query_params.get('code')
    error = request.query_params.get('error')
    error_description = request.query_params.get('error_description')
    state = request.query_params.get('state')
    
    log_structured_event(
        "oauth_callback_params",
        has_code=code is not None,
        code_length=len(code) if code else 0,
        has_error=error is not None,
        error=error,
        error_description=error_description,
        state=state
    )
    
    if error:
        log_structured_event("oauth_callback_error", "ERROR", 
                            github_error=error, 
                            github_error_description=error_description)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"GitHub OAuth error: {error} - {error_description or 'No description provided'}"
        )
    
    if not code:
        log_structured_event("oauth_callback_missing_code", "ERROR")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authorization code not provided"
        )
    
    # Log session state
    try:
        session_keys = list(request.session.keys())
        stored_redirect_uri = request.session.get('oauth_redirect_uri')
        stored_pp_callback = request.session.get('power_platform_callback')
        
        log_structured_event(
            "session_state_check",
            session_keys=session_keys,
            has_stored_redirect=stored_redirect_uri is not None,
            has_pp_callback=stored_pp_callback is not None
        )
        
    except Exception as e:
        log_structured_event("session_read_failed", "ERROR", error=str(e))
        stored_redirect_uri = None
    
    # Determine which redirect URI to use for token exchange
    redirect_uri_for_token = stored_redirect_uri or REDIRECT_URI_API
    
    log_structured_event(
        "token_exchange_prep",
        redirect_uri_for_token=redirect_uri_for_token,
        using_fallback=stored_redirect_uri is None
    )
    
    # Debug mode check
    debug = request.query_params.get('debug')
    if debug and debug.lower() == 'true':
        log_structured_event("debug_mode_activated")
        debug_info = {
            "message": "Debug mode is active",
            "code": code,
            "redirect_uri": redirect_uri_for_token,
            "client_id": CLIENT_ID,
            "session_data": {
                "oauth_redirect_uri": stored_redirect_uri,
                "power_platform_callback": stored_pp_callback,
                "session_keys": session_keys
            }
        }
        return JSONResponse(content=debug_info, status_code=status.HTTP_200_OK)
    
    try:
        log_structured_event("token_exchange_starting")
        
        # Exchange code for token using stored redirect URI
        token = await get_access_token(code, redirect_uri_for_token)
        
        log_structured_event(
            "token_exchange_success",
            token_length=len(token) if token else 0
        )
        
        # Store token in session
        try:
            request.session['github_token'] = token
            log_structured_event("token_stored_in_session")
        except Exception as e:
            log_structured_event("token_storage_failed", "ERROR", error=str(e))
            raise
        
        # Get user info to verify token
        user_info = await get_github_user(request)
        
        log_structured_event(
            "user_validation_success",
            user_login=user_info.get('login', 'Unknown'),
            user_id=user_info.get('id'),
            user_name=user_info.get('name')
        )
        
        # Check if this came from Power Platform and redirect back
        power_platform_callback = request.session.get('power_platform_callback')
        
        if power_platform_callback:
            log_structured_event(
                "power_platform_redirect_prep",
                callback_url=power_platform_callback,
                user_login=user_info['login']
            )
            
            try:
                # Clean up session
                del request.session['power_platform_callback']
                if 'oauth_redirect_uri' in request.session:
                    del request.session['oauth_redirect_uri']
                log_structured_event("session_cleanup_success")
            except Exception as e:
                log_structured_event("session_cleanup_failed", "WARNING", error=str(e))
            
            # Redirect back to Power Platform with success indicator
            callback_url = f"{power_platform_callback}?auth_success=true&user={user_info['login']}"
            
            log_structured_event(
                "power_platform_redirect_executed",
                final_callback_url=callback_url
            )
            
            logger.info("=== CALLBACK ENDPOINT END - REDIRECTING TO POWER PLATFORM ===")
            return RedirectResponse(callback_url)
        
        log_structured_event("direct_api_response", user_login=user_info['login'])
        
        # Direct API access - return JSON response
        response_data = StandardResponse(
            success=True,
            message="GitHub authentication successful",
            data={
                "authenticated": True,
                "user": user_info
            }
        )
        
        logger.info("=== CALLBACK ENDPOINT END - SUCCESS ===")
        return response_data
        
    except Exception as e:
        log_structured_event(
            "oauth_callback_error", "ERROR",
            error=str(e),
            error_type=type(e).__name__
        )
        
        # Log stack trace for debugging
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        
        # If from Power Platform, redirect back with error
        power_platform_callback = request.session.get('power_platform_callback')
        if power_platform_callback:
            log_structured_event(
                "power_platform_error_redirect",
                callback_url=power_platform_callback,
                error=str(e)
            )
            
            try:
                del request.session['power_platform_callback']
            except:
                pass
            
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
    Uses the provided redirect URI for token exchange
    """
    log_structured_event(
        "token_exchange_initiated",
        code_length=len(code) if code else 0,
        redirect_uri=redirect_uri
    )
    
    data = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': code,
        'redirect_uri': redirect_uri
    }
    
    # Validate required parameters
    if not CLIENT_ID:
        log_structured_event("token_exchange_validation_failed", "ERROR", missing_param="CLIENT_ID")
        raise Exception("GitHub CLIENT_ID is not configured")
    
    if not CLIENT_SECRET:
        log_structured_event("token_exchange_validation_failed", "ERROR", missing_param="CLIENT_SECRET")
        raise Exception("GitHub CLIENT_SECRET is not configured")
    
    if not code:
        log_structured_event("token_exchange_validation_failed", "ERROR", missing_param="code")
        raise Exception("Authorization code is required")
    
    if not redirect_uri:
        log_structured_event("token_exchange_validation_failed", "ERROR", missing_param="redirect_uri")
        raise Exception("Redirect URI is required")
    
    try:
        log_structured_event(
            "github_token_request",
            token_url=TOKEN_URL,
            client_id=CLIENT_ID,
            redirect_uri=redirect_uri
        )
        
        response = requests.post(
            TOKEN_URL, 
            headers={'Accept': 'application/json'}, 
            data=data,
            timeout=30
        )
        
        log_structured_event(
            "github_token_response",
            status_code=response.status_code,
            response_content_length=len(response.content),
            response_headers=dict(response.headers)
        )
        
        if response.status_code != 200:
            log_structured_event(
                "github_token_error", "ERROR",
                status_code=response.status_code,
                response_text=response.text
            )
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        try:
            response_data = response.json()
            log_structured_event(
                "github_token_parsed",
                response_keys=list(response_data.keys())
            )
        except ValueError as e:
            log_structured_event("github_token_parse_failed", "ERROR", error=str(e))
            raise Exception(f"Invalid JSON response from GitHub: {str(e)}")
        
        # Check for error in response
        if 'error' in response_data:
            error_msg = response_data.get('error_description', response_data.get('error', 'Unknown error'))
            log_structured_event(
                "github_oauth_error", "ERROR",
                github_error=response_data.get('error'),
                github_error_description=response_data.get('error_description')
            )
            raise Exception(f"GitHub OAuth error: {error_msg}")
        
        # Check for access token
        if 'access_token' not in response_data:
            log_structured_event(
                "github_token_missing", "ERROR",
                available_keys=list(response_data.keys())
            )
            raise Exception(f"No access token in response: {response_data}")
        
        access_token = response_data['access_token']
        token_type = response_data.get('token_type', 'bearer')
        scope = response_data.get('scope', 'unknown')
        
        log_structured_event(
            "github_token_success",
            token_type=token_type,
            token_scope=scope,
            token_length=len(access_token)
        )
        
        return access_token
        
    except requests.RequestException as e:
        log_structured_event(
            "github_token_network_error", "ERROR",
            error=str(e),
            error_type=type(e).__name__
        )
        raise Exception(f"Network error during token exchange: {str(e)}")
        
    except Exception as e:
        log_structured_event(
            "github_token_general_error", "ERROR",
            error=str(e),
            error_type=type(e).__name__
        )
        
        # Log stack trace for debugging
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
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
        # user_info = await get_github_user(request)       
        # github_token = request.session['github_token']

        # Use the flexible authentication
        user_info = await get_github_user_flexible(request)
        github_token = await get_github_token_from_request(request)

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
        # user_info = await get_github_user(request)
        # github_token = request.session['github_token']

        # Use the flexible authentication
        user_info = await get_github_user_flexible(request)
        github_token = await get_github_token_from_request(request)
        
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

@app.post("/clone_template_with_token", response_model=StandardResponse)
@limiter.limit("10/minute")
async def clone_template_with_token(
    request: Request, 
    github_token: str,
    clone_request: Optional[CloneTemplateRequest] = None
):
    """
    Clone template with direct token - no session required
    Perfect for Power Platform Agent integration
    """
    try:
        # Validate token and get user info
        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        user_response = requests.get("https://api.github.com/user", headers=headers)
        if user_response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid GitHub token"
            )
        
        user_info = user_response.json()
        
        # Rest of your clone_template logic here...
        # Use the same logic but with direct token instead of session
        
        return StandardResponse(
            success=True,
            message="Repository created successfully from template",
            data={
                "new_repository": f"{user_info['login']}/{new_repo_name}",
                "user": user_info['login']
            }
        )
        
    except Exception as e:
        logger.error(f"Clone template with token error: {str(e)}")
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

@app.post("/authenticate", response_model=StandardResponse)
@limiter.limit("10/minute")
async def authenticate_with_token(request: Request, github_token: str):
    """
    Direct authentication with GitHub personal access token
    Better for Power Platform Agent integration
    """
    try:
        # Validate the token directly
        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json"
        }
        
        response = requests.get("https://api.github.com/user", headers=headers)
        
        if response.status_code == 200:
            user_data = response.json()
            
            # Store token in session
            request.session['github_token'] = github_token
            
            return StandardResponse(
                success=True,
                message="Authentication successful",
                data={
                    "authenticated": True,
                    "user": {
                        "login": user_data.get("login"),
                        "name": user_data.get("name"),
                        "id": user_data.get("id"),
                        "avatar_url": user_data.get("avatar_url")
                    }
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid GitHub token"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}"
        )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
