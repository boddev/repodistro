from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import OAuth2AuthorizationCodeBearer
import requests
from starlette.middleware.sessions import SessionMiddleware
from github import Github

app = FastAPI()

# Replace these with your own values
CLIENT_ID = 'Ov23li8CQAivGoBQeDol'  
CLIENT_SECRET = '2ce7252b751c6e1094f6d525a2513a4742fe684b' 
REDIRECT_URI = 'http://127.0.0.1:8000/callback'

# GitHub OAuth URLs
AUTHORIZE_URL = 'https://github.com/login/oauth/authorize'
TOKEN_URL = 'https://github.com/login/oauth/access_token'
API_URL = 'https://api.github.com'

# Add SessionMiddleware
app.add_middleware(SessionMiddleware, secret_key=CLIENT_SECRET)


oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=AUTHORIZE_URL,
    tokenUrl=TOKEN_URL
)

g = ''

@app.get("/")
async def home():
    return {"message": "Go to /login to authenticate with GitHub"}

@app.get("/login")
async def login():
    return RedirectResponse(f'{AUTHORIZE_URL}?scope=repo&client_id={CLIENT_ID}')  # client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&scope=repo')

@app.get("/callback")
async def callback(request: Request):
    code = request.query_params.get('code')
    token = await get_access_token(code)
    request.session['github_token'] = token
    #return RedirectResponse(url=f"/clone-template?token={token}")  # (url="/clone-template") #
    try:
        edit = request.session['edit']
        if edit:
            print('Editing repo')
            request.session['edit'] = False
            response = await modify_repo(request)#
        else:
            return {"message": "You are authenticated"}  # (url="/") #
        
    except: 
        response = await clone_template(request)
    return response

async def get_access_token(code: str):
    response = requests.post(TOKEN_URL, headers={'Accept': 'application/json'}, data={
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': code,
        'redirect_uri': REDIRECT_URI
    })
    response_data = response.json()
    return response_data['access_token']

@app.post("/modify_repo")
async def modify_repo(request: Request): 
    print('Modifying repo')
    data = await request.json()
    repo_name = data.get("repo_name")
    file_path = data.get("file_path")
    file_content = data.get("file_content")
    print(repo_name, file_path, file_content)
    try:
        token = request.session['github_token']
    except:
        print('No token')
        request.session['edit'] = True
        return RedirectResponse(url="/login")
    
    g = Github(token)
    user = g.get_user()

    repo = g.get_repo(f'{user}/{repo_name}')
    
    retries = 5
    for i in range(retries):
        try:
            contents = repo.get_contents(file_path)
            repo.update_file(contents.path, "Updating file", file_content, contents.sha)
            break
        except:
            repo.create_file(file_path, "Creating new file", file_content)
            break
    return {"message": "File added or updated successfully"}

@app.post("/clone_template")
async def clone_template(request: Request):
    if "github_token" not in request.session:
        return RedirectResponse(url="/login")

    #data = await request.json()
    template_owner = 'boddev'# data["template_owner"]
    template_repo = 'GraphConnectorApiTemplate' #data["template_repo"]
    new_repo_owner = 'bodoutlook' #data["new_repo_owner"]
    new_repo_name = 'myGcTemplate' #data["new_repo_name"]

    headers = {
        "Authorization": f"Bearer {request.session['github_token']}",
        "Accept": "application/vnd.github+json",
        'X-GitHub-Api-Version': '2022-11-28'
    }

    payload = {
        "owner": new_repo_owner,
        "name": new_repo_name,
        "description": "Repo created from a template",
        "include_all_branches": False,
        "private": False
    }

    url = f"{API_URL}/repos/{template_owner}/{template_repo}/generate"
    response = requests.post(
        url=url,
        headers=headers,
        json=payload
    )

    if response.status_code == 201:
        return JSONResponse(status_code=201, content={"message": "Repository created successfully"})
    else:
        return JSONResponse(status_code=response.status_code, content=response.json())


@app.get("/get_schema")
async def get_endpoint_schema(url: str):
    response = requests.get(url=url)
    if response.status_code == 200:
        return JSONResponse(status_code=200, content=response.json())
    else:
        return JSONResponse(status_code=response.status_code, content=response.json())

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
