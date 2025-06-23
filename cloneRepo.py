# from fastapi import FastAPI, Request, Depends
# from fastapi.responses import RedirectResponse, JSONResponse
# from fastapi.security import OAuth2AuthorizationCodeBearer
# import requests
# import os

# app = FastAPI()

# # Replace these with your own values
# CLIENT_ID = 'your_client_id'
# CLIENT_SECRET = 'your_client_secret'
# REDIRECT_URI = 'http://localhost:8000/callback'

# # GitHub OAuth URLs
# AUTHORIZE_URL = 'https://github.com/login/oauth/authorize'
# TOKEN_URL = 'https://github.com/login/oauth/access_token'
# API_URL = 'https://api.github.com'

# oauth2_scheme = OAuth2AuthorizationCodeBearer(
#     authorizationUrl=AUTHORIZE_URL,
#     tokenUrl=TOKEN_URL
# )

# @app.get("/")
# async def home():
#     return {"message": "Go to /login to authenticate with GitHub"}

# @app.get("/login")
# async def login():
#     return RedirectResponse(f'{AUTHORIZE_URL}?client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&scope=repo')

# @app.get("/callback")
# async def callback(request: Request):
#     code = request.query_params.get('code')
#     token = await get_access_token(code)
#     return RedirectResponse(url=f"/create_repo?token={token}")

# async def get_access_token(code: str):
#     response = requests.post(TOKEN_URL, headers={'Accept': 'application/json'}, data={
#         'client_id': CLIENT_ID,
#         'client_secret': CLIENT_SECRET,
#         'code': code,
#         'redirect_uri': REDIRECT_URI
#     })
#     response_data = response.json()
#     return response_data['access_token']

# @app.get("/create_repo")
# async def create_repo(token: str = Depends(oauth2_scheme)):
#     headers = {
#         'Authorization': f'token {token}',
#         'Accept': 'application/vnd.github.baptiste-preview+json'
#     }
#     data = {
#         'owner': 'your_github_username',
#         'name': 'new_repo_name',
#         'description': 'This is a new repository created from a template',
#         'private': False
#     }
#     response = requests.post(f'{API_URL}/repos/template_owner/template_repo/generate', headers=headers, json=data)
    
#     if response.status_code == 201:
#         return JSONResponse(content={"message": "Repository created successfully!"})
#     else:
#         return JSONResponse(content={"error": f"Failed to create repository: {response.status_code}", "details": response.json()})

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
