import requests
import json

url = 'https://kauth.kakao.com/oauth/token'
client_id = ''
redirect_uri = 'https://example.com/oauth'
code = ''

data = {
    'grant_type':'authorization_code',
    'client_id':client_id,
    'redirect_uri':redirect_uri,
    'code': code,
    }

response = requests.post(url, data=data)
tokens = response.json()

#발행된 토큰 저장
with open("token.json","w") as kakao:
    json.dump(tokens, kakao)