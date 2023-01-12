import requests
import json

#발행한 토큰 불러오기
with open("./token.json","r") as kakao:
    tokens = json.load(kakao)

url="https://kapi.kakao.com/v2/api/talk/memo/default/send"

headers={
    "Authorization" : "Bearer " + tokens["access_token"]
}

data = {
       'object_type': 'text',
       'text': '1epoch 종료',
       'link': {
           'web_url': 'https://developers.kakao.com',
           'mobile_web_url': 'https://developers.kakao.com'
       },
       'button_title': 'epoch alert'
   }
   

# pre_url = "https://kauth.kakao.com/oauth/authorize?client_id=e8623c7a81b0db7fcfb663fe8bd62a5f&redirect_uri=https://example.com/oauth&response_type=code&scope=talk_message"
# pre_response = requests.post(pre_url, headers=headers)
# print(pre_response.status_code)
data = {'template_object': json.dumps(data)}
response = requests.post(url, headers=headers, data=data)
print(response.status_code)
print(response.text)

