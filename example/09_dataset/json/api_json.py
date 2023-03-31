import requests
# url = 'http://apis.data.go.kr/1360000/WthrWrnInfoService/getWthrWrnMsg'
# key = 'RyU8vp09GyAdmsH9z57XcTivzDrtN2nO52XqA8t4yZW+rEuNZKNDmrtTidRCdf2xh5xV3PgBNBiCkGbtgjraDg=='

url = 'http://apis.data.go.kr/1360000/WthrWrnInfoService/getWthrWrnList'
key = 'QjITnZtxSg5+hzh+WR8hYLMstCDRuf1REcb5E59648Wy77+7z8aQBHgv95ylOhyoP31mFZWlyiqd2TrMu7HTuw=='

params ={'serviceKey' : key,
        'pageNo': '1',
        'numOfRows':'30',
        'dataType': 'JSON',
        'stnId':'108',
        'fromTmFc':'20221201',
        'toTmFc':'20221205'}

response = requests.get(url, params=params)

result = response.json()
print(result)