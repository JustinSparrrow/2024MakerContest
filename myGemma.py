import base64
import urllib
import requests
import json
import time

API_KEY = "y8SxSzo6f4rw2mD2E4G9loga"
SECRET_KEY = "3y9if5vKzzCPpQvgq0RfBpMmQHkuxLrK"

def submit_request(path, urlencoded=False):
    url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/image-understanding/request?access_token=" + get_access_token()
    
    # image 可以通过 get_file_content_as_base64("C:\fakepath\example.jpg",False) 方法获取
    payload = json.dumps({
        "image": get_file_content_as_base64(path, urlencoded),
        "question": "请用优雅的语言表达鉴赏一下图片内容",
        "output_CHN": True
    })
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    
    response_data = response.text
    data = json.loads(response_data)
    task_id = data['result']['task_id']
    return task_id

def get_request(path, urlencoded=False):
    task_id = submit_request(path, urlencoded)
    
    while True:
        url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/image-understanding/get-result?access_token=" + get_access_token()
        
        payload = json.dumps({
            "task_id": task_id
        })
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload)
        
        response_data = response.text
        data = json.loads(response_data)
        
        # 检查 ret_code
        ret_code = data['result']['ret_code']
        if ret_code == 0:
            response_data = response.text
            data = json.loads(response_data)
            text = data['result']['description']
            print("Final Result:", text)
            break
        else:
            print("Processing... Retrying in 5 seconds.")
            time.sleep(5)  # 等待5秒后再检查

def get_file_content_as_base64(path, urlencoded=False):
    """
    获取文件base64编码
    :param path: 文件路径
    :param urlencoded: 是否对结果进行urlencoded 
    :return: base64编码信息
    """
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

if __name__ == '__main__':
    path = '/Users/moqi/Desktop/竞赛/2024创客赛/example.jpg'
    get_request(path, False)