import requests

class AuthenticationEngine:
    def __init__(self, auth_url):
        self.auth_url = auth_url


    def call_login_token(self, username, password):
        try:
            headers = {
                'Content-Type': 'application/json'
            }
            data = {
                'username': username,
                'password': password
            }
            response = requests.post(self.auth_url, json=data, headers=headers, timeout=5, verify=False)
            if response.status_code == 200:
                token = response.json().get('token')
                return token
            else:
                print(f"Authentication Error: {response.status_code} - {response.text}")
                return None


        except requests.exceptions.RequestException as e:
            print(f"Error during authentication: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error during authentication: {e}")
            return None