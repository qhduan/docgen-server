import requests

inputs = '''
def is_prime(n):
    for i in range(2, n - 1):
        if n % i == 0:
            return False
    return True

# docstring
"""'''

ret = requests.post('http://localhost:8000', json={
    'context': inputs
})
res = ret.json()

print(res)
print(res.get('text'))
