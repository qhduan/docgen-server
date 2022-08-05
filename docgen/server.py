import traceback
import onnxruntime
from fastapi import FastAPI, Request
from .infer import infer

app = FastAPI()


@app.get('/')
async def hello():
    return onnxruntime.get_available_providers()


@app.post("/")
async def infer_api(request: Request):
    """Infer text from request body.

    Args:
        request: request object

    Return:
        json object with 'ok' and 'text' keys
    """
    try:
        body = await request.json()
        text = infer(**body)
        return {
            'ok': True,
            'text': text
        }
    except KeyboardInterrupt:
        raise
    except:
        traceback.print_exc()
        return {
            'ok': False,
            'error': traceback.format_exc()
        }
