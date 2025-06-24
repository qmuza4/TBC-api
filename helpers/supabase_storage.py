from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

client = create_client(url, key)

def uploadToStorage(image, imagename):
    res = client.storage.from_("diagnosis-image").upload(imagename, image.getvalue(), {"content-type": "image/jpeg"})
    path = res.fullPath
    src = f"{url}/storage/v1/object/public/{path}"
    return src
    