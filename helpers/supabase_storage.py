from supabase import create_client
import os
from dotenv import load_dotenv
import json

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
service_key = os.getenv("SUPABASE_SERVICE_KEY")

client = create_client(url, key)

service_client = create_client(url, service_key)

def uploadToStorage(image, imagename):
    res = client.storage.from_("diagnosis-image").upload(imagename, image.getvalue(), {"content-type": "image/jpeg"})
    path = res.fullPath
    src = f"{url}/storage/v1/object/public/{path}"
    return src

def isAdmin(uuid):
    res = client.from_("users").select("role").eq("auth_uuid", uuid).maybe_single().execute()
    try:
        return (res.model_dump()["data"]["role"] == "admin")
    except:
        return False

def createUser(email, password, role="user"):
    user_metadata = {"role": role}
    try:
        res = service_client.auth.admin.create_user({
            "email": email,
            "password": password,
            "user_metadata": user_metadata,
            "email_confirm": True
        })
        return res.model_dump()
    except:
        return {"error": { "message": "error creating user" }}

    