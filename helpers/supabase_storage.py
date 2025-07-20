def uploadToStorage(client, url, image, imagename):
    try:
        res = client.storage.from_("diagnosis-image").upload(imagename, image.getvalue(), {"content-type": "image/jpeg"})
        path = res.fullPath
        src = f"{url}/storage/v1/object/public/{path}"
        return src
    except:
        return None

def isAdmin(client, uuid):
    try:
        res = client.from_("users").select("role").eq("auth_uuid", uuid).maybe_single().execute()
        return (res.model_dump()["data"]["role"] == "admin")
    except:
        return False

def createUser(service_role_client, email, password, role="user"):
    user_metadata = {"role": role}
    try:
        res = service_role_client.auth.admin.create_user({
            "email": email,
            "password": password,
            "user_metadata": user_metadata,
            "email_confirm": True
        })
        return res.model_dump() # convert SimpleApiResponse to JSON
    except:
        return {"error": { "message": "error creating user" }}
    
def updateUser(service_role_client, target_user_id, updated_fields):
    try:
        res = service_role_client.auth.admin.update_user_by_id(target_user_id, updated_fields)
        return res.model_dump() # convert SimpleApiResponse to JSON
    except:
        return {"error": { "message": "error updating user" }}
    
def deleteUser(service_role_client, target_user_id):
    try:
        res = service_role_client.auth.admin.delete_user(target_user_id)
        return {"user": None} # response from delete_user() is None, we cant use model_dump()
    except:
        return {"error": { "message": "error deleting user" }}

    