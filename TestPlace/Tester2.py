import certifi
from dotenv import load_dotenv, find_dotenv
import os
from pymongo import MongoClient
import pprint


ca = certifi.where()
load_dotenv(find_dotenv())
password = os.environ.get("MONGO_PWD")
connection_string = f"mongodb+srv://thuyluu9595:{password}@ecommerce.tmleeao.mongodb.net/?retryWrites=true&w=majority&authSource=admin"
client = MongoClient(connection_string, tlsCAFile=ca)
db = client.test


def createUser():
    user_validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "title": "User Object Validation",
            "required": ["email", "name", "password", "isAdmin"],
            "properties": {
                "name": {
                    "bsonType": "string",
                    "description": "'name' must be a string and is required"
                },
                "email": {
                    "bsonType": "string",
                    "description": "'email' is required"
                },
                "password": {
                    "bsonType": "string",
                    "description": "password"
                },
                "isAdmin": {
                    "bsonType": "bool",
                    "description": "isAdmin"
                }
            }
        }
    }
    try:
        db.create_collection("User")
    except Exception as e:
        print(e)

    db.command("collMod", "User", validator=user_validator)
    User = db.User
    return User

myuser = createUser()
data = {
    "name": "Thuy",
    "email": "thuyluu9595@gmail.com",
    "password":"12345",
    "isAdmin": True
}
insert_id = myuser.insert_one(data)
print(insert_id)