import base64
import cv2
import numpy as np
import uvicorn
from starlette.responses import RedirectResponse
from typing import Optional, Callable
from database import *
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from routers import contact
from fastapi.middleware.cors import CORSMiddleware

from fastapi import Body, FastAPI, Request, Response
from fastapi.routing import APIRoute

# from models.contact import *


app = FastAPI(title='Contact.ly', description='APIs for contact Apis', version='0.1')
# app.router.route_class = GzipRoute
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
    "https://0.0.0.0:8000",
    "https://localhost:449",
    "http://localhost:63342",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["DELETE", "GET", "POST", "PUT"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


class Contact(BaseModel):
    contact_id: int
    first_name: str
    last_name: str
    user_name: str
    password: str

    class Config:
        schema_extra = {
            "example": {
                "contact_id": 1,
                "first_name": "Jhon",
                "last_name": "Doe",
                "user_name": "jhon_123",
            }
        }


class Face(BaseModel):
    face_id: Optional[str] = None
    img_str: str


class ContactOut(BaseModel):
    contact_id: int
    first_name: str
    last_name: str
    user_name: str


@app.get("/")
def main():
    return RedirectResponse(url="/docs/")


@app.post("/convert/{img_str}")
async def img_to_string(img_str: str):
    with open("imgToString.jpeg", "wb") as new_file:
        new_file.write(base64.b64decode(img_str))
    return new_file


@app.post("/view/anhtostring2")
async def convert_str_img(face: Face):
    img_str = face.img_str
    img_name = face.face_id + '.jpeg'
    img_path = 'images/' + img_name
    with open(img_path, "wb") as new_file:
        new_file.write(base64.b64decode(img_str))
    return new_file


@app.get("/view/anhtostring")
async def convert_str_img(strvar: str):
    return strvar


def chuyen_base64_sang_anh(anh_base64):
    try:
        anh_base64 = np.fromstring(base64.b64decode(anh_base64), dtype=np.uint8)
        anh_base64 = cv2.imdecode(anh_base64, cv2.IMREAD_ANYCOLOR)
    except:
        return None
    return anh_base64


def dem_so_mat(face):
    # Khoi tao bo phat hien khuon mat
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Chuyen gray
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # Phat hien khuon mat trong anh
    faces = face_cascade.detectMultiScale(gray, 1.2, 10)

    so_mat = len(faces)
    return so_mat


@app.post("/view/nhandienkhuonmat")
async def nhandienkhuonmat(face: Face):
    face_numbers = 0
    # Doc anh tu client gui len
    # facebase64 = await request.form()
    # facebase64 = face.img_str
    # Chuyen base 64 ve OpenCV Format
    # face = chuyen_base64_sang_anh(facebase64)
    face = cv2.imread('images/WIN_20210603_16_28_36_Pro.jpg')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Chuyen gray
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # Phat hien khuon mat trong anh
    faces = face_cascade.detectMultiScale(gray, 1.2, 10)

    so_mat = len(faces)
    # Đếm số mặt trong ảnh
    # face_numbers = dem_so_mat(face)

    # Trả về
    # return "Số mặt là = " + str(face_numbers)
    return so_mat
#####################################################################
# app.include_router(contact.router_contacts)


# @app.on_event("startup")
# async def startup():
#     print("Connecting...")
#     if conn.is_closed():
#         conn.connect()
#
#
# @app.on_event("shutdown")
# async def shutdown():
#     print("Closing...")
#     if not conn.is_closed():
#         conn.close()
