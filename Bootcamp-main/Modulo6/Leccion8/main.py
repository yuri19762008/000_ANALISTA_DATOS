from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import nbformat
from nbconvert import HTMLExporter

app = FastAPI()

NOTEBOOK_PATH = r"D:\000_ANALISTA DATOS , TALENTO DIGITAL\Bootcamp-main\Modulo6\Leccion8\Index_M6_AE8.ipynb"

@app.get("/", response_class=HTMLResponse)
async def home():
    return "<h2>Ir a /notebook para ver el .ipynb</h2>"

@app.get("/notebook", response_class=HTMLResponse)
async def render_notebook():
    # 1) Leer el notebook
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # 2) Convertir a HTML
    html_exporter = HTMLExporter()
    body, _ = html_exporter.from_notebook_node(nb)

    # 3) Devolver HTML al navegador
    return HTMLResponse(content=body)



#INICIAR EL SERVIDOR CON: uvicorn main:app --reload
#Luego ir a http://127.0.0.1:8000/notebook para ver el notebook renderizado en HTML.
