import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from src.graph.graphbuilder import GraphBuilder
from src.llms.groqllm import GroqLLM


import os
from dotenv import load_dotenv
load_dotenv()


app = FastAPI(
    title="Battlefield Triage System",
    version="1.0.0"
)

@app.post('/triage')
async def triage_system(request:Request):
    try:
        data = await request.json()
        soldier_id = data.get("soldier_id")
        vitals = data.get("vitals")
        injury = data.get("injury")

        if not all([soldier_id, vitals, injury]):
            return JSONResponse(status_code=400, content={"error": "Missing required fields"})


        # Get the llm object 
        groqllm = GroqLLM()
        llm = groqllm.get_llm()
        graphbuilder = GraphBuilder(llm)

        compiled_graph =graphbuilder.setup_graph()
        result = compiled_graph.invoke(data)
        return {"data": result}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__=="__main__":
    uvicorn.run("app:app",host="0.0.0.0",port=8000,reload=True)
