from typing_extensions import List
from pydantic import BaseModel, Field
from typing import TypedDict

# This is the way in which my LLM model(Agent has to give the output as)
class TriageState(BaseModel):
    soldier_id: str= Field(description="The id of the soldier who is injured")
    vitals: dict
    injury: str= Field(description ="This will describe the type and the severity of the injury")
    classification: str = Field(description="This will describe the classification of the triage")
    rule_category: str 
    rule_priority: str
    reasoning: str
    treatment: str = Field(description= "This will give me the complete treatment plan for the soldier")

#This is the actual state schema
class triage(TypedDict):
    soldier_id: str
    vitals: dict
    injury: str
    classification: str
    rule_category: str
    rule_priority: str
    details : TriageState
    treatment: str = Field(description= "This will give me the complete treatment plan for the soldier")