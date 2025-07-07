from langgraph.graph import StateGraph, START, END
from src.llms.groqllm import  GroqLLM
from src.state.triagestate import TriageState, triage
from src.node.triage_node import triageNode


class GraphBuilder:
    def __init__(self, llm):
        self.llm = llm
        self.graph = StateGraph(triage)


    def build_triage_graph(self):
        """
            Build a graph to generate the treatment plans based on the vitals of the patients
        """

        self.triage_node_obj = triageNode(self.llm)

        # Creating the nodes required 
        self.graph.add_node("classification_patient" , self.triage_node_obj.classification_node)
        self.graph.add_node("rule_logic" , self.triage_node_obj.node_rule_based_category)
        self.graph.add_node("generate_reasoning", self.triage_node_obj.node_generating_reasoning)
        self.graph.add_node("generate_treatment", self.triage_node_obj.node_generate_treatment)


        # Now defining the workflow 
        self.graph.add_edge(START, "classification_patient")
        self.graph.add_edge("classification_patient", "rule_logic")
        self.graph.add_edge("rule_logic", "generate_reasoning")
        self.graph.add_edge("generate_reasoning", "generate_treatment")
        self.graph.add_edge("generate_treatment", END)


       

        return self.graph
    
    def setup_graph(self):
        logic = self.build_triage_graph()
        return logic.compile()



#Below code is for the langsmith langgrpah studio
llm =GroqLLM().get_llm()

# get the grpaph
grpah_builder = GraphBuilder(llm)
graph = grpah_builder.build_triage_graph().compile()