from urllib import response

from ollama import chat
MODEL_NAME = "qwen3.6"





class blinders_framework:
  def __init__(self, model_name):
    self._model_name = model_name
    nutritionist_role = "nutritionist"
    user_handler_role = "user_handler"
    
    self._roles ={
}


    self._tools_for_agents = {
    }
    self._tools_for_agents_description = {
    }

    self._goals_for_roles = {
    }

    self._main_agent_role = None
    
    

  def add_agent(self, role_name, role_description, tool_func_list,tool_descriptions, goal_for_role, main_agent = False):
    """Add a new agent role to the framework."""
    self._roles[role_name] = role_description

    self._tools_for_agents[role_name] = {}

    for t in tool_func_list:
      self._tools_for_agents[role_name][t.__name__] = t

    for t_desc in tool_descriptions:
      self._tools_for_agents_description[role_name] = t_desc


    self._goals_for_roles[role_name] = goal_for_role

    if main_agent:
        self._main_agent_role = role_name

  def run_framework(self, query):
    """Run the framework with the main agent role."""
    if not self._main_agent_role:
        raise ValueError("No main agent role defined. Please set main_agent=True for one of the agents.")


    prompt = f"""
            ### role:
              {self._roles[self._main_agent_role]}

            ### goal:
              {self._goals_for_roles[self._main_agent_role]}

            ### call content:
              {query}
            """

    
    messages = [{"role": "user", "content": prompt}]


    response = chat(
        model=self._model_name,
        messages=messages,
        tools=self._tools_for_agents[self._main_agent_role].values(),
        think=False
    )


    if response.message.tool_calls:
        call = response.message.tool_calls[0]
        tool_name = call.function.name
        arguments = dict(call.function.arguments or {})


        # Pass the user's query as an argument to the tool
        result = self._tools_for_agents[self._main_agent_role][tool_name](**arguments)

        function_output_back_prompt = f"""
                  ### role:
                    {self._roles[self._main_agent_role]}
                    
                  ### call content:
                    {result.message.content}
                  """

        
        messages = [{"role": "user", "content": prompt + function_output_back_prompt}]
        

        final_response = chat(model=self._model_name, messages=messages, think=False)
        print(" ----- final response ----- ")
        print(final_response)

        output = final_response.message.content
        return output
    else:
      output = "WARNING: non tool call is not yet supported! Main agent tried to speak without"


    return output



class diet_planner_agentic_agent(blinders_framework):


  def __init__(self, model_name):
    super().__init__(model_name)


    nutrition_goal = """Generate a healthy diet plan for weight loss based on the user's requirements.
                      The diet plan should be for a week and include breakfast, lunch, and dinner for each day.
                      Make the text nice and readable from the terminal.
                      Keep the plan small and concise, and do not include any additional information or explanations.
                      """

    user_assistant_goal = """Inform the user of the decisions made by the underlying agentic system diet app.
                          Only call the nutritionist tool to generate a diet plan based on the user's requirements.
                          Do not make any decisions on your own."""
    
    nutrition_call_tool = {
                  "type": "function",
                  "function": {
                      "name": "call_nutritionist_tool",
                      "description": "Call the nutritionist tool to generate a diet plan based on user requirements.",
                      "parameters": {
                          "type": "object",
                          "properties": {
                              "user_requirements": {
                                  "type": "string",
                                  "description": "The user's requirements for the diet plan (e.g., 'I want to lose weight and eat healthy food')."
                              }
                          },
                          "required": ["user_requirements"]
                      }
                  }
              }

    
    self.add_agent(
        role_name="user assistant",
        role_description="This user handles communication with the user and calls " \
        "the nutritionist tool to generate a diet plan based on the user's requirements",
        tool_func_list=[self.call_nutritionist_tool],
        tool_descriptions = [nutrition_call_tool],
        goal_for_role=user_assistant_goal,
        main_agent=True
    )



    self.add_agent(
        role_name="nutritionist",
        role_description="This user generates a nutrition plan based on the user's requirements",
        tool_func_list=[],
        tool_descriptions = [],
        goal_for_role=nutrition_goal
    )



  def call_nutritionist_tool(self, user_requirements: str):
    """Call the nutritionist tool with the user's requirements."""
    print("called the nutritionist")

    prompt = f"""
    ### role:
      {self._roles['nutritionist']}

    ### goal:

    ### user requirements:
      {user_requirements}
    """

    messages = [{"role": "user", "content": prompt}]
    response = chat(model=self._model_name, messages=messages, think=False)
    print("nutritionist response: ")
    print(response)



    return response



if __name__ == "__main__":
  agentic_system = diet_planner_agentic_agent(MODEL_NAME)

  query = "I want a diet plan for a week, I want to lose weight and I want to eat healthy food"

  response_val = agentic_system.run_framework(query)

  print(" ----- final response ----- ")
  print(response_val)