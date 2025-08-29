import datetime

class SimpleLLMAgent:
    def __init__(self, llm_model):
        self.llm = llm_model
        self.memory = [] # Simple short-term memory
        self.tools = {
            "search": self._search_tool,
            "calculate_days": self._calculate_days_tool
        }

    def _search_tool(self, query: str) -> str:
        """Simulates a search engine tool."""
        print(f"DEBUG: Using search tool for: '{query}'")
        if "2024年巴黎奥运会开幕日期" in query:
            return "2024年巴黎奥运会开幕日期是2024年7月26日。"
        elif "今天的日期" in query or "今天是" in query:
            return f"今天是 {datetime.date.today().strftime('%Y年%m月%d日')}。"
        return "未找到相关信息。"

    def _calculate_days_tool(self, date_str: str) -> str:
        """Calculates days from today to a given date string."""
        print(f"DEBUG: Using calculate_days tool for: '{date_str}'")
        try:
            # Attempt to parse date in common formats
            formats = ["%Y年%m月%d日", "%Y-%m-%d", "%Y/%m/%d"]
            target_date = None
            for fmt in formats:
                try:
                    target_date = datetime.datetime.strptime(date_str, fmt).date()
                    break
                except ValueError:
                    continue
            
            if target_date is None:
                raise ValueError("Date format not recognized.")

            today = datetime.date.today()
            delta = target_date - today
            if delta.days >= 0:
                return f"距离{date_str}还有{delta.days}天。"
            else:
                return f"{date_str}已过去{-delta.days}天。"
        except ValueError as e:
            return f"日期格式不正确或无法计算: {e}"

    def run(self, task: str) -> str:
        self.memory.append(f"用户任务: {task}")
        print(f"\nAgent 启动，处理任务: '{task}'")

        # Simplified planning: LLM decides actions based on prompt
        # In a real agent, the LLM would output structured actions (e.g., JSON)
        # Here, we simulate simple action selection based on keywords.

        current_thought = "Initial thought: I need to determine the best action to fulfill the user's request."
        
        for step in range(5): # Limit steps to prevent infinite loops in simplified simulation
            print(f"\n--- Step {step + 1} ---")
            print(f"Agent Thought: {current_thought}")

            # Simulate LLM's decision-making
            action_decision = self.llm.invoke(f"Current task: '{task}'. Current memory/context: {self.memory[-1]}. Available tools: {list(self.tools.keys())}. Based on this, what should I do next? (e.g., 'search 2024年巴黎奥运会开幕日期', 'calculate_days 2024年7月26日', or 'final_answer ...')")
            
            print(f"LLM Decision: {action_decision}")

            if "final_answer" in action_decision:
                final_answer_text = action_decision.replace("final_answer ", "").strip()
                print(f"Agent Final Answer: {final_answer_text}")
                return final_answer_text
            
            elif action_decision.startswith("search"):
                query = action_decision[len("search "):].strip()
                observation = self.tools["search"](query)
                self.memory.append(f"Observation (search): {observation}")
                current_thought = f"Observed: '{observation}'. Now, how do I use this information?"
            
            elif action_decision.startswith("calculate_days"):
                date_str = action_decision[len("calculate_days "):].strip()
                observation = self.tools["calculate_days"](date_str)
                self.memory.append(f"Observation (calculate_days): {observation}")
                current_thought = f"Observed: '{observation}'. Now, how do I formulate the final answer?"
            else:
                print(f"ERROR: Unknown action decision from LLM: {action_decision}")
                return "Agent encountered an unknown action and cannot proceed."

        return "Agent could not resolve the task within the given steps."

# Simulate a very simple LLM that makes decisions based on keywords in its prompt
class MockLLM:
    def invoke(self, prompt: str) -> str:
        print(f"DEBUG: LLM Received Prompt (first 100 chars): {prompt[:100]}...")
        if "巴黎奥运会开幕日期" in prompt and "search" in prompt:
            return "search 2024年巴黎奥运会开幕日期"
        elif "2024年巴黎奥运会开幕日期是2024年7月26日" in prompt and "calculate_days" in prompt:
            return "calculate_days 2024年7月26日"
        elif "距离2024年7月26日还有" in prompt and "final_answer" in prompt:
            # Extract days from the observation in prompt
            import re
            match = re.search(r"距离(\d{4}年\d{1,2}月\d{1,2}日)还有(\d+)天", prompt)
            if match:
                date_found = match.group(1)
                days_left = match.group(2)
                return f"final_answer 2024年巴黎奥运会将于{date_found}开幕，距离今天还有{days_left}天。"
            return "final_answer 2024年巴黎奥运会开幕日期已找到，但无法精确计算剩余天数。"
        elif "今天的日期" in prompt and "search" in prompt:
            return "search 今天的日期"
        elif "今天是" in prompt and "final_answer" in prompt:
            import re
            match = re.search(r"今天是 (\d{4}年\d{1,2}月\d{1,2}日)", prompt)
            if match:
                date_today = match.group(1)
                return f"final_answer 今天的日期是{date_today}。"
            return "final_answer 无法确定今天的日期。"

        return "final_answer 我未能找到足够的信息或合适的工具来完成您的请求。"

if __name__ == "__main__":
    llm = MockLLM()
    agent = SimpleLLMAgent(llm)

    # Example 1: Find Olympic date and calculate days
    print("--- Running Example 1: Olympic Date ---")
    result1 = agent.run("查找2024年巴黎奥运会开幕日期，并告诉我距离今天还有多少天。")
    print(f"\nFinal Result 1: {result1}")
    print("\n" + "="*80 + "\n")

    # Example 2: Just ask for today's date
    llm_for_ex2 = MockLLM() # Reset LLM for clean run
    agent_for_ex2 = SimpleLLMAgent(llm_for_ex2)
    print("--- Running Example 2: Today's Date ---")
    result2 = agent_for_ex2.run("告诉我今天的日期。")
    print(f"\nFinal Result 2: {result2}")
    print("\n" + "="*80 + "\n")

    # Example 3: Unresolvable query
    llm_for_ex3 = MockLLM() # Reset LLM for clean run
    agent_for_ex3 = SimpleLLMAgent(llm_for_ex3)
    print("--- Running Example 3: Unresolvable Query ---")
    result3 = agent_for_ex3.run("请预测下周三彩票的中奖号码。")
    print(f"\nFinal Result 3: {result3}")
    print("\n" + "="*80 + "\n")
