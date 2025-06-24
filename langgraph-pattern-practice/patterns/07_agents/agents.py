"""
Agents パターン（自律エージェント）
================================

このパターンは、LLMが自律的に判断し、ツールを使用して
複雑なタスクを実行するエージェントシステムです。

特徴：
- 自律的な意思決定: エージェントが次に取るべきアクションを自分で決定
- ツール使用: 外部APIやツールを動的に選択・使用
- 環境からのフィードバック: 実行結果を基に次のアクションを調整
- 目標指向: 与えられた目標を達成するまで継続的に動作
- エラーハンドリング: 失敗から学習し、別のアプローチを試行

例：
- コーディングエージェント: 複数ファイルの編集、テスト実行、デバッグ
- 研究エージェント: 複数ソースからの情報収集と分析
- 自動化エージェント: 複雑なワークフローの実行

このパターンの利点：
- 複雑で予測困難なタスクに対応
- 人間の監督なしで長時間動作可能
- 柔軟な問題解決アプローチ
- スケーラブルな自動化
"""

import json
import re
import time
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List

from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain.schema import SystemMessage
from langchain_openai import ChatOpenAI

# ===== 環境変数の読み込み =====
load_dotenv()


class ToolResult:
    """
    ツール実行結果を表すクラス
    """

    def __init__(
        self,
        success: bool,
        result: Any = None,
        error: str = None,
        tool_name: str = "",
        execution_time: float = 0,
    ):
        self.success = success
        self.result = result
        self.error = error
        self.tool_name = tool_name
        self.execution_time = execution_time
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "tool_name": self.tool_name,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        }


class AgentTool:
    """
    エージェントが使用できるツールの基底クラス
    """

    def __init__(self, name: str, description: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.usage_count = 0

    def execute(self, **kwargs) -> ToolResult:
        """
        ツールを実行する（サブクラスで実装）

        Returns:
            ToolResult: 実行結果
        """
        raise NotImplementedError("Subclasses must implement execute method")

    def get_schema(self) -> Dict[str, Any]:
        """
        ツールのスキーマを取得

        Returns:
            Dict[str, Any]: ツールのスキーマ情報
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class CalculatorTool(AgentTool):
    """
    計算ツール
    """

    def __init__(self):
        super().__init__(
            name="calculator",
            description="数学的な計算を実行します。Python式を評価できます。",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "計算式（例: '2 + 3 * 4', 'math.sqrt(16)'）",
                }
            },
        )

    def execute(self, expression: str) -> ToolResult:
        start_time = time.time()
        self.usage_count += 1

        try:
            # セキュリティ上の理由で、利用可能な関数を制限
            import math

            allowed_names = {
                "__builtins__": {},
                "math": math,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "len": len,
            }

            result = eval(expression, allowed_names)
            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                result=result,
                tool_name=self.name,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                error=f"計算エラー: {str(e)}",
                tool_name=self.name,
                execution_time=execution_time,
            )


class WebSearchTool(AgentTool):
    """
    Web検索ツール（模擬実装）
    """

    def __init__(self):
        super().__init__(
            name="web_search",
            description="インターネットで情報を検索します。",
            parameters={
                "query": {"type": "string", "description": "検索クエリ"},
                "num_results": {
                    "type": "integer",
                    "description": "取得する結果数（デフォルト: 5）",
                },
            },
        )

    def execute(self, query: str, num_results: int = 5) -> ToolResult:
        start_time = time.time()
        self.usage_count += 1

        # 模擬検索結果を生成
        mock_results = [
            {
                "title": f"{query}に関する詳細情報 - 専門サイト",
                "url": f"https://example.com/search/{query.replace(' ', '-')}",
                "snippet": f"{query}についての包括的な情報を提供します。最新の研究成果や実用的な知識を網羅。",
            },
            {
                "title": f"{query}の基礎知識 - 教育リソース",
                "url": f"https://education.example.com/{query}",
                "snippet": f"{query}の基本的な概念から応用まで、初心者にも分かりやすく解説。",
            },
            {
                "title": f"{query}の最新動向 - ニュースサイト",
                "url": f"https://news.example.com/latest/{query}",
                "snippet": f"{query}に関する最新のニュースと動向。業界の専門家による分析。",
            },
        ]

        results = mock_results[:num_results]
        execution_time = time.time() - start_time

        return ToolResult(
            success=True,
            result={"query": query, "results": results, "total_found": len(results)},
            tool_name=self.name,
            execution_time=execution_time,
        )


class FileOperationTool(AgentTool):
    """
    ファイル操作ツール（模擬実装）
    """

    def __init__(self):
        super().__init__(
            name="file_operation",
            description="ファイルの読み書き、作成、削除を行います。",
            parameters={
                "operation": {
                    "type": "string",
                    "description": "操作の種類（read, write, create, delete, list）",
                },
                "file_path": {"type": "string", "description": "ファイルパス"},
                "content": {
                    "type": "string",
                    "description": "書き込む内容（write/create操作の場合）",
                },
            },
        )
        # 仮想ファイルシステム（メモリ内）
        self.virtual_fs = {}

    def execute(
        self, operation: str, file_path: str, content: str = None
    ) -> ToolResult:
        start_time = time.time()
        self.usage_count += 1

        try:
            if operation == "read":
                if file_path in self.virtual_fs:
                    result = self.virtual_fs[file_path]
                else:
                    raise FileNotFoundError(f"ファイル {file_path} が見つかりません")

            elif operation == "write" or operation == "create":
                if content is None:
                    raise ValueError("書き込み内容が指定されていません")
                self.virtual_fs[file_path] = content
                result = f"ファイル {file_path} に書き込み完了"

            elif operation == "delete":
                if file_path in self.virtual_fs:
                    del self.virtual_fs[file_path]
                    result = f"ファイル {file_path} を削除しました"
                else:
                    raise FileNotFoundError(f"ファイル {file_path} が見つかりません")

            elif operation == "list":
                result = list(self.virtual_fs.keys())

            else:
                raise ValueError(f"不明な操作: {operation}")

            execution_time = time.time() - start_time
            return ToolResult(
                success=True,
                result=result,
                tool_name=self.name,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name,
                execution_time=execution_time,
            )


class DataAnalysisTool(AgentTool):
    """
    データ分析ツール（模擬実装）
    """

    def __init__(self):
        super().__init__(
            name="data_analysis",
            description="データの分析と統計を行います。",
            parameters={
                "data": {
                    "type": "array",
                    "description": "分析するデータ（数値のリスト）",
                },
                "analysis_type": {
                    "type": "string",
                    "description": "分析の種類（basic_stats, correlation, trend）",
                },
            },
        )

    def execute(
        self, data: List[float], analysis_type: str = "basic_stats"
    ) -> ToolResult:
        start_time = time.time()
        self.usage_count += 1

        try:
            if not data or not isinstance(data, list):
                raise ValueError("有効なデータが提供されていません")

            # 数値に変換
            numeric_data = [float(x) for x in data]

            if analysis_type == "basic_stats":
                result = {
                    "count": len(numeric_data),
                    "mean": sum(numeric_data) / len(numeric_data),
                    "min": min(numeric_data),
                    "max": max(numeric_data),
                    "sum": sum(numeric_data),
                }

                # 標準偏差を計算
                mean = result["mean"]
                variance = sum((x - mean) ** 2 for x in numeric_data) / len(
                    numeric_data
                )
                result["std_dev"] = variance**0.5

            elif analysis_type == "trend":
                if len(numeric_data) < 2:
                    raise ValueError(
                        "トレンド分析には少なくとも2つのデータポイントが必要"
                    )

                # 簡単な線形トレンド
                n = len(numeric_data)
                x_vals = list(range(n))

                # 最小二乗法で傾きを計算
                x_mean = sum(x_vals) / n
                y_mean = sum(numeric_data) / n

                numerator = sum(
                    (x_vals[i] - x_mean) * (numeric_data[i] - y_mean) for i in range(n)
                )
                denominator = sum((x - x_mean) ** 2 for x in x_vals)

                slope = numerator / denominator if denominator != 0 else 0

                result = {
                    "trend": "増加"
                    if slope > 0.1
                    else "減少"
                    if slope < -0.1
                    else "横ばい",
                    "slope": slope,
                    "data_points": n,
                }

            else:
                raise ValueError(f"未対応の分析タイプ: {analysis_type}")

            execution_time = time.time() - start_time
            return ToolResult(
                success=True,
                result=result,
                tool_name=self.name,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=self.name,
                execution_time=execution_time,
            )


class AutonomousAgent:
    """
    自律エージェントのメインクラス
    """

    def __init__(
        self,
        name: str = "Agent",
        max_iterations: int = 10,
        model_name: str = "gpt-3.5-turbo",
    ):
        self.name = name
        self.max_iterations = max_iterations
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,  # 一貫した意思決定のため低温度
        )

        # ===== ツールの初期化 =====
        self.tools = {
            "calculator": CalculatorTool(),
            "web_search": WebSearchTool(),
            "file_operation": FileOperationTool(),
            "data_analysis": DataAnalysisTool(),
        }

        # ===== 実行状態 =====
        self.current_goal = None
        self.execution_history = []
        self.working_memory = {}
        self.is_running = False
        self.iteration_count = 0

        # ===== 統計情報 =====
        self.total_executions = 0
        self.successful_executions = 0
        self.total_tools_used = 0

    def _get_available_tools_description(self) -> str:
        """
        利用可能ツールの説明を生成

        Returns:
            str: ツールの説明
        """
        descriptions = []
        for tool in self.tools.values():
            schema = tool.get_schema()
            param_desc = ", ".join(
                [
                    f"{param}: {info['description']}"
                    for param, info in schema["parameters"].items()
                ]
            )
            descriptions.append(
                f"- {schema['name']}: {schema['description']} パラメータ: {param_desc}"
            )

        return "\n".join(descriptions)

    def _parse_action_from_response(self, response: str) -> Dict[str, Any]:
        """
        LLMの応答からアクションを解析

        Args:
            response (str): LLMの応答

        Returns:
            Dict[str, Any]: 解析されたアクション
        """

        # ===== JSONブロックを探す =====
        json_pattern = r"```json\s*(\{.*?\})\s*```"
        json_match = re.search(json_pattern, response, re.DOTALL)

        if json_match:
            try:
                action_data = json.loads(json_match.group(1))
                return action_data
            except json.JSONDecodeError:
                pass

        # ===== フォールバック: キーワードベースの解析 =====
        if "FINAL_ANSWER:" in response:
            answer_match = re.search(r"FINAL_ANSWER:\s*(.*)", response, re.DOTALL)
            if answer_match:
                return {
                    "action": "final_answer",
                    "content": answer_match.group(1).strip(),
                }

        # ===== ツール使用パターンを探す =====
        tool_patterns = {
            "calculator": r'calculator.*?expression["\']?\s*:\s*["\']([^"\']+)["\']',
            "web_search": r'web_search.*?query["\']?\s*:\s*["\']([^"\']+)["\']',
            "file_operation": r'file_operation.*?operation["\']?\s*:\s*["\']([^"\']+)["\']',
            "data_analysis": r'data_analysis.*?analysis_type["\']?\s*:\s*["\']([^"\']+)["\']',
        }

        for tool_name, pattern in tool_patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return {
                    "action": "use_tool",
                    "tool": tool_name,
                    "parameters": {"expression": match.group(1)}
                    if tool_name == "calculator"
                    else {},
                }

        # ===== デフォルト: 思考として処理 =====
        return {"action": "think", "content": response}

    def _execute_action(self, action: Dict[str, Any]) -> ToolResult:
        """
        アクションを実行

        Args:
            action (Dict[str, Any]): 実行するアクション

        Returns:
            ToolResult: 実行結果
        """

        action_type = action.get("action", "unknown")

        if action_type == "use_tool":
            tool_name = action.get("tool")
            parameters = action.get("parameters", {})

            if tool_name in self.tools:
                print(f"🔧 ツール使用: {tool_name}")
                self.total_tools_used += 1
                return self.tools[tool_name].execute(**parameters)
            else:
                return ToolResult(
                    success=False,
                    error=f"不明なツール: {tool_name}",
                    tool_name=tool_name,
                )

        elif action_type == "think":
            print(f"💭 思考: {action.get('content', '')[:100]}...")
            return ToolResult(
                success=True, result="思考を記録しました", tool_name="思考"
            )

        elif action_type == "final_answer":
            print("🎯 最終回答生成")
            return ToolResult(
                success=True, result=action.get("content", ""), tool_name="最終回答"
            )

        else:
            return ToolResult(
                success=False,
                error=f"不明なアクション: {action_type}",
                tool_name="不明",
            )

    def execute_goal(self, goal: str, context: str = "") -> Dict[str, Any]:
        """
        目標を実行

        Args:
            goal (str): 実行する目標
            context (str): 追加のコンテキスト

        Returns:
            Dict[str, Any]: 実行結果
        """

        start_time = time.time()
        self.current_goal = goal
        self.is_running = True
        self.iteration_count = 0
        self.execution_history = []
        self.working_memory = {"goal": goal, "context": context}

        print(f"🚀 エージェント {self.name} が目標実行を開始")
        print(f"🎯 目標: {goal}")
        print(f"📝 コンテキスト: {context}")
        print("-" * 60)

        final_answer = None

        while self.is_running and self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            print(f"\n🔄 反復 {self.iteration_count}/{self.max_iterations}")

            iteration_start = time.time()

            # ===== システムプロンプトを構築 =====
            system_prompt = f"""
            あなたは自律的なAIエージェント「{self.name}」です。
            与えられた目標を達成するために、以下のツールを使用できます：
            
            {self._get_available_tools_description()}
            
            行動指針：
            1. 目標を分析し、必要なステップを特定する
            2. 適切なツールを選択して実行する
            3. 結果を評価し、次のアクションを決定する
            4. 目標が達成されたら FINAL_ANSWER: で最終回答を提供する
            
            アクションは以下の形式で指定してください：
            
            ```json
            {{
                "action": "use_tool",
                "tool": "tool_name",
                "parameters": {{"param1": "value1", "param2": "value2"}},
                "reasoning": "このツールを使用する理由"
            }}
            ```
            
            または最終回答の場合：
            FINAL_ANSWER: あなたの最終回答
            
            効率的で論理的なアプローチを心がけてください。
            """

            # ===== 現在の状況を含むユーザープロンプト =====
            user_prompt = f"""
            【目標】
            {goal}
            
            【コンテキスト】
            {context}
            
            【現在の状況】
            - 反復回数: {self.iteration_count}/{self.max_iterations}
            - 実行済みアクション数: {len(self.execution_history)}
            """

            # ===== 実行履歴があれば追加 =====
            if self.execution_history:
                user_prompt += "\n\n【これまでの実行履歴】\n"
                for i, entry in enumerate(
                    self.execution_history[-3:], 1
                ):  # 最新3件のみ
                    action_desc = entry.get("action_description", "不明なアクション")
                    result = entry.get("result", {})
                    if result.get("success"):
                        user_prompt += f"{i}. ✅ {action_desc} → {str(result.get('result', ''))[:100]}\n"
                    else:
                        user_prompt += f"{i}. ❌ {action_desc} → エラー: {result.get('error', '不明')}\n"

            user_prompt += f"""
            
            【作業メモリ】
            {json.dumps(self.working_memory, ensure_ascii=False, indent=2)}
            
            次に実行すべきアクションを決定してください。
            目標を達成するために最も効果的なアプローチを選択してください。
            """

            # ===== LLMに問い合わせ =====
            try:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]

                response = self.llm.invoke(messages)
                response_text = response.content

                print(f"🤖 エージェントの応答: {response_text[:200]}...")

                # ===== アクションを解析 =====
                action = self._parse_action_from_response(response_text)
                print(f"📋 解析されたアクション: {action.get('action', 'unknown')}")

                # ===== アクションを実行 =====
                result = self._execute_action(action)

                # ===== 履歴に記録 =====
                history_entry = {
                    "iteration": self.iteration_count,
                    "action": action,
                    "action_description": f"{action.get('action', 'unknown')} - {action.get('tool', action.get('content', '')[:50])}",
                    "result": result.to_dict(),
                    "llm_response": response_text,
                    "iteration_time": time.time() - iteration_start,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                self.execution_history.append(history_entry)

                # ===== 作業メモリを更新 =====
                if result.success and result.result:
                    memory_key = (
                        f"step_{self.iteration_count}_{action.get('action', 'unknown')}"
                    )
                    self.working_memory[memory_key] = result.result

                # ===== 最終回答チェック =====
                if action.get("action") == "final_answer":
                    final_answer = result.result
                    self.is_running = False
                    print("🎉 目標達成！最終回答を生成しました")
                    break

                # ===== 失敗が続く場合の対処 =====
                recent_failures = sum(
                    1
                    for entry in self.execution_history[-3:]
                    if not entry["result"]["success"]
                )

                if recent_failures >= 3:
                    print("⚠️ 連続する失敗が検出されました。アプローチを変更します。")
                    self.working_memory["retry_needed"] = True

                print(
                    f"⏱️ 反復 {self.iteration_count} 完了 ({time.time() - iteration_start:.2f}秒)"
                )

            except Exception as e:
                print(f"❌ 反復 {self.iteration_count} でエラー: {str(e)}")
                self.execution_history.append(
                    {
                        "iteration": self.iteration_count,
                        "error": str(e),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

        # ===== 実行完了 =====
        self.is_running = False
        total_time = time.time() - start_time

        # ===== 統計を更新 =====
        self.total_executions += 1
        if final_answer:
            self.successful_executions += 1

        # ===== 結果をまとめる =====
        result = {
            "goal": goal,
            "context": context,
            "completed_iterations": self.iteration_count,
            "max_iterations": self.max_iterations,
            "success": final_answer is not None,
            "final_answer": final_answer,
            "execution_history": self.execution_history,
            "working_memory": self.working_memory,
            "total_execution_time": total_time,
            "tools_used": self.total_tools_used,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        print(f"\n{'=' * 60}")
        print("🏁 エージェント実行完了")
        print(f"⏱️ 総実行時間: {total_time:.2f}秒")
        print(f"🔄 実行反復数: {self.iteration_count}")
        print(f"🛠️ 使用ツール数: {self.total_tools_used}")
        print(f"{'✅ 成功' if final_answer else '❌ 失敗'}")

        return result

    def get_agent_statistics(self) -> Dict[str, Any]:
        """
        エージェントの統計情報を取得

        Returns:
            Dict[str, Any]: 統計情報
        """

        # ===== ツール使用統計 =====
        tool_stats = {}
        for tool_name, tool in self.tools.items():
            tool_stats[tool_name] = {
                "usage_count": tool.usage_count,
                "description": tool.description,
            }

        success_rate = (
            (self.successful_executions / self.total_executions * 100)
            if self.total_executions > 0
            else 0
        )

        return {
            "agent_name": self.name,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "success_rate": success_rate,
            "total_tools_used": self.total_tools_used,
            "available_tools": len(self.tools),
            "tool_usage_statistics": tool_stats,
            "max_iterations": self.max_iterations,
        }


# ===== 使用例 =====
def main():
    """
    Agentsパターンのデモンストレーション
    """
    print("=== Agents パターン（自律エージェント）のデモ ===\n")

    # ===== エージェントのインスタンスを作成 =====
    agent = AutonomousAgent(name="ResearchBot", max_iterations=8)

    # ===== デモ1: 数学的問題解決 =====
    print("🧮 デモ1: 数学的問題解決エージェント")
    print("=" * 50)

    math_goal = """
    以下の数学的問題を解決してください：
    
    1. 複利計算: 元本100万円、年利3%、5年間の複利計算
    2. 統計分析: データ [10, 15, 20, 25, 30, 35, 40] の基本統計
    3. 最適化問題: 長方形の周囲の長さが20の時、面積を最大化する縦と横の長さ
    
    各問題を順番に解決し、最終的に全ての答えをまとめた報告書を作成してください。
    """

    result1 = agent.execute_goal(
        goal=math_goal,
        context="数学計算、統計分析、最適化問題を含む複合的な問題解決タスク",
    )

    print("\n📊 数学問題解決結果:")
    print(f"- 成功: {'はい' if result1['success'] else 'いいえ'}")
    print(f"- 実行反復数: {result1['completed_iterations']}")
    print(f"- 実行時間: {result1['total_execution_time']:.2f}秒")
    print(f"- 使用ツール数: {result1['tools_used']}")

    if result1["final_answer"]:
        print("\n🎯 最終回答:")
        print(
            result1["final_answer"][:500] + "..."
            if len(result1["final_answer"]) > 500
            else result1["final_answer"]
        )

    # ===== デモ2: 研究・情報収集エージェント =====
    print("\n\n🔍 デモ2: 研究・情報収集エージェント")
    print("=" * 50)

    research_goal = """
    「人工知能の最新トレンドと将来の影響」について包括的な調査を行い、
    以下の構成でレポートを作成してください：
    
    1. 現在のAI技術の主要トレンド
    2. 産業への影響と応用事例
    3. 社会への潜在的な影響（ポジティブ・ネガティブ）
    4. 将来の展望と予測
    5. 推奨される対策や戦略
    
    情報を収集し、分析し、構造化されたレポートとして提示してください。
    """

    # 新しいエージェントインスタンスを作成（独立した実行のため）
    research_agent = AutonomousAgent(name="ResearchAnalyst", max_iterations=6)

    result2 = research_agent.execute_goal(
        goal=research_goal,
        context="学術的レポート作成、情報収集と分析、将来予測を含む研究タスク",
    )

    print("\n📊 研究レポート作成結果:")
    print(f"- 成功: {'はい' if result2['success'] else 'いいえ'}")
    print(f"- 実行反復数: {result2['completed_iterations']}")
    print(f"- 実行時間: {result2['total_execution_time']:.2f}秒")
    print(f"- 使用ツール数: {result2['tools_used']}")

    if result2["final_answer"]:
        print("\n🎯 研究レポート:")
        print(
            result2["final_answer"][:600] + "..."
            if len(result2["final_answer"]) > 600
            else result2["final_answer"]
        )

    # ===== デモ3: ファイル管理・データ処理エージェント =====
    print("\n\n📁 デモ3: ファイル管理・データ処理エージェント")
    print("=" * 50)

    file_goal = """
    以下のタスクを実行してください：
    
    1. 売上データファイル "sales_data.csv" を作成し、以下の内容を保存：
       月,売上高,前年同月比
       1月,1000000,105
       2月,1200000,108
       3月,1100000,102
       4月,1300000,110
       5月,1250000,107
    
    2. データを読み込み、基本統計を計算
    3. トレンド分析を実行
    4. 結果をまとめたレポートファイル "sales_report.txt" を作成
    5. 作成されたファイルのリストを表示
    
    各ステップの結果を検証しながら進めてください。
    """

    # ファイル操作専用エージェント
    file_agent = AutonomousAgent(name="FileManager", max_iterations=10)

    result3 = file_agent.execute_goal(
        goal=file_goal,
        context="ファイル操作、データ処理、レポート生成を含む総合的なデータ管理タスク",
    )

    print("\n📊 ファイル管理タスク結果:")
    print(f"- 成功: {'はい' if result3['success'] else 'いいえ'}")
    print(f"- 実行反復数: {result3['completed_iterations']}")
    print(f"- 実行時間: {result3['total_execution_time']:.2f}秒")
    print(f"- 使用ツール数: {result3['tools_used']}")

    if result3["final_answer"]:
        print("\n🎯 ファイル管理結果:")
        print(
            result3["final_answer"][:500] + "..."
            if len(result3["final_answer"]) > 500
            else result3["final_answer"]
        )

    # ===== 全体統計の表示 =====
    print("\n\n📈 エージェント統計")
    print("=" * 40)

    agents = [agent, research_agent, file_agent]
    for i, ag in enumerate(agents, 1):
        stats = ag.get_agent_statistics()
        print(f"\n{i}. {stats['agent_name']}")
        print(f"   - 総実行回数: {stats['total_executions']}")
        print(f"   - 成功回数: {stats['successful_executions']}")
        print(f"   - 成功率: {stats['success_rate']:.1f}%")
        print(f"   - 総ツール使用回数: {stats['total_tools_used']}")

        print("   - ツール使用統計:")
        for tool_name, tool_stats in stats["tool_usage_statistics"].items():
            if tool_stats["usage_count"] > 0:
                print(f"     * {tool_name}: {tool_stats['usage_count']}回")


if __name__ == "__main__":
    main()
