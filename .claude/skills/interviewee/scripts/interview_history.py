"""
面试历史管理脚本
用于记录和检索面试问题历史
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys

HISTORY_FILE = "interview_history.json"


def get_history_path() -> str:
    """获取历史文件路径"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "..", HISTORY_FILE)


def load_history() -> Dict[str, Any]:
    """加载面试历史"""
    history_path = get_history_path()
    if not os.path.exists(history_path):
        return {"sessions": []}

    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(history: Dict[str, Any]) -> None:
    """保存面试历史"""
    history_path = get_history_path()
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def add_question(question: str, answer: str, session_id: Optional[str] = None) -> str:
    """
    添加面试问题和回答到历史记录

    Args:
        question: 面试问题
        answer: 回答内容
        session_id: 会话ID（可选）

    Returns:
        会话ID
    """
    history = load_history()

    if not session_id:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    timestamp = datetime.now().isoformat()

    # 查找或创建会话
    session = None
    for s in history["sessions"]:
        if s["session_id"] == session_id:
            session = s
            break

    if not session:
        session = {
            "session_id": session_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "questions": [],
        }
        history["sessions"].append(session)

    # 添加问题和回答
    session["questions"].append({"question": question, "answer": answer, "timestamp": timestamp})

    save_history(history)
    return session_id


def search_questions(keyword: str) -> List[Dict[str, Any]]:
    """
    搜索包含关键词的问题

    Args:
        keyword: 搜索关键词

    Returns:
        匹配的问题列表
    """
    history = load_history()
    results = []

    for session in history["sessions"]:
        for q in session["questions"]:
            if keyword.lower() in q["question"].lower() or keyword.lower() in q["answer"].lower():
                results.append(
                    {
                        "session_id": session["session_id"],
                        "date": session["date"],
                        "question": q["question"],
                        "answer": q["answer"],
                    }
                )

    return results


def list_sessions() -> List[Dict[str, Any]]:
    """列出所有面试会话"""
    history = load_history()
    return [
        {"session_id": s["session_id"], "date": s["date"], "question_count": len(s["questions"])}
        for s in history["sessions"]
    ]


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """获取特定会话的详细信息"""
    history = load_history()
    for session in history["sessions"]:
        if session["session_id"] == session_id:
            return session
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python interview_history.py list              # 列出所有会话")
        print("  python interview_history.py search <keyword>  # 搜索问题")
        print("  python interview_history.py get <session_id>  # 获取会话详情")
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        sessions = list_sessions()
        for s in sessions:
            print(f"[{s['date']}] Session {s['session_id']}: {s['question_count']} questions")

    elif command == "search":
        if len(sys.argv) < 3:
            print("Please provide a keyword")
            sys.exit(1)
        keyword = sys.argv[2]
        results = search_questions(keyword)
        for r in results:
            print(f"\n[{r['date']}] Session {r['session_id']}")
            print(f"Q: {r['question']}")
            print(f"A: {r['answer'][:100]}...")

    elif command == "get":
        if len(sys.argv) < 3:
            print("Please provide a session_id")
            sys.exit(1)
        session_id = sys.argv[2]
        session = get_session(session_id)
        if session:
            print(f"\nSession: {session['session_id']}")
            print(f"Date: {session['date']}")
            print(f"Questions: {len(session['questions'])}")
            for i, q in enumerate(session["questions"], 1):
                print(f"\n--- Question {i} ---")
                print(f"Q: {q['question']}")
                print(f"A: {q['answer']}")
        else:
            print(f"Session {session_id} not found")

    else:
        print(f"Unknown command: {command}")
