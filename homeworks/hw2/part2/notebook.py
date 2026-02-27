"""
FTEC5660 Homework 2 - Part 2: Moltbook Social Agent
====================================================

This agent interacts with the Moltbook platform (a Reddit-like social network 
for AI agents) via its REST API. Tasks:
1. Authenticate with API key
2. Subscribe to /m/ftec5660
3. Like (upvote) and comment on a specific post

Usage:
    python homework2_part2.py
"""

import json
import os
import re
import time
import requests
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)

# =====================================================
#  Configuration
# =====================================================
# DeepSeek API 配置（确保余额充足）
DEEPSEEK_API_KEY = "sk-92a5ca8a9c33429f8fbe9bbdaf350b6d"  # TODO: 确保余额充足
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# ==========================================
# !! 重要：请按照以下步骤配置 Moltbook API Key !!
# 1. 打开 Google Colab 启动代码
# 2. 运行 encode_student_id() 得到编码后的学生ID
# 3. 运行注册代码获取 moltbook_sk_xxx 格式的 API Key
# 4. 访问 claim_url 完成认证
# 5. 将 API Key 粘贴到下方
# ==========================================
MOLTBOOK_API_KEY = "moltbook_sk_w-my4MQDLnLyxuvvM5kWf68Fb3oIwpkp"  # Moltbook API Key (agent: ql_69204331)
MOLTBOOK_BASE_URL = "https://www.moltbook.com/api/v1"

# Student ID and target post
STUDENT_ID = 1155249660
TARGET_POST_ID = "47ff50f3-8255-4dee-87f4-2c3637c7351c"
TARGET_SUBMOLT = "ftec5660"

MAX_AGENT_TURNS = 12


# =====================================================
#  Student ID Encoder
# =====================================================
def encode_student_id(student_id: int) -> str:
    """
    Reversibly encode a student ID using an affine cipher.
    E(x) = (a * x + b) mod m
    """
    a = 11
    b = 10025379
    m = 10**10
    return str((a * student_id + b) % m)


# =====================================================
#  Moltbook API Headers
# =====================================================
HEADERS = {
    "Authorization": f"Bearer {MOLTBOOK_API_KEY}",
    "Content-Type": "application/json",
}


# =====================================================
#  Moltbook Tool Definitions (for LangChain Agent)
# =====================================================
@tool
def get_agent_profile() -> dict:
    """Get the current agent's Moltbook profile and status."""
    r = requests.get(
        f"{MOLTBOOK_BASE_URL}/agents/me",
        headers=HEADERS,
        timeout=15,
    )
    return r.json()


@tool
def check_claim_status() -> dict:
    """Check if the agent has been claimed by a human."""
    r = requests.get(
        f"{MOLTBOOK_BASE_URL}/agents/status",
        headers=HEADERS,
        timeout=15,
    )
    return r.json()


@tool
def get_feed(sort: str = "new", limit: int = 10) -> dict:
    """Fetch the Moltbook feed. Sort options: hot, new, top, rising."""
    r = requests.get(
        f"{MOLTBOOK_BASE_URL}/feed",
        headers=HEADERS,
        params={"sort": sort, "limit": limit},
        timeout=15,
    )
    return r.json()


@tool
def get_post(post_id: str) -> dict:
    """Get a single post by its ID."""
    r = requests.get(
        f"{MOLTBOOK_BASE_URL}/posts/{post_id}",
        headers=HEADERS,
        timeout=15,
    )
    return r.json()


@tool
def upvote_post(post_id: str) -> dict:
    """Upvote (like) a post by its ID."""
    r = requests.post(
        f"{MOLTBOOK_BASE_URL}/posts/{post_id}/upvote",
        headers=HEADERS,
        timeout=15,
    )
    return r.json()


@tool
def add_comment(post_id: str, content: str) -> dict:
    """Add a comment on a post. Returns verification challenge if required."""
    r = requests.post(
        f"{MOLTBOOK_BASE_URL}/posts/{post_id}/comments",
        headers=HEADERS,
        json={"content": content},
        timeout=15,
    )
    return r.json()


@tool
def subscribe_to_submolt(submolt_name: str) -> dict:
    """Subscribe to a submolt (community) by its name."""
    r = requests.post(
        f"{MOLTBOOK_BASE_URL}/submolts/{submolt_name}/subscribe",
        headers=HEADERS,
        timeout=15,
    )
    return r.json()


@tool
def get_submolt_info(submolt_name: str) -> dict:
    """Get information about a submolt by its name."""
    r = requests.get(
        f"{MOLTBOOK_BASE_URL}/submolts/{submolt_name}",
        headers=HEADERS,
        timeout=15,
    )
    return r.json()


@tool
def list_submolts() -> dict:
    """List all available submolts on Moltbook."""
    r = requests.get(
        f"{MOLTBOOK_BASE_URL}/submolts",
        headers=HEADERS,
        timeout=15,
    )
    return r.json()


@tool
def submit_verification(verification_code: str, answer: str) -> dict:
    """
    Submit an answer to a verification challenge.
    The answer should be a number with 2 decimal places (e.g., '15.00').
    """
    r = requests.post(
        f"{MOLTBOOK_BASE_URL}/verify",
        headers=HEADERS,
        json={
            "verification_code": verification_code,
            "answer": answer,
        },
        timeout=15,
    )
    return r.json()


@tool
def search_moltbook(query: str, search_type: str = "all", limit: int = 20) -> dict:
    """
    Semantic search across Moltbook posts and comments.
    search_type: 'posts', 'comments', or 'all'.
    """
    r = requests.get(
        f"{MOLTBOOK_BASE_URL}/search",
        headers=HEADERS,
        params={"q": query, "type": search_type, "limit": limit},
        timeout=15,
    )
    return r.json()


@tool
def get_post_comments(post_id: str, sort: str = "new") -> dict:
    """Get comments on a post. Sort: top, new, controversial."""
    r = requests.get(
        f"{MOLTBOOK_BASE_URL}/posts/{post_id}/comments",
        headers=HEADERS,
        params={"sort": sort},
        timeout=15,
    )
    return r.json()


# =====================================================
#  Verification Challenge Solver
# =====================================================
def solve_verification_challenge(challenge_text: str) -> Optional[str]:
    """
    Solve the obfuscated math verification challenge.
    
    The challenge is an obfuscated math word problem with two numbers 
    and one operation (+, -, *, /). Example:
    "A] lO^bSt-Er S[wImS aT/ tW]eNn-Tyy mE^tE[rS aNd] SlO/wS bY^ fI[vE"
    → "A lobster swims at twenty meters and slows by five" 
    → 20 - 5 = 15.00
    """
    # Use LLM to solve the challenge
    solver_llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.0,
    )

    prompt = f"""You are solving an obfuscated math verification challenge from Moltbook.

The challenge text uses alternating caps, scattered symbols (^, [, ], /, -), and broken words to hide a simple math problem with two numbers and one operation (+, -, *, /).

Challenge text: "{challenge_text}"

Instructions:
1. First, clean up the text by removing symbols and fixing the alternating caps
2. Identify the two numbers (they may be written as words like "twenty", "five", etc.)
3. Identify the operation (addition, subtraction, multiplication, or division)
4. Compute the answer
5. Return ONLY the numeric answer with exactly 2 decimal places (e.g., "15.00", "-3.50", "84.00")

Your answer (number only with 2 decimal places):"""

    response = solver_llm.invoke([HumanMessage(content=prompt)])
    answer_text = response.content.strip()

    # Extract number from response
    match = re.search(r"(-?\d+\.?\d*)", answer_text)
    if match:
        num = float(match.group(1))
        return f"{num:.2f}"

    return answer_text


# =====================================================
#  LLM Setup
# =====================================================
llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.3,
    max_tokens=2048,
)

# =====================================================
#  System Prompt
# =====================================================
SYSTEM_PROMPT = f"""You are an autonomous Moltbook Social Agent. Your mission is to interact with the Moltbook platform (a Reddit-like social network for AI agents) and complete the following tasks:

## Tasks (in order):
1. **Check your profile** - Verify your agent identity and claim status
2. **Subscribe to /m/ftec5660** - Subscribe to the ftec5660 submolt
3. **Get the target post** - Retrieve the specific post: {TARGET_POST_ID}
4. **Upvote the post** - Like the target post
5. **Comment on the post** - Add a thoughtful comment on the target post

## Important Notes:
- After creating content (comments), you may receive a verification challenge
- The verification challenge is an obfuscated math word problem
- Use the `submit_verification` tool to solve challenges
- If you get a verification challenge, solve the math problem and submit the answer as a number with 2 decimal places (e.g., "15.00")
- Always use `submit_verification` with the verification_code from the response and your answer

## Response to verification challenges:
When you see a verification object in a response, DO NOT IGNORE IT:
1. Read the `challenge_text` carefully
2. Remove the scattered symbols and fix alternating caps to reveal the math problem
3. Compute the answer
4. Call `submit_verification` with the `verification_code` and your answer formatted as "X.XX"

Proceed step by step and report the result of each action.
"""


# =====================================================
#  Agent Loop
# =====================================================
def run_agent(instruction: str, max_turns: int = MAX_AGENT_TURNS) -> str:
    """Run the Moltbook agent loop."""

    # All available tools
    all_tools = [
        get_agent_profile,
        check_claim_status,
        get_feed,
        get_post,
        upvote_post,
        add_comment,
        subscribe_to_submolt,
        get_submolt_info,
        list_submolts,
        submit_verification,
        search_moltbook,
        get_post_comments,
    ]

    llm_with_tools = llm.bind_tools(all_tools)
    tool_map = {t.name: t for t in all_tools}

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=instruction),
    ]

    for turn in range(max_turns):
        print(f"\n[Turn {turn + 1}/{max_turns}]")

        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            print(f"  [ERROR] LLM call failed: {e}")
            time.sleep(2)
            continue

        messages.append(response)

        # Print LLM's text content
        if response.content:
            content_text = response.content if isinstance(response.content, str) else str(response.content)
            if content_text.strip():
                print(f"  [Agent]: {content_text[:500]}")

        # Process tool calls
        if response.tool_calls:
            print(f"  [Tool calls]: {[tc['name'] for tc in response.tool_calls]}")

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc["id"]

                print(f"    -> {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:200]})")

                try:
                    result = tool_map[tool_name].invoke(tool_args)
                    result_str = json.dumps(result, ensure_ascii=False, default=str) if isinstance(result, dict) else str(result)
                    print(f"    <- Result: {result_str[:300]}")

                    # Auto-handle verification challenges
                    if isinstance(result, dict):
                        verification = None

                        # Check nested structures for verification
                        for key in ["post", "comment", "submolt"]:
                            if key in result and isinstance(result[key], dict):
                                if "verification" in result[key]:
                                    verification = result[key]["verification"]
                                    break

                        if "verification" in result and isinstance(result["verification"], dict):
                            verification = result["verification"]

                        if verification and "challenge_text" in verification:
                            print(f"\n  [VERIFICATION CHALLENGE DETECTED]")
                            print(f"    Challenge: {verification['challenge_text']}")

                            answer = solve_verification_challenge(verification["challenge_text"])
                            print(f"    Computed answer: {answer}")

                            verify_result = submit_verification.invoke({
                                "verification_code": verification["verification_code"],
                                "answer": answer,
                            })
                            print(f"    Verification result: {json.dumps(verify_result, ensure_ascii=False)}")

                            # Add verification result as additional context
                            result_str += f"\n\n[AUTO-VERIFICATION] Submitted answer '{answer}'. Result: {json.dumps(verify_result, ensure_ascii=False)}"

                except Exception as e:
                    result_str = f"Error: {str(e)}"
                    print(f"    <- Error: {e}")

                messages.append(ToolMessage(content=result_str, tool_call_id=tool_id))
        else:
            # No tool calls - agent finished
            final_content = response.content if isinstance(response.content, str) else str(response.content)
            print(f"\n[Agent finished]")
            return final_content

    return "Agent reached maximum turns."


# =====================================================
#  Direct API Operations (Fallback)
# =====================================================
def direct_subscribe(submolt_name: str) -> dict:
    """Directly subscribe to a submolt."""
    r = requests.post(
        f"{MOLTBOOK_BASE_URL}/submolts/{submolt_name}/subscribe",
        headers=HEADERS,
        timeout=15,
    )
    return r.json()


def direct_upvote(post_id: str) -> dict:
    """Directly upvote a post."""
    r = requests.post(
        f"{MOLTBOOK_BASE_URL}/posts/{post_id}/upvote",
        headers=HEADERS,
        timeout=15,
    )
    return r.json()


def direct_comment(post_id: str, content: str) -> dict:
    """Directly add a comment."""
    r = requests.post(
        f"{MOLTBOOK_BASE_URL}/posts/{post_id}/comments",
        headers=HEADERS,
        json={"content": content},
        timeout=15,
    )
    return r.json()


def direct_verify(verification_code: str, challenge_text: str) -> dict:
    """Solve verification challenge and submit."""
    answer = solve_verification_challenge(challenge_text)
    print(f"  Verification answer: {answer}")

    r = requests.post(
        f"{MOLTBOOK_BASE_URL}/verify",
        headers=HEADERS,
        json={
            "verification_code": verification_code,
            "answer": answer,
        },
        timeout=15,
    )
    return r.json()


# =====================================================
#  Main Execution
# =====================================================
def main():
    encoded_id = encode_student_id(STUDENT_ID)
    print("=" * 60)
    print("  FTEC5660 - Moltbook Social Agent")
    print("=" * 60)
    print(f"  Student ID: {STUDENT_ID}")
    print(f"  Encoded ID: {encoded_id}")
    print(f"  Agent naming format: nickname_{encoded_id}")
    print(f"  Target submolt: /m/{TARGET_SUBMOLT}")
    print(f"  Target post: {TARGET_POST_ID}")
    print()

    # Step 1: Check authentication
    print("[Step 1] Checking agent authentication...")
    try:
        profile = requests.get(
            f"{MOLTBOOK_BASE_URL}/agents/me",
            headers=HEADERS,
            timeout=15,
        ).json()
        print(f"  Profile: {json.dumps(profile, indent=2, ensure_ascii=False)[:500]}")
    except Exception as e:
        print(f"  [ERROR] Auth check failed: {e}")
        print("  Please verify your MOLTBOOK_API_KEY is correct.")
        return

    # Step 2: Subscribe to /m/ftec5660
    print("\n[Step 2] Subscribing to /m/ftec5660...")
    try:
        sub_result = direct_subscribe(TARGET_SUBMOLT)
        print(f"  Result: {json.dumps(sub_result, ensure_ascii=False)}")
    except Exception as e:
        print(f"  [ERROR] Subscribe failed: {e}")

    # Step 3: Get the target post
    print(f"\n[Step 3] Getting target post {TARGET_POST_ID}...")
    try:
        post = requests.get(
            f"{MOLTBOOK_BASE_URL}/posts/{TARGET_POST_ID}",
            headers=HEADERS,
            timeout=15,
        ).json()
        print(f"  Post: {json.dumps(post, indent=2, ensure_ascii=False)[:500]}")
    except Exception as e:
        print(f"  [ERROR] Get post failed: {e}")

    # Step 4: Upvote the post
    print(f"\n[Step 4] Upvoting the target post...")
    try:
        upvote_result = direct_upvote(TARGET_POST_ID)
        print(f"  Result: {json.dumps(upvote_result, ensure_ascii=False)}")
    except Exception as e:
        print(f"  [ERROR] Upvote failed: {e}")

    # Step 5: Comment on the post
    print(f"\n[Step 5] Commenting on the target post...")
    comment_text = (
        f"Hello from FTEC5660! This is an autonomous agent comment. "
        f"I'm exploring the Moltbook platform as part of my coursework on "
        f"Agentic AI for Business and FinTech. Excited to be part of this community!"
    )
    try:
        comment_result = direct_comment(TARGET_POST_ID, comment_text)
        print(f"  Result: {json.dumps(comment_result, indent=2, ensure_ascii=False)[:500]}")

        # Handle verification challenge if present
        if isinstance(comment_result, dict):
            verification = None

            # Check for verification in nested structures
            for key in ["comment", "post"]:
                if key in comment_result and isinstance(comment_result[key], dict):
                    if "verification" in comment_result[key]:
                        verification = comment_result[key]["verification"]
                        break

            if "verification" in comment_result and isinstance(comment_result["verification"], dict):
                verification = comment_result["verification"]

            if verification and "challenge_text" in verification:
                print(f"\n  [VERIFICATION REQUIRED]")
                print(f"  Challenge: {verification['challenge_text']}")

                verify_result = direct_verify(
                    verification["verification_code"],
                    verification["challenge_text"],
                )
                print(f"  Verification result: {json.dumps(verify_result, ensure_ascii=False)}")
            elif comment_result.get("verification_required"):
                # Check if verification info is at top level
                if "verification" in comment_result:
                    v = comment_result["verification"]
                    if "challenge_text" in v:
                        print(f"\n  [VERIFICATION REQUIRED]")
                        print(f"  Challenge: {v['challenge_text']}")

                        verify_result = direct_verify(
                            v["verification_code"],
                            v["challenge_text"],
                        )
                        print(f"  Verification result: {json.dumps(verify_result, ensure_ascii=False)}")
    except Exception as e:
        print(f"  [ERROR] Comment failed: {e}")

    # Step 6: Run LLM agent for additional tasks (optional)
    print(f"\n[Step 6] Running LLM agent for intelligent interaction...")
    try:
        result = run_agent(
            f"""Complete the following tasks on Moltbook:
1. Check your profile status
2. Verify that you are subscribed to /m/{TARGET_SUBMOLT}
3. Confirm the upvote on post {TARGET_POST_ID}
4. Check if your comment was published on post {TARGET_POST_ID}

Report the status of each task.""",
            max_turns=8,
        )
        print(f"\n  Agent report: {result[:1000]}")
    except Exception as e:
        print(f"  [NOTE] LLM agent step skipped: {e}")

    print("\n" + "=" * 60)
    print("  All tasks completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
