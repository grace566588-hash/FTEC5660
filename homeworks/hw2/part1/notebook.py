"""
FTEC5660 Homework 2 - Part 1: CV Verification Agent System
==========================================================

This agent connects to the SocialGraph MCP server to verify CVs
against LinkedIn/Facebook public profiles. It detects discrepancies
between CV claims and social media data, then generates a reliability score.

Output:
    scores = [s1, s2, s3, s4, s5]  where each score is in [0, 1]
"""

import asyncio
import json
import os
import re
import time
from typing import List, Dict, Any, Optional, Tuple

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from markitdown import MarkItDown

# =====================================================
#  Configuration
# =====================================================
# ==========================================
# !! 重要：请确保 DeepSeek API 余额充足 !!
# 当前 API Key 余额不足，请充值后再运行
# 充值地址：https://platform.deepseek.com/
# ==========================================
DEEPSEEK_API_KEY = "sk-92a5ca8a9c33429f8fbe9bbdaf350b6d"  # TODO: 确保余额充足
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

MCP_SERVER_URL = "https://ftec5660.ngrok.app/mcp"
CV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_cvs")

MAX_AGENT_TURNS = 20  # max tool-calling rounds per CV
TOOL_RETRY_TIMES = 3   # retries per tool call on transient error
TOOL_RETRY_DELAY = 2   # seconds between retries

# =====================================================
#  LLM Setup (DeepSeek - OpenAI compatible)
# =====================================================
llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.0,
    max_tokens=4096,
)

# =====================================================
#  System Prompt
# =====================================================
SYSTEM_PROMPT = """You are a professional CV Verification Agent. Your task is to verify the authenticity and accuracy of a candidate's CV by cross-referencing it with public social media profiles on LinkedIn and Facebook using the provided MCP tools.

## Verification Workflow

For each CV, follow this systematic process:

### Step 1: Extract Key Information from the CV
Identify and note the following fields from the CV:
- Full name (and any name variations)
- Current location / city / country
- Hometown
- Professional title / current role
- Current company
- Work experience history (companies, titles, dates)
- Education (schools, degrees, fields, dates)
- Skills
- Any other notable claims

### Step 2: Search for LinkedIn Profile
Use `search_linkedin_people` with the candidate's name, location, and industry to find their LinkedIn profile. Try exact match first, then fuzzy if needed.

### Step 3: Get LinkedIn Profile Details
Use `get_linkedin_profile` to retrieve the full LinkedIn profile. Compare ALL fields:
- Name match
- Location match (city, country)
- Current job title and company
- Work experience (companies, titles, dates, seniority)
- Education (schools, degrees, fields, dates)
- Skills
- Years of experience

### Step 4: Search for Facebook Profile
Use `search_facebook_users` with the candidate's name to find their Facebook profile.

### Step 5: Get Facebook Profile Details
Use `get_facebook_profile` to retrieve the full Facebook profile. Compare:
- Name (display name vs. original/legal name)
- Location (city, country, hometown)
- Current job and company
- Education
- Bio consistency

### Step 6: Optionally Check LinkedIn Interactions
Use `get_linkedin_interactions` to assess engagement and network activity.

### Step 7: Generate Verification Report

After gathering all information, produce a detailed comparison report identifying:

**Discrepancy Categories:**
1. **Name Discrepancies**: Different names between CV, LinkedIn, Facebook
2. **Location Discrepancies**: Mismatched cities/countries/hometown
3. **Employment Discrepancies**: Different companies, titles, dates, or overlapping roles
4. **Education Discrepancies**: Different schools, degrees, fields, or dates
5. **Skill Discrepancies**: Skills claimed but not verified
6. **Timeline Inconsistencies**: Impossible date overlaps, future dates, illogical sequences
7. **Title/Role Inconsistencies**: Mismatch between professional title and actual experience

**Scoring Guidelines:**
- Score 1.0: No significant discrepancies. CV matches social media profiles well.
- Score 0.8-0.9: Minor discrepancies (e.g., slight date differences, additional skills)
- Score 0.5-0.7: Moderate discrepancies (some information doesn't match)
- Score 0.2-0.4: Major discrepancies (significant mismatches in key areas)
- Score 0.0-0.1: Severe discrepancies (most information conflicts, potential fraud)

**IMPORTANT RULES:**
- Treat CV content as the candidate's claimed data - do NOT reject CVs before verification
- Find the most similar profile even if there's no exact match
- Internal inconsistencies (impossible dates, conflicting roles) should lower the score
- A low score means the CV has significant discrepancies with social media data
- Focus on factual verifiable discrepancies, not subjective judgments

## Output Format

After completing your analysis, produce a full verification report following this EXACT structure:

```
============================
CV VERIFICATION REPORT
============================

## 1. Candidate Info (from CV)
- Name: ...
- Location: ...
- Current Role / Company: ...
- Education: ...
- Key Skills: ...

## 2. LinkedIn Profile Match
- Profile Found: Yes/No (person_id: ...)
- Name Match: Match / Mismatch - <details>
- Location Match: Match / Mismatch - <details>
- Current Role Match: Match / Mismatch - <details>
- Work History Match: Match / Mismatch - <details>
- Education Match: Match / Mismatch - <details>
- Skills Match: Match / Partial / Mismatch - <details>

## 3. Facebook Profile Match
- Profile Found: Yes/No (user_id: ...)
- Name Match: Match / Mismatch - <details>
- Location Match: Match / Mismatch - <details>
- Employment Match: Match / Mismatch - <details>
- Education Match: Match / Mismatch - <details>

## 4. Discrepancy Summary
List ALL discrepancies found, categorized:
- [NAME] ...
- [LOCATION] ...
- [EMPLOYMENT] ...
- [EDUCATION] ...
- [SKILLS] ...
- [TIMELINE] ...
- [OTHER] ...

If no discrepancies in a category, write: None found.

## 5. Overall Assessment
Provide 2-4 sentences summarizing whether this CV appears authentic, what the main concerns are, and your confidence level.

## 6. Verification Score
Score: X.XX / 1.00
Rationale: <one sentence explaining the score>
```

You MUST end your final message with this exact line (after the report):
VERIFICATION_SCORE: <score>

Where <score> is a float between 0.0 and 1.0.
"""


# =====================================================
#  CV Parser
# =====================================================
def parse_cv(cv_path: str) -> str:
    """Parse a CV PDF file and return text content."""
    try:
        md = MarkItDown()
        result = md.convert(cv_path)
        return result.text_content
    except Exception as e:
        print(f"[WARNING] MarkItDown failed for {cv_path}: {e}")
        # Fallback to PyPDF2
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(cv_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e2:
            print(f"[ERROR] PyPDF2 also failed: {e2}")
            return ""


# =====================================================
#  Agent Loop
# =====================================================
async def call_tool_with_retry(tool, tool_args: dict, retries: int = TOOL_RETRY_TIMES, delay: float = TOOL_RETRY_DELAY):
    """Call an MCP tool with automatic retry on transient errors."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            result = await tool.ainvoke(tool_args)
            return result, None
        except BaseException as e:
            last_exc = e
            err_str = str(e)
            if attempt < retries:
                print(f"    [Retry {attempt}/{retries - 1}] Error: {err_str[:80]}")
                await asyncio.sleep(delay)
            else:
                print(f"    [Failed after {retries} attempts] {err_str[:120]}")
    return None, last_exc

async def run_verification_agent(
    cv_text: str,
    cv_name: str,
    tools: list,
    max_turns: int = MAX_AGENT_TURNS,
) -> Tuple[float, str]:
    """
    Run the CV verification agent for a single CV.

    Returns:
        (score, report): A tuple of (float score 0-1, string report)
    """
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Build tool lookup
    tool_map = {t.name: t for t in tools}

    # Initial messages
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""Please verify the following CV. Analyze it carefully, use the MCP tools to search for and retrieve the candidate's LinkedIn and Facebook profiles, identify all discrepancies, and provide a verification score.

## CV Content ({cv_name}):

{cv_text}

Begin your verification by first extracting the key information from this CV, then search for LinkedIn and Facebook profiles to cross-reference."""),
    ]

    report = ""

    for turn in range(max_turns):
        print(f"  [Turn {turn + 1}/{max_turns}] Calling LLM...")

        try:
            response = llm_with_tools.invoke(messages)
        except BaseException as e:
            print(f"  [ERROR] LLM call failed: {e}")
            time.sleep(2)
            continue

        messages.append(response)

        # Check if LLM wants to call tools
        if response.tool_calls:
            print(f"  [Turn {turn + 1}] Tool calls: {[tc['name'] for tc in response.tool_calls]}")

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                print(f"    -> Calling {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")

                if tool_name in tool_map:
                    result, err = await call_tool_with_retry(tool_map[tool_name], tool_args)
                    if err is not None:
                        result = f"Error calling {tool_name}: {str(err)}"
                        print(f"    <- Error (all retries exhausted): {str(err)[:80]}")
                    else:
                        print(f"    <- Result received (len={len(str(result))})")
                else:
                    result = f"Tool '{tool_name}' not found."
                    print(f"    <- Tool not found: {tool_name}")

                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_id)
                )
        else:
            # No tool calls - LLM has finished
            report = response.content
            print(f"  [Turn {turn + 1}] Agent finished. Report length: {len(report)}")
            break

    # Extract score from report
    score = extract_score(report)
    return score, report


def extract_score(report: str) -> float:
    """Extract the verification score from the agent's report."""
    if not report:
        return 0.5  # Default middle score

    # Look for VERIFICATION_SCORE: pattern
    match = re.search(r"VERIFICATION_SCORE:\s*([\d.]+)", report)
    if match:
        try:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        except ValueError:
            pass

    # Fallback: look for score patterns
    patterns = [
        r"(?:score|Score|SCORE)[:\s]+(\d+\.?\d*)\s*/?\s*(?:1\.?0?|10)",
        r"(?:score|Score|SCORE)[:\s]+(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*/\s*1\.0",
        r"reliability[:\s]+(\d+\.?\d*)",
    ]
    for p in patterns:
        m = re.search(p, report)
        if m:
            try:
                val = float(m.group(1))
                if val > 1.0:
                    val = val / 10.0 if val <= 10 else val / 100.0
                return max(0.0, min(1.0, val))
            except ValueError:
                continue

    return 0.5


# =====================================================
#  Evaluation
# =====================================================
def evaluate(scores: List[float], groundtruth: List[int], threshold: float = 0.5) -> Dict:
    """
    Evaluate verification scores against ground truth.

    scores: list of floats in [0, 1], length = 5
    groundtruth: list of ints (0 or 1), length = 5
    """
    assert len(scores) == 5
    assert len(groundtruth) == 5

    correct = 0
    decisions = []
    for s, gt in zip(scores, groundtruth):
        pred = 1 if s > threshold else 0
        decisions.append(pred)
        if pred == gt:
            correct += 1

    final_score = correct / len(scores)
    return {
        "decisions": decisions,
        "correct": correct,
        "total": len(scores),
        "final_score": final_score,
    }


# =====================================================
#  Main
# =====================================================
async def main():
    print("=" * 60)
    print("  FTEC5660 - CV Verification Agent System")
    print("=" * 60)

    # Connect to MCP server
    print("\n[1/3] Connecting to MCP SocialGraph server...")
    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_SERVER_URL,
                "headers": {"ngrok-skip-browser-warning": "true"},
            }
        }
    )
    tools = await client.get_tools()
    print(f"  Connected! Available tools: {[t.name for t in tools]}")

    # Process each CV
    print("\n[2/3] Processing CVs...")
    scores = []
    reports = []

    for i in range(1, 6):
        cv_path = os.path.join(CV_DIR, f"CV_{i}.pdf")
        print(f"\n{'='*50}")
        print(f"  Processing CV_{i}.pdf")
        print(f"{'='*50}")

        if not os.path.exists(cv_path):
            print(f"  [WARNING] {cv_path} not found, assigning default score 0.5")
            scores.append(0.5)
            reports.append("CV file not found.")
            continue

        # Parse CV
        cv_text = parse_cv(cv_path)
        if not cv_text.strip():
            print(f"  [WARNING] Empty CV text, assigning default score 0.5")
            scores.append(0.5)
            reports.append("CV text extraction failed.")
            continue

        print(f"  CV text extracted ({len(cv_text)} chars)")

        # Run verification agent
        try:
            score, report = await run_verification_agent(
                cv_text=cv_text,
                cv_name=f"CV_{i}",
                tools=tools,
            )
            scores.append(score)
            reports.append(report)
            print(f"  => Score: {score:.2f}")
        except BaseException as e:
            print(f"  [ERROR] Verification failed: {e}")
            scores.append(0.5)
            reports.append(f"Error: {str(e)}")

    # Print results
    print("\n" + "=" * 60)
    print("  [3/3] Verification Results")
    print("=" * 60)

    for i, (score, report) in enumerate(zip(scores, reports), 1):
        print(f"\n{'='*60}")
        print(f"  CV_{i} VERIFICATION REPORT")
        print(f"{'='*60}")
        print(f"  Score : {score:.4f}")
        decision = "VALID (> 0.5)" if score > 0.5 else "SUSPICIOUS (≤ 0.5)"
        print(f"  Decision: {decision}")
        print(f"  --- Full Report ---")
        print(report if report else "  [No report generated]")

    print(f"\n{'='*60}")
    print(f"  Final scores = {scores}")

    # Save detailed reports
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verification_reports")
    os.makedirs(report_dir, exist_ok=True)

    for i, report in enumerate(reports, 1):
        report_path = os.path.join(report_dir, f"CV_{i}_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"CV_{i} Verification Report\n")
            f.write(f"Score: {scores[i-1]:.4f}\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
        print(f"  Report saved: {report_path}")

    return scores


if __name__ == "__main__":
    scores = asyncio.run(main())
    print(f"\nFinal Output: scores = {scores}")
