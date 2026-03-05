"""
demo.py
-------
Integration demo showing the custom tools being called in an "agent/workflow" style.

Run:
  python demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import datetime
import re



from tool import Tool, get_default_tools



def extract_dates_from_text(text: str, current_year: int = 2026) -> List[str]:
    """
    Extract potential date strings from text and normalize them for date_normalizer tool.
    
    Looks for patterns like:
    - "28 February" -> "28 February 2026"
    - "March 5" -> "March 5 2026" 
    - "02/28" -> "02/28/2026"
    """
    dates = []
    
    # Pattern 1: "28 February" (day month)
    day_month_pattern = r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\b'
    for match in re.finditer(day_month_pattern, text, re.IGNORECASE):
        day, month = match.groups()
        dates.append(f"{day} {month} {current_year}")
    
    # Pattern 2: "February 28" (month day) 
    month_day_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})\b'
    for match in re.finditer(month_day_pattern, text, re.IGNORECASE):
        month, day = match.groups()
        dates.append(f"{month} {day} {current_year}")
    
    # Pattern 3: "02/28" or "2/28" (MM/DD or DD/MM)
    numeric_pattern = r'\b(\d{1,2})/(\d{1,2})\b'
    for match in re.finditer(numeric_pattern, text):
        part1, part2 = match.groups()
        dates.append(f"{part1}/{part2}/{current_year}")
    
    return dates



@dataclass
class SimpleAgent:
    """
    A minimal agent that selects tools manually via lightweight routing rules.

    This intentionally avoids external LLM dependencies while demonstrating:
    - Tool registration
    - Tool selection
    - Successful execution
    - Structured error handling on bad input
    """

    tools: List[Tool]

    def get_tool(self, name: str) -> Tool:
        for t in self.tools:
            if t.name == name:
                return t
        raise ValueError(f"Unknown tool: {name}")

    def route(self, task: str) -> str:
        t = task.lower()
        if any(k in t for k in ["date", "when", "normalize", "yyyy", "mm/", "yesterday", "tomorrow", "today"]):
            return "date_normalizer"
        if any(k in t for k in ["keyword", "topics", "key terms", "extract terms"]):
            return "keyword_extractor"
        if any(k in t for k in ["stats", "readability", "word count", "sentence"]):
            return "text_statistics"
        # default to keywords for news/business text
        return "keyword_extractor"

    def run(self, task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = self.route(task)
        tool = self.get_tool(tool_name)
        result = tool.execute(**payload)
        return {"task": task, "selected_tool": tool.name, "result": result}


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    tools = get_default_tools()
    agent = SimpleAgent(tools=tools)

    article = """International Markets Shake After “Ghost” IPO Rumor – USD/€ Drops 3.5% 📉

LONDON — A mysterious “SourceTech” startup reportedly filed for a $3 billion IPO on 31 June 2025 (nonsense date) just hours before the London Stock Exchange suspended all trading.

Company spokeswoman Jane Doe said, “We expect to close the round by 29‑02‑2024,” which is also impossible (Feb 29 exists only on leap years).

Analysts noted an earnings estimate of -€5.2M (negative revenue?), 12e3 employees, and a novel pricing scheme: “$0.00 (free)” – is that even a number?

Highlights:

Quarterly growth: 110% ↑ vs. last quarter
“Next quarter” guidance pegged at Q13 2026 🧮
CEO tweet: “#Growth 🚀🚀🚀” (hashtags, emojis)
The press release included a raw HTML snippet: <div>Alert: server overload</div> and malformed URL http//invalid[dot]com.

Contact info: +44 (0)20 7946 0958, email: info@source-tech..com

Published: “2025/02/31” —
Updated: “the day before yesterday”"""

    # Extract dates from article for dynamic demo
    current_date = datetime.date.today()  # Use current date as reference
    extracted_dates = extract_dates_from_text(article, current_year=2026)
    
    import os
    # debug: show current file path and beginning of file that Python reads
    print(f"Running script file: {__file__}")
    try:
        with open(__file__, 'r') as f:
            print('File head---')
            for _ in range(10):
                print(f.readline().rstrip())
            print('---end head')
    except Exception as e:
        print('Could not read file:', e)
    print(f"Article: {article}")
    print(f"Extracted dates: {extracted_dates}")
    print()

    # 1) Successful executions
    print_section("DEMO: Successful tool calls via agent routing")

    out1 = agent.run("Extract key terms from this news snippet", {"text": article, "top_k": 8})
    print("Selected tool:", out1["selected_tool"])
    print("Result:", out1["result"])

    out2 = agent.run("Get text stats and readability", {"text": article})
    print("\nSelected tool:", out2["selected_tool"])
    print("Result:", out2["result"])

    # Normalize extracted dates from the article
    if extracted_dates:
        extracted_date = extracted_dates[0]  # Use first extracted date
        out3 = agent.run(f"Normalize the date '{extracted_date}' extracted from article", {"text": extracted_date, "dayfirst": False})
        print("\nSelected tool:", out3["selected_tool"])
        print("Result:", out3["result"])
    else:
        print("\nNo dates found in article to normalize")

    out4 = agent.run("Normalize the date expression 'yesterday' using current date as reference", {"text": "yesterday", "reference_date": current_date.isoformat()})
    print("\nSelected tool:", out4["selected_tool"])
    print("Result:", out4["result"])

    # 2) Error handling: problematic elements from the article
    print_section("DEMO: Error handling from article content")

    # Try to normalize invalid dates from the article
    invalid_date1 = agent.run("Normalize invalid date 'February 30, 2026' from article", {"text": "February 30, 2026", "dayfirst": False})
    print("Invalid date (Feb 30 doesn't exist) ->", invalid_date1["result"])

    invalid_date2 = agent.run("Normalize invalid date '02/30/2026' from article", {"text": "02/30/2026", "dayfirst": False})
    print("\nInvalid date (02/30 doesn't exist) ->", invalid_date2["result"])

    # Try to normalize date with time and timezone
    complex_date = agent.run("Normalize complex date '02/30/2026 at 4:30 PM PST' from article", {"text": "02/30/2026 at 4:30 PM PST", "dayfirst": False})
    print("\nComplex date with time/timezone ->", complex_date["result"])

    # Try to extract keywords from scientific notation
    scientific_notation = agent.run("Extract keywords from scientific notation '2.341e11'", {"text": "2.341e11", "top_k": 5})
    print("\nScientific notation text ->", scientific_notation["result"])

    # Try to get stats from text with emojis
    emoji_text = agent.run("Get stats from text with emojis '🚀🚀🚀'", {"text": "🚀🚀🚀"})
    print("\nText with only emojis ->", emoji_text["result"])



if __name__ == "__main__":
    main()

