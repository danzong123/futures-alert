"""
Global news module - collect market context using verified AKShare APIs.
"""
import akshare as ak
from datetime import datetime
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def collect_global_context() -> Dict:
    """Collect global market background (news headlines, etc.)"""
    context = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "news_headlines": [],
        "summary": "",
    }

    # Tonghuashun global flash news - includes futures-related news
    try:
        news_df = ak.stock_info_global_ths()
        if not news_df.empty and "title" in news_df.columns:
            context["news_headlines"] = news_df["title"].head(6).tolist()
    except Exception as e:
        logger.warning(f"Failed to get Tonghuashun news: {e}")

    # Summary
    if context["news_headlines"]:
        context["summary"] = f"Finance headlines: {len(context['news_headlines'])} items"
    else:
        context["summary"] = "No news available"

    return context


def get_context_summary_text(context: Dict) -> str:
    """Generate global context text"""
    lines = [f"Global Context ({context['timestamp']})", ""]
    if context["news_headlines"]:
        lines.append("[Finance Headlines]")
        for i, title in enumerate(context["news_headlines"], 1):
            lines.append(f"  {i}. {title}")
    return "\n".join(lines)
