# agent.py
import os
import re
import json
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from datetime import datetime, timedelta
import calendar

import pandas as pd
import requests
from pydantic import BaseModel, Field

from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from langgraph.graph import StateGraph, END

from settings import (
    OPENAI_API_KEY,
    APP_SCRIPT_URL,
    PERSIST_DIR,
    MODEL,
)

logger = logging.getLogger("uvicorn.error")

# Simple in-memory chat history
CONVERSATION_HISTORY: List[Dict[str, str]] = []
CONTEXT = {"last_sheet": None}

# -------- Helpers --------
def _normalize_name(name: str) -> str:
    return (
        str(name).lower()
        .replace("sheet", "")
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )

def _safe_float(x: Any) -> float:
    try:
        if x in (None, ""):
            return 0.0
        return float(str(x).replace(",", "").strip())
    except Exception:
        return 0.0

def _normalize_status(val: Any) -> str:
    val = str(val).strip().lower()
    if val == "" or val in ("pending", "not done", "todo", "incomplete"):
        return "pending"
    if val in ("done", "completed", "finished"):
        return "done"
    return val

def _date_range(keyword: str) -> Tuple[datetime.date, datetime.date]:
    today = datetime.now().date()
    kw = keyword.strip().lower()
    if kw == "today":
        return today, today
    if kw == "yesterday":
        y = today - timedelta(days=1)
        return y, y
    if kw in ("this week", "current week"):
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
        return start, end
    if kw == "last week":
        end = today - timedelta(days=today.weekday() + 1)
        start = end - timedelta(days=6)
        return start, end
    if kw in ("this month", "current month"):
        start = today.replace(day=1)
        _, last_day = calendar.monthrange(today.year, today.month)
        end = today.replace(day=last_day)
        return start, end
    if kw == "last month":
        first = today.replace(day=1)
        last_month_end = first - timedelta(days=1)
        start = last_month_end.replace(day=1)
        end = last_month_end
        return start, end
    return today, today

# -------- Router schema --------
class Condition(BaseModel):
    column: str
    op: Literal["==", "!=", ">", ">=", "<", "<=", "contains", "between"] = "=="
    value: Any

class RoutePlan(BaseModel):
    intent: Literal["tabular_agg", "semantic_answer", "lookup"]
    sheet: Optional[str] = None
    operation: Optional[Literal["COUNT", "SUM", "AVG", "MIN", "MAX"]] = None
    target_column: Optional[str] = None
    conditions: List[Condition] = []
    lookup_values: Optional[List[str]] = None

# -------- Agent --------
class SheetRAGAgent:
    def __init__(self) -> None:
        self._index = None
        self._parser = SimpleNodeParser()
        os.makedirs(PERSIST_DIR, exist_ok=True)
        self._graph = self._build_graph()

    # --- Data & Index ---
    def fetch_sheet_data(self) -> Dict[str, List[Dict[str, Any]]]:
        resp = requests.get(APP_SCRIPT_URL, timeout=40)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError("Apps Script must return {sheetName: [rows...]}")
        return data

    def _docs_from_data(self, data: Dict[str, List[Dict[str, Any]]]) -> List[Document]:
        docs: List[Document] = []
        for sheet, rows in data.items():
            for i, row in enumerate(rows):
                docs.append(
                    Document(
                        text=json.dumps(row, ensure_ascii=False),
                        metadata={**row, "sheetName": sheet},
                        doc_id=f"{sheet}-row-{i+1}",
                    )
                )
        return docs

    def _load_index_if_exists(self) -> bool:
        docstore = Path(PERSIST_DIR) / "docstore.json"
        if not docstore.exists():
            return False
        try:
            storage_ctx = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            self._index = load_index_from_storage(storage_ctx)
            logger.info("[Index] Loaded from disk.")
            return True
        except Exception as e:
            logger.warning("[Index] Failed to load, rebuilding. Reason: %s", e)
            return False

    def ensure_index(self) -> None:
        if self._index is not None:
            return
        if self._load_index_if_exists():
            return
        self.rebuild_index()

    def rebuild_index(self) -> None:
        logger.info("[Index] Building fresh from Google Sheets…")
        data = self.fetch_sheet_data()
        docs = self._docs_from_data(data)
        nodes = self._parser.get_nodes_from_documents(docs)

        # Create embedding model
        embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

        # tqdm progress bar in logs
        for node in tqdm(nodes, desc="Embedding nodes", unit="row"):
            node.embedding = embed_model.get_text_embedding(node.get_content())

        storage_ctx = StorageContext.from_defaults()
        self._index = VectorStoreIndex(nodes, storage_context=storage_ctx)
        self._index.storage_context.persist(persist_dir=PERSIST_DIR)

        logger.info(
            "[Index] Built with %d rows from %d sheets.",
            sum(len(v) for v in data.values()),
            len(data),
        )

    # --- Tabular Engine ---
    def _build_dataframes(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        data = self.fetch_sheet_data()
        dfs: Dict[str, pd.DataFrame] = {}
        name_map: Dict[str, str] = {}
        for sheet, rows in data.items():
            df = pd.DataFrame(rows)
            df.columns = [str(c) for c in df.columns]

            if "Status" in df.columns:
                df["Status"] = df["Status"].apply(_normalize_status)
            if "Quantity" in df.columns and "Rate" in df.columns:
                df["Cost"] = df["Quantity"].apply(_safe_float) * df["Rate"].apply(_safe_float)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date

            dfs[sheet] = df
            name_map[_normalize_name(sheet)] = sheet
        return dfs, name_map

    def _apply_conditions(self, df: pd.DataFrame, conditions: List[Condition]) -> pd.DataFrame:
        if df is None or df.empty or not conditions:
            return df
        mask = pd.Series([True] * len(df))
        for cond in conditions:
            cols_map = {_normalize_name(c): c for c in df.columns}
            real_col = cols_map.get(_normalize_name(cond.column))
            if real_col is None:
                continue
            series = df[real_col]
            try:
                if cond.op == "contains":
                    mask &= series.astype(str).str.contains(str(cond.value), case=False, na=False)
                elif cond.op == "==":
                    if real_col.lower() == "status":
                        series = series.apply(_normalize_status)
                        mask &= (series == str(cond.value).strip().lower())
                    else:
                        mask &= (series.astype(str).str.strip().str.lower() == str(cond.value).strip().lower())
                elif cond.op == "between":
                    series_dt = pd.to_datetime(series, errors="coerce").dt.date
                    start, end = cond.value
                    mask &= (series_dt >= start) & (series_dt <= end)
                else:
                    left = series.apply(_safe_float)
                    right = _safe_float(cond.value)
                    if cond.op == ">": mask &= (left > right)
                    if cond.op == ">=": mask &= (left >= right)
                    if cond.op == "<": mask &= (left < right)
                    if cond.op == "<=": mask &= (left <= right)
            except Exception:
                pass
        return df[mask]

    def _normalize_plan_dates(self, plan: RoutePlan) -> RoutePlan:
        if not plan or not plan.conditions:
            return plan
        for cond in plan.conditions:
            if _normalize_name(cond.column) == "date":
                if isinstance(cond.value, str):
                    cond.op = "between"
                    cond.value = _date_range(cond.value)
        return plan

    def _run_tabular_agg(self, plan: RoutePlan) -> str:
        plan = self._normalize_plan_dates(plan)
        dfs, name_map = self._build_dataframes()
        if not plan.sheet:
            return "I need a sheet name to analyze."
        real_sheet = name_map.get(_normalize_name(plan.sheet))
        if not real_sheet:
            return f"I could not find a sheet called '{plan.sheet}'."
        df = dfs.get(real_sheet)
        if df is None or df.empty:
            return f"The sheet '{real_sheet}' is empty."
        df_f = self._apply_conditions(df, plan.conditions)
        op = (plan.operation or "COUNT").upper()
        if op == "COUNT":
            if not plan.conditions:  # no filters → give total rows
                return f"There are {len(df)} rows in the '{real_sheet}' sheet."
            else:  # filters applied → give filtered count
                return f"There are {len(df_f)} rows in '{real_sheet}' that match the criteria."
        if not plan.target_column:
            return "I need a target column for this calculation."
        cols_map = {_normalize_name(c): c for c in df_f.columns}
        real_col = cols_map.get(_normalize_name(plan.target_column))
        if not real_col:
            return f"I could not find a column '{plan.target_column}' in '{real_sheet}'."
        series = df_f[real_col].apply(_safe_float)
        if op == "SUM": return f"The total sum of {real_col} is {series.sum()}."
        if op == "AVG": return f"The average of {real_col} is {series.mean()}."
        if op == "MIN": return f"The minimum value in {real_col} is {series.min()}."
        if op == "MAX": return f"The maximum value in {real_col} is {series.max()}."
        return f"I cannot process the operation '{op}'."

    # --- Router ---
    def _normalize_plan(self, plan: dict) -> dict:
        if "operation" in plan and isinstance(plan["operation"], str):
            plan["operation"] = plan["operation"].upper()
        if "conditions" in plan and isinstance(plan["conditions"], list):
            for cond in plan["conditions"]:
                if cond.get("op") in ("=", "eq"):
                    cond["op"] = "=="
                if cond.get("op") == "neq":
                    cond["op"] = "!="
        return plan

    def _route(self, question: str, available_sheets: List[str]) -> RoutePlan:
        q_lower = question.lower()

        # quick regex-based lookup detection
        lookup_matches = re.findall(r"(?:id|task id|po\s*no|employee|name)\s*([A-Za-z0-9\-_]+)", q_lower)
        if lookup_matches:
            return RoutePlan(
                intent="lookup",
                lookup_values=lookup_matches
            )

        llm = OpenAI(model=MODEL, api_key=OPENAI_API_KEY)
        sys = (
            "You are Diya, an AI Agent for Botivate LLP. "
            "Your role is to analyze business data and provide clear, full-sentence answers. "
            "Never give one-word answers. Never ask follow-ups. "
            "If numeric (count/sum/avg/min/max), choose intent='tabular_agg'. "
            "If user asks about details of a row (by id, task, po no, employee, name, etc.), use intent='lookup'. "
            "Otherwise, use intent='semantic_answer'. Respond in JSON only."
        )
        history_text = "\n".join([f"User: {h['user']}\nDiya: {h['diya']}" for h in CONVERSATION_HISTORY[-5:]])
        user = f"Conversation so far:\n{history_text}\n\nSheets: {available_sheets}\nQuestion: {question}"
        raw = llm.complete(prompt=f"{sys}\n\n{user}")
        txt = raw.text.strip()
        logger.debug("[Router RAW] %s", txt)

        try:
            data = json.loads(txt)
            data = self._normalize_plan(data)
            return RoutePlan(**data)
        except Exception as e:
            return RoutePlan(intent="semantic_answer")

    def _semantic_answer(self, question: str) -> str:
        self.ensure_index()
        llm = OpenAI(model=MODEL, api_key=OPENAI_API_KEY)
        qe = self._index.as_query_engine(llm=llm)
        resp = qe.query(question)
        return str(resp)

    def _node_lookup(self, state: "SheetRAGAgent.State") -> "SheetRAGAgent.State":
        def _format_value(k: str, v: str) -> str:
            """Clean and format raw values for nicer presentation."""
            # convert ISO date to human-friendly
            if "date" in k.lower() or "timestamp" in k.lower():
                try:
                    dt = pd.to_datetime(v)
                    return dt.strftime("%d %B %Y")
                except Exception:
                    return v
            # round delay values
            if "delay" in k.lower():
                try:
                    return f"{round(float(v))} days"
                except Exception:
                    return v
            return v

        try:
            dfs, _ = self._build_dataframes()
            found_rows = []

            for sheet, df in dfs.items():
                for lookup_val in (state.route.lookup_values or []):
                    for col in df.columns:
                        matches = df[df[col].astype(str).str.strip().str.lower() == str(lookup_val).lower()]
                        if not matches.empty:
                            for _, row in matches.iterrows():
                                details = {k: _format_value(k, str(v)) for k, v in row.items() if pd.notna(v) and v != ""}
                                found_rows.append((sheet, col, lookup_val, details))

            if not found_rows:
                state.answer = f"Sorry, I couldn’t find any details for {', '.join(state.route.lookup_values or [])}."
                return state

            summaries = []
            llm = OpenAI(model=MODEL, api_key=OPENAI_API_KEY)

            for sheet, col, val, details in found_rows:
                kv_text = "\n".join([f"{k}: {v}" for k, v in details.items()])
                prompt = (
                    f"Summarize the following task details in a clear, natural sentence:\n\n{kv_text}\n\n"
                    f"Be concise, human-friendly, and highlight only the most important fields "
                    f"(Assigned To, Given By, Department, Task Description, Start Date, Status, Delay)."
                )
                resp = llm.complete(prompt=prompt)
                nice_summary = resp.text.strip()

                # format details as markdown table
                table_rows = "\n".join([f"| {k} | {v} |" for k, v in details.items()])
                raw_details = f"| Field | Value |\n|-------|-------|\n{table_rows}"

                text = (
                    f"### 🔎 Lookup `{val}` (from sheet **{sheet}**, matched in '{col}')\n\n"
                    f"**Summary:** {nice_summary}\n\n"
                    f"**Full Details:**\n{raw_details}"
                )
                summaries.append(text)

            state.answer = "\n\n---\n\n".join(summaries)
        except Exception as e:
            state.answer = f"I encountered a lookup error: {e}"
        return state

    # --- LangGraph orchestration ---
    class State(BaseModel):
        question: str
        route: Optional[RoutePlan] = None
        answer: Optional[str] = None

    def _node_route(self, state: "SheetRAGAgent.State") -> "SheetRAGAgent.State":
        data = self.fetch_sheet_data()
        plan = self._route(state.question, list(data.keys()))
        state.route = plan
        return state

    def _node_tabular(self, state: "SheetRAGAgent.State") -> "SheetRAGAgent.State":
        try:
            state.answer = self._run_tabular_agg(state.route)  # type: ignore
        except Exception as e:
            state.answer = f"I encountered a tabular error: {e}"
        return state

    def _node_rag(self, state: "SheetRAGAgent.State") -> "SheetRAGAgent.State":
        try:
            state.answer = self._semantic_answer(state.question)
        except Exception as e:
            state.answer = f"I encountered a RAG error: {e}"
        return state

    def _should_tabular(self, state: "SheetRAGAgent.State") -> str:
        if state.route and state.route.intent == "tabular_agg":
            return "tabular"
        if state.route and state.route.intent == "lookup":
            return "lookup"
        return "rag"

    def _build_graph(self):
        g = StateGraph(self.State)
        g.add_node("route", self._node_route)
        g.add_node("tabular", self._node_tabular)
        g.add_node("rag", self._node_rag)
        g.add_node("lookup", self._node_lookup)
        g.set_entry_point("route")
        g.add_conditional_edges("route", self._should_tabular, {
            "tabular": "tabular",
            "rag": "rag",
            "lookup": "lookup",
        })
        g.add_edge("tabular", END)
        g.add_edge("rag", END)
        g.add_edge("lookup", END)
        return g.compile()

    # --- Public API ---
    def chat(self, message: str) -> str:
        msg_lower = message.lower().strip()

        # --- Handle greetings & system date/time ---
        if msg_lower in ["hi", "hello", "hey"]:
            reply = "Hello, I’m Diya, your AI Agent from Botivate LLP. How can I assist you with your business data today?"
        elif "today" in msg_lower and "date" in msg_lower:
            reply = f"Today’s date is {datetime.now().date().isoformat()}."
        elif "time" in msg_lower:
            reply = f"The current time is {datetime.now().strftime('%H:%M:%S')}."
        elif "day" in msg_lower and "today" in msg_lower:
            reply = f"Today is {datetime.now().strftime('%A')}."
        else:
            init = self.State(question=message)
            final_state = self._graph.invoke(init)
            reply = final_state.get("answer") if isinstance(final_state, dict) else getattr(final_state, "answer", "No answer.")

            # --- Update sheet memory if router selected a sheet ---
            if final_state and isinstance(final_state, dict):
                route = final_state.get("route")
            else:
                route = getattr(final_state, "route", None)

            if route and route.sheet:
                CONTEXT["last_sheet"] = route.sheet

            # --- If no sheet but we have last_sheet, re-run with it ---
            if (("pending" in msg_lower or "rows" in msg_lower) 
                and (not route or not route.sheet) 
                and CONTEXT["last_sheet"]):
                logger.debug("Reusing last sheet: %s", CONTEXT["last_sheet"])
                # Force route with last sheet
                plan = self._route(message, [CONTEXT["last_sheet"]])
                plan.sheet = CONTEXT["last_sheet"]
                reply = self._run_tabular_agg(plan)

        # Save conversation
        CONVERSATION_HISTORY.append({"user": message, "diya": reply})
        return reply

    def refresh(self, force: bool = True) -> None:
        if force:
            self.rebuild_index()
        else:
            self.ensure_index()
