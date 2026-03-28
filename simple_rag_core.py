from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


class SmartDocAgent:
    def __init__(
        self,
        pdf_retrievers: dict[str, Any],
        llm: Any,
        max_history: int = 8,
        pdf_vectorstores: dict[str, Any] | None = None,
    ):
        self.pdf_retrievers = pdf_retrievers
        self.pdf_vectorstores = pdf_vectorstores or {}
        self.llm = llm
        self.name = "SmartDoc Agent"
        self.max_history = max_history
        self.history: list[dict[str, str]] = []
        self.question_history: list[str] = []
        self.important_points: list[str] = []

    def _format_history(self) -> str:
        if not self.history:
            return "None"
        lines: list[str] = []
        for item in self.history[-self.max_history:]:
            lines.append(f"Q: {item['question']}")
            lines.append(f"A: {item['answer_summary']}")
        return "\n".join(lines)

    def _remember_turn(self, question: str, answer_text: str) -> None:
        summary = answer_text.replace("\n", " ").strip()[:300]
        self.history.append({"question": question, "answer_summary": summary})
        self.history = self.history[-self.max_history:]

    def _add_important(self, note: str) -> bool:
        cleaned = note.strip()
        if not cleaned:
            return False

        lower_existing = {point.lower() for point in self.important_points}
        if cleaned.lower() in lower_existing:
            return False

        self.important_points.append(cleaned)
        self.important_points = self.important_points[-20:]
        return True

    def _extract_note_from_query(self, query: str) -> str | None:
        lowered = query.lower().strip()
        prefixes = ("remember:", "note:", "important:")
        for prefix in prefixes:
            if lowered.startswith(prefix):
                return query[len(prefix):].strip()

        trigger_words = ["important", "remember", "note", "key point", "main point"]
        if any(word in lowered for word in trigger_words):
            return query.strip()
        return None

    def _is_identity_query(self, query: str) -> bool:
        lowered = query.lower().strip()
        identity_triggers = (
            "what is your name",
            "what's your name",
            "whats your name",
            "who are you",
            "your name",
            "agent name",
            "tell me your name",
        )
        return any(trigger in lowered for trigger in identity_triggers)

    def show_memory(self) -> str:
        recent_questions = self.question_history[-10:]
        question_lines = [f"- {question}" for question in recent_questions] if recent_questions else ["- None"]
        important_lines = [f"- {point}" for point in self.important_points] if self.important_points else ["- None"]

        return (
            "📌 Memory Status\n"
            f"Stored previous questions: {len(self.question_history)}\n"
            f"Stored important notes: {len(self.important_points)}\n\n"
            "Recent Questions:\n" + "\n".join(question_lines) + "\n\n"
            "Important Notes:\n" + "\n".join(important_lines)
        )

    def _enforce_format(self, source_name: str, answer: str) -> str:
        required = ["📄 Document:", "📌 Summary:", "📚 Key Points:", "💡 Example:"]
        if all(marker in answer for marker in required):
            return answer

        lines = [line.strip(" -•\t") for line in answer.splitlines() if line.strip()]
        summary_points = lines[:3] if lines else ["No clear summary available from model output."]
        key_points = lines[3:9] if len(lines) > 3 else ["No additional key points extracted."]
        example_line = lines[-1] if lines else "No clear example generated."

        summary_text = "\n".join([f"• {point}" for point in summary_points])
        key_points_text = "\n".join([f"• {point}" for point in key_points])

        return (
            f"📄 Document: {source_name}\n\n"
            "📌 Summary:\n"
            f"{summary_text}\n\n"
            "📚 Key Points:\n"
            f"{key_points_text}\n\n"
            "💡 Example:\n"
            f"{example_line}"
        )

    def _dedupe_docs(self, docs: list[Any], limit: int = 6) -> list[Any]:
        seen: set[tuple[Any, str]] = set()
        unique_docs: list[Any] = []

        for document in docs:
            page = document.metadata.get("page", "N/A")
            signature = re.sub(r"\s+", " ", document.page_content.strip())[:220]
            key = (page, signature)
            if key in seen:
                continue
            seen.add(key)
            unique_docs.append(document)
            if len(unique_docs) >= limit:
                break

        return unique_docs

    def _source_name_match_bonus(self, query: str, source_name: str) -> float:
        query_tokens = {token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) >= 3}
        source_tokens = {token for token in re.findall(r"[a-z0-9]+", Path(source_name).stem.lower()) if len(token) >= 3}
        if not query_tokens or not source_tokens:
            return 0.0
        overlap = len(query_tokens.intersection(source_tokens))
        return float(overlap) * 0.35

    def _query_keywords(self, query: str) -> set[str]:
        stop_words = {
            "what", "which", "when", "where", "who", "whom", "whose", "why", "how",
            "is", "are", "was", "were", "be", "being", "been",
            "a", "an", "the", "of", "to", "in", "on", "for", "from", "with", "about",
            "tell", "give", "explain", "describe", "please", "pdf", "document", "summary",
            "short", "points", "point", "and", "or", "this", "that", "it",
        }
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        return {token for token in tokens if len(token) >= 3 and token not in stop_words}

    def _context_keyword_overlap(self, query: str, docs: list[Any]) -> tuple[int, int, float]:
        keywords = self._query_keywords(query)
        if not keywords:
            return 0, 0, 0.0

        context_text = " ".join(document.page_content.lower() for document in docs[:4])
        matched = {token for token in keywords if token in context_text}
        match_count = len(matched)
        total_keywords = len(keywords)
        ratio = match_count / total_keywords if total_keywords else 0.0
        return match_count, total_keywords, ratio

    def _has_strong_support(self, query: str, docs: list[Any], score: float | None, explicit_source: bool) -> bool:
        if not docs:
            return False

        if explicit_source:
            return True

        match_count, total_keywords, ratio = self._context_keyword_overlap(query, docs)

        if total_keywords > 0 and match_count == 0:
            return False

        if total_keywords >= 2 and ratio < 0.25:
            return False

        if score is not None and score > 1.10 and ratio < 0.40:
            return False

        return True

    def _get_scored_docs(self, source_name: str, query: str) -> tuple[list[Any], float | None]:
        store = self.pdf_vectorstores.get(source_name)
        if store is not None and hasattr(store, "similarity_search_with_score"):
            try:
                docs_scores = store.similarity_search_with_score(query, k=6)
                if docs_scores:
                    docs = [document for document, _ in docs_scores]
                    best_score = float(docs_scores[0][1])
                    adjusted_score = best_score - self._source_name_match_bonus(query, source_name)
                    return docs, adjusted_score
            except Exception:
                pass

        retriever = self.pdf_retrievers.get(source_name)
        if retriever is None:
            return [], None

        try:
            docs = list(retriever.invoke(query))
            return docs, None
        except Exception:
            return [], None

    def _select_relevant_sources(self, query: str) -> list[str]:
        if not self.pdf_retrievers:
            return []

        lowered_query = query.lower()
        explicit_matches: list[str] = []
        for source_name in self.pdf_retrievers:
            stem = Path(source_name).stem.lower()
            normalized = re.sub(r"[^a-z0-9]+", " ", stem).strip()
            if not normalized:
                continue
            if stem in lowered_query or normalized in lowered_query:
                explicit_matches.append(source_name)

        if explicit_matches:
            return explicit_matches

        candidates: list[tuple[str, float]] = []
        scored_found = False

        for source_name in self.pdf_retrievers:
            docs, score = self._get_scored_docs(source_name, query)
            if not docs:
                continue
            if score is not None:
                scored_found = True
                candidates.append((source_name, score))

        if scored_found and candidates:
            candidates.sort(key=lambda item: item[1])
            best_source, best_score = candidates[0]

            selected = [best_source]
            if len(candidates) > 1:
                second_source, second_score = candidates[1]
                if second_score <= best_score + 0.25:
                    selected.append(second_source)
            return selected

        return list(self.pdf_retrievers.keys())

    def _has_explicit_source_match(self, query: str, source_name: str) -> bool:
        lowered_query = query.lower()
        stem = Path(source_name).stem.lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", stem).strip()
        if not normalized:
            return False
        return stem in lowered_query or normalized in lowered_query

    def run(self, query: str) -> str:
        cleaned_query = query.strip()
        if not cleaned_query:
            return "Please enter a question."

        lowered = cleaned_query.lower()
        if lowered in {"show memory", "/memory", "memory"}:
            return self.show_memory()

        if self._is_identity_query(cleaned_query):
            return f"My name is {self.name}."

        self.question_history.append(cleaned_query)
        self.question_history = self.question_history[-30:]

        note = self._extract_note_from_query(cleaned_query)
        note_added = self._add_important(note) if note else False

        if lowered.startswith(("remember:", "note:", "important:")):
            return "Noted. I saved this in memory.\n\n" + self.show_memory()

        if not self.pdf_retrievers:
            return "No PDF retrievers found. Please upload/load PDFs first."

        history_text = self._format_history()
        important_text = "\n".join(self.important_points[-10:]) if self.important_points else "None"
        answers: list[str] = []
        relevant_sources = self._select_relevant_sources(cleaned_query)
        if not relevant_sources:
            return "I could not find relevant content in your uploaded PDFs for this question."

        for source_name in relevant_sources:
            source_retriever = self.pdf_retrievers[source_name]
            source_docs, base_score = self._get_scored_docs(source_name, cleaned_query)

            alt_query = cleaned_query.replace("?", "").strip()
            if alt_query and alt_query != cleaned_query:
                try:
                    source_docs.extend(list(source_retriever.invoke(alt_query)))
                except Exception:
                    pass

            source_docs = self._dedupe_docs(source_docs, limit=6)
            explicit_source = self._has_explicit_source_match(cleaned_query, source_name)
            if not self._has_strong_support(cleaned_query, source_docs, base_score, explicit_source=explicit_source):
                continue

            pdf_context = "\n\n".join([
                f"(Page {document.metadata.get('page', '?')}) {document.page_content}"
                for document in source_docs
            ])

            if not pdf_context.strip():
                answers.append(
                    "\n".join([
                        "=" * 94,
                        f"📄 Document: {source_name}",
                        "",
                        "📌 Summary:",
                        "• No relevant content found for this question.",
                        "",
                        "📚 Key Points:",
                        "• Not enough matching text in this PDF.",
                        "",
                        "💡 Example:",
                        "No clear example available from this PDF context.",
                        "",
                        "🔎 Citations:",
                        "• Not available",
                    ])
                )
                continue

            citations: list[str] = []
            for document in source_docs:
                page = document.metadata.get("page", "N/A")
                if isinstance(page, int):
                    page = page + 1
                citations.append(f"• {source_name} (Page {page})")

            prompt = f"""
You are {self.name}, a helpful AI assistant.

Rules:
- Use ONLY this PDF context.
- Do NOT guess.
- If the context is insufficient, explicitly say you cannot find enough information in this document.
- Do not use conversation history as evidence for facts.
- Keep language simple and clear.
- Add short real-life examples.
- Mention page numbers where possible.
- Output EXACTLY this structure:

📄 Document: {source_name}

📌 Summary:
• ...
• ...
• ...

📚 Key Points:
• ...
• ...
• ...

💡 Example:
...

Conversation History (previous Q&A):
{history_text}

Important User Notes to Remember:
{important_text}

PDF Context:
{pdf_context}

Current Question: {cleaned_query}

Answering policy:
- First, ground each point in the provided context.
- If the question asks something not present in this document, state that clearly in Summary and Key Points.
- Never invent details.
"""

            try:
                raw_answer = self.llm.invoke(prompt).content.strip()
            except Exception as error:
                error_text = str(error)
                if "invalid_api_key" in error_text or "Invalid API Key" in error_text or "AuthenticationError" in error_text:
                    return (
                        "RAG system is not ready. Status: Invalid GROQ_API_KEY (401). "
                        "Please set a valid GROQ_API_KEY and try again."
                    )
                return f"RAG runtime error: {error_text}"
            answer = self._enforce_format(source_name, raw_answer)
            citation_text = "\n".join(dict.fromkeys(citations)) if citations else "• Not available"
            answers.append("=" * 94 + "\n" + answer + "\n\n🔎 Citations:\n" + citation_text)

        if not answers:
            no_match_message = (
                "I could not find enough relevant information in your uploaded PDFs for this question. "
                "Please ask a question that matches your uploaded documents or mention the PDF name."
            )
            self._remember_turn(cleaned_query, no_match_message)
            return no_match_message

        final_answer = "\n\n".join(answers) + f"\n\n{'=' * 94}\n\n❓ Question:\n{cleaned_query}"
        self._remember_turn(cleaned_query, final_answer)

        if note_added:
            final_answer = "(Saved one new important note from your query.)\n\n" + final_answer

        return final_answer


def build_agent_from_docs(docs_by_file: dict[str, list[Any]], groq_api_key: str | None = None) -> tuple[SmartDocAgent | None, str]:
    if not docs_by_file:
        return None, "No PDFs loaded"

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_groq import ChatGroq
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except Exception:
            from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception as error:
        return None, f"Missing dependency: {error}"

    api_key = (
        groq_api_key
        or os.getenv("GROQ_API_KEY")
        or ""
    ).strip()
    if not api_key:
        return None, "Missing GROQ_API_KEY"

    all_docs: list[Any] = []
    for docs in docs_by_file.values():
        all_docs.extend(docs)

    if not all_docs:
        return None, "No PDF pages found"

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(all_docs)

    source_to_chunks: defaultdict[str, list[Any]] = defaultdict(list)
    for chunk in splits:
        source_path = chunk.metadata.get("source", "unknown")
        source_name = Path(source_path).name
        source_to_chunks[source_name].append(chunk)

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"batch_size": 8},
        )
    except Exception as error:
        return None, f"Embedding initialization failed: {error}"

    pdf_retrievers: dict[str, Any] = {}
    pdf_vectorstores: dict[str, Any] = {}
    for source_name, chunks in source_to_chunks.items():
        try:
            source_store = FAISS.from_documents(chunks, embeddings)
            pdf_vectorstores[source_name] = source_store
            pdf_retrievers[source_name] = source_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 6, "fetch_k": 24, "lambda_mult": 0.35},
            )
        except Exception as error:
            return None, f"Vector index creation failed for {source_name}: {error}"

    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
    except Exception as error:
        return None, f"Groq client initialization failed: {error}"

    return (
        SmartDocAgent(pdf_retrievers=pdf_retrievers, pdf_vectorstores=pdf_vectorstores, llm=llm),
        f"Ready ({len(pdf_retrievers)} PDFs indexed)",
    )


def parse_uploaded_pdfs(
    uploaded_files_data: list[tuple[str, bytes]],
    upload_dir: str | Path = "uploaded_pdfs",
    persist_files: bool = True,
) -> tuple[dict[str, list[Any]], int, int, list[str]]:
    target_dir = Path(upload_dir)
    if persist_files:
        target_dir.mkdir(parents=True, exist_ok=True)

    docs_by_file: dict[str, list[Any]] = {}
    total_files = 0
    total_pages = 0
    errors: list[str] = []

    for file_name, file_bytes in uploaded_files_data:
        safe_name = "_".join(file_name.split())
        target_path = target_dir / safe_name

        if persist_files:
            target_path.write_bytes(file_bytes)
        else:
            original_path = target_dir / file_name
            if original_path.exists():
                target_path = original_path
            elif not target_path.exists():
                errors.append(f"{file_name}: file not found in {target_dir}")
                continue

        try:
            loader = PyPDFLoader(str(target_path))
            file_docs = loader.load()
        except Exception as error:
            errors.append(f"{target_path.name}: {error}")
            continue

        docs_by_file[target_path.name] = file_docs
        total_files += 1
        total_pages += len(file_docs)

    return docs_by_file, total_files, total_pages, errors
