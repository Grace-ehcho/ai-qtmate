import json
import os
import time
import traceback
from datetime import datetime

import requests
import streamlit as st
from google import genai
from google.genai import types, errors as genai_errors
from google.api_core import exceptions as google_exceptions

# ── 상수 ──────────────────────────────────────────────────
FLASH_MODEL        = "gemini-2.5-flash"
PRO_MODEL          = "gemini-2.5-flash"
MAX_HISTORY        = 10
EXTRACT_TIMEOUT    = 20  # seconds
QUOTA_RETRY_WAITS  = [15, 30]  # 429 재시도 대기(초): 1차 15s, 2차 30s

BIBLE_BOOKS = {
    "창세기": 1, "출애굽기": 2, "레위기": 3, "민수기": 4, "신명기": 5,
    "여호수아": 6, "사사기": 7, "룻기": 8, "사무엘상": 9, "사무엘하": 10,
    "열왕기상": 11, "열왕기하": 12, "역대상": 13, "역대하": 14, "에스라": 15,
    "느헤미야": 16, "에스더": 17, "욥기": 18, "시편": 19, "잠언": 20,
    "전도서": 21, "아가": 22, "이사야": 23, "예레미야": 24, "예레미야애가": 25,
    "에스겔": 26, "다니엘": 27, "호세아": 28, "요엘": 29, "아모스": 30,
    "오바댜": 31, "요나": 32, "미가": 33, "나훔": 34, "하박국": 35,
    "스바냐": 36, "학개": 37, "스가랴": 38, "말라기": 39,
    "마태복음": 40, "마가복음": 41, "누가복음": 42, "요한복음": 43,
    "사도행전": 44, "로마서": 45, "고린도전서": 46, "고린도후서": 47,
    "갈라디아서": 48, "에베소서": 49, "빌립보서": 50, "골로새서": 51,
    "데살로니가전서": 52, "데살로니가후서": 53, "디모데전서": 54, "디모데후서": 55,
    "디도서": 56, "빌레몬서": 57, "히브리서": 58, "야고보서": 59,
    "베드로전서": 60, "베드로후서": 61, "요한일서": 62, "요한이서": 63,
    "요한삼서": 64, "유다서": 65, "요한계시록": 66,
}

BIBLE_CHAPTERS = {
    "창세기": 50, "출애굽기": 40, "레위기": 27, "민수기": 36, "신명기": 34,
    "여호수아": 24, "사사기": 21, "룻기": 4, "사무엘상": 31, "사무엘하": 24,
    "열왕기상": 22, "열왕기하": 25, "역대상": 29, "역대하": 36, "에스라": 10,
    "느헤미야": 13, "에스더": 10, "욥기": 42, "시편": 150, "잠언": 31,
    "전도서": 12, "아가": 8, "이사야": 66, "예레미야": 52, "예레미야애가": 5,
    "에스겔": 48, "다니엘": 12, "호세아": 14, "요엘": 3, "아모스": 9,
    "오바댜": 1, "요나": 4, "미가": 7, "나훔": 3, "하박국": 3,
    "스바냐": 3, "학개": 2, "스가랴": 14, "말라기": 4,
    "마태복음": 28, "마가복음": 16, "누가복음": 24, "요한복음": 21,
    "사도행전": 28, "로마서": 16, "고린도전서": 16, "고린도후서": 13,
    "갈라디아서": 6, "에베소서": 6, "빌립보서": 4, "골로새서": 4,
    "데살로니가전서": 5, "데살로니가후서": 3, "디모데전서": 6, "디모데후서": 4,
    "디도서": 3, "빌레몬서": 1, "히브리서": 13, "야고보서": 5,
    "베드로전서": 5, "베드로후서": 3, "요한일서": 5, "요한이서": 1,
    "요한삼서": 1, "유다서": 1, "요한계시록": 22,
}

DEFAULT_CHARACTER = {
    "name": "지혜로운 조언자",
    "role": "성경의 지혜를 전하는 영적 안내자",
    "description": (
        "성경 전체의 가르침을 바탕으로 따뜻하고 지혜롭게 조언해주는 영적 안내자입니다. "
        "오늘 읽은 말씀을 삶에 연결할 수 있도록 도와줍니다."
    ),
}


# ── 유틸 ──────────────────────────────────────────────────
def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except Exception:
        pass


def safe_str(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value) if not isinstance(value, str) else value


# ── 세션 초기화 (브라우저 탭마다 완전 독립) ────────────────
def init_session() -> None:
    defaults: dict = {
        "api_key":                    "",
        # 직접 입력 탭
        "direct_text":                "",
        "direct_extraction_done":     False,
        "direct_characters":          [],
        "direct_selected_character":  None,
        "direct_messages":            [],
        # 성경 구절 불러오기 탭
        "fetch_sv":                   1,
        "fetch_ev":                   1,
        "fetch_preview":              "",
        "fetch_extraction_done":      False,
        "fetch_characters":           [],
        "fetch_selected_character":   None,
        "fetch_messages":             [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── API 키 관리 ───────────────────────────────────────────
def get_api_key() -> str | None:
    """우선순위: st.secrets → 환경변수 → 사용자 세션 입력"""
    sources = [
        lambda: st.secrets.get("GEMINI_API_KEY"),
        lambda: os.environ.get("GEMINI_API_KEY", ""),
        lambda: st.session_state.get("api_key", ""),
    ]
    for source in sources:
        try:
            key = (source() or "").strip()
            if key:
                return key
        except Exception:
            pass
    return None


def server_key_set() -> bool:
    """서버(secrets/env)에 키가 설정되어 있으면 True — 사용자 입력창 숨김 여부 결정"""
    for source in [
        lambda: st.secrets.get("GEMINI_API_KEY"),
        lambda: os.environ.get("GEMINI_API_KEY", ""),
    ]:
        try:
            if (source() or "").strip():
                return True
        except Exception:
            pass
    return False


def make_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


# ── 성경 불러오기 콜백 ────────────────────────────────────
def _reset_fetch_analysis() -> None:
    st.session_state.fetch_extraction_done    = False
    st.session_state.fetch_characters         = []
    st.session_state.fetch_selected_character = None
    st.session_state.fetch_messages           = []

def _reset_chapter_and_verses() -> None:
    st.session_state.fetch_chapter = 1
    st.session_state.fetch_sv = 1
    st.session_state.fetch_ev = 1
    st.session_state.fetch_preview = ""
    _reset_fetch_analysis()

def _reset_verses() -> None:
    st.session_state.fetch_sv = 1
    st.session_state.fetch_ev = 1
    _reset_fetch_analysis()


# ── 성경 불러오기 ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_verse_count(book_id: int, chapter: int) -> int:
    url = f"https://bolls.life/get-text/KRV/{book_id}/{chapter}/"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return len(data) if data else 30
    except Exception:
        return 30


def fetch_bible_text(book_id: int, chapter: int, start_verse: int, end_verse: int) -> str:
    url = f"https://bolls.life/get-text/KRV/{book_id}/{chapter}/"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        verses = resp.json()
        lines = [
            f"{v['verse']}절 {v['text'].strip()}"
            for v in verses
            if start_verse <= v["verse"] <= end_verse
        ]
        return "\n".join(lines)
    except Exception as e:
        _log(f"[성경불러오기] 오류: {e}")
        return ""


# ── 인물 추출 (Flash) ──────────────────────────────────────
def extract_characters(api_key: str, bible_text: str) -> list[dict]:
    _log(f"[인물분석] 시작 — {FLASH_MODEL}, {len(bible_text)}자")
    prompt = (
        "성경 본문의 주요 등장인물을 최대 5명 추출해 JSON 배열로만 반환하세요. "
        "인물이 없으면 []를 반환하세요.\n\n"
        f"본문: {bible_text}\n\n"
        '형식: [{"name":"이름","role":"역할","description":"설명(1문장)"}]'
    )
    client = make_client(api_key)
    raw = ""
    for attempt, wait in enumerate([0] + QUOTA_RETRY_WAITS):
        if wait:
            _log(f"[인물분석] quota 초과 — {wait}초 대기 후 재시도 ({attempt}/{len(QUOTA_RETRY_WAITS)})")
            time.sleep(wait)
        try:
            resp = client.models.generate_content(
                model=FLASH_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    http_options=types.HttpOptions(timeout=EXTRACT_TIMEOUT * 1000),
                ),
            )
            raw = safe_str(resp.text).strip()
            break
        except genai_errors.ClientError as e:
            if getattr(e, "code", None) == 429 and attempt < len(QUOTA_RETRY_WAITS):
                continue
            raise
        except google_exceptions.DeadlineExceeded:
            raise TimeoutError(f"AI 응답이 {EXTRACT_TIMEOUT}초를 초과했습니다.")

    _log(f"[인물분석] 응답: {raw[:120]}")

    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
        chars = result if isinstance(result, list) else []
        _log(f"[인물분석] {len(chars)}명 추출 완료")
        return chars
    except json.JSONDecodeError:
        return []


# ── 시스템 프롬프트 ────────────────────────────────────────
def build_system_prompt(character: dict, bible_text: str) -> str:
    if character["name"] == DEFAULT_CHARACTER["name"]:
        return f"""당신은 성경의 지혜를 전하는 '지혜로운 조언자'입니다.

[역할]
- 성경 전체의 가르침을 바탕으로 따뜻하고 지혜롭게 조언합니다.
- 오늘 묵상한 성경 본문을 중심으로 사용자가 말씀을 깊이 이해하고 삶에 적용하도록 돕습니다.

[오늘의 묵상 본문]
{bible_text}

[대화 가이드라인]
1. 항상 성경적 근거를 바탕으로 답변하세요.
2. 현대적 맥락과 연결하되, 성경의 본질적 진리를 벗어나지 마세요.
3. 모르는 것은 솔직하게 인정하고, 성경에서 찾을 수 있는 부분을 안내하세요.
4. 따뜻하고 격려적인 어조를 유지하세요.
5. 신학적 논쟁보다 실제 적용과 영적 성장에 초점을 맞추세요.
6. 특정 교단의 교리보다 성경 본문 자체에 집중하세요.

[경계]
- 성경에 없는 내용을 창작하거나 추가하지 마세요.
- 예언이나 개인적 계시를 주장하지 마세요.
- 의학적·법적·재정적 조언은 삼가세요."""

    return f"""당신은 성경에 등장하는 '{character["name"]}'입니다.

[인물 정보]
- 이름: {character["name"]}
- 역할: {character["role"]}
- 설명: {character["description"]}

[오늘의 묵상 본문]
{bible_text}

[역할극 가이드라인]
1. 성경에 기록된 {character["name"]}의 관점에서 말하고 생각하세요.
2. 성경에 실제로 기록된 당신의 경험, 말씀, 행적을 바탕으로 답변하세요.
3. 당신이 살았던 시대적·문화적 배경을 반영하여 이야기하세요.
4. 사용자가 현대적 질문을 하면, 당신의 성경적 경험을 통해 연결점을 찾아 주세요.

[성경적 진실성 유지]
- 성경에 기록되지 않은 내용은 "성경에는 기록되지 않았지만, 제 경험으로는..."처럼 명확히 구분하세요.
- 과도한 창작이나 성경 외적인 내용은 삼가세요.
- 하나님의 말씀과 뜻에 항상 순종하는 자세를 유지하세요.
- 이단적이거나 성경에 반하는 내용은 절대 말하지 마세요.

[엄격한 경계]
- 당신은 성경 속 인물이지, 현대의 예언자나 신탁을 주는 존재가 아닙니다.
- 직접적 계시나 예언을 주장하지 마세요.
- 성경적 가르침과 충돌하는 요청은 정중히 거절하세요.

[언어와 어조]
- 한국어로 대화하세요.
- 당신의 인물 특성에 맞는 어조와 표현을 사용하세요."""


# ── 대화 (Pro) ────────────────────────────────────────────
def chat_with_character(
    api_key: str,
    user_message: str,
    character: dict,
    bible_text: str,
    history: list[dict],
) -> str:
    system_prompt = build_system_prompt(character, bible_text)
    client = make_client(api_key)

    gemini_history = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["content"])],
        )
        for m in history[-MAX_HISTORY:]
    ]
    chat = client.chats.create(
        model=PRO_MODEL,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            http_options=types.HttpOptions(timeout=30_000),
        ),
        history=gemini_history,
    )
    return safe_str(chat.send_message(user_message).text)


# ── 에러 핸들링 ───────────────────────────────────────────
def show_api_error(e: Exception) -> None:
    if isinstance(e, genai_errors.ClientError):
        code = getattr(e, "code", None)
        if code in (401, 403):
            st.error("🔑 **API 키가 올바르지 않습니다.**")
            st.info(
                "**해결 방법**\n"
                "1. 사이드바에서 API 키를 다시 확인해주세요.\n"
                "2. [Google AI Studio](https://aistudio.google.com/app/apikey)에서 새 키를 발급받으세요.\n"
                "3. 키를 복사할 때 앞뒤 공백이 없는지 확인하세요."
            )
        elif code == 429:
            st.error("⏳ **API 호출 한도(quota)를 초과했습니다.**")
            st.info(
                "**해결 방법**\n"
                "1. 1~2분 후 다시 시도해주세요.\n"
                "2. 계속 초과된다면 [Google AI Studio](https://aistudio.google.com/app/apikey)에서 결제 설정을 확인하세요."
            )
        else:
            st.error("⚠️ **API 요청 중 오류가 발생했습니다.**")
            st.info("잠시 후 다시 시도해주세요.")
    elif isinstance(e, genai_errors.ServerError):
        code = getattr(e, "code", None)
        if code == 503:
            st.error("🔄 **AI 서버가 일시적으로 혼잡합니다.**")
            st.info("30초~1분 후 다시 시도해주세요.")
        elif code == 504:
            st.error("⏱️ **AI 서버 응답 시간이 초과되었습니다. (504)**")
            st.info(
                "서버가 일시적으로 지연되고 있습니다.\n"
                "잠시 후 다시 시도하거나, 성경 본문을 더 짧게 입력해보세요."
            )
        else:
            st.error("🔄 **AI 서버에서 일시적인 오류가 발생했습니다.**")
            st.info("잠시 후 다시 시도해주세요.")
    elif isinstance(e, TimeoutError):
        st.error("⏱️ **응답 시간이 초과되었습니다.**")
        st.info("네트워크 상태를 확인하거나 잠시 후 다시 시도해주세요.")
    else:
        st.error("⚠️ **일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.**")
        _log(f"[오류] {type(e).__name__}: {e}\n{traceback.format_exc()}")


# ── 대화 이력 텍스트 빌드 ─────────────────────────────────
def build_download_text(char: dict, bible_text: str, messages: list) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    sep = "=" * 50
    lines = [
        sep,
        "AI 큐티 메이트 — 대화 이력",
        f"저장 일시: {now}",
        f"대화 상대: {char['name']}",
        sep,
    ]
    if bible_text:
        lines += ["", "[오늘의 말씀]", bible_text]
    lines += ["", "[대화 내용]"]
    for msg in messages:
        label = "나" if msg["role"] == "user" else char["name"]
        lines += ["", f"{label}:", msg["content"]]
    lines.append("\n" + sep)
    return "\n".join(lines)


def render_tab_flow(prefix: str, bible_text_val: str) -> None:
    """탭별 독립 인물 분석 → 인물 선택 → 채팅 흐름"""
    done_key  = f"{prefix}_extraction_done"
    chars_key = f"{prefix}_characters"
    sel_key   = f"{prefix}_selected_character"
    msgs_key  = f"{prefix}_messages"

    if not st.session_state.get(done_key):
        return

    # ── Step 2: 인물 선택 ────────────────────────────────
    st.header("👥 대화 상대 선택")
    options = [DEFAULT_CHARACTER] + st.session_state[chars_key]
    names   = [c["name"] for c in options]

    selected_name = st.radio(
        "대화하고 싶은 인물을 선택하세요:", names, horizontal=True,
        key=f"{prefix}_char_radio",
    )
    selected = next((c for c in options if c["name"] == selected_name), DEFAULT_CHARACTER)

    if st.session_state[sel_key] != selected:
        st.session_state[sel_key] = selected
        st.session_state[msgs_key] = []

    with st.expander(f"📋 {selected['name']} 소개", expanded=False):
        st.markdown(f"**역할:** {selected['role']}")
        st.markdown(f"**설명:** {selected['description']}")

    # ── Step 3: 채팅 ──────────────────────────────────────
    char = st.session_state[sel_key]
    if not char:
        return

    st.header(f"💬 {char['name']}와(과) 대화")

    if not st.session_state[msgs_key]:
        st.info(f"안녕하세요, {char['name']}입니다. 오늘의 말씀에 대해 무엇이든 질문해 보세요.")

    for msg in st.session_state[msgs_key]:
        avatar = None if msg["role"] == "user" else "✝"
        with st.chat_message("user" if msg["role"] == "user" else "assistant", avatar=avatar):
            st.write(msg["content"])

    if user_input := st.chat_input(f"{char['name']}에게 질문하세요...", key=f"{prefix}_chat_input"):
        st.session_state[msgs_key].append({"role": "user", "content": user_input})
        if len(st.session_state[msgs_key]) > MAX_HISTORY * 2:
            st.session_state[msgs_key] = st.session_state[msgs_key][-(MAX_HISTORY * 2):]

        with st.spinner(f"{char['name']}이(가) 말씀을 묵상하며 답변하는 중..."):
            answer = error = None
            history = st.session_state[msgs_key][:-1]
            waits = [0] + QUOTA_RETRY_WAITS
            for attempt, wait in enumerate(waits):
                if wait:
                    time.sleep(wait)
                try:
                    answer = chat_with_character(
                        get_api_key(), user_input, char, bible_text_val, history,
                    )
                    break
                except genai_errors.ClientError as e:
                    if getattr(e, "code", None) == 429 and attempt < len(QUOTA_RETRY_WAITS):
                        continue
                    error = e
                    break
                except genai_errors.ServerError as e:
                    if getattr(e, "code", None) == 503 and attempt < len(QUOTA_RETRY_WAITS):
                        time.sleep(3)
                        continue
                    error = e
                    break
                except Exception as e:
                    error = e
                    break

        if answer is not None:
            st.session_state[msgs_key].append({"role": "assistant", "content": answer})
            st.rerun()
        else:
            st.session_state[msgs_key].pop()
            if error:
                show_api_error(error)

    if st.session_state[msgs_key]:
        col_dl, col_reset = st.columns(2)
        with col_dl:
            filename = f"큐티대화_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            st.download_button(
                "💾 대화 이력 저장",
                data=build_download_text(char, bible_text_val, st.session_state[msgs_key]).encode("utf-8"),
                file_name=filename,
                mime="text/plain",
                use_container_width=True,
                key=f"{prefix}_download",
            )
        with col_reset:
            if st.button("🗑️ 대화 초기화", type="secondary", use_container_width=True, key=f"{prefix}_reset"):
                st.session_state[msgs_key] = []
                st.rerun()


# ── 사이드바 ──────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        st.header("⚙️ 설정")

        if server_key_set():
            st.success("Gemini API 키가 서버에 설정되어 있습니다 ✓")
        else:
            st.markdown("**🔑 Gemini API 키**")
            entered = st.text_input(
                "API 키",
                type="password",
                value=st.session_state.api_key,
                placeholder="AIza...",
                label_visibility="collapsed",
            )
            if entered != st.session_state.api_key:
                st.session_state.api_key = entered

            if not st.session_state.api_key:
                st.caption(
                    "API 키가 없으신가요?  \n"
                    "[Google AI Studio](https://aistudio.google.com/app/apikey)에서 "
                    "무료로 발급받을 수 있습니다."
                )

        st.divider()
        st.markdown("**📖 사용 방법**")
        st.markdown(
            "1. 오늘 읽은 성경 본문을 입력하세요.\n"
            "2. **인물 분석** 버튼을 누르세요.\n"
            "3. 대화하고 싶은 인물을 선택하세요.\n"
            "4. 질문을 입력하고 대화를 시작하세요!"
        )
        st.divider()
        st.caption(f"인물 분석: `{FLASH_MODEL}`")
        st.caption(f"대화: `{PRO_MODEL}`")


# ── 메인 ──────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="AI 큐티 메이트", page_icon="✝", layout="wide")
    init_session()
    render_sidebar()

    st.title("✝ AI 큐티 메이트")
    st.caption("오늘의 말씀과 함께하는 AI 성경 묵상 파트너")

    # API 키 없음 → 안내 후 조기 종료
    if not get_api_key():
        st.warning(
            "⚠️ **Gemini API 키가 필요합니다.**\n\n"
            "왼쪽 사이드바에 API 키를 입력하면 바로 시작할 수 있습니다.\n\n"
            "[🔗 Google AI Studio에서 무료 API 키 발급하기](https://aistudio.google.com/app/apikey)"
        )
        st.stop()

    # ── UI 스타일 ─────────────────────────────────────────
    st.markdown("""
<style>
/* ── 페이지 배경 ── */
.stApp { background-color: #fdfbff !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(170deg, #f0e9ff 0%, #faf6ff 100%) !important;
}

/* ── 탭 버튼: 각 50% 너비 ── */
[data-baseweb="tab-list"] {
    display: flex !important;
    background: #ede8fb !important;
    border-radius: 16px !important;
    padding: 5px !important;
    gap: 5px !important;
    border: none !important;
    width: 100% !important;
}
button[role="tab"] {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    border-radius: 12px !important;
    padding: 13px 10px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: #7c3aed !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.25s ease !important;
    white-space: nowrap !important;
}
button[role="tab"]:hover:not([aria-selected="true"]) {
    background: rgba(124,58,237,0.09) !important;
}
button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%) !important;
    color: white !important;
    box-shadow: 0 4px 16px rgba(124,58,237,0.35) !important;
}
[data-baseweb="tab-highlight"],
[data-baseweb="tab-border"] { display: none !important; }
[data-baseweb="tab-panel"] { padding: 1.5rem 0.1rem 0 !important; }

/* ── 셀렉트박스 ── */
[data-baseweb="select"] > div:first-child {
    border-radius: 12px !important;
    border: 1.5px solid #ddd4f8 !important;
    background: linear-gradient(135deg, #faf7ff, #f4eeff) !important;
    transition: border-color .2s, box-shadow .2s !important;
}
[data-baseweb="select"] > div:first-child:hover {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.13) !important;
}
.stSelectbox label {
    font-weight: 700 !important;
    color: #5b21b6 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── 텍스트 영역 ── */
.stTextArea textarea {
    border-radius: 12px !important;
    border: 1.5px solid #ddd4f8 !important;
    background: linear-gradient(135deg, #faf7ff, #f4eeff) !important;
    transition: border-color .2s, box-shadow .2s !important;
    font-size: 0.93rem !important;
}
.stTextArea textarea:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.13) !important;
}
.stTextArea label {
    font-weight: 700 !important;
    color: #5b21b6 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── 버튼: Primary ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 16px rgba(124,58,237,0.32) !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 24px rgba(124,58,237,0.48) !important;
    transform: translateY(-1px) !important;
}

/* ── 버튼: Secondary ── */
.stButton > button[kind="secondary"] {
    border-radius: 12px !important;
    border: 1.5px solid #ddd4f8 !important;
    color: #6d28d9 !important;
    background: white !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #f5f0ff !important;
    border-color: #8b5cf6 !important;
}

/* ── 제목 색상 ── */
.stApp h1 { color: #3b0764 !important; }
.stApp h2 { color: #4c1d95 !important; }
.stApp h3 { color: #5b21b6 !important; }

/* ── 라디오 버튼 — primaryColor(#7c3aed)로 테마 자동 적용 ── */
[data-baseweb="radio"] [role="radio"] > div:first-child {
    border-color: #c4b5fd !important;
    transition: border-color .2s !important;
}
[data-baseweb="radio"] [role="radio"][aria-checked="true"] > div:first-child {
    border-color: #7c3aed !important;
    background-color: #7c3aed !important;
}
[data-baseweb="radio"] [role="radio"][aria-checked="false"] > div:first-child {
    background-color: white !important;
}
[data-baseweb="radio"] [role="radio"]:hover > div:first-child {
    border-color: #8b5cf6 !important;
}

/* ── 알림/인포박스 (파란색 → 보라색) ── */
[data-testid="stAlert"] > div {
    border-radius: 12px !important;
    background: linear-gradient(135deg, #f5f0ff, #fdf8ff) !important;
    border-left: 4px solid #8b5cf6 !important;
    color: #4c1d95 !important;
}
[data-testid="stAlert"] p { color: #4c1d95 !important; }

/* ── 경고박스 ── */
[data-testid="stAlert"][data-type="warning"] > div {
    background: linear-gradient(135deg, #fffbeb, #fef9e7) !important;
    border-left-color: #f59e0b !important;
    color: #78350f !important;
}

/* ── 에러박스 ── */
[data-testid="stAlert"][data-type="error"] > div {
    background: linear-gradient(135deg, #fff1f2, #fef2f2) !important;
    border-left-color: #f43f5e !important;
    color: #9f1239 !important;
}

/* ── 익스팬더 ── */
[data-testid="stExpander"] {
    border-radius: 12px !important;
    border: 1.5px solid #ddd4f8 !important;
    background: white !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    color: #5b21b6 !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] summary:hover {
    background: #f5f0ff !important;
}
[data-testid="stExpander"] summary svg { color: #8b5cf6 !important; }

/* ── 채팅 입력창 wrapper 배경 제거 ── */
[data-testid="stChatInput"] {
    background: transparent !important;
    padding: 0 !important;
}
[data-testid="stBottom"],
[data-testid="stBottom"] > div {
    background: transparent !important;
    box-shadow: none !important;
    border-top: none !important;
}
/* ── 채팅 입력창 pill 스타일 ── */
[data-testid="stChatInput"] > div {
    border-radius: 20px !important;
    border: 2px solid #c4b5fd !important;
    background: rgba(255,255,255,0.95) !important;
    box-shadow: 0 2px 16px rgba(124,58,237,0.12) !important;
    transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.15), 0 4px 20px rgba(124,58,237,0.18) !important;
}
/* 내부 컨테이너: grid로 textarea(좌) | 버튼(우) 배치 */
[data-testid="stChatInput"] > div > div {
    display: grid !important;
    grid-template-columns: 1fr auto !important;
    grid-template-rows: auto !important;
    align-items: center !important;
}
[data-testid="stChatInput"] > div > div > div:first-child {
    grid-column: 1 !important;
    grid-row: 1 !important;
    min-width: 0 !important;
}
[data-testid="stChatInput"] > div > div > div:last-child {
    grid-column: 2 !important;
    grid-row: 1 !important;
    align-self: center !important;
    padding-right: 0.35rem !important;
}
[data-testid="stChatInputSubmitButton"] button {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    border-radius: 50% !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(124,58,237,0.35) !important;
}

/* ── 다운로드 버튼 ── */
[data-testid="stDownloadButton"] button {
    border-radius: 12px !important;
    border: 1.5px solid #ddd4f8 !important;
    color: #6d28d9 !important;
    background: white !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: #f5f0ff !important;
    border-color: #8b5cf6 !important;
    color: #4c1d95 !important;
}

/* ── 채팅 메시지 버블 ── */
[data-testid="stChatMessage"] {
    border-radius: 14px !important;
    border: 1px solid #ede8fb !important;
    background: white !important;
    box-shadow: 0 2px 8px rgba(124,58,237,0.06) !important;
    padding-right: 1.2rem !important;
}

/* ── 스피너 ── */
[data-testid="stSpinner"] > div {
    color: #7c3aed !important;
}

/* ── 인물 분석 완료 뱃지 ── */
.badge-done {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: linear-gradient(135deg, #ede9fe, #ddd6fe);
    color: #5b21b6;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
    border: 1px solid #c4b5fd;
    white-space: nowrap;
}

/* ─────────────────────────────────────
   모바일 반응형
   ───────────────────────────────────── */

/* 태블릿 / 대형 폰 (≤ 768px) */
@media screen and (max-width: 768px) {
    .stApp h1 { font-size: 1.6rem !important; }

    button[role="tab"] {
        font-size: 0.88rem !important;
        padding: 11px 8px !important;
    }

    /* 가로 라디오 — 줄바꿈 허용 */
    [data-testid="stRadio"] > div:last-child {
        flex-wrap: wrap !important;
        gap: 6px 12px !important;
    }
}

/* 일반 폰 (≤ 480px) */
@media screen and (max-width: 480px) {
    /* 제목 */
    .stApp h1 { font-size: 1.35rem !important; }
    .stApp h2 { font-size: 1.1rem !important; }
    .stApp h3 { font-size: 0.95rem !important; }

    /* 탭 리스트 */
    [data-baseweb="tab-list"] {
        padding: 4px !important;
        gap: 4px !important;
        border-radius: 12px !important;
    }
    /* 탭 버튼 — 텍스트 줄바꿈 허용 */
    button[role="tab"] {
        font-size: 0.75rem !important;
        padding: 9px 5px !important;
        border-radius: 9px !important;
        white-space: normal !important;
        word-break: keep-all !important;
        line-height: 1.3 !important;
        text-align: center !important;
    }

    /* 라디오 — 줄바꿈 강제 */
    [data-testid="stRadio"] > div:last-child {
        flex-wrap: wrap !important;
        gap: 6px 10px !important;
    }

    /* 셀렉트박스 */
    .stSelectbox label {
        font-size: 0.72rem !important;
    }

    /* 텍스트 영역 */
    .stTextArea textarea { font-size: 0.85rem !important; }
    .stTextArea label   { font-size: 0.72rem !important; }

    /* "Press ⌘+Enter to apply" 힌트 텍스트 — 모바일 키보드 미지원으로 불필요, 숨김 */
    [data-testid="InputInstructions"] { display: none !important; }

    /* 버튼 */
    .stButton > button,
    [data-testid="stDownloadButton"] button {
        font-size: 0.82rem !important;
    }

    /* 뱃지 */
    .badge-done {
        font-size: 0.72rem !important;
        padding: 3px 8px !important;
    }

    /* 채팅 메시지 패딩 */
    [data-testid="stChatMessage"] {
        padding-right: 0.5rem !important;
    }

    /* 채팅 입력창 pill 반경 */
    [data-testid="stChatInput"] > div {
        border-radius: 24px !important;
    }
}

/* 초소형 폰 (≤ 360px) */
@media screen and (max-width: 360px) {
    .stApp h1 { font-size: 1.15rem !important; }
    .stApp h2 { font-size: 0.98rem !important; }

    [data-baseweb="tab-list"] {
        padding: 3px !important;
        gap: 3px !important;
    }
    button[role="tab"] {
        font-size: 0.68rem !important;
        padding: 8px 4px !important;
        border-radius: 8px !important;
    }

    .badge-done {
        font-size: 0.66rem !important;
        padding: 3px 6px !important;
    }

    .stButton > button,
    [data-testid="stDownloadButton"] button {
        font-size: 0.75rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

    # ── Step 1~3: 탭별 완전 독립 ─────────────────────────
    st.header("📖 오늘의 말씀")
    tab_input, tab_fetch = st.tabs(["✍️ 직접 입력", "📥 성경 구절 불러오기"])

    with tab_input:
        st.text_area(
            "성경 본문",
            height=180,
            key="direct_text",
            label_visibility="collapsed",
            placeholder=(
                "예) 요한복음 3:16  하나님이 세상을 이처럼 사랑하사 독생자를 주셨으니 "
                "이는 그를 믿는 자마다 멸망하지 않고 영생을 얻게 하려 하심이라.\n\n"
                "오늘 읽은 성경 구절이나 단락을 입력하세요."
            ),
        )
        c_btn, c_status = st.columns([1, 3])
        with c_btn:
            direct_analyze = st.button("🔍 인물 분석", type="primary", use_container_width=True, key="analyze_direct")
        with c_status:
            if st.session_state.direct_extraction_done:
                st.markdown('<span class="badge-done">✓ 인물 분석 완료</span>', unsafe_allow_html=True)

        if direct_analyze:
            if not st.session_state.direct_text.strip():
                st.warning("성경 본문을 먼저 입력해주세요.")
            else:
                st.session_state.direct_extraction_done    = False
                st.session_state.direct_selected_character = None
                st.session_state.direct_messages           = []
                with st.spinner("성경 본문에서 등장인물을 분석하는 중..."):
                    try:
                        chars = extract_characters(get_api_key(), st.session_state.direct_text)
                        st.session_state.direct_characters      = chars
                        st.session_state.direct_extraction_done = True
                    except Exception as e:
                        show_api_error(e)
                if st.session_state.direct_extraction_done:
                    if not st.session_state.direct_characters:
                        st.info("명확한 등장인물을 찾지 못했습니다. '지혜로운 조언자'와 대화를 시작할 수 있습니다.")
                    st.rerun()

        render_tab_flow("direct", st.session_state.direct_text)

    with tab_fetch:
        book_name = st.selectbox(
            "성경책", list(BIBLE_BOOKS.keys()),
            key="fetch_book",
            on_change=_reset_chapter_and_verses,
        )
        max_ch = BIBLE_CHAPTERS.get(book_name, 50)
        c1, c2, c3 = st.columns(3)
        with c1:
            chapter_num = st.selectbox(
                "장", list(range(1, max_ch + 1)),
                key="fetch_chapter",
                on_change=_reset_verses,
            )

        verse_count = get_verse_count(BIBLE_BOOKS[book_name], chapter_num)
        verse_options = list(range(1, verse_count + 1))

        sv_cur = st.session_state.get("fetch_sv", 1)
        sv_idx = verse_options.index(sv_cur) if sv_cur in verse_options else 0
        with c2:
            sv = st.selectbox("시작 절", verse_options, index=sv_idx)
        if sv != st.session_state.fetch_sv:
            _reset_fetch_analysis()
        st.session_state.fetch_sv = sv

        end_options = [v for v in verse_options if v >= sv]
        ev_cur = st.session_state.get("fetch_ev", sv)
        if ev_cur not in end_options:
            ev_cur = sv
        ev_idx = end_options.index(ev_cur)
        with c3:
            ev = st.selectbox("끝 절", end_options, index=ev_idx)
        if ev != st.session_state.fetch_ev:
            _reset_fetch_analysis()
        st.session_state.fetch_ev = ev

        if st.button("📥 말씀 불러오기", type="primary"):
            with st.spinner("말씀을 불러오는 중..."):
                fetched = fetch_bible_text(BIBLE_BOOKS[book_name], chapter_num, sv, ev)
            if fetched:
                header = f"{book_name} {chapter_num}:{sv}" + (f"~{ev}" if ev > sv else "")
                st.session_state.fetch_preview = f"[{header}]\n{fetched}"
            else:
                st.error("말씀을 불러오지 못했습니다. 장/절 번호를 확인해주세요.")

        st.text_area(
            "📖 말씀 미리보기",
            value=st.session_state.fetch_preview,
            height=180,
            disabled=True,
            placeholder="성경책, 장, 절을 선택하고 '말씀 불러오기'를 누르세요.",
        )

        c_btn2, c_status2 = st.columns([1, 3])
        with c_btn2:
            fetch_analyze = st.button("🔍 인물 분석", type="primary", use_container_width=True, key="analyze_fetch")
        with c_status2:
            if st.session_state.fetch_extraction_done:
                st.markdown('<span class="badge-done">✓ 인물 분석 완료</span>', unsafe_allow_html=True)

        if fetch_analyze:
            if not st.session_state.fetch_preview.strip():
                st.warning("말씀을 먼저 불러와주세요.")
            else:
                st.session_state.fetch_extraction_done    = False
                st.session_state.fetch_selected_character = None
                st.session_state.fetch_messages           = []
                with st.spinner("성경 본문에서 등장인물을 분석하는 중..."):
                    try:
                        chars = extract_characters(get_api_key(), st.session_state.fetch_preview)
                        st.session_state.fetch_characters      = chars
                        st.session_state.fetch_extraction_done = True
                    except Exception as e:
                        show_api_error(e)
                if st.session_state.fetch_extraction_done:
                    if not st.session_state.fetch_characters:
                        st.info("명확한 등장인물을 찾지 못했습니다. '지혜로운 조언자'와 대화를 시작할 수 있습니다.")
                    st.rerun()

        render_tab_flow("fetch", st.session_state.fetch_preview)


if __name__ == "__main__":
    main()
