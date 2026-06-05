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
FLASH_MODEL     = "gemini-2.5-flash-lite"
PRO_MODEL       = "gemini-2.5-flash"
MAX_HISTORY     = 10
EXTRACT_TIMEOUT = 15  # seconds

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
        "api_key":            "",
        "bible_text":         "",
        "characters":         [],
        "selected_character": None,
        "messages":           [],
        "extraction_done":    False,
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


# ── 성경 불러오기 ─────────────────────────────────────────
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
    try:
        resp = client.models.generate_content(
            model=FLASH_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                http_options=types.HttpOptions(timeout=EXTRACT_TIMEOUT * 1000),
            ),
        )
        raw = safe_str(resp.text).strip()
    except genai_errors.ClientError:
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
            st.error(f"API 오류 (코드 {code}): {e}")
    elif isinstance(e, genai_errors.ServerError) and getattr(e, "code", None) == 503:
        st.error("🔄 **AI 서버가 일시적으로 혼잡합니다.**")
        st.info("30초~1분 후 다시 시도해주세요.")
    elif isinstance(e, TimeoutError):
        st.error("⏱️ **응답 시간이 초과되었습니다.**")
        st.info("네트워크 상태를 확인하거나 잠시 후 다시 시도해주세요.")
    else:
        st.error(f"오류가 발생했습니다: {type(e).__name__}: {e}")
        with st.expander("🔍 상세 오류 정보"):
            st.code(traceback.format_exc())


# ── 대화 이력 텍스트 빌드 ─────────────────────────────────
def build_download_text() -> str:
    char = st.session_state.selected_character or DEFAULT_CHARACTER
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")
    sep  = "=" * 50
    lines = [
        sep,
        "AI 큐티 메이트 — 대화 이력",
        f"저장 일시: {now}",
        f"대화 상대: {char['name']}",
        sep,
    ]
    if st.session_state.bible_text:
        lines += ["", "[오늘의 말씀]", st.session_state.bible_text]
    lines += ["", "[대화 내용]"]
    for msg in st.session_state.messages:
        label = "나" if msg["role"] == "user" else char["name"]
        lines += ["", f"{label}:", msg["content"]]
    lines.append("\n" + sep)
    return "\n".join(lines)


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

    # ── Step 1: 성경 본문 입력 ────────────────────────────
    st.header("📖 오늘의 말씀")

    fetch_mode = st.toggle("📥 성경 구절 불러오기", value=False)

    if fetch_mode:
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        with c1:
            book_name = st.selectbox("성경책", list(BIBLE_BOOKS.keys()), label_visibility="collapsed")
        with c2:
            chapter_num = st.number_input("장", min_value=1, max_value=150, value=1, step=1)
        with c3:
            sv = st.number_input("시작 절", min_value=1, value=1, step=1)
        with c4:
            ev = st.number_input("끝 절", min_value=1, value=1, step=1)

        if st.button("📥 말씀 불러오기", type="secondary"):
            with st.spinner("말씀을 불러오는 중..."):
                fetched = fetch_bible_text(BIBLE_BOOKS[book_name], chapter_num, sv, ev)
            if fetched:
                header = f"{book_name} {chapter_num}:{sv}" + (f"~{ev}" if ev > sv else "")
                st.session_state.bible_text = f"[{header}]\n{fetched}"
                st.session_state.extraction_done = False
                st.session_state.selected_character = None
                st.session_state.messages = []
                st.rerun()
            else:
                st.error("말씀을 불러오지 못했습니다. 장/절 번호를 확인해주세요.")

    bible_text = st.text_area(
        "성경 본문",
        value=st.session_state.bible_text,
        height=180,
        label_visibility="collapsed",
        placeholder=(
            "예) 요한복음 3:16  하나님이 세상을 이처럼 사랑하사 독생자를 주셨으니 "
            "이는 그를 믿는 자마다 멸망하지 않고 영생을 얻게 하려 하심이라.\n\n"
            "오늘 읽은 성경 구절이나 단락을 입력하세요."
        ),
    )

    col_btn, col_status = st.columns([1, 4])
    with col_btn:
        analyze_clicked = st.button("🔍 인물 분석", type="primary", use_container_width=True)
    with col_status:
        if st.session_state.extraction_done:
            st.caption("✅ 인물 분석 완료")

    if analyze_clicked:
        if not bible_text.strip():
            st.warning("성경 본문을 먼저 입력해주세요.")
        else:
            st.session_state.bible_text         = bible_text
            st.session_state.messages           = []
            st.session_state.selected_character = None
            st.session_state.extraction_done    = False

            with st.spinner("성경 본문에서 등장인물을 분석하는 중..."):
                try:
                    chars = extract_characters(get_api_key(), bible_text)
                    st.session_state.characters      = chars
                    st.session_state.extraction_done = True
                    if not chars:
                        st.info(
                            "명확한 등장인물을 찾지 못했습니다. "
                            "'지혜로운 조언자'와 대화를 시작할 수 있습니다."
                        )
                except Exception as e:
                    show_api_error(e)

    # ── Step 2: 인물 선택 ────────────────────────────────
    if st.session_state.extraction_done:
        st.header("👥 대화 상대 선택")
        options = [DEFAULT_CHARACTER] + st.session_state.characters
        names   = [c["name"] for c in options]

        selected_name = st.radio("대화하고 싶은 인물을 선택하세요:", names, horizontal=True)
        selected = next((c for c in options if c["name"] == selected_name), DEFAULT_CHARACTER)

        if st.session_state.selected_character != selected:
            st.session_state.selected_character = selected
            st.session_state.messages           = []

        with st.expander(f"📋 {selected['name']} 소개", expanded=False):
            st.markdown(f"**역할:** {selected['role']}")
            st.markdown(f"**설명:** {selected['description']}")

    # ── Step 3: 채팅 ──────────────────────────────────────
    if not st.session_state.selected_character:
        return

    char = st.session_state.selected_character
    st.header(f"💬 {char['name']}와(과) 대화")

    if not st.session_state.messages:
        st.info(f"안녕하세요, {char['name']}입니다. 오늘의 말씀에 대해 무엇이든 질문해 보세요.")

    for msg in st.session_state.messages:
        avatar = None if msg["role"] == "user" else "✝"
        with st.chat_message("user" if msg["role"] == "user" else "assistant", avatar=avatar):
            st.write(msg["content"])

    if user_input := st.chat_input(f"{char['name']}에게 질문하세요..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        if len(st.session_state.messages) > MAX_HISTORY * 2:
            st.session_state.messages = st.session_state.messages[-(MAX_HISTORY * 2):]

        with st.spinner(f"{char['name']}이(가) 말씀을 묵상하며 답변하는 중..."):
            answer = error = None
            history = st.session_state.messages[:-1]

            for attempt in range(3):
                try:
                    answer = chat_with_character(
                        get_api_key(), user_input, char,
                        st.session_state.bible_text, history,
                    )
                    break
                except genai_errors.ServerError as e:
                    if getattr(e, "code", None) == 503 and attempt < 2:
                        time.sleep(3)
                        continue
                    error = e
                    break
                except Exception as e:
                    error = e
                    break

        if answer is not None:
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()
        else:
            st.session_state.messages.pop()
            if error:
                show_api_error(error)

    # 하단 버튼 (대화 이력이 있을 때만)
    if st.session_state.messages:
        col_dl, col_reset = st.columns(2)
        with col_dl:
            filename = f"큐티대화_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            st.download_button(
                "💾 대화 이력 저장",
                data=build_download_text().encode("utf-8"),
                file_name=filename,
                mime="text/plain",
                use_container_width=True,
            )
        with col_reset:
            if st.button("🗑️ 대화 초기화", type="secondary", use_container_width=True):
                st.session_state.messages = []
                st.rerun()


if __name__ == "__main__":
    main()
