import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Set, Optional

from ltp import LTP

CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-\+]{1,}")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？])")

PUNCT_STRIP = " \t\r\n\"'“”‘’（）()《》〈〉【】[]{}!?？！：；、，．.·•-—~～"

ASCII_WHITELIST = {
    'O3', 'AI', 'AGI', 'GRL', 'GaoZheng', 'GNLA', 'PFB', 'DERI', 'GCPOLAA', 'Git', 'arXiv',
    'Mathematica'
}

ASCII_STOPWORDS = {
    'CEO', 'CFO', 'PDF', 'PPT', 'ID', 'IP', 'PK', 'VIP', 'DIY', 'OK', 'OKR', 'HR', 'PR', 'QA',
    'QR', 'GDP', 'CPI', 'KPI', 'USB', 'CPU', 'GPU', 'HTTP', 'HTTPS'
}

TAG_MAP = {
    'Nh': 'person',
    'Ns': 'location',
    'Ni': 'organization'
}

REMOVE_PREFIXES = ('老', '小')
REMOVE_SUFFIXES = (
    '老', '总', '们', '家', '老师', '教授', '同学', '同志', '先生', '小姐', '太太', '伯', '叔', '姨',
    '舅', '爸', '妈', '爷', '奶', '哥', '姐', '嫂', '弟', '妹', '博士', '大'
)

INVALID_TRAIL_CHARS = set('会地飞重静你呢啊呀吧么吶啦呐儿们着了过')
REPETITIVE_SUBSTRINGS = ('静静', '重重')

SINGLE_SURNAME = set(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐"
    "费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄"
    "穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜"
    "阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞"
    "万支柯咎管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪包诸左石崔吉钮龚程"
    "嵇邢滑裴陆荣翟谭贡劳逄姬申扶堵冉宰郦雍郤璩桑桂濮牛寿通边扈燕冀郏"
    "浦尚农温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都"
    "耿满弘匡国文寇广禄阙东欧利蔚越夔隆师鞠顾渠"
)

DOUBLE_SURNAME = {
    '欧阳', '司马', '上官', '诸葛', '东方', '西门', '皇甫', '尉迟', '公孙', '夏侯',
    '长孙', '宇文', '轩辕', '令狐', '钟离', '闾丘', '南宫', '北堂', '东郭', '第五',
    '羊舌', '司徒', '司空', '万俟', '呼延', '王孙', '澹台', '公冶', '太史', '端木',
    '谷梁', '左丘', '公良', '拓跋', '独孤', '南门', '百里', '申屠', '仲孙', '亓官',
    '司寇', '巫马', '公西', '公孟', '乐正', '壤驷', '宰父', '夹谷', '鲜于', '钟仪',
    '段干', '随侯', '羊角', '王官'
}

LOCATION_WHITELIST = {
    '中国', '美国', '德国', '挪威', '泰山', '奥斯陆', '欧洲', '北美', '华南', '华南区', '华东区',
    '华尔街', '斯德哥尔摩', '瑞士', '瑞典', '东欧', '临海', '临海市', '云南', '京北', '京北市',
    '京海', '京海市', '新安', '哥廷根', '哥本哈根', '黄金岛', '太平洋', '上甘岭', '普洱',
    '武夷山'
}

ORGANIZATION_WHITELIST = {
    '京北大学', '京海大学', '文学院', '法学院', '瑞典皇家科学院', '欧洲核子研究中心', '国家博物馆',
    '国际会议中心', '马普所', '科研院所', '微软', '马普研究所'
}

BANNED_TERMS = {
    '大学', '中心', '安逸', '荣休', '阳光', '金钟罩', '雷霆万钧', '滑向深渊', '滑铁卢',
    '蓦然回首', '毛头小子', '网络安全', '终得宝', '石之声', '申请专利', '旧版本', '新生事物',
    '无底洞', '朝圣地', '文明史', '花心思', '宣告成立', '安全性', '张开', '余温',
    'offer', 'kcal', 'gaozheng', 'jinghai-university', '宋体', '雷霆万', '维特', '达摩', '赵董', '高正那', 'A4', 'Author'
}

MANUAL_NAMES = {
    '维特根斯坦': 'person',
    '达摩克利斯': 'person'
}


def normalize(token: str) -> str:
    cleaned = token.strip(PUNCT_STRIP)
    cleaned = re.sub(r'^[\s\W_]+', '', cleaned)
    cleaned = re.sub(r'[\s\W_]+$', '', cleaned)
    return cleaned


def chunk_text(text: str, max_len: int = 200) -> List[str]:
    sentences = [s for s in SENTENCE_SPLIT_RE.split(text) if s]
    chunks: List[str] = []
    current = ''
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) > max_len and current:
            chunks.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        chunks.append(current)
    if not chunks:
        return [text]
    return chunks


def keep_ascii(token: str) -> bool:
    if not token:
        return False
    if token in ASCII_WHITELIST or token.upper() in ASCII_WHITELIST:
        return True
    if token.upper() in ASCII_STOPWORDS:
        return False
    if token.islower():
        return False
    if any(ch.isdigit() for ch in token):
        return True
    letters = [ch for ch in token if ch.isalpha()]
    if not letters:
        return False
    has_lower = any(ch.islower() for ch in letters)
    has_upper = any(ch.isupper() for ch in letters)
    if has_lower and has_upper:
        return True
    return len(token) > 3


def extract_ascii_tokens(text: str) -> Counter:
    counter: Counter[str] = Counter()
    for match in ASCII_TOKEN_RE.finditer(text):
        token = match.group()
        if keep_ascii(token):
            counter[token] += 1
    return counter


def chinese_length(token: str) -> int:
    return len(CHINESE_RE.findall(token))


def load_common_words(path: Path) -> Set[str]:
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return set()
    words: Set[str] = set()
    if isinstance(raw, dict):
        words.update(str(key).strip() for key in raw.keys() if str(key).strip())
        return words
    if not isinstance(raw, list):
        return words
    for item in raw:
        if isinstance(item, str):
            token = item.strip()
            if token:
                words.add(token)
        elif isinstance(item, dict):
            token = str(item.get('word', '')).strip()
            if token:
                words.add(token)
    return words


def strip_trailing_particles(base: str) -> str:
    while base:
        if len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]
            continue
        if base[-1] in INVALID_TRAIL_CHARS:
            base = base[:-1]
            continue
        break
    return base


def refine_person_name(raw: str, common_words: Set[str]) -> Optional[str]:
    base = ''.join(CHINESE_RE.findall(raw.replace('·', '').replace('•', '')))
    if len(base) < 2:
        return None
    base = strip_trailing_particles(base)
    if len(base) < 2:
        return None
    max_len = min(4, len(base))
    surname_candidates: List[tuple[int, str]] = []
    fallback_candidates: List[tuple[int, str]] = []
    for length in range(2, max_len + 1):
        candidate = base[:length]
        if any(sub in candidate for sub in REPETITIVE_SUBSTRINGS):
            continue
        if candidate in common_words:
            continue
        if candidate[-1] in INVALID_TRAIL_CHARS:
            continue
        if any(candidate.startswith(ds) for ds in DOUBLE_SURNAME):
            surname_candidates.append((length, candidate))
        elif candidate[0] in SINGLE_SURNAME:
            surname_candidates.append((length, candidate))
        elif length == 2:
            fallback_candidates.append((length, candidate))
    if surname_candidates:
        surname_candidates.sort(key=lambda item: (-item[0], item[1]))
        return surname_candidates[0][1]
    if base not in common_words and len(base) <= 4:
        return base
    if fallback_candidates:
        fallback_candidates.sort(key=lambda item: (-item[0], item[1]))
        return fallback_candidates[0][1]
    return None


def is_valid_output(token: str, tags: Set[str], common_words: Set[str]) -> bool:
    if token in BANNED_TERMS:
        return False
    if 'ascii' not in tags:
        if chinese_length(token) <= 1:
            return False
        if any(token.endswith(suffix) for suffix in REMOVE_SUFFIXES):
            return False
        if any(token.startswith(prefix) for prefix in REMOVE_PREFIXES) and chinese_length(token) <= 2:
            return False
        if 'person' in tags and token in common_words:
            return False
        if 'organization' in tags and token in common_words and len(token) <= 2:
            return False
        if 'location' in tags and token in common_words and token not in LOCATION_WHITELIST:
            return False
    return True


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    text_path = base_dir / 'data' / 'sample_article.txt'
    output_path = base_dir / 'data' / 'chinese_name_frequency_word.json'
    frequency_source = base_dir / 'data' / 'chinese_frequency_word.json'

    text = text_path.read_text(encoding='utf-8')
    common_words = load_common_words(frequency_source)
    ltp = LTP()

    chunks = []
    for paragraph in text.split('\n\n'):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        chunks.extend(chunk_text(paragraph))

    counter: Counter[str] = Counter()
    tag_counter: defaultdict[str, Set[str]] = defaultdict(set)

    batch_size = 16
    for idx in range(0, len(chunks), batch_size):
        batch = chunks[idx:idx + batch_size]
        result = ltp.pipeline(batch, tasks=['cws', 'ner'])
        for ner_list in result.ner:
            for tag, entity, _start, _end in ner_list:
                if tag not in TAG_MAP:
                    continue
                cleaned = normalize(entity)
                if not cleaned:
                    continue
                tag_type = TAG_MAP[tag]
                if tag_type == 'person':
                    refined = refine_person_name(cleaned, common_words)
                    if not refined:
                        continue
                    cleaned = refined
                elif tag_type == 'location':
                    if cleaned not in LOCATION_WHITELIST and cleaned in BANNED_TERMS:
                        continue
                elif tag_type == 'organization':
                    if cleaned not in ORGANIZATION_WHITELIST and cleaned in BANNED_TERMS:
                        continue
                counter[cleaned] += 1
                tag_counter[cleaned].add(tag_type)

    ascii_counter = extract_ascii_tokens(text)
    for token, freq in ascii_counter.items():
        counter[token] += freq
        tag_counter[token].add('ascii')

    for name, tag in MANUAL_NAMES.items():
        occurrences = len(re.findall(name, text))
        if occurrences:
            counter[name] += occurrences
            tag_counter[name].add(tag)

    filtered_items = [
        (name, freq)
        for name, freq in counter.items()
        if is_valid_output(name, tag_counter[name], common_words)
    ]

    sorted_names = [name for name, _ in sorted(filtered_items, key=lambda item: (-item[1], item[0]))]
    output_path.write_text(json.dumps(sorted_names, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Extracted {len(sorted_names)} names into {output_path}')


if __name__ == '__main__':
    main()
