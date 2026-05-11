import math
from collections import defaultdict

SKILL_ALIASES = {
    "python": "python", "pyhton": "python",
    "java": "java",
    "javascript": "javascript", "javascrpit": "javascript", "js": "javascript",
    "typescript": "typescript", "typescrpit": "typescript",
    "c++": "cpp", "cpp": "cpp",
    "r": "r", "kotlin": "kotlin",
    "machinelearning": "machine_learning", "machine learning": "machine_learning",
    "ml": "machine_learning", "sklearn": "machine_learning",
    "deeplearning": "deep_learning", "deep learning": "deep_learning", "deep-learning": "deep_learning",
    "tensorflow": "tensorflow", "pytorch": "pytorch", "keras": "keras",
    "nlp": "nlp", "bert": "bert", "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics", "stats": "statistics",
    "regression": "regression", "clustering": "clustering",
    "data-viz": "data_visualization", "data visualization": "data_visualization",
    "data viz": "data_visualization", "matplotlib": "data_visualization",
    "tableau": "data_visualization", "power-bi": "data_visualization",
    "power bi": "data_visualization", "powerbi": "data_visualization",
    "pandas": "pandas", "numpy": "numpy",
    "react": "react", "reacts": "react", "reactjs": "react",
    "vue": "vue", "vue.js": "vue", "vuejs": "vue",
    "redux": "redux", "tailwind": "tailwind",
    "html/css": "html_css", "html css": "html_css", "html": "html_css", "css": "html_css",
    "jest": "jest", "graphql": "graphql",
    "node.js": "nodejs", "nodejs": "nodejs", "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot", "springboot": "spring_boot",
    "rest api": "rest_api", "rest": "rest_api", "restapi": "rest_api",
    "microservices": "microservices",
    "sql": "sql", "mysql": "mysql", "mysq": "mysql",
    "postgresql": "postgresql", "postgres": "postgresql",
    "mongodb": "mongodb", "redis": "redis",
    "docker": "docker",
    "kubernetes": "kubernetes", "kubernates": "kubernetes", "k8s": "kubernetes",
    "ci/cd": "ci_cd", "cicd": "ci_cd", "ci cd": "ci_cd",
    "aws": "aws",
    "android": "android", "firebase": "firebase",
    "algorithms": "algorithms", "algoritms": "algorithms",
    "data structure": "data_structures", "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    "ui/ux": "ui_ux", "ui ux": "ui_ux", "figma": "figma",
}

MULTI_WORD = sorted([k for k in SKILL_ALIASES if ' ' in k], key=len, reverse=True)

def normalize(raw):
    text = raw.lower()
    for phrase in MULTI_WORD:
        if phrase in text:
            text = text.replace(phrase, '|||' + SKILL_ALIASES[phrase] + '|||')
    tokens = [t.strip() for t in text.split(',')]
    result = []
    for token in tokens:
        if '|||' in token:
            for part in token.split('|||'):
                part = part.strip()
                if part and part in set(SKILL_ALIASES.values()):
                    result.append(part)
        else:
            token = token.strip()
            if token in SKILL_ALIASES:
                result.append(SKILL_ALIASES[token])
    seen = set()
    deduped = []
    for s in result:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped

resumes = [
    ("Arjun Sharma",    "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"),
    ("Priya Nair",      "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"),
    ("Rahul Gupta",     "Java, Spring Boot, MySql, Microservices, Docker, kubernates"),
    ("Sneha Patel",     "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"),
    ("Vikram Singh",    "C++, Algoritms, Data Structure, competitive programming, python"),
    ("Ananya Krishnan", "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"),
    ("Karan Mehta",     "Python, Sklearn, XGboost, feature engineering, SQL, tableau"),
    ("Deepika Rao",     "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"),
    ("Aditya Kumar",    "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"),
    ("Meera Iyer",      "python, R, statistics, ML, regression, clustering, Power-BI"),
]

jds = [
    ("JD1", "Kakao", "ML Engineer",
     "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization, NLP, BERT, Feature Engineering, Statistics"),
    ("JD2", "Naver", "Backend Engineer",
     "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes, REST API, CI/CD, Redis"),
    ("JD3", "Line", "Frontend Engineer",
     "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS, Node.js, GraphQL, Redux, Jest, AWS"),
]

normalized = [(name, normalize(raw)) for name, raw in resumes]

all_skills = set()
for _, skills in normalized:
    all_skills.update(skills)
vocab = sorted(all_skills)
vocab_idx = {s: i for i, s in enumerate(vocab)}

df = defaultdict(int)
for _, skills in normalized:
    for s in skills:
        df[s] += 1

tfidf_vectors = []
for name, skills in normalized:
    n = len(skills)
    vec = [0.0] * len(vocab)
    for s in skills:
        tf = 1.0 / n
        idf = math.log(10 / df[s])
        vec[vocab_idx[s]] = tf * idf
    tfidf_vectors.append((name, vec))

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

print("=" * 50)
print("FINAL RESULTS")
print("=" * 50)

labels = {"JD1": "JD-1 — Kakao (ML Engineer)",
          "JD2": "JD-2 — Naver (Backend Engineer)",
          "JD3": "JD-3 — Line (Frontend Engineer)"}

for jd_id, company, role, raw_skills in jds:
    jd_skills = normalize(raw_skills)
    jd_vec = [1.0 if vocab[i] in jd_skills else 0.0 for i in range(len(vocab))]
    scores = [(name, cosine(vec, jd_vec)) for name, vec in tfidf_vectors]
    scores.sort(key=lambda x: (-round(x[1], 2), x[0]))
    top3 = scores[:3]
    print(f"\n{labels[jd_id]}")
    print(", ".join(f"{n}({s:.2f})" for n, s in top3))
