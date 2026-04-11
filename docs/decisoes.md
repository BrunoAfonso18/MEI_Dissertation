# Tecnologias e Ferramentas

- Linguagem: Python 3.11+
- Backend API: FastAPI (opcional no MVP – pode começar só com scripts)
- ABSA: Hugging Face Transformers + modelo multilingual
- Fuzzy Logic: scikit-fuzzy
- Base de Dados: PostgreSQL DW
- Cache/Real-time: Redis
- Frontend: React
- Visualização: Plotly + custom components para representação difusa


## ABSA Module

>SemEval-2014 EN (all 3 labels)
        ↓ auto-translate to PT
SemEval-2016 PT (aspect + polarity) ← merge/augment
        ↓ fine-tune
XLM-RoBERTa or BERTimbau
        ↓ evaluate on
your own held-out PT reviews

**For this step in this module theres a couple of things that must be done. First manually adding opinion terms to the complete reviews dataset. After that convert the all dataset into BIO format by tokens. Than fine tune the model to correctly identify sentiment labels.**