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

Foram usadas BIO Tags para codificar o dataset para fine tune do modelo. Foi uma escolha entre o método BIO e BILOU, visto que os outputs sao semelhantes foi escolhido BIO por simplicidade.

O modelo não está a ter a performance esperada e o modelo utilizada pelo sent. analys. vis apenas analisa sentimentos ao nível da frase e não por aspetos. O sistema de backend e bd ja estao operacionais!

É necessário melhorar a performance do modelo antes de passar à próxima fase (fuzzy logic)

- Prompt 1
Quero adicionar o mapa de portugal para drill down por distritos na pagina principal. O que sugeres que use para o fazer. O mapa deve estar bem formatado e deve mostrar uma overview de todos os distritos com o numero de reviews e percentagem positiva negativa e neutra quando passar o rato por cima de cada um. O mapa nao deve ser suscetivel aos filtros de distritos apenas aos outros.

- Prompt 2
Preciso de melhorar a performance do modelo nomeadamente nos dados de treino. Preciso que adiciones ao dataset de treino final mais reviews nomeadamente com mais de 2 aspetos por review e termos de opiniao e termos aspeto com maiss de uma palavra que é o que esta a falhar no modelo neste momento. De seguida da me os passos para treinar o modelo desde o inicio e acrescenta esses passos numa secção do Readme

- Prompt 3
Preciso de implementar no modelo também um mecanismo para detetar ironias e negações.

- Prompt 4 
Preciso de melhorar a classificação das categorias de aspeto. Primeiro preciso de ter uma categoria chamada General para oss termos de aspeto que nao se encuadrarem em nenhuma das categorias pre selecionadas. De seguida preciso de implementar um fuzzy comparitor para que palavras semelhantes ou derivadas sejam corretaMENTE categorizadas (ex: serviço, servente etc...)
