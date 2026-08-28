# MEI_Dissertation

Sistema de análise de sentimentos baseada em aspetos (ABSA) para reviews de
restaurantes, com uma camada de lógica difusa para produzir um score de
sentimento contínuo e um dashboard em Dash para exploração analítica dos
resultados (Data Warehouse em esquema estrela, filtros dinâmicos,
drill-downs por distrito/restaurante/categoria/aspeto).

- `app/absa_module` — construção do dataset BIO, treino e avaliação do
  modelo de extração de aspetos/opiniões (`xlm-roberta-base` fine-tuned).
- `app/backend` — API FastAPI: pipeline de inferência (extração → correção
  de negação/ironia → lógica difusa → categorização) e endpoints analíticos
  sobre o Data Warehouse.
- `app/frontend` — dashboard Dash (submissão de reviews, Visão Geral,
  Categorias, Aspetos & Tendência, Restaurantes).

## Correr o projeto

```bash
cd app
docker compose up --build
```

Dashboard em `http://localhost:8050`, API em `http://localhost:8000`.

## Negação e ironia/sarcasmo

O classificador de token avalia cada termo isoladamente, pelo que falha em
casos como "o serviço não foi nada simpático" (negação) ou elogios ditos a
doer ("ah, claro, o empregado foi *super* rápido..." depois de descrever 40
minutos de espera). `app/backend/models/linguistic_adjustments.py`
implementa uma camada de pós-processamento baseada em regras — não um
modelo aprendido, dado que não existe corpus anotado de negação/ironia em
português para este domínio — aplicada entre a extração de aspetos e a
fuzzificação (`main.py::_analyse_text`):

- **Negação**: procura marcadores de negação portugueses ("não", "nunca",
  "nem", "sem", "nada", ...) numa janela de até 4 palavras antes do termo
  de opinião (ou de aspeto, na sua ausência), sem atravessar pontuação
  forte. Se encontrada **e a confiança do modelo for inferior a 0.8**,
  inverte a polaridade positiva/negativa e reduz ligeiramente a confiança;
  acima desse limiar assume-se que o modelo (que vê a frase inteira, não só
  o termo isolado) já resolveu a negação corretamente por si só — testado
  em casos reais como "o serviço não foi nada simpático", onde o modelo já
  prevê "negative" com confiança 0.91 sem qualquer ajuda heurística, e uma
  inversão cega aqui produziria uma dupla negação incorreta. Inclui uma
  exceção para a construção de superlativo "nunca ... tão/tanto" (ex.
  "nunca esteve tão boa"), que não é uma negação apesar de conter "nunca".
- **Ironia**: sinaliza expressões idiomáticas de sarcasmo comuns em
  português ("claro que", "para variar", "como sempre", ...), aspas de
  distanciamento à volta do termo de opinião, e o padrão de contradição
  (termo de opinião positivo previsto numa frase com forte carga negativa
  no resto do texto). Só inverte a polaridade quando esta é "positive"
  (elogio sarcástico → crítica real é, de longe, o padrão mais comum em
  reviews de restaurantes; o inverso não é tratado, para não arriscar
  inverter queixas genuínas que só por acaso partilham uma das expressões).

Estes dois indicadores (`negation_detected`, `irony_detected`) são
devolvidos por aspeto em `POST /query`.

## Notas soltas

```bash
docker system prune -a --volumes
```

```bash
.\venv\Scripts\activate.bat
```
