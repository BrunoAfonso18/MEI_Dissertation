# Arquitetura do Sistema - Versão 1.0 (MVP)

**Projeto:** Expressão Gráfica em Tempo Real de Sentimentos Difusos Baseados em Aspetos  
**Autor:** Bruno de Sousa Afonso  
**Versão:** 1.0 (MVP - Março/Abril 2026)  
**Estado:** Em desenvolvimento (Sprint 0–1)

## 1. Visão Geral

O sistema implementa um pipeline completo de **Análise de Sentimentos Baseada em Aspetos (ABSA)** com **modelação difusa** e **visualização em tempo real**, conforme definido na pré-dissertação.

A arquitetura segue uma abordagem **modular em camadas** como apresentado abaixo.

### Diagrama de Arquitetura
```mermaid
flowchart TD
    A[Reviews Textuais<br/>Streaming ou Batch] --> B[ABSA Module<br/>Extração de aspetos, polaridade e intensidade]
    B --> C[Fuzzy Logic Module<br/>Fuzzificação + Agregação Multiaspeto]
    C --> D[Camada de Persistência<br/>Data Warehouse + Redis Cache]
    D <--> E[Dashboard em Tempo Real]

    subgraph "Tempo Real"
        F[Streaming / Redis Streams]
    end
    F --> A
```