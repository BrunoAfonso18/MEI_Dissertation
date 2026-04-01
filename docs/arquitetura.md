# Arquitetura do Sistema - Versão 1.0 (MVP)

**Projeto:** Expressão Gráfica em Tempo Real de Sentimentos Difusos Baseados em Aspetos  
**Autor:** Bruno de Sousa Afonso  
**Versão:** 1.0 (MVP - Março/Abril 2026)  
**Estado:** Em desenvolvimento (Sprint 0–1)

## 1. Visão Geral

O sistema implementa um pipeline completo de **Análise de Sentimentos Baseada em Aspetos (ABSA)** com **modelação difusa** e **visualização em tempo real**, conforme definido na pré-dissertação.

A arquitetura segue uma abordagem **modular em camadas**, permitindo desenvolvimento incremental, testes unitários e fácil substituição de componentes.

### Diagrama de Arquitetura

```mermaid
flowchart TD
    A[Reviews Textuais\n(JSON / Streaming)] 
    --> B[Módulo ABSA\n(Extração de Aspetos + Polaridade + Intensidade)]
    
    B --> C[Módulo Lógica Difusa\n(Fuzzificação + Agregação Multiaspeto + Defuzzificação)]
    
    C --> D[Camada de Persistência\n(Data Warehouse + Redis Cache)]
    
    D <--> E[Dashboard em Tempo Real\n(Visualização Interativa)]
    
    subgraph "Tempo Real"
        F[Simulador de Streaming\n(ou Kafka/Redis Streams)]
    end
    
    F --> A