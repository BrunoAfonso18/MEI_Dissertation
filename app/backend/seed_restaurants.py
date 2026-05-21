"""
Run once to populate the dim_restaurant table with Portuguese restaurants.
Usage (from app/backend/):
    python seed_restaurants.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from database.connection import SessionLocal, engine
from database.models import Base, DimRestaurant

RESTAURANTS = [
    {
        "id_restaurant": 1,
        "name": "O Bacalhau d'Ouro",
        "district": "Lisboa",
        "category": "Marisqueira",
        "address": "Av. da Liberdade, 120, Lisboa",
        "inspection_grade": "A",
    },
    {
        "id_restaurant": 2,
        "name": "Pizzeria Roma",
        "district": "Porto",
        "category": "Italiana",
        "address": "Rua de Santa Catarina, 45, Porto",
        "inspection_grade": "B",
    },
    {
        "id_restaurant": 3,
        "name": "Sushi Kyoto",
        "district": "Lisboa",
        "category": "Japonesa",
        "address": "Rua Garrett, 10, Lisboa",
        "inspection_grade": "A+",
    },
    {
        "id_restaurant": 4,
        "name": "A Tasca do Zé",
        "district": "Braga",
        "category": "Portuguesa",
        "address": "Rua do Souto, 22, Braga",
        "inspection_grade": "B",
    },
    {
        "id_restaurant": 5,
        "name": "Casa da Ribeira",
        "district": "Aveiro",
        "category": "Marisqueira",
        "address": "Rua João Mendonça, 5, Aveiro",
        "inspection_grade": "A",
    },
    {
        "id_restaurant": 6,
        "name": "Restaurante Solar",
        "district": "Évora",
        "category": "Alentejana",
        "address": "Praça do Giraldo, 8, Évora",
        "inspection_grade": "A",
    },
    {
        "id_restaurant": 7,
        "name": "O Churrasqueiro",
        "district": "Coimbra",
        "category": "Churrascaria",
        "address": "Av. Sá da Bandeira, 30, Coimbra",
        "inspection_grade": "B",
    },
    {
        "id_restaurant": 8,
        "name": "Verde e Saúde",
        "district": "Lisboa",
        "category": "Vegetariana",
        "address": "Rua do Loreto, 15, Lisboa",
        "inspection_grade": "A+",
    },
    {
        "id_restaurant": 9,
        "name": "Hamburgueria Lisboeta",
        "district": "Lisboa",
        "category": "Hamburguer",
        "address": "Av. Almirante Reis, 55, Lisboa",
        "inspection_grade": "B",
    },
    {
        "id_restaurant": 10,
        "name": "Quinta da Palha",
        "district": "Sintra",
        "category": "Fine Dining",
        "address": "Estrada da Pena, 2, Sintra",
        "inspection_grade": "A+",
    },
    {
        "id_restaurant": 11,
        "name": "Mar e Sol",
        "district": "Faro",
        "category": "Marisqueira",
        "address": "Rua de Santo António, 12, Faro",
        "inspection_grade": "A",
    },
    {
        "id_restaurant": 12,
        "name": "Cervejaria Central",
        "district": "Porto",
        "category": "Portuguesa",
        "address": "Rua do Bonjardim, 100, Porto",
        "inspection_grade": "B",
    },
    {
        "id_restaurant": 13,
        "name": "Cantinho Algarvio",
        "district": "Faro",
        "category": "Algarvia",
        "address": "Av. Tomás Cabreira, 78, Portimão",
        "inspection_grade": "A",
    },
    {
        "id_restaurant": 14,
        "name": "Taberna do Norte",
        "district": "Viana do Castelo",
        "category": "Portuguesa",
        "address": "Rua Grande, 33, Viana do Castelo",
        "inspection_grade": "B",
    },
    {
        "id_restaurant": 15,
        "name": "Restaurante Douro",
        "district": "Porto",
        "category": "Portuguesa",
        "address": "Av. de Diogo Leite, 22, Vila Nova de Gaia",
        "inspection_grade": "A",
    },
]


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        added = 0
        for r in RESTAURANTS:
            exists = db.query(DimRestaurant).filter(
                DimRestaurant.id_restaurant == r["id_restaurant"]
            ).first()
            if not exists:
                db.add(DimRestaurant(**r))
                added += 1
        db.commit()
        print(f"Seed concluído: {added} restaurante(s) inserido(s), "
              f"{len(RESTAURANTS) - added} já existia(m).")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
