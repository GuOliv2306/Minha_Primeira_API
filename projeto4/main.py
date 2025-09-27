# API Framework
from fastapi import FastAPI, HTTPException

from fastapi.responses import HTMLResponse

# Manipulação de dados (se necessário)
from src.sanitize_df import *
from src.optimize_simulator_endpoints import *
import pandas as pd
import numpy as np
from pathlib import Path
# Para execução da API
import uvicorn

df = pd.read_csv('https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv')


DOCS_PATH = Path(__file__).resolve().parent / "docs" / "api_documentation.html"

try:
    DOCUMENTATION_HTML = DOCS_PATH.read_text(encoding="utf-8")
except Exception:
    DOCUMENTATION_HTML = (
        "<h1>Documentação indisponível</h1>"
        "<p>Não foi possível carregar o arquivo <code>api_documentation.html</code>.</p>"
    )




app = FastAPI(title="API de Extração e Análise - FGV Comunicações")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTMLResponse(content=DOCUMENTATION_HTML)


@app.get("/passengers")
def list_passengers(limit: int = 50, offset: int = 0):
    total = len(df)
    sliced_df = df.iloc[offset: offset + limit]
    records = sanitize_records(sliced_df)
    return {"total": total, "count": len(records), "data": records}


@app.get("/passengers/statistics")
def get_passenger_statistics():
    return {
        "total_passengers": int(len(df)),
        "survival_rate": float(df["Survived"].mean())
    }


@app.get("/passengers/by-class")
def get_passengers_by_class():
    return df["Pclass"].value_counts().to_dict()


@app.get("/passengers/age-distribution")
def get_passengers_age_distribution():
    return df["Age"].describe().to_dict()


@app.get("/passengers/survival-heatmap")
def get_passengers_survival_heatmap():
    return df.groupby("Pclass")["Survived"].mean().to_dict()


@app.get("/passenger/{passenger_id}")
def get_passenger(passenger_id: int):
    passenger_df = df[df["PassengerId"] == passenger_id]
    if passenger_df.empty:
        raise HTTPException(status_code=404, detail="Passageiro não encontrado")
    return sanitize_records(passenger_df)[0]


# Instância global do otimizador
optimizer = EvacuationOptimizer()


@app.post("/api/v1/optimize/evacuation-plan", response_model=EvacuationPlan)
async def optimize_evacuation_plan(request: EvacuationRequest):
    """
    Otimiza plano de evacuação do Titanic baseado nos dados dos passageiros
    
    Este endpoint usa algoritmos de otimização e simulação Monte Carlo para:
    - Determinar a melhor ordem de evacuação
    - Atribuir passageiros aos botes salva-vidas
    - Calcular tempos estimados de evacuação
    - Avaliar equidade e eficiência do plano
    """
    try:
        # Carrega os dados reais do Titanic
        # Em produção, você carregaria do banco de dados
        try:
            passengers_data = pd.read_csv('https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv')
        except:
            # Fallback para dados simulados se não conseguir carregar
            passengers_data = pd.DataFrame({
                'PassengerId': range(1, 101),
                'Survived': np.random.choice([0, 1], 100),
                'Pclass': np.random.choice([1, 2, 3], 100, p=[0.24, 0.21, 0.55]),
                'Name': [f'Passenger_{i}' for i in range(100)],
                'Sex': np.random.choice(['male', 'female'], 100, p=[0.65, 0.35]),
                'Age': np.random.normal(30, 12, 100),
                'SibSp': np.random.poisson(0.5, 100),
                'Parch': np.random.poisson(0.4, 100),
                'Ticket': [f'TICKET_{i}' for i in range(100)],
                'Fare': np.random.lognormal(3, 1, 100),
                'Cabin': [None] * 100,
                'Embarked': np.random.choice(['C', 'Q', 'S'], 100, p=[0.2, 0.1, 0.7])
            })
        
        # Otimiza o plano de evacuação
        evacuation_plan = optimizer.optimize_evacuation(request, passengers_data)

        return sanitize_data(evacuation_plan)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na otimização: {str(e)}")


@app.get("/api/v1/optimize/passengers/summary")
async def get_passengers_summary():
    """Retorna resumo dos dados dos passageiros do Titanic"""
    try:
        passengers_data = pd.read_csv('https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv')
        
        summary = {
            "total_passengers": len(passengers_data),
            "by_class": passengers_data['Pclass'].value_counts().to_dict(),
            "by_gender": passengers_data['Sex'].value_counts().to_dict(),
            "by_embarked": passengers_data['Embarked'].value_counts().to_dict(),
            "survival_rate": passengers_data['Survived'].mean(),
            "age_stats": {
                "mean": passengers_data['Age'].mean(),
                "min": passengers_data['Age'].min(),
                "max": passengers_data['Age'].max(),
                "missing": passengers_data['Age'].isna().sum()
            },
            "fare_stats": {
                "mean": passengers_data['Fare'].mean(),
                "min": passengers_data['Fare'].min(),
                "max": passengers_data['Fare'].max(),
                "missing": passengers_data['Fare'].isna().sum()
            },
            "families": {
                "with_siblings_spouse": (passengers_data['SibSp'] > 0).sum(),
                "with_parents_children": (passengers_data['Parch'] > 0).sum(),
                "alone": ((passengers_data['SibSp'] == 0) & (passengers_data['Parch'] == 0)).sum()
            }
        }
        
        return sanitize_data(summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao carregar dados: {str(e)}")


@app.get("/api/v1/optimize/historical-survival")
async def get_historical_survival():
    """Retorna dados históricos de sobrevivência por diferentes categorias"""
    try:
        passengers_data = pd.read_csv('https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv')
        
        survival_data = {
            "by_class": passengers_data.groupby('Pclass')['Survived'].agg(['mean', 'count']).to_dict('index'),
            "by_gender": passengers_data.groupby('Sex')['Survived'].agg(['mean', 'count']).to_dict('index'),
            "by_age_group": passengers_data.assign(
                AgeGroup=pd.cut(passengers_data['Age'], 
                              bins=[0, 12, 18, 30, 50, 100], 
                              labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Elder'],
                              include_lowest=True)
            ).groupby('AgeGroup')['Survived'].agg(['mean', 'count']).to_dict('index'),
            "by_embarked": passengers_data.groupby('Embarked')['Survived'].agg(['mean', 'count']).to_dict('index'),
            "family_vs_alone": passengers_data.assign(
                FamilyStatus=passengers_data.apply(lambda x: 'With Family' if x['SibSp'] + x['Parch'] > 0 else 'Alone', axis=1)
            ).groupby('FamilyStatus')['Survived'].agg(['mean', 'count']).to_dict('index')
        }
        
        return sanitize_data(survival_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar dados históricos: {str(e)}")


@app.get("/api/v1/optimize/evacuation-plan/constraints-info")
async def get_constraints_info():
    """Retorna informações sobre as opções de configuração do otimizador"""
    return {
        "priority_systems": {
            "women_children_first": "Prioriza mulheres e crianças (protocolo histórico)",
            "class_based": "Prioriza por classe social",
            "age_based": "Prioriza por idade (mais jovens e idosos primeiro)",
            "disability_first": "Prioriza pessoas com mobilidade reduzida",
            "optimal_survival": "Baseado em análise de dados históricos de sobrevivência"
        },
        "optimization_targets": {
            "max_survivors": "Maximiza número de sobreviventes",
            "min_time": "Minimiza tempo total de evacuação",
            "fairness": "Maximiza equidade entre diferentes grupos",
            "balanced": "Equilibra todos os fatores"
        },
        "lifeboat_info": {
            "total_boats": len(optimizer.lifeboat_capacities),
            "total_capacity": sum(optimizer.lifeboat_capacities.values()),
            "regular_boats": 16,
            "collapsible_boats": 4
        }
    }


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="localhost", port=8014)
    except Exception as e:
        print(f"Erro ao iniciar o servidor: {str(e)}")

